import tensorflow as tf
import numpy as np


def get_embeddings(hparams):
    initializer = tf.random_normal_initializer(-0.25, 0.25)
    return tf.get_variable(
            "word_embbedings",
            shape=[hparams.vocab_size, hparams.embedding_dim],
            initializer=initializer)

def birnn_att_model(
        hparams,
        mode,
        document,
        document_len,
        query,
        query_len,
        targets):

    embeddings_W = get_embeddings(hparams)

    document_embedded = tf.nn.embedding_lookup(
            embeddings_W, document, name="embed_document")
    query_embedded = tf.nn.embedding_lookup(
            embeddings_W, query, name="embed_query")

    with tf.variable_scope("q_bigru"):
        cell = tf.contrib.rnn.GRUCell(hparams.gru_dim)
        _, (fw_f_u, bw_f_u) = tf.nn.bidirectional_dynamic_rnn(
                cell, cell,
                query_embedded, query_len,
                dtype='float', scope='gru')
        u = tf.concat([fw_f_u, bw_f_u], 1)

    with tf.variable_scope("d_bigru"):
        #tf.get_variable_scope().reuse_variables()
        dcell = tf.contrib.rnn.GRUCell(hparams.gru_dim)
        (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(
                dcell, dcell,
                document_embedded, document_len,
                dtype='float', scope='gru')
        h = tf.concat([fw_h, bw_h], 2)

    with tf.variable_scope('att'):
        u_norm = tf.nn.l2_normalize(u, 1)
        h_norm = tf.nn.l2_normalize(h, 2)
                # tf.reshape(h, [-1, 2 * hparams.embedding_dim]), 1)

        w = tf.get_variable("weight1",
                [2 * hparams.gru_dim, 2 * hparams.gru_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0))
        b = tf.get_variable("bias1",
                [2 * hparams.gru_dim],
                initializer=tf.truncated_normal_initializer(stddev=1.0))

        # TODO
        q_repr = tf.expand_dims(tf.matmul(u_norm, w) + b, 1)
        before_pow = tf.matmul(q_repr, h_norm, False, True)
        s = tf.reduce_sum(tf.square(tf.square(before_pow)), 2)
#        s = tf.squeeze(tf.reduce_sum(
#                tf.square(tf.square(before_pow)), 2), [2])

        w2 = tf.get_variable("weight2", [1],
                initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("bias2", [1],
                initializer=tf.random_normal_initializer())
        logits = w2 * s + b2
        probs = tf.sigmoid(logits)

        losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=tf.to_float(targets))

    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    return probs, mean_loss


def get_id_feature(features, key, len_key, max_len):
    ids = features[key]
    ids_len = tf.squeeze(features[len_key], [1])
    ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
    return ids, ids_len

def create_train_op(loss, hparams):
    train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=hparams.learning_rate,
            clip_gradients=10.0,
            optimizer=hparams.optimizer)
    return train_op


def create_model_fn(hparams, model_impl):

    def model_fn(features, targets, mode):
        document, document_len = get_id_feature(
                features,
                "document", "document_len", hparams.max_document_len)

        query, query_len = get_id_feature(
                features,
                "query", "query_len", hparams.max_query_len)

        batch_size = targets.get_shape().as_list()[0]

        if tf.contrib.learn.ModeKeys.TRAIN == mode:
            probs, loss = model_impl(
                    hparams, mode,
                    document, document_len,
                    query, query_len, targets)

            train_op = create_train_op(loss, hparams)
            return probs, loss, train_op

        if tf.contrib.learn.ModeKeys.EVAL == mode:
            probs, loss = model_impl(
                    hparams, mode,
                    document, document_len,
                    query, query_len, targets)
            return probs, loss, None

        assert False

    return model_fn
