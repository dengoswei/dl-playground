# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf

def lstm_cell(
        num_units,
        dropout_rate,
        is_training,
        scope=None,
        reuse=None):
    with tf.variable_scope(scope):
        cell = tf.contrib.rnn.LSTMCell(
                num_units, reuse=reuse)
        if is_training and 0 < dropout_rate:
            cell = tf.contrib.rnn.DropoutWrapper(
                    cell, input_keep_prob=1-dropout_rate)
        return cell

def lstm_stack(
        num_units,
        dropout_rate,
        is_training,
        reuse=None):
    return tf.contrib.rnn.MultiRNNCell(
            [lstm_cell(num_units,
                dropout_rate, is_training,
                "lstm_cell_{}".format(0),
                reuse=reuse),
            lstm_cell(num_units,
                0, is_training,
                "lstm_cell_{}".format(1),
                reuse=reuse)])



class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                # TODO get_batch_data
                self.x, self.y, self.num_batch = get_batch_data()
            else:
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # TODO: define decoder input

            # TODO: encode vocab vs decode vocab
            encode2idx, idx2encode = load_encode_vocab()
            decode2idx, idx2decode = load_decode_vocab();

            x_len = tf.reduce_sum(tf.sign(self.x), 1)
            y_len = tf.reduce_sum(tf.sign(self.y), 1)

            with tf.variable_scope("encoder"):
                self.enc = embedding(
                        self.x, vocab_size=len(encode2idx),
                        num_units=hp.embedding_dim,
                        scale=False,
                        scope="encode_embed")

                # bi-LSTM -> drop-out -> bi-LSTM
                # hidden unit size = 600;
                cell = lstm_stack(
                        hp.hidden_units,
                        hp.dropout_rate,
                        is_training)
                (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell, cell, self.enc, x_len, dtype='float')

                # bt: use for decoder attention compute
                # => reduce: only; shape: N, max_sent_size, 2 * hidden_size
                bt = tf.concat(
                        [fw_h[-1, :, :, :], bw_h[-1, :, :, :]], -1)

                self.enc = tf.concat(
                        fw_h[:, :, -1, :], bw_h[:, :, 0, :], -1)

            with tf.variable_scope("decoder"):
                self.dec = embedding(
                        self.y, vocab_size=len(decode2idx),
                        num_units=hp.embedding_dim,
                        scale=False,
                        scope="decode_embed")

                # LSTM
                cell = lstm_stack(
                        hp.hidden_units,
                        hp.dropout_rate,
                        is_training)

                # N, max_ques_size, hidden_units
                h, _ = tf.nn.dynamic_rnn(
                        cell, self.y, y_len,
                        initial_state=self.enc, dtype='float')

            with tf.variable_scope("attention"):
                wb = tf.get_variable("wb",
                        [2 * hp.hidden_units, hp.hidden_units],
                        initializer=tf.truncated_normal_initializer(stddev=1.0))
                # att shape: N, max_ques_size, max_sent_size
                logits = tf.matmul(
                        h, tf.matmul(bt, tf.expand_dims(wb, 0)),
                        transpose_b=True)
                logits_masks = tf.sign(tf.abs(logits))
                # construct negative infi..
                paddings = tf.ones_like(logits_masks) * (-2**32+1)
                logits = tf.where(tf.equal(logits_masks, 0), paddings, logits)

                att = tf.nn.softmax(logits)
                att_masks = tf.sign(self.y)
                att_masks = tf.tranpose(att_masks, perm=[0, 2, 1])
                att_masks = tf.tile(att_masks, [1, 1, tf.shape(self.x)[-1]])
                paddings = tf.zeros(att)
                # N, max_ques_size, max_sent_size
                att = tf.where(tf.equal(att_masks, 0), paddings, att)

                # N, max_ques_size, 2 * hidden_size
                c = tf.matmul(att, bt)
                c_masks = tf.sign(self.y)
                c_masks = tf.tranpose(c_masks, perm=[0, 2, 1])
                c_masks = tf.tile(c_masks, [1, 1, tf.shape(self.y)[-1]])
                paddings = tf.zeros(c)
                c = tf.where(tf.equal(c_masks, 0), paddings, c)

            with tf.variable_scope("prob"):
                combine = tf.concat([h, c], 2)
                wt = tf.get_variable("wt",
                        [2 * hp.hidden_units, hp.hidden_units],
                        initializer=tf.truncated_normal_initializer(stddev=1.0))

                logits = tf.matmul(
                        combine, tf.expand_dims(wt, 0))
                # tanh(0) == 0 => so no masks..
                # N, max_ques_size, hidden_units
                logits = tanh(logits)

                ws = tf.get_variable("ws",
                        [hp.hidden_units, len(decode2idx)],
                        initializer=tf.truncated_normal_initializer(stddev=1.0))
                logits = tf.matmul(
                        logits, tf.expand_dims(ws, 0))

                # N, max_ques_size, len(decode2idx)
                probs = tf.nn.softmax(logits)

                preds = tf.argmax(probs, 2)
                if is_training:
                    flat_probs = tf.reshape(probs, [-1, len(decode2idx)])
                    indices = tf.range(tf.shape(flat_probs)[0])
                    indices = tf.concat([indices, tf.reshape(self.y, [-1, 1])], 1)

                    y_probs = tf.gather_nd(probs, indices)
                    y_probs = tf.where(
                            tf.equal(tf.reshape(self.y, [-1, 1]), 0),
                            tf.zeros(y_probs, dtype='float'), y_probs)
                    self.loss = tf.log(y_probs)
                    self.loss = -tf.reduce_sum(self.loss)
                else:





if __name__ == "__main__":

    g = Graph("train"); print("Graph loaded")

    sv = tf.train.Supervisor(
            graph=g.graph,
            logdir=hp.logdir, save_model_secs=0)

    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1):
            if sv.should_stop():
                break
            for step in tqdm(range(g.num_batch),
                    total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)

                gs = sess.run(g.global_step)
                sv.saver.save(sess,
                        hp.logdir +
                        "/model_epoch_%02d_gs_%d" % (epoch, gs))
    print("Done")

