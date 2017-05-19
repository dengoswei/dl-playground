
import json
import sys
import tensorflow as tf
import numpy as np

flags = tf.app.flags

# - max_sent_size
# - max_ques_size
# // - max_word_size

flags.DEFINE_string("shared_data", "./shared.json", "[word2vec, word2idx")
# config.word_vocab_size
# config.word_emb_size

def simple_cross_entropy(y, a):
    return tf.reduce_mean(-y * tf.log(a) - (1-y) * tf.log(1-a))

def get_initializer(matrix):
    def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer

class Model(object):
    def __init__(self, config):
        self.config = config
	batch_size = config.batch_size

        self.global_step = tf.get_variable(
                'global_step', shape=[], dtype='int32',
                initializer=tf.constant_initializer(0), trainable=False)

        self.x = tf.placeholder(
                'int32', [batch_size, config.max_sent_size], name='x')
        # x_mask: actual sent size per sample
        self.x_mask = tf.placeholder(
                'bool', [batch_size, config.max_sent_size], name='x_mask')
        self.q = tf.placeholder(
                'int32', [batch_size, config.max_ques_size], name='q')
        # q_mask: actual ques size per sample
        self.q_mask = tf.placeholder(
                'bool', [batch_size, config.max_ques_size], name='q_mask')

        self.y = tf.placeholder('bool', [batch_size], name='y')

        self._build_model()

    def _build_model(self):
        #
        config = self.config
        word_vocab_size = config.word_vocab_size
        word_emb_size = config.word_emb_size
        max_sent_size = config.max_sent_size
        with tf.variable_scope("emb"), tf.device("/cpu:0"):
            word_emb_mat = tf.get_variable(
                    "word_emb_mat", dtype='float',
                    shape=[word_vocab_size, word_emb_size],
                    initializer=get_initializer(config.emb_mat))
            xx = tf.nn.embedding_lookup(word_emb_mat, self.x)
            qq = tf.nn.embedding_lookup(word_emb_mat, self.q)

        # TODO: add highway network ?? ref bidaf
        cell = tf.contrib.rnn.GRUCell(word_emb_size)
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 1)
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)

        # TODO: add drop-out
        with tf.variable_scope("bigru"):
            # query: query with drop out ??? bi-daf
            # [-, max_ques_size, word_emb_size]
            _, (fw_f_u, bw_f_u) = tf.nn.bidirectional_dynamic_rnn(
                        cell, cell, qq, q_len, dtype='float', scope='gru')
            # TODO: check concat ???
            # [-, max_ques_size, 2 * word_emb_size]
            u = tf.concat([fw_f_u, bw_f_u], 1)

            # context
            tf.get_variable_scope().reuse_variables()
            (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell, cell, xx, x_len, dtype='float', scope='gru')
            # TODO: check concat ???
            # [-, max_sent_size, 2 * word_emb_size]
            h = tf.concat([fw_h, bw_h], 2)

        with tf.variable_scope("att"):
            u_norm = tf.nn.l2_normalize(u, 1)
            h_norm = tf.nn.l2_normalize(
                    tf.reshape(h, [-1, 2 * word_emb_size]), 1)

            w = tf.get_variable("weight1",
                    [2 * word_emb_size, 2 * word_emb_size],
                    initializer=tf.random_normal_initializer(stddev=1.0))
            b = tf.get_variable("bias1",
                    [2 * word_emb_size],
                    initializer=tf.random_normal_initializer(stddev=1.0))

            q_repr = tf.matmul(u_norm, w) + b
            q_repr_tile = tf.reshape(
                    tf.tile(q_repr, [1, max_sent_size]), [-1, 2 * word_emb_size])
            att = tf.reduce_sum(tf.multiply(q_repr_tile, h_norm), 1)
            N = att.shape.dims[0].value
            s = tf.reduce_sum(
                    tf.reshape(
                        tf.pow(att, 4 * tf.ones([N], tf.float32)), [-1, max_sent_size]), 1)

        w2 = tf.get_variable("weight2", [1],
            initializer=tf.random_normal_initializer(stddev=1.0))
        b2 = tf.get_variable("bias2", [1],
            initializer=tf.random_normal_initializer(stddev=1.0))
        p = tf.sigmoid(w2 * s + b2)

        self.loss = simple_cross_entropy(tf.cast(self.y, 'float'), p)


    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_feed_dict(self, batch):
        config = self.config

        x = np.zeros(
            [config.batch_size, config.max_sent_size], dtype='int32')
        x_mask = np.zeros(
            [config.batch_size, config.max_sent_size], dtype='bool')
        q = np.zeros(
            [config.batch_size, config.max_ques_size], dtype='int32')
        q_mask = np.zeros(
            [config.batch_size, config.max_ques_size], dtype='bool')

        y = np.zeros([config.batch_size], dtype='bool')

        feed_dict = {}
        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.q] = q
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.y] = y

        # x
        for i, xi in enumerate(batch['X']):
            for j, xij in enumerate(xi):
                x[i, j] = xij
        # q
        for i, qi in enumerate(batch['Q']):
            for j, qij in enumerate(qi):
                q[i, j] = qij
        # y
        for i, yi in enumerate(batch['Y']):
            y[i] = yi

        return feed_dict


def random_split(train_data, batch_size):
    batches = []


def fake_config(config):
    # batch_size
    config.batch_size = 100
    config.word_vocab_size = 10
    config.word_emb_size = 100
    config.emb_mat = np.array(
            [np.random.multivariate_normal(
                np.zeros(config.word_emb_size),
                np.eye(config.word_emb_size))
                for idx in range(config.word_vocab_size)])

    config.max_ques_size = 15
    config.max_sent_size = 15


def fake_train_data(config):
    train_data = {}
    x = []
    q = []
    y = []
    for i in range(config.batch_size):
        size = np.random.randint(10, 15)
        x.append(np.random.randint(1, 10, size))
        q_size = np.random.randint(5, 11)
        q.append(np.random.randint(1, 10, q_size))

        if 0.8 <= np.random.ranf():
            y.append(True)
        else:
            y.append(False)
    train_data['X'] = x
    train_data['Q'] = q
    train_data['Y'] = y
    return train_data




def main():
    config = flags.FLAGS

    #shared_data = read_shared(config.shared_data)
    #word2vec_dict = shared['word2vec']
    #word2idx_dict = shared['word2idx']
    #idx2vec_dict = {\
    #        word2idx_dict[word]: \
    #            vec for word, vec in word2vec_dict.items() \
    #                if word in word2idx_dict }
    #emb_mat = np.array(
    #        [idx2vec_dict[idx] if idx in idx2vec_dict
    #            else np.random.multivariate_normal(
    #                np.zeros(config.word_emb_size),
    #                np.eye(config.word_emb_size))
    #            for idx in range(config.word_vocab_size)])
    #config.emb_mat = emb_mat
    # for test: tmp


    # TODO:
    fake_config(config)
    train_data = fake_train_data(config)

    model = Model(config)

    train_op = tf.train.GradientDescentOptimizer(0.005).minimize(model.get_loss())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(1000):
            feed_dict = model.get_feed_dict(train_data)
            loss, train_res = sess.run(
			[model.get_loss(), train_op],
                        feed_dict=feed_dict)
            print loss

if __name__ == "__main__":
    main()



