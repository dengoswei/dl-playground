
import json
import sys
import tensorflow as tf
import numpy as np
import read_data

flags = tf.app.flags

# - max_sent_size
# - max_ques_size
# // - max_word_size

flags.DEFINE_string("shared_data", "./shared.json", "[word2vec, word2idx")
# config.word_vocab_size
# config.word_emb_size

def simple_cross_entropy(y, a):
    return -tf.reduce_sum(y * tf.log(a))

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

        predict = tf.greater(p, 0.5)
        self.correct = tf.equal(predict, self.y)



    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_feed_dict(self, mq, mp, ml):
        assert len(mq) == len(mp)
        assert len(mq) == len(ml)
        assert len(mp) == self.config.batch_size

        batch_size = len(mq)
        config = self.config
        x = np.zeros(
            [batch_size, config.max_sent_size], dtype='int32')
        x_mask = np.zeros(
            [batch_size, config.max_sent_size], dtype='bool')
        q = np.zeros(
            [batch_size, config.max_ques_size], dtype='int32')
        q_mask = np.zeros(
            [batch_size, config.max_ques_size], dtype='bool')

        y = np.zeros([batch_size], dtype='bool')
        feed_dict = {}
        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.q] = q
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.y] = y

        for i, xi in enumerate(mp):
            for j, xij in enumerate(xi):
                x[i, j] = xij

        for i, qi in enumerate(mq):
            for j, qij in enumerate(qi):
                q[i, j] = qij

        for i, yi in enumerate(ml):
            assert (0 == yi) or (2 == yi)
            y[i] = 2 == yi

        return feed_dict

def random_split(train_data, batch_size):
    batches = []


def update_config(config, q, p):
    # max_ques_size
    config.max_ques_size = max(map(len, q))
    config.max_sent_size = max(map(len, p))

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
    config.word_emb_size = 200
    config.word_vocab_size = 200000
    config.num_steps = 1000
    config.batch_size = 50

    word2vec, word2idx = read_data.load_word_embedding(
            "./data/ready.train.1.json", config.word_vocab_size)
    print len(word2vec), len(word2idx)
    q, p, l = read_data.load_sougou_data("./data/ready.train.1.json", word2idx)
    q, p, l = read_data.filter_sougou_data(q, p, l)
    print len(q), len(p), len(l)

    update_config(config, q, p)

    vq, vp, vl, q, p, l = read_data.split_data(q, p, l, 0.2)
    print len(vq), len(vp), len(vl), len(q), len(p), len(l)

    idx2vec_dict = { word2idx[w]: vec
            for w, vec in word2vec.items() if w in word2idx }
    print len(idx2vec_dict)
#    emb_mat = np.array([np.zeros(config.word_emb_size) for idx in xrange(config.word_vocab_size)])
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict else np.zeros(config.word_emb_size) for idx in xrange(config.word_vocab_size)])
    config.emb_mat = emb_mat

    model = Model(config)

    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(model.get_loss())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        for mq, mp, ml in read_data.get_mini_batches(
                q, p, l, config.num_steps, config.batch_size):
            assert len(mq) == len(mp)
            assert len(mq) == len(ml)
            feed_dict = model.get_feed_dict(mq, mp, ml)
            loss, train_res = sess.run(
			[model.get_loss(), train_op],
                        feed_dict=feed_dict)
            print "loss", loss
            step += 1
            if 0 == step % 1000:
                cors = []
                try_cnt = 0
                for vmq, vmp, vml in read_data.get_mini_batches(
                        vq, vp, vl, 1, config.batch_size):
                    feed_dict = model.get_feed_dict(vmq, vmp, vml)
                    correct = sess.run([model.correct], feed_dict=feed_dict)
                    cors.extend(correct)
                    try_cnt += 1
                    if 10 < try_cnt:
                        break
                accuracy = np.mean(cors)
                print "step: ", step, " accuracy: ", accuracy, " / ", len(cors)

if __name__ == "__main__":
    main()



