import tensorflow as tf
from collections import namedtuple


tf.flags.DEFINE_integer(
        "vocab_size",
        200000, "size of the vocabulary")

tf.flags.DEFINE_integer(
        "embedding_dim", 100, "dimensionality of embedding")
tf.flags.DEFINE_integer(
        "gru_dim", 128, "dimensionality of the gru cell")
tf.flags.DEFINE_integer(
        "max_query_len", 30, "truncate query len to ")
tf.flags.DEFINE_integer(
        "max_document_len", 200, "truncate document len to ")
tf.flags.DEFINE_string(
        "vocab_path", None, "path to vocabulary.txt file")

# training parameters:
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_integer("batch_size", 32, "batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 128,
        "batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam",
        "optimizer name (Adam, Adagrad, etc)")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
        "HParams",
        [
            'batch_size',
            'embedding_dim',
            'eval_batch_size',
            'learning_rate',
            'max_query_len',
            'max_document_len',
            'optimizer',
            'gru_dim',
            'vocab_size',
            'vocab_path',
        ])

def create_hparams():
    return HParams(
            batch_size=FLAGS.batch_size,
            embedding_dim=FLAGS.embedding_dim,
            eval_batch_size=FLAGS.eval_batch_size,
            learning_rate=FLAGS.learning_rate,
            max_query_len=FLAGS.max_query_len,
            max_document_len=FLAGS.max_document_len,
            optimizer=FLAGS.optimizer,
            gru_dim=FLAGS.gru_dim,
            vocab_size=FLAGS.vocab_size,
            vocab_path=FLAGS.vocab_path)

