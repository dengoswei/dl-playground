import os
import time
import tensorflow as tf
import numpy as np
import read_data
import new_hparams
import model


tf.flags.DEFINE_string(
        "input_dir", "./data", "input data (tfrecords)")
tf.flags.DEFINE_string(
        "model_dir", None, "store model checkpoints (./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "# of training epochs.")
tf.flags.DEFINE_integer(
        "eval_every", 2000, "evaluate every # train steps")

FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())
if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    MODEL_DIR = os.path.abspath(
            os.path.join("./runs", str(TIMESTAMP)))

TRAIN_FILE = os.path.abspath(
        os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(
        os.path.join(FLAGS.input_dir, "validation.tfrecords"))
tf.logging.set_verbosity(FLAGS.loglevel)


def create_evaluation_metrics():
    eval_metrics = {}
    eval_metrics['accuracy'] = tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)

def main(unused_argv):
    hparams = new_hparams.create_hparams()
    model_fn = model.create_model_fn(
            hparams, model_impl=model.birnn_att_model)

    estimator = tf.contrib.learn.Estimator(
            model_fn=model_fn,
            model_dir=MODEL_DIR,
            config=tf.contrib.learn.RunConfig())

    input_fn_train = read_data.create_input_fn(
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            input_files=[TRAIN_FILE],
            batch_size=hparams.batch_size,
            num_epochs=FLAGS.num_epochs)

    input_fn_eval = read_data.create_input_fn(
            mode=tf.contrib.learn.ModeKeys.EVAL,
            input_files=[VALIDATION_FILE],
            batch_size=hparams.eval_batch_size,
            num_epochs=1)

    eval_metrics = create_evaluation_metrics()
    eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=input_fn_eval,
            every_n_steps=FLAGS.eval_every,
            metrics=eval_metrics)
    # TODO: add eval_monitor metrics
    estimator.fit(
            input_fn=input_fn_train,
            steps=None,
            monitors=None)
            #monitors=[eval_monitor])
    # ?

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = 2
    tf.app.run()


