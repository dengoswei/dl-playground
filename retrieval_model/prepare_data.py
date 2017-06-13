import os
import json
import functools
import itertools
import tensorflow as tf
import numpy as np


tf.flags.DEFINE_integer(
        "min_word_frequency", 5, "mininum frequency of words in vocab")

tf.flags.DEFINE_integer(
        "max_sentence_len", 200, "maximum sentence length")

tf.flags.DEFINE_string(
        "input_dir", os.path.abspath("./data"),
        "input directory")

tf.flags.DEFINE_string(
        "output_dir", os.path.abspath("./data"),
        "output directory")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.json")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "valid.json")


def tokenizer_fn(iterator):
    return (x.split(" ") for x in iterator)

def create_vocab(input_iter, min_frequency):
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
            FLAGS.max_sentence_len,
            min_frequency=min_frequency,
            tokenizer_fn=tokenizer_fn)
    vocab_processor.fit(input_iter)
    return vocab_processor


def create_json_iter(filename):
    with open(filename) as jsonfile:
        data = json.load(jsonfile)
        assert len(data['q']) == len(data['p'])
        assert len(data['q']) == len(data['l'])
        for idx in xrange(len(data['q'])):
            if 1 == int(data['l'][idx]):
                continue
            if 200 < len(data['p'][idx].split()):
                continue
            yield (data['q'][idx].encode('utf-8'),
                    data['p'][idx].encode('utf-8'), int(data['l'][idx]))

def write_vocabulary(vocab_processor, outfile):
    vocab_size = len(vocab_processor.vocabulary_)
    with open(outfile, 'w') as vocabfile:
        for id in xrange(vocab_size):
            word = vocab_processor.vocabulary_._reverse_mapping[id]
            vocabfile.write(word + "\n")
    print("save vocabulary to {}".format(outfile))

def transform_sentence(sequence, vocab_processor):
    return next(vocab_processor.transform([sequence])).tolist()

def create_example_train(row, vocab):
    query, document, label = row
    query_transformed = transform_sentence(query, vocab)
    document_transformed = transform_sentence(document, vocab)
    # TODO: next ???
    query_len = len(next(vocab._tokenizer([query])))
    document_len = len(next(vocab._tokenizer([document])))
    label = int(label)

    example = tf.train.Example()
    example.features.feature[
            "query"].int64_list.value.extend(query_transformed)
    example.features.feature[
            "document"].int64_list.value.extend(document_transformed)
    example.features.feature[
            "query_len"].int64_list.value.extend([query_len])
    example.features.feature[
            "document_len"].int64_list.value.extend([document_len])
    example.features.feature[
            "label"].int64_list.value.extend([label])
    return example


def create_tfrecords_file(input_filename, output_filename, example_fn):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for i, row in enumerate(create_json_iter(input_filename)):
        x = example_fn(row)
        writer.write(x.SerializeToString())
    writer.close()
    print("writtern {}".format(output_filename))


if __name__ == "__main__":
    input_iter = create_json_iter(TRAIN_PATH)
    input_iter = (x[0] + " " + x[1] for x in input_iter)
    vocab = create_vocab(
            input_iter, min_frequency=FLAGS.min_word_frequency)
    print("vocabulary size: {}".format(len(vocab.vocabulary_)))

    # create vocabulary txt file
    write_vocabulary(
            vocab, os.path.join(FLAGS.output_dir, "vocabulary.txt"))

    # save vocab processor
    vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

    # create validation.tfrecords
    create_tfrecords_file(
            input_filename=VALIDATION_PATH,
            output_filename=os.path.join(
                FLAGS.output_dir, "validataion.tfrecords"),
                example_fn=functools.partial(
                    create_example_train, vocab=vocab))

    # create train.tfrecords
    create_tfrecords_file(
            input_filename=TRAIN_PATH,
            output_filename=os.path.join(
                FLAGS.output_dir, "train.tfrecords"),
            example_fn=functools.partial(
                create_example_train, vocab=vocab))


