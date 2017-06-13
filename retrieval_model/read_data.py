import tensorflow as tf


# TODO: ? value fit for sougou data
TEXT_FEATURE_SIZE = 200

def get_feature_columns(mode):
    feature_columns = []

    feature_columns.append(
            tf.contrib.layers.real_valued_column(
                column_name="query",
                dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
    feature_columns.append(
            tf.contrib.layers.real_valued_column(
                column_name="query_len",
                dimension=1, dtype=tf.int64))
    feature_columns.append(
            tf.contrib.layers.real_valued_column(
                column_name="document",
                dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
    feature_columns.append(
            tf.contrib.layers.real_valued_column(
                column_name="document_len",
                dimension=1, dtype=tf.int64))
    #if mode == tf.contrib.learn.ModeKeys.TRAIN:
    feature_columns.append(
            tf.contrib.layers.real_valued_column(
                column_name="label",
                dimension=1, dtype=tf.int64))
    # TODO: for eval mode ?
    return set(feature_columns)


def create_input_fn(
        mode, input_files, batch_size, num_epochs):
    def input_fn():
        features = tf.contrib.layers.create_feature_spec_for_parsing(
                get_feature_columns(mode))

        feature_map = tf.contrib.learn.io.read_batch_features(
                file_pattern=input_files,
                batch_size=batch_size,
                features=features,
                reader=tf.TFRecordReader,
                randomize_input=True,
                num_epochs=num_epochs,
                queue_capacity=200000 + batch_size * 10,
                name="read_batch_features_{}".format(mode))

        # TODO: bug in tf.learn ? ref: udc_input.py
        target = feature_map.pop('label')
#        if mode == tf.contrib.learn.ModeKeys.TRAIN:
#            target = feature_map.pop("label")
#        else:
#            # TODO ?
#            target = tf.zeros([batch_size, 1], dtype=tf.int64)
        return feature_map, target

    return input_fn

