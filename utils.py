import tensorflow as tf


def gen_normal(shape):
    # TODO: change this to open for `lower` and `upper`.
    return tf.truncated_normal(shape, mean=0.0, stddev=1.0)
