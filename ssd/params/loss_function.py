import tensorflow as tf


def multinominal_logistic_reggression(labels, score):
    # cross entropy
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=score))
