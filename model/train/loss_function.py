from ..common.utils.argchecker import *
from ..common.error.warnmsg import _wmsg_enum
from .params import LossFunctionParams, LossFuncType

import tensorflow as tf
import logging

def get_loss_function(gt_labels, predict_labels, loss_params, ins):
    lossfunction = check_type(loss_params, 'loss_params', LossFunctionParams, ins)

    # create loss
    if lossfunction.func_type == LossFuncType.square_error:
        loss = square_error(gt_labels, predict_labels)
    elif lossfunction.func_type == LossFuncType.softmax_cross_entropy:
        loss = softmax_cross_entropy(gt_labels, predict_labels)
    elif lossfunction.func_type == LossFuncType.smoothL1:
        loss = smoothL1(gt_labels, predict_labels)
    else:
        message = _wmsg_enum(lossfunction.func_type, LossFuncType)
        logging.warning(message)

    return loss

def square_error(gt_labels, predict_labels):
    return tf.reduce_mean(tf.pow(predict_labels - gt_labels, 2))


# for one hot vector
def softmax_cross_entropy(gt_labels, predict_labels):
    # cross entropy
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_labels, logits=predict_labels))

"""
# for label
def softmax_cross_entropy(gt_labels, predict_labels):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_labels, logits=predict_labels)
"""
def smoothL1(gt_labels, predict_labels):
    def _smoothL1(x):
        return tf.where(tf.less(tf.abs(x), 1.0), tf.abs(x) - 0.5, 0.5 * tf.pow(x, 2.0))
    return tf.reduce_mean(_smoothL1(gt_labels - predict_labels))

