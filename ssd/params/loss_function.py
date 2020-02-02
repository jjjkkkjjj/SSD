from ..utils.error.argchecker import *
from ..utils.error.warnmsg import _wmsg_enum
from ..params.training import LossFunction, LossFuncType, LossRegularizationType

import tensorflow as tf
import logging

def get_loss_function(labels, score, loss_params, cls):
    lossfunction = check_type(loss_params, 'loss_params', LossFunction, cls)

    # create loss
    if lossfunction.func_type == LossFuncType.square_error:
        loss = square_error(labels, score)
        pass
    elif lossfunction.func_type == LossFuncType.multinominal_logistic_regression:
        loss = multinominal_logistic_reggression(labels, score)
    else:
        message = _wmsg_enum(lossfunction.func_type, LossFuncType)
        logging.warning(message)

    return loss

def square_error(labels, score):
    return tf.reduce_mean(tf.pow(score - labels, 2))

def multinominal_logistic_reggression(labels, score):
    # cross entropy
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=score))

