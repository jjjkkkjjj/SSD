from ..utils.error.argchecker import *
from ..utils.error.warnmsg import _wmsg_enum
from ..params.training import LossFunctionParams, LossFuncType, LossRegularizationType

import tensorflow as tf
import logging

def get_loss_function(y_true, score, loss_params, ins):
    lossfunction = check_type(loss_params, 'loss_params', LossFunctionParams, ins)

    # create loss
    if lossfunction.func_type == LossFuncType.square_error:
        loss = square_error(y_true, score)
        pass
    elif lossfunction.func_type == LossFuncType.multinominal_logistic_regression:
        loss = multinominal_logistic_reggression(y_true, score)
    else:
        message = _wmsg_enum(lossfunction.func_type, LossFuncType)
        logging.warning(message)

    return loss

def square_error(y_true, score):
    return tf.reduce_mean(tf.pow(score - y_true, 2))

def multinominal_logistic_reggression(y_true, score):
    # cross entropy
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=score))

