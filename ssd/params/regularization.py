from ..utils.error.argchecker import *
from ..utils.error.warnmsg import _wmsg_enum
from ..params.training import LossFunctionParams, LossFuncType, LossRegularizationType

import tensorflow as tf
import logging

def get_loss_added_regularization(weights, loss, loss_params, ins):
    lossfunction = check_type(loss_params, 'loss_params', LossFunctionParams, ins)

    # add regularization term
    if lossfunction.reg_type == LossRegularizationType.none:
        ret_loss = loss
    elif lossfunction.reg_type == LossRegularizationType.l1:
        ret_loss = add_l1(lossfunction.decay, weights, loss)
    elif lossfunction.reg_type == LossRegularizationType.l2:
        ret_loss = add_l2(lossfunction.decay, weights, loss)
    else:
        message = _wmsg_enum(lossfunction.reg_type, LossRegularizationType)
        logging.warning(message)

    return ret_loss

def add_l1(decay, weights, loss):
    ret_loss = loss #+ decay * tf.reduce_sum(weights)
    # Sigma |w_i|
    l1_weights = [tf.reduce_sum(tf.compat.v1.abs(weight)) for weight in weights]
    return ret_loss + decay * tf.compat.v1.add_n(l1_weights)

def add_l2(decay, weights, loss):
    ret_loss = loss #+ decay * tf.reduce_sum(tf.abs(weights))
    # 1/2 * w**2
    l2_weights = [tf.nn.l2_loss(weight) for weight in weights]
    return ret_loss + decay * tf.compat.v1.add_n(l2_weights)