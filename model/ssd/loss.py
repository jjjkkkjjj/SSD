from ..train.loss_function import get_loss_function
from ..train.params import LossFunctionParams, LossFuncType

import tensorflow as tf

def get_loss(indicator, confidence, predict_locations, gt_locations, ins):
    return

def loss_confidence(indicator, confidence):
    loss_params = LossFunctionParams(func=LossFuncType.softmax_cross_entropy)

    # positive
    loss_pos =  indicator *  get_loss_function(gt_locations, predict_locations, loss_params, ins)
    # negative
    loss_neg = get_loss_function(gt_locations, predict_locations, loss_params, ins)

    return

def loss_location(indicator, predict_locations, gt_locations, ins):
    loss_params = LossFunctionParams(func=LossFuncType.smoothL1)
    return indicator * get_loss_function(gt_locations, predict_locations, loss_params, ins)