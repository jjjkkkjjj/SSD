from ..base.object import Object
from ..utils.error.argchecker import *

from enum import Enum


class LossFuncType(Enum):
    square_error = 0
    multinominal_logistic_regression = 1


class LossRegularizationType(Enum):
    none = 0
    l1 = 1
    l2 = 2

class LossFunction(Object):
    # enum may be better than str?
    func_list = ['square_error', 'multinominal_logistic_regression']

    reg_type_list = ['none', 'l1', 'l2']

    func_type: LossFuncType
    reg_type: LossRegularizationType

    #loss function and regularization
    def __init__(self, func=LossFuncType.square_error, reg_type=LossRegularizationType.none, decay=10e-3):
        self.func_type = check_enum(func, 'func', LossFuncType, LossFunction, '__init__')
        self.reg_type = check_enum(reg_type, 'reg_type', LossRegularizationType, LossFunction, '__init__')
        self.decay = check_type(decay, 'decay', float, LossFunction, '__init__')



class Iteration(Object):
    # epoch and batch size
    def __init__(self, epoch=50, batch_size=256):
        self.epoch = check_type(epoch, 'epoch', int, Iteration, '__init__')
        self.batch_size = check_type(batch_size, 'batch_size', int, Iteration, '__init__')

class Optimization(Object):
    # momentum, learning rate
    def __init__(self, learning_rate=10e-2, momentum=0.8):
        self.learning_rate = check_type(learning_rate, 'learning_rate', float, Optimization, '__init__')
        self.momentum = check_type(momentum, 'momentum', float, Optimization, '__init__')

class TrainingParams(Object):
    loss: LossFunction
    iteration: Iteration
    optimization: Optimization

    def __init__(self, lossfunction=None, iteration=None, optimization=None):
        self.loss = check_type_including_none(lossfunction, 'lossfunction', LossFunction, TrainingParams, default=LossFunction(), funcnames='__init__')
        self.iteration = check_type_including_none(iteration, 'iteration', Iteration, TrainingParams, default=Iteration(), funcnames='__init__')
        self.optimization = check_type_including_none(optimization, 'optimization', Optimization, TrainingParams, default=Optimization(), funcnames='__init__')

    # getter and setter will be implemented in future
    """
    @property
    def epoch(self):
        return self.iteration.epoch
    
    @epoch.setter
    def epoch(self, value):
        return check_type(value, 'epoch', int, Iteration)
    """