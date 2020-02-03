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

class LossFunctionParams(Object):
    # enum may be better than str?
    func_list = ['square_error', 'multinominal_logistic_regression']

    reg_type_list = ['none', 'l1', 'l2']

    func_type: LossFuncType
    reg_type: LossRegularizationType

    #loss function and regularization
    def __init__(self, func=LossFuncType.square_error, reg_type=LossRegularizationType.none, decay=10e-3):
        self.func_type = check_enum(func, 'func', LossFuncType, self, '__init__')
        self.reg_type = check_enum(reg_type, 'reg_type', LossRegularizationType, self, '__init__')
        self.decay = check_type(decay, 'decay', float, self, '__init__')



class IterationParams(Object):
    # epoch and batch size
    # batch_size is None, which means online training
    epoch: int
    batch_size: int
    def __init__(self, epoch=50, batch_size=256):
        self.epoch = check_type(epoch, 'epoch', int, self, '__init__')
        self.batch_size = check_type_including_none(batch_size, 'batch_size', int, self, funcnames='__init__', default=None)

class OptimizationParams(Object):
    # momentum, learning rate
    learning_rate: float
    momentum: float
    def __init__(self, learning_rate=10e-2, momentum=0.8):
        self.learning_rate = check_type(learning_rate, 'learning_rate', float, self, '__init__')
        self.momentum = check_type(momentum, 'momentum', float, self, '__init__')

class TrainingParams(Object):
    lossfunc_params: LossFunctionParams
    iter_params: IterationParams
    opt_params: OptimizationParams

    def __init__(self, lossfunc_params=None, iter_params=None, opt_params=None):
        self.lossfunc_params = check_type_including_none(lossfunc_params, 'lossfunc_params', LossFunctionParams, self, default=LossFunctionParams(), funcnames='__init__')
        self.iter_params = check_type_including_none(iter_params, 'iter_params', IterationParams, self, default=IterationParams(), funcnames='__init__')
        self.opt_params = check_type_including_none(opt_params, 'opt_params', OptimizationParams, self, default=OptimizationParams(), funcnames='__init__')

    # getter and setter will be implemented in future
    """
    @property
    def epoch(self):
        return self.iteration.epoch
    
    @epoch.setter
    def epoch(self, value):
        return check_type(value, 'epoch', int, IterationParams)
    """