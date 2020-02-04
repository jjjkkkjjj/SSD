from ..core.object import Object
from ..common.utils.argchecker import *

from enum import Enum
from tensorflow.compat.v1.train import Optimizer

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


class OptimizationParams(Object):
    # optimizer
    optimizer: Optimizer
    # epoch and batch size
    # batch_size is None, which means online training
    epoch: int
    batch_size: int
    def __init__(self, optimizer, epoch, batch_size, random_by_epoch=True):
        self.optimizer = check_type(optimizer, 'optimizer', Optimizer, self, '__init__')
        self.epoch = check_type(epoch, 'epoch', int, self, '__init__')
        self.batch_size = check_type_including_none(batch_size, 'batch_size', int, self, funcnames='__init__',
                                                    default=None)
        self.random_by_epoch = check_type(random_by_epoch, 'random_by_epoch', bool, self, '__init__')


class TrainingParams(Object):
    lossfunc_params: LossFunctionParams
    opt_params: OptimizationParams

    def __init__(self, lossfunc_params, opt_params):
        self.lossfunc_params = check_type(lossfunc_params, 'lossfunc_params', LossFunctionParams, self, funcnames='__init__')
        self.opt_params = check_type(opt_params, 'opt_params', OptimizationParams, self, funcnames='__init__')

    # getter and setter will be implemented in future
    """
    @property
    def epoch(self):
        return self.iteration.epoch
    
    @epoch.setter
    def epoch(self, value):
        return check_type(value, 'epoch', int, IterationParams)
    """