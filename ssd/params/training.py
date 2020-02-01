from ..base.object import Object
from ..utils.error.argchecker import *

class LossFunction(Object):
    func_list = ['square_error', 'multinominal_logistic_regression']

    reg_type_list = ['none', 'l1', 'l2']
    #loss function and regularization
    def __init__(self, func='', reg_type=None, decay=10e-3):
        self.func = check_name(func, 'func', self.func_list, LossFunction)
        self.reg_type = check_name_including_none(reg_type, 'reg_type', self.reg_type_list, LossFunction, default='none')
        self.decay = check_type(decay, 'decay', float, LossFunction)

class Iteration(Object):
    # epoch and batch size
    def __init__(self, epoch=50, batch_size=256):
        self.epoch = check_type(epoch, 'epoch', int, Iteration)
        self.batch_size = check_type(batch_size, 'batch_size', int, Iteration)

class Optimization(Object):
    # momentum, learning rate
    def __init__(self, learning_rate=10e-2, momentum=0.8):
        self.learning_rate = check_type(learning_rate, 'learning_rate', float, Optimization)
        self.momentum = check_type(momentum, 'momentum', float, Optimization)

class TrainingParams(Object):
    loss: LossFunction
    iteration: Iteration
    optimization: Optimization

    def __init__(self, lossfunction=LossFunction(), iteration=Iteration(), optimization=Optimization()):
        self.loss = check_type(lossfunction, 'lossfunction', LossFunction, TrainingParams)
        self.iteration = check_type(iteration, 'iteration', Iteration, TrainingParams)
        self.optimization = check_type(optimization, 'optimization', Optimization, TrainingParams)

    # getter and setter will be implemented in future
    """
    @property
    def epoch(self):
        return self.iteration.epoch
    
    @epoch.setter
    def epoch(self, value):
        return check_type(value, 'epoch', int, Iteration)
    """