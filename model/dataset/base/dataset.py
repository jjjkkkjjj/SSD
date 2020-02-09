from ...core.base.object import Object
from ...common.utils.argchecker import check_type
from ...train.params import OptimizationParams

import numpy as np

class BaseDataSet(Object):
    def __init__(self, train_X, train_labels, test_X, test_labels):
        self.train_X = np.array(check_type(train_X, 'train_X', (list, np.ndarray), self, funcnames='__init__'))
        self.train_labels = np.array(check_type(train_labels, 'train_labels', (list, np.ndarray), self, funcnames='__init__'))
        # error handling when X and labels size is not same

        self.test_X = np.array(check_type(test_X, 'test_X', (list, np.ndarray), self, funcnames='__init__'))
        self.test_labels = np.array(check_type(test_labels, 'test_labels', (list, np.ndarray), self, funcnames='__init__'))

        #self.type =

    @property
    def count_train(self):
        return len(self.train_labels)
    @property
    def count_test(self):
        return len(self.test_labels)

    # iterator
    def epoch_iterator(self, opt_params):
        from .iterator import EpochIterator

        _ = check_type(opt_params, 'iter_params', OptimizationParams, self, 'train')

        return EpochIterator(opt_params, self)