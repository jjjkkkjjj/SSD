from ..core.object import Object
from ..common.utils.argchecker import check_type
from ..train.params import OptimizationParams

import numpy as np


class DataSet(Object):
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

class DatasetEncoder(DataSet):
    def __init__(self, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = check_type(shape, 'shape', (list, np.ndarray, tuple), self, funcnames='__init__')


    def epoch_iterator(self, opt_params):
        from .iterator import EpochIteratorEncoder
        _ = super().epoch_iterator(opt_params)

        return EpochIteratorEncoder(opt_params, self)

class DatasetClassification(DataSet):
    def __init__(self, class_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_num = check_type(class_num, 'class_num', int, self, funcnames='__init__')

    @property
    def train_one_hotted_labels(self):
        return self.__one_hot_encode(self.train_labels)

    @property
    def test_one_hotted_labels(self):
        return self.__one_hot_encode(self.test_labels)

    def __one_hot_encode(self, labels):
        labels_count = len(labels)
        ret = np.zeros((labels_count, self.class_num))
        ret[range(labels_count), labels] = 1
        return ret

    # iterator
    def epoch_iterator(self, opt_params):
        from .iterator import EpochIteratorClassification
        _ = super().epoch_iterator(opt_params)

        return EpochIteratorClassification(opt_params, self)

class DatasetObjectRecognition(DataSet):
    pass