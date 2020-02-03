from ..base.object import Object
from ..utils.error.argchecker import check_type
from ..params.training import IterationParams

import numpy as np

class DataSet(Object):
    def __init__(self, train_X, train_labels, test_X, test_labels):
        self.train_X = np.array(check_type(train_X, 'train_X', (list, np.ndarray), DataSet, funcnames='__init__'))
        self.train_labels = np.array(check_type(train_labels, 'train_labels', (list, np.ndarray), DataSet, funcnames='__init__'))
        # error handling when X and labels size is not same

        self.test_X = np.array(check_type(test_X, 'test_X', (list, np.ndarray), DataSet, funcnames='__init__'))
        self.test_labels = np.array(check_type(test_labels, 'test_labels', (list, np.ndarray), DataSet, funcnames='__init__'))

    @property
    def count_train(self):
        return len(self.train_labels)
    @property
    def count_test(self):
        return len(self.test_labels)

    # iterator
    def epoch_iterator(self, iter_params, random_by_epoch=True):
        from .iterator import EpochIterator

        _ = check_type(iter_params, 'iter_params', IterationParams, DataSet, 'train')
        _ = check_type(random_by_epoch, 'random_by_epoch', bool, DataSet, 'train')

        return EpochIterator(iter_params, self, random_by_epoch)

class DatasetEncoder(DataSet):
    def __init__(self, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = check_type(shape, 'shape', (list, np.ndarray, tuple), DatasetEncoder, funcnames='__init__')


    def epoch_iterator(self, iter_params, random_by_epoch=True):
        from .iterator import EpochIteratorEncoder
        _ = super().epoch_iterator(iter_params, random_by_epoch)

        return EpochIteratorEncoder(iter_params, self, random_by_epoch)

class DatasetClassification(DataSet):
    def __init__(self, class_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_num = check_type(class_num, 'class_num', int, DatasetClassification, funcnames='__init__')

    @property
    def train_one_hotted_labels(self):
        return self.__one_hot_encode(self.train_labels)

    @property
    def test_one_hotted_labels(self):
        return self.__one_hot_encode(self.train_labels)

    def __one_hot_encode(self, labels):
        labels_count = len(labels)
        ret = np.zeros((labels_count, self.class_num))
        ret[range(labels_count), labels] = 1
        return ret

    # iterator
    def epoch_iterator(self, iter_params, random_by_epoch=True):
        from .iterator import EpochIteratorClassification
        _ = super().epoch_iterator(iter_params, random_by_epoch)

        return EpochIteratorClassification(iter_params, self, random_by_epoch)