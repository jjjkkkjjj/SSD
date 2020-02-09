__all__ = ['DatasetEncoder', 'DatasetClassification', 'DatasetObjectDetection']
from .base.dataset import BaseDataSet
from ..common.utils.argchecker import check_type

import numpy as np

class DatasetEncoder(BaseDataSet):
    def __init__(self, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = check_type(shape, 'shape', (list, np.ndarray, tuple), self, funcnames='__init__')


    def epoch_iterator(self, opt_params):
        from .epoch import EpochIteratorEncoder
        _ = super().epoch_iterator(opt_params)

        return EpochIteratorEncoder(opt_params, self)

class DatasetClassification(BaseDataSet):
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
        from .epoch import EpochIteratorClassification
        _ = super().epoch_iterator(opt_params)

        return EpochIteratorClassification(opt_params, self)

class DatasetObjectDetection(BaseDataSet):
    def __init__(self, class_num,  train_X, train_norm_boxes, train_labels, test_X, test_norm_boxes, test_labels):
        train_ls = np.vstack((train_labels, train_norm_boxes)) # shape(2, *)
        test_ls = np.vstack((test_labels, test_norm_boxes)) # shape(2, *)
        super().__init__(train_X, train_ls, test_X, test_ls)