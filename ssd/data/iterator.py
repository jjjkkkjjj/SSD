from ..params.training import IterationParams
from .dataset import DataSet
from ..utils.error.argchecker import check_type

import numpy as np

"""
Property naming rule: hoge or hoge_now
    hoge    : maximum count of hoge
    hoge_now: current count of hoge
"""
class TrainIterator:
    _iter_params: IterationParams
    _dataset: DataSet
    random_by_epoch: bool

    def __init__(self, iter_params, dataset, random_by_epoch):
        # check is already finished in _dataset method
        #self.iter_params = check_type(iter_params, 'iter_params', IterationParams, Iterator, '__init__')
        #self.dataset = check_type(dataset, 'dataset', DataSet, Iterator, '__init__')
        #self.random_by_epoch = check_type(random_by_epoch, 'random_by_epoch', bool, Iterator, '__init__')

        self._iter_params = iter_params
        self._dataset = dataset
        self.random_by_epoch = random_by_epoch

        # iteration parameter
        self._iter_num = 0
    
    # for _dataset
    @property
    def test_X(self):
        return self._dataset.test_X
    @property
    def test_one_hotted_labels(self):
        return self._dataset.test_one_hotted_labels
    
    # for parameter
    @property
    def iteration(self):
        return int(np.ceil(self._dataset.count_train / self.batch_size))
    @property
    def epoch(self):
        return self._iter_params.epoch
    @property
    def batch_size(self):
        return self._iter_params.batch_size

    def __iter__(self):
        pass
    
    def __next__(self):
        pass
        # End of IterationParams
        #raise StopIteration

class EpochIterator(TrainIterator):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # for _dataset
    @property
    def onlineTrain_X(self):
        return self._dataset.train_X
    @property
    def onlineTrain_one_hotted_labels(self):
        return self._dataset.train_one_hotted_labels

    # for parameter
    @property
    def epoch_now(self):
        return self._iter_num
    @epoch_now.setter
    def epoch_now(self, value):
        self._iter_num = value


    def batch_iterator(self):

        return BatchIterator(self.epoch_now, self._iter_params, self._dataset, self.random_by_epoch)

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch_now < self.epoch:
            self.epoch_now += 1
            return self
        else:
            raise StopIteration



class BatchIterator(TrainIterator):
    epoch_num: int

    def __init__(self, epoch_now, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_now = epoch_now

        # p = numpy.random.permutation(len(a))
        # a[p], b[p]
        if self.random_by_epoch:
            self.__indices = np.random.permutation(self._dataset.count_train)
        else:
            self.__indices = np.arange(self._dataset.count_train)

    @property
    def iteration_now(self):
        return self._iter_num
    @iteration_now.setter
    def iteration_now(self, value):
        self._iter_num = value


    def __iter__(self):
        return self

    """
    :returns
        train_X         : np.ndarray
        train_one_hotted_labels    : np.ndarray
    """
    def __next__(self):
        if self.iteration_now < self.iteration:
            
            init_index = self.iteration_now * self.batch_size
            indices = self.__indices[init_index:init_index + self.batch_size]

            train_X = self._dataset.train_X[indices]
            train_labels = self._dataset.train_one_hotted_labels[indices]
            """
            below condition isn't needed. see below example
            >>> a=np.arange(10)
            >>> a
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> a[:15]
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            """
            """
            if self.iteration_now < self.iteration - 1:
                indices = self.__indices[init_index:init_index + self.batch_size]

                train_X = self._dataset.train_X[indices]
                train_labels = self._dataset.train_labels[indices]
            else: # self.iteration_now == self.iteration - 1 means last iteration
                indices = self.__indices[init_index:init_index + self.batch_size]

                train_X = self._dataset.train_X[indices]
                train_labels = self._dataset.train_labels[indices]
            """
            self.iteration_now += 1

            return train_X, train_labels, self
        else:
            raise StopIteration
