from model.train.params import OptimizationParams
from model.dataset.dataset import DataSet, DatasetClassification, DatasetEncoder

import numpy as np

"""
Property naming rule: hoge or hoge_now
    hoge    : maximum count of hoge
    hoge_now: current count of hoge
"""
class EpochIterator:
    _opt_params: OptimizationParams
    _dataset: DataSet
    random_by_epoch: bool

    def __init__(self, opt_params, dataset):
        # check is already finished in _dataset method
        #self.iter_params = check_type(iter_params, 'iter_params', IterationParams, Iterator, '__init__')
        #self.dataset = check_type(dataset, 'dataset', DataSet, Iterator, '__init__')
        #self.random_by_epoch = check_type(random_by_epoch, 'random_by_epoch', bool, Iterator, '__init__')

        self._opt_params = opt_params
        self._dataset = dataset

        # iteration parameter
        self._iter_num = 0
    
    # for _dataset
    @property
    def onlineTrain_X(self):
        return self._dataset.train_X
    @property
    def test_X(self):
        return self._dataset.test_X

    @property
    def onlineTrain_labels(self):
        return self._dataset.train_labels
    @property
    def test_labels(self):
        return self._dataset.test_labels

    @property
    def count_train(self):
        return len(self.onlineTrain_labels)
    @property
    def count_test(self):
        return len(self.test_labels)

    # for parameter
    @property
    def random_by_epoch(self):
        return self._opt_params.random_by_epoch

    @property
    def epoch(self):
        return self._opt_params.epoch

    @property
    def epoch_now(self):
        return self._iter_num

    @epoch_now.setter
    def epoch_now(self, value):
        self._iter_num = value

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.epoch_now < self.epoch:
            self.epoch_now += 1
            return self
        else:
            raise StopIteration

    def batch_iterator(self):
        return BatchIterator(self, self._opt_params)
"""
epoch

"""
class EpochIteratorEncoder(EpochIterator):
    _dataset: DatasetEncoder

    def __iter__(self):
        return self

    def __next__(self):
        _ = super().__next__()
        return self

    def batch_iterator(self):
        return BatchIteratorEncoder(self, self._opt_params)

class EpochIteratorClassification(EpochIterator):
    _dataset: DatasetClassification
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # for _dataset
    @property
    def onlineTrain_one_hotted_labels(self):
        return self._dataset.train_one_hotted_labels

    @property
    def test_one_hotted_labels(self):
        return self._dataset.test_one_hotted_labels


    def batch_iterator(self):
        return BatchIteratorClassification(self, self._opt_params)

    def __iter__(self):
        return self

    def __next__(self):
        _ = super().__next__()
        return self

"""
batch

"""
class BatchIterator:
    _epoch_iterator: EpochIterator
    _opt_params: OptimizationParams
    batch_indices: np.ndarray

    def __init__(self, epoch_iterator, opt_params):
        self._epoch_iterator = epoch_iterator
        self._opt_params = opt_params

        # p = numpy.random.permutation(len(a))
        # a[p], b[p]
        if self.random_by_epoch:
            self.__indices = np.random.permutation(self.count_train)
        else:
            self.__indices = np.arange(self.count_train)

        self.batch_indices = np.ndarray([])

        # iteration parameter
        self._iter_num = 0

    @property
    def random_by_epoch(self):
        return self._epoch_iterator.random_by_epoch

    @property
    def count_train(self):
        return self._epoch_iterator.count_train
    @property
    def batch_size(self):
        return self._opt_params.batch_size

    @property
    def epoch(self):
        return self._opt_params.epoch
    @property
    def epoch_now(self):
        return self._epoch_iterator.epoch_now

    @property
    def iteration(self):
        return int(np.ceil(self.count_train / self.batch_size))
    @property
    def iteration_now(self):
        return self._iter_num
    @iteration_now.setter
    def iteration_now(self, value):
        self._iter_num = value


    @property
    def X(self):
        return self._epoch_iterator.onlineTrain_X[self.batch_indices]
    @property
    def labels(self):
        return self._epoch_iterator.onlineTrain_labels[self.batch_indices]

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
            self.batch_indices = self.__indices[init_index:init_index + self.batch_size]
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

            return self
        else:
            raise StopIteration


class BatchIteratorEncoder(BatchIterator):
    _epoch_iterator: EpochIteratorEncoder

    def __iter__(self):
        return self

    def __next__(self):
        _ = super().__next__()
        return self


class BatchIteratorClassification(BatchIterator):
    _epoch_iterator: EpochIteratorClassification

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @property
    def one_hotted_labels(self):
        return self._epoch_iterator.onlineTrain_one_hotted_labels[self.batch_indices]

    def __iter__(self):
        return self

    def __next__(self):
        _ = super().__next__()
        return self