__all__ = ['EpochIteratorEncoder', 'EpochIteratorClassification']
from .dataset import *
from .base.iterator import EpochIterator

class EpochIteratorEncoder(EpochIterator):
    _dataset: DatasetEncoder

    def __iter__(self):
        return self

    def __next__(self):
        _ = super().__next__()
        return self

    def batch_iterator(self):
        from  .batch import BatchIteratorEncoder
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
        from .batch import BatchIteratorClassification
        return BatchIteratorClassification(self, self._opt_params)

    def __iter__(self):
        return self

    def __next__(self):
        _ = super().__next__()
        return self