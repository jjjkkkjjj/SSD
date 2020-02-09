__all__ = ['BatchIteratorClassification', 'BatchIteratorEncoder']
from .epoch import *
from .base.iterator import BatchIterator

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