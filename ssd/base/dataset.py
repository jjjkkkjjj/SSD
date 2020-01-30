from .object import Object

class DataSet(Object):
    def __init__(self, X, labels, test_X, test_labels):
        self.X = X
        self.labels = labels

        self.test_X = test_X
        self.test_labels = test_labels

    @property
    def count(self):
        return len(self.labels)
    """
    must make iterator
    """
    def batch(self, size, epoch):
        if epoch + size < self.count:
            return self.X[epoch:epoch+size].reshape(size, 32, 32, 3), self.labels[epoch:epoch+size]
        else:
            return self.X[epoch:].reshape(-1, 32, 32, 3), self.labels[epoch:]