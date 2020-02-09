from .object import Object
from ...train.params import TrainingParams

import tensorflow as tf

class BaseOptMixin(Object):
    input_layer: tf.compat.v1.placeholder
    score: tf.Tensor
    weights: list
    params: TrainingParams

    def train(self, dataset, params, savedir=None):
        pass

    def __del__(self):
        self.session = None

    def load(self, path):
        self.session = tf.compat.v1.Session()

        tf.compat.v1.train.Saver().restore(self.session, path)

    def predict(self, X):
        pass