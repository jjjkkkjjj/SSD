from .architecture import *
from .opt import OptimezerMixin
from ..utils import build

import tensorflow as tf
import logging

"""
Attributes
params: 
    __layers    : array-like of built layer
    verbose     : boolean about printing calculation information or not
    weights     : list of tf.Variable is weights
    biases      : list of tf.Variable is bias
"""

class Model(Architecture, OptimezerMixin):

    def __init__(self, input_model, hidden_models, output_model, verbose=True):
        super().__init__(input_model, hidden_models, output_model)
        self.__layers = []

        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.DEBUG)

        self.weights = []
        self.biases = []

    @property
    def layers(self):
        return self.__layers

    def build(self):

        # input layer model
        self.layers.append(build.input(self.input_model))

        # hidden layer models
        for layer_model in self.hidden_models:
            input = self.layers[-1]
            layer = self.__get_layer(input, layer_model)
            self.layers.append(layer)

        # output layer model
        input = self.layers[-1]
        layer = self.__get_layer(input, self.output_model)
        self.layers.append(layer)

        self.weights, self.biases = self.__get_weights_biases()
        exit()
        logging.debug("\nBuilding model was succeeded.\n")


    @property
    def score(self):
        return self.layers[-1]

    """
    :returns
        layer   : tf.Tensor represents layer
        weights : tf.Tensor represents weights
    """
    def __get_layer(self, input, layer_model):
        if layer_model.type == Layer.LayerType.convolution:
            return build.convolution(input, layer_model)

        elif layer_model.type == Layer.LayerType.maxpooling:
            return build.maxPooling(input, layer_model)

        elif layer_model.type == Layer.LayerType.flatten:
            return build.flatten(input, layer_model)

        elif layer_model.type == Layer.LayerType.fullyconnection:
            return build.fully_connection(input, layer_model)

        elif layer_model.type == Layer.LayerType.dropout:
            return build.dropout(input, layer_model)

        else:
            raise SyntaxError('This was bug...')

    """
    :returns
        weights : list of tf.Variable is weights
        biases  : list of tf.Variable is bias
    """
    def __get_weights_biases(self):
        weights = []
        biases = []

        for layer in self.layers:
            print(layer.name)
        return  weights, biases