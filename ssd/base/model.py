from .architecture import *
from .opt import OptimezerMixin
from ..utils import build

import tensorflow as tf

"""
Attributes
params: 
    __layers    : array-like of built layer
    verbose     : boolean about printing calculation information or not
"""

class Model(Architecture, OptimezerMixin):

    def __init__(self, input_model, hidden_models, output_model, verbose=True):
        super().__init__(input_model, hidden_models, output_model)
        self.__layers = []
        self.verbose = verbose
        #self.create_model()

    @property
    def layers(self):
        return self.__layers

    def build(self):

        # input layer model
        self.layers.append(build.input(self.input_model))

        # hidden layer models
        for layer_model in self.hidden_models[1:-1]:
            input = self.layers[-1]
            layer = self.__get_layer(input, layer_model)
            self.layers.append(layer)

        # output layer model
        input = self.layers[-1]
        layer = self.__get_layer(input, self.output_model)
        self.layers.append(layer)

        if self.verbose:
            print("\nBuilding model was succeeded.\n")


    @property
    def score(self):
        return self.layers[-1]

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
