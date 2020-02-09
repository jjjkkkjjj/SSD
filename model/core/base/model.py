from .object import Object
from ...common.utils.argchecker import *
from ...common.utils.typename import _get_typename
from ..layer_models import Input, Layer
from .. import build

import logging

"""
Attributes
params: array-like of Layer
"""
class BaseLayerModel(Object):
    def __init__(self, models):
        _ = check_type(models, 'models', list, self, '__init__')
        if len(models) > 0:
            try:
                _ = check_type(models[0], 'models[0]', Input, self, '__init__')
            except ArgumentTypeError:
                message = 'models\' first element must be Input, but got {0}'.format(_get_typename(models[0]))
                raise ArgumentTypeError(message)
        self._models = models

    @property
    def all_models(self):
        return self._models

    @property
    def input_model(self):
        if len(self._models) == 0:
            return None
        else:
            return self._models[0]

    @property
    def output_model(self):
        if len(self._models) == 0:
            return None
        else:
            return self._models[-1]

    @property
    def hidden_models(self):
        if len(self._models) < 3:
            return None
        else:
            return self._models[1:-1]

    """
    :return
        tuple  : output shape
    """

    @property
    def output_shape(self):
        if self.output_model.type == Layer.LayerType.fullyconnection:
            return (self.output_model.outputnums)
        else:
            logging.warning('Cannot get output shape because implementation has not be defined')

"""
Attributes
params: 
    __layers    : array-like of built layer
    verbose     : boolean about printing calculation information or not
    weights     : list of tf.Variable is weights
    biases      : list of tf.Variable is bias
"""

class BaseModel(BaseLayerModel):

    def __init__(self, models, verbose=True):
        super().__init__(models)
        self.__layers = []

        logging.basicConfig()
        self.verbose = verbose

        self.__weights = []
        self.__biases = []

    @property
    def verbose(self):
        return self.__verbose
    @verbose.setter
    def verbose(self, value):
        self.__verbose = value
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.WARNING)

    @property
    def layers(self):
        return self.__layers
    @property
    def input_layer(self):
        return self.layers[0]
    @property
    def weights(self):
        return self.__weights
    @property
    def biases(self):
        return self.__biases

    def build(self):

        # input layer model
        self.layers.append(build.input(self.input_model))
        self.set_layer_attribute(self.layers[-1], self.input_model)

        # hidden layer models
        for layer_model in self.hidden_models:
            input = self.layers[-1]
            layer = self.get_layer(input, layer_model)
            self.layers.append(layer)
            self.set_layer_attribute(layer, layer_model)

        # output layer model
        input = self.layers[-1]
        layer = self.get_layer(input, self.output_model)
        self.layers.append(layer)
        self.set_layer_attribute(layer, self.output_model)

        logging.debug("\nBuilding model was succeeded.\n")

        return self

    @property
    def isBuilt(self):
        return len(self.layers) > 0

    @property
    def score(self):
        if not self.isBuilt:
            message = 'score function must be called after calling build function'
            raise UnBuiltError(message)
        return self.layers[-1]

    """
    :returns
        layer   : tf.Tensor represents layer
        weights : tf.Tensor represents weights
    """
    def get_layer(self, input, layer_model):
        if layer_model.type == Layer.LayerType.convolution:
            layer, weights, bias = build.convolution(input, layer_model)
            self.__weights.append(weights)
            self.__biases.append(bias)
            return layer

        elif layer_model.type == Layer.LayerType.maxpooling:
            return build.maxPooling(input, layer_model)

        elif layer_model.type == Layer.LayerType.flatten:
            return build.flatten(input, layer_model)

        elif layer_model.type == Layer.LayerType.fullyconnection:
            layer, weights, bias = build.fully_connection(input, layer_model)
            self.__weights.append(weights)
            self.__biases.append(bias)
            return layer

        elif layer_model.type == Layer.LayerType.dropout:
            return build.dropout(input, layer_model)

        elif layer_model.type == Layer.LayerType.atrous_convolution:
            layer, weights, bias = build.atrous_convolution(input, layer_model)
            self.__weights.append(weights)
            self.__biases.append(bias)
            return layer

        else:
            raise SyntaxError('This was bug...')

    def set_layer_attribute(self, layer, layer_model):
        self.set_layer_attribute_name(layer, layer_model.name)

    def set_layer_attribute_name(self, layer, name):
        if hasattr(self, name):
            message = 'layer\'s name \'{0}\' is invalid because this name will be used for layer value'.format(layer_model.name)
            raise ArgumentNameError(message)
        setattr(self, name, layer)
