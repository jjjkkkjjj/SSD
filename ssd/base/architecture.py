from ..utils.argchecker import *
from .object import Object

from enum import Enum
import numpy as np

class Layer(Object):
    def __init__(self, name, layertype):
        self.name = name
        self.type = layertype

    class LayerType(Enum):
        input = 0
        convolution = 1
        maxpooling = 2
        flatten = 3
        fullyconnection = 4
        dropout = 5

class Input(Layer):
    def __init__(self, name, shape):
        super().__init__(name, layertype=Layer.LayerType.input)
        self.shape = check_type(shape, 'shape', (list, np.ndarray), Input)

    @property
    def width(self):
        return self.shape[1]
    @property
    def height(self):
        return self.shape[0]
"""
Attributes
kernel  :array-like
bias    :int
strides :array-like
"""
class Convolution(Layer):
    def __init__(self, name, kernel, kernelnums, strides, padding='SAME'):
        super().__init__(name, layertype=Layer.LayerType.convolution)
        self.kernel = check_type(kernel, 'kernel', (list, np.ndarray), Convolution)
        self.kernelnums = check_type(kernelnums, 'kernelnums', int, Convolution)
        self.strides = check_type(strides, 'strides', (list, np.ndarray), Convolution)
        self.padding = check_name(padding, 'padding', ['SAME'], Convolution)

    @property
    def kernel_width(self):
        return self.kernel[1]
    @property
    def kernel_height(self):
        return self.kernel[0]
    @property
    def stride_width(self):
        return self.strides[1]
    @property
    def stride_height(self):
        return self.strides[0]
"""
below class must be duplicated
"""
class MaxPooling(Layer):
    def __init__(self, name, kernel, strides, padding='VALID'):
        super().__init__(name, layertype=Layer.LayerType.maxpooling)
        self.kernel = check_type(kernel, 'kernel', (list, np.ndarray), MaxPooling)
        self.strides = check_type(strides, 'strides', (list, np.ndarray), MaxPooling)
        self.padding = check_name(padding, 'padding', ['VALID'], MaxPooling)

    @property
    def kernel_width(self):
        return self.kernel[1]
    @property
    def kernel_height(self):
        return self.kernel[0]
    @property
    def stride_width(self):
        return self.strides[1]
    @property
    def stride_height(self):
        return self.strides[0]

class Flatten(Layer):
    def __init__(self, name):
        super().__init__(name, layertype=Layer.LayerType.flatten)

class FullyConnection(Layer):
    def __init__(self, name, outputnums, activationfunc='relu'):
        super().__init__(name, layertype=Layer.LayerType.fullyconnection)
        self.outputnums = check_type(outputnums, 'outputnums', int, FullyConnection)
        self.activationfunc = check_name(activationfunc, 'activationfunc', ['relu'], FullyConnection)

class DropOut(Layer):
    def __init__(self, name, rate):
        super().__init__(name, layertype=Layer.LayerType.dropout)
        self.rate = float(check_type(rate, 'rate', (float, int), DropOut))
"""
Attributes
params: array-like of Layer
"""
class Architecture(Object):
    def __init__(self, input_model, layer_models, output_model):
        self.__input_model = check_type(input_model, 'input_model', Input, Architecture)
        self.__layer_models = check_layer_models(layer_models)
        self.__output_model = check_type(output_model, 'output_model', (FullyConnection), Architecture)


    def input_model(self):
        return self.__input_model

    @property
    def layer_models(self):
        return self.__layer_models


