from ..common.utils.argchecker import *
from .base.object import Object

from enum import Enum
import numpy as np
#import warnings
#warnings.filterwarnings('ignore',category=FutureWarning)


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
        atrous_convolution = 6

class Input(Layer):
    def __init__(self, name, rect, channel):
        super().__init__(name, layertype=Layer.LayerType.input)
        self.rect = check_type(rect, 'rect', (list, tuple, np.ndarray), self, '__init__')
        self.channel = check_type(channel, 'channel', int, self, '__init__')

    @property
    def width(self):
        return self.rect[1]
    @property
    def height(self):
        return self.rect[0]

    @property
    def shape(self):
        shape = []
        shape.extend(self.rect)
        shape.append(self.channel)
        return shape
"""
Attributes
kernel  :array-like
bias    :int
strides :array-like
"""
class Convolution(Layer):
    def __init__(self, name, kernel, kernelnums, strides, padding='VALID'):
        super().__init__(name, layertype=Layer.LayerType.convolution)
        self.kernel = check_type(kernel, 'kernel', (list, tuple, np.ndarray), self, '__init__')
        self.kernelnums = check_type(kernelnums, 'kernelnums', int, self, '__init__')
        self.strides = check_type(strides, 'strides', (list, tuple, np.ndarray), self, '__init__')
        self.padding = check_name(padding, 'padding', ['SAME', 'VALID'], self, '__init__')

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

class AtrousConvolution(Convolution):
    def __init__(self, name, dilation_rate, kernel, kernelnums, strides, padding='VALID'):
        super().__init__(name, kernel, kernelnums, strides, padding)
        self.type = Layer.LayerType.atrous_convolution
        self.dilation_rate = dilation_rate

"""
below class must be duplicated
"""
class MaxPooling(Layer):
    def __init__(self, name, kernel, strides, padding='VALID'):
        super().__init__(name, layertype=Layer.LayerType.maxpooling)
        self.kernel = check_type(kernel, 'kernel', (list, tuple, np.ndarray), self, '__init__')
        self.strides = check_type(strides, 'strides', (list, tuple, np.ndarray), self, '__init__')
        self.padding = check_name(padding, 'padding', ['VALID', 'SAME'], self, '__init__')

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
        self.outputnums = check_type(outputnums, 'outputnums', int, self, '__init__')
        self.activationfunc = check_name_including_none(activationfunc, 'activationfunc', ['relu', 'softmax'], self, default=None, funcnames='__init__')

class DropOut(Layer):
    def __init__(self, name, rate):
        super().__init__(name, layertype=Layer.LayerType.dropout)
        self.rate = float(check_type(rate, 'rate', (float, int), self, '__init__'))