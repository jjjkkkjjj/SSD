from .checkargment import *
from .object import Object
import tensorflow as tf

class Layer(Object):
    def __init__(self, name):
        self.name = name

class Input(Layer):
    def __init__(self, name, shape):
        super().__init__(name)
        self.shape = check_strides(shape)

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
        super().__init__(name)
        self.kernel = check_kernel(kernel)
        self.kernelnums = check_kernelnums(kernelnums)
        self.strides = check_strides(strides)
        self.padding = padding

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
        super().__init__(name)
        self.kernel = check_kernel(kernel)
        self.strides = check_strides(strides)
        self.padding = padding

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
        super().__init__(name)

class FullyConnection(Layer):
    def __init__(self, name, outputnums, activationfunc='relu'):
        super().__init__(name)
        self.outputnums = check_outputnums(outputnums)
        self.activationfunc = activationfunc

class DropOut(Layer):
    def __init__(self, name, rate):
        super().__init__(name)
        self.rate = rate
"""
Attributes
params: array-like of Layer
"""
class Architecture(Object):
    def __init__(self, params):
        self.params = check_params(params)
