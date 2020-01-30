import numpy as np

def _type_error_message(cls, arg, instances, name):
    if not isinstance(arg, instances):
        raise cls.ArgumentError('{0} must be list or ndarray, but got {1}'.format(name, type(arg).__name__))
    return arg

"""
Convolution in architecture.py
"""
"""
belows function must be duplicated
"""
# return array-like when it's valid, otherwise, raise KernelError
def check_kernel(kernel):
    from .architecture import Convolution
    return _type_error_message(Convolution, kernel, (list, np.ndarray), 'kernel')

# return int when it's valid, otherwise, raise BiasError
def check_kernelnums(kernelnums):
    from .architecture import Convolution

    return _type_error_message(Convolution, kernelnums, int, 'kernelnums')

# return array-like when it's valid, otherwise, raise StridesError
def check_strides(strides):
    from .architecture import Convolution
    return _type_error_message(Convolution, strides, (list, np.ndarray), 'strides')

"""
FullyConnection in architecture.py
"""
# return int when it's valid, otherwise, raise BiasError
def check_outputnums(outputnums):
    from .architecture import FullyConnection

    return _type_error_message(FullyConnection, outputnums, int, 'outputnums')

"""
Architecture in architecture.py
"""
# return array-like when it's valid, otherwise, raise StridesError
def check_params(params):
    from .architecture import Architecture, Layer

    params = _type_error_message(Architecture, params, (list, np.ndarray), 'params')
    if not all(isinstance(param, Layer) for param in params):
        raise Architecture.ArgumentError('params must be composed of layer')

    return params


"""
Model in object.py
"""
def check_architecture(architecture):
    from .architecture import Architecture
    from .model import Model
    return _type_error_message(Model, architecture, Architecture, 'architecture')