from ..error.errormsg import _emsg_type_check, _emsg_name_check, _emsg_check_layers_all, _emsg_enum_check
from ..error.exception import *

import numpy as np


"""
param:
    arg         : the argument you want to check
    argname     : arg's name
    classes     : list of static class arg must be inherited
    layercls    : static class

return:
    arg         : if arg's instance contains 'instances', return value
    
raise:
    ArgmentError: if arg's instance doesn't contain 'instances', raise ArgmentError
"""
def check_type(arg, argname, classes, ins, funcnames=''):
    if not isinstance(arg, classes):
        message = _emsg_type_check(arg, argname, classes, ins, funcnames)
        raise ArgumentTypeError(message)
    return arg
"""
see above
param:
    default : if arg is None, return default
"""
def check_type_including_none(arg, argname, classes, ins, default, funcnames=''):
    if arg is None:
        return default
    else:
        return check_type(arg, argname, classes, ins, funcnames)

"""
param:
    arg         : the argument you want to check
    argname     : arg's name
    valid_names : list of names
    layercls    : static class

return:
    arg         : if valid_names contains arg, return arg

raise:
    ArgumentNameError   : if valid_names doesn't contains arg, raise ArgumentNameError
"""
def check_name(arg, argname, valid_names, ins, funcnames=''):
    check_type(arg, argname, str, ins, funcnames)

    if not arg in valid_names:
        message = _emsg_name_check(arg, argname, valid_names, ins, funcnames)
        raise ArgumentNameError(message)

    return arg
"""
see above
param:
    default : if arg is None, return default
"""
def check_name_including_none(arg, argname, valid_names, ins, default, funcnames=''):
    if arg is None:
        return default
    else:
        return check_name(arg, argname, valid_names, ins, funcnames)

"""
param:
    layers          : list of Layer

return:
    layers          : if all layers' elements is inherited Layer, return layers

raise:
    ArgumentTypeError   : if all layers' elements isn't inherited Layer, raise ArgumentTypeError
"""
def check_layer_models(layer_models, ins):
    from ...core.architecture import Layer
    check_type(layer_models, 'layers', (list, np.ndarray), ins)
    if not all(isinstance(layer_model, Layer) for layer_model in layer_models):
        message = _emsg_check_layers_all(ins)
        raise ArgumentTypeError(message)

    """
    if not (len(layer_models) > 0 and layer_models[0].type == Layer.LayerType.input):
        message = _emsg_check_layers_input(layer_models[0])
        raise Architecture.ArgumentTypeError(message)
    """

    return layer_models


"""
param:
    arg         : the argument you want to check
    argname     : arg's name
    classes     : list of static class arg must be inherited
    layercls    : static class

return:
    arg         : if arg's instance contains 'instances', return value

raise:
    ArgmentEnumError: if arg's instance doesn't contain 'instances', raise ArgmentEnumError
"""


def check_enum(arg, argname, enum, ins, funcnames=''):
    if not isinstance(arg, enum):
        message = _emsg_enum_check(argname, enum, ins, funcnames)
        raise ArgumentEnumError(message)
    return arg