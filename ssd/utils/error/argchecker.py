from ...utils.error.errormsg import _emsg_type_check, _emsg_name_check, _emsg_check_layers_all

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
def check_type(arg, argname, classes, layercls, funcnames=''):
    if not isinstance(arg, classes):
        message = _emsg_type_check(arg, argname, classes)
        if funcnames != '':
            message = 'In {0}, '.format(funcnames) + message
        raise layercls.ArgumentError(message)
    return arg


"""
param:
    arg         : the argument you want to check
    argname     : arg's name
    valid_names : list of names
    layercls    : static class

return:
    arg         : if valid_names contains arg, return arg

raise:
    NameError   : if valid_names doesn't contains arg, raise NameError
"""
def check_name(arg, argname, valid_names, layercls):
    check_type(arg, argname, str, layercls.ArgumentError)

    if not arg in valid_names:
        message = _emsg_name_check(arg, argname, valid_names)
        raise layercls.NameError(message)

    return arg
"""
see above
param:
    default : if arg is None, return default
"""
def check_name_including_none(arg, argname, valid_names, layercls, default):
    if arg is None:
        return default
    else:
        return check_name(arg, argname, valid_names, layercls)

"""
param:
    layers          : list of Layer

return:
    layers          : if all layers' elements is inherited Layer, return layers

raise:
    ArgumentError   : if all layers' elements isn't inherited Layer, raise ArgumentError
"""
def check_layer_models(layer_models):
    from ...base.architecture import Architecture, Layer
    check_type(layer_models, 'layers', (list, np.ndarray), Architecture)
    if not all(isinstance(layer_model, Layer) for layer_model in layer_models):
        message = _emsg_check_layers_all()
        raise Architecture.ArgumentError(message)

    """
    if not (len(layer_models) > 0 and layer_models[0].type == Layer.LayerType.input):
        message = _emsg_check_layers_input(layer_models[0])
        raise Architecture.ArgumentError(message)
    """

    return layer_models
