from ..utils.typename import _get_typename_from_class, _get_typename

# return instances' names for error message
def _emsg_type_check(arg, argname, classes, ins, funcnames):

    if not isinstance(classes, (list, tuple)):
        clsstr = _get_typename_from_class(classes)

    elif len(classes) == 1:
        clsstr = _get_typename_from_class(classes[0])
    elif len(classes) == 2:
        clsstr = '{0} or {1}'.format(_get_typename_from_class(classes[0]), _get_typename_from_class(classes[1]))
    else:
        ret = ''
        for _class in classes[:-1]:
            ret += '{0}, '.format(_get_typename_from_class(_class))
        clsstr = ret + 'or {0}'.format(_get_typename_from_class(classes[-1]))

    message = '{0} must be {1}, but got {2}'.format(argname, clsstr, _get_typename(arg))
    if funcnames != '':
        message = 'In {0}, '.format(funcnames) + message

    return '{0}: {1}'.format(_get_typename(ins), message)

def _emsg_name_check(arg, argname, valid_names, ins, funcnames):
    message = '{0} must be {1}, but got \'{2}\''.format(argname, valid_names, arg)
    if funcnames != '':
        message = 'In {0}, '.format(funcnames) + message
    return '{0}: {1}'.format(_get_typename(ins), message)

def _emsg_check_layers_all(ins):
    return '{0}: all layers\' elements must inherit Layer'.format(_get_typename(ins))

def _emsg_check_layers_input(firstlayer):
    return 'layers\' first element must be inherited Input, but got {0}'.format(firstlayer.type)

def _emsg_enum_check(argname, enum, ins, funcnames):
    names = [e.name for e in enum]

    message = 'Argument \'{0}\' was invalid, select one in {1}'.format(argname, names)
    if funcnames != '':
        message = 'In {0}, '.format(funcnames) + message
    return '{0}: {1}'.format(_get_typename(ins), message)