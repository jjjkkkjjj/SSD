from ssd.utils.argutils import _get_typename_from_class, _get_typename

# return instances' names for error message
def _emsg_type_check(arg, argname, classes):

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

    return '{0} must be {1}, but got {2}'.format(argname, clsstr, _get_typename(arg))

def _emsg_name_check(arg, argname, valid_names):
    return '{0} must be {1}, but got \'{2}\''.format(argname, valid_names, arg)

def _emsg_check_layers_all():
    return 'all layers\' elements must inherit Layer'

def _emsg_check_layers_input(firstlayer):
    return 'layers\' first element must be inherited Input, but got {0}'.format(firstlayer.type)

def _emsg_enum_check(argname, enum):
    names = [e.name for e in enum]
    return 'Argument \'{0}\' was invalid, select one in {1}'.format(argname, names)