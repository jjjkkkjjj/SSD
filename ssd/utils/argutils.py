
def _get_typename(instance):
    return type(instance).__name__

def _get_typename_from_class(cls):
    return _get_typename(cls())