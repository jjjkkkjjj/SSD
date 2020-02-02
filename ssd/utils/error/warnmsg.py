
def _wmsg_enum(enumname, enum):
    names = [e.name for e in enum]
    return '{0} isn\'t supported. Supported enum is {1}.'.format(enumname, names)