import argparse
import dataclasses

__all__ = ('Arg', 'Int', 'Float', 'Bool', 'Str', 'Choice', 'parse_to')

class Arg:
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class Int(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=int, **kwargs)


class Float(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=float, **kwargs)


class Bool(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=bool, **kwargs)


class Str(Arg):
    def __init__(self, **kwargs):
        super().__init__(type=str, **kwargs)


class _MetaChoice(type):
    def __getitem__(self, item):
        return self(choices=list(item), type=item)


class Choice(Arg, metaclass=_MetaChoice):
    def __init__(self, choices, **kwargs):
        super().__init__(choices=choices, **kwargs)


def parse_to(container_class, **kwargs):
    def mangle_name(name):
        return '--' + name.replace('_', '-')

    parser = argparse.ArgumentParser(description=container_class.__doc__)
    for field in dataclasses.fields(container_class):
        name = field.name
        default = field.default
        value_or_class = field.type
        if isinstance(value_or_class, type):
            value = value_or_class(default=default)
        else:
            value = value_or_class
            value.kwargs['default'] = default
        parser.add_argument(
            mangle_name(name), **value.kwargs)

    arg_dict = parser.parse_args(**kwargs)
    return container_class(**vars(arg_dict))