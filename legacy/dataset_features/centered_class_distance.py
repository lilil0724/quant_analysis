"""Compatibility import for :mod:`codebase.raw.centered_class_distance`."""
from codebase.raw import centered_class_distance as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    _implementation.main()
