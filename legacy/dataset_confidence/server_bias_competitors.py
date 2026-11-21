"""Compatibility import for :mod:`codebase.raw.server_bias_competitors`."""
from codebase.raw import server_bias_competitors as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    _implementation.main()
