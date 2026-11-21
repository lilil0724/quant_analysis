"""Compatibility import for :mod:`codebase.raw.tsne_paired`."""
from codebase.raw import tsne_paired as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == '__main__':
    _implementation.main()
