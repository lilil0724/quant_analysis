"""Compatibility import for :mod:`codebase.raw.common`."""
from codebase.raw import common as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
