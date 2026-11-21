"""Compatibility import for :mod:`codebase.raw.paired_inputs`."""
from codebase.raw import paired_inputs as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
