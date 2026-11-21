"""Compatibility import for :mod:`codebase.raw.metrics`."""
from codebase.raw import metrics as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
