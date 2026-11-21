"""Compatibility import for :mod:`codebase.raw.bias_competitor_core`."""
from codebase.raw import bias_competitor_core as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
