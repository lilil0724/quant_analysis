"""Compatibility entry point for the DAFB raw-export analysis."""
import sys
from pathlib import Path

from codebase.raw import dafb as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def main(argv=None):
    module = Path(__file__).resolve().parent
    defaults = ['--result-root', str(module / 'result'), '--report-root', str(module / 'reports')]
    _implementation.main([*defaults, *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    main()
