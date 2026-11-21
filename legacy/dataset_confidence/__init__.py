"""Matched analysis workflow for TGDA HQQ confidence artifacts."""

import sys
from pathlib import Path

# The historical module commands run from legacy/, outside the repository import root.
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

