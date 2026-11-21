"""Source snapshots and run metadata for the raw-export analyses."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .common import sha256

RAW_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RAW_ROOT.parents[1]


def raw_source_files():
    return (*RAW_ROOT.glob('*.py'), RAW_ROOT / 'run_nano4.sh')


def snapshot_source(out, paths):
    records = {}
    for path in sorted(paths):
        relative = path.relative_to(REPO_ROOT)
        destination = out / 'provenance' / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records[relative.as_posix()] = sha256(destination)
    return records


def write_receipt(path, value):
    partial = path.with_suffix('.json.tmp')
    partial.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    partial.replace(path)


def environment():
    versions = {}
    for package in ('numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn', 'threadpoolctl'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions)


def git_identity():
    try:
        values = []
        for arguments in (['rev-parse','HEAD'], ['status','--short']):
            process = subprocess.Popen(['git','-C',str(REPO_ROOT),*arguments], stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True)
            output, _ = process.communicate()
            if process.returncode:
                return dict(commit=None, dirty=None)
            values.append(output.strip())
        return dict(commit=values[0], dirty=bool(values[1]), status=values[1].splitlines())
    except OSError:
        return dict(commit=None, dirty=None)
