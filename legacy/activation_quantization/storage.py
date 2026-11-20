"""Safe paths and immutable snapshot validation for activation analyses."""
from pathlib import Path
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent


def validate_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('IDs must use letters, digits, underscores or hyphens')
    return value


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f'Manifest path escapes its directory: {name}')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'source_manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('status', 'complete') != 'complete' or not manifest.get('files'):
        raise ValueError('Input snapshot is incomplete or has no file manifest')
    for name, info in manifest['files'].items():
        path = contained(directory, name)
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
    return manifest
