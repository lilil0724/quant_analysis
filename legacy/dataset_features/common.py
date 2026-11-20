"""Dataset scope and file helpers shared by the active paired workflows.

Helpers retain the legacy implementations so extracting them does not change
CSV serialization or source checksums. No analysis or plotting belongs here.
"""

import csv
import hashlib


DATASETS = ('aircraft', 'cars', 'cub', 'dogs', 'flowers', 'food', 'inat17', 'moe', 'nabirds', 'pets', 'soyageing', 'soygene', 'soyglobal', 'soylocal', 'vegfru', 'cotton')


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
