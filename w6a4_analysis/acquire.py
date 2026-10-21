"""Import a verified local snapshot without changing its historical receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

BASE = Path(__file__).resolve().parent
REPO = BASE.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('IDs must use letters, digits, underscores or hyphens')
    return value


def import_snapshot(input_id, selected, target, reference):
    destination = BASE / 'data' / validate_id(input_id)
    if destination.exists():
        raise FileExistsError(destination)
    selected, target, reference = map(lambda p: Path(p).resolve(), (selected, target, reference))
    cohort_receipt = selected.with_name('manifest.json')
    target_receipt = target.with_name('source_manifest.json')
    cohort = json.loads(cohort_receipt.read_text(encoding='utf-8-sig'))
    target_source = json.loads(target_receipt.read_text(encoding='utf-8-sig'))
    if cohort['status'] != 'complete' or sha256(selected) != cohort['artifacts']['selected_runs.jsonl']:
        raise ValueError('Incomplete or changed baseline cohort')
    if sha256(target) != target_source['files']['serial_20007_runs.json']['sha256']:
        raise ValueError('Changed serial 20007 source')
    reference_receipt = reference.with_name('verification.json')
    verified = json.loads(reference_receipt.read_text(encoding='utf-8-sig'))
    if verified['status'] != 'complete' or sha256(reference) != verified['artifacts']['acc_ratio_pairs.csv']:
        raise ValueError('Reference pairs are not a verified completed result')
    sources = {'selected_runs.jsonl': selected, 'cohort_receipt.json': cohort_receipt,
               'serial_20007_runs.json': target, 'target_receipt.json': target_receipt,
               'reference_pairs.csv': reference, 'reference_receipt.json': reference_receipt}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'snapshot'
        staging.mkdir()
        files = {}
        for name, source in sources.items():
            expected = sha256(source)
            copied = staging / name
            shutil.copy2(source, copied)
            if sha256(copied) != expected:
                raise ValueError(f'Copy checksum mismatch: {source}')
            files[name] = dict(source=str(source), bytes=copied.stat().st_size, sha256=expected)
        manifest = dict(status='complete', input_id=input_id,
                        acquired_at=datetime.now(timezone.utc).isoformat(),
                        source_type='verified local W&B config/summary snapshots; no new download',
                        project='nycu_pcs/TiT', serial=20007, expected_contexts=144,
                        files=files, historical_receipts='preserved byte-for-byte; their original paths are provenance')
        (staging / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-id', required=True)
    parser.add_argument('--selected-file', type=Path, default=REPO / 'data/baseline_recovery/selected_20261021/selected_runs.jsonl')
    parser.add_argument('--target-file', type=Path, default=REPO / 'outputs/serial_20007/data/snapshot_20261021/serial_20007_runs.json')
    parser.add_argument('--reference-file', type=Path, default=REPO / 'outputs/serial_20007/results/main/acc_ratio_pairs.csv')
    args = parser.parse_args()
    print(import_snapshot(args.input_id, args.selected_file, args.target_file, args.reference_file))


if __name__ == '__main__':
    main()
