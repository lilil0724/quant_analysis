"""Import a verified local snapshot without changing its historical receipts."""
import argparse
import collections
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


def read_manifest(directory):
    """Verify an immutable local input, including containment and file sizes."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'source_manifest.json').read_text(encoding='utf-8-sig'))
    if manifest['status'] != 'complete':
        raise ValueError('Input snapshot is not complete')
    for name, entry in manifest['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f'Invalid input manifest path: {name}')
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Input checksum mismatch: {name}')
    return manifest


def acquire_serials(input_id, serials, baseline_input_id='baseline_20261021',
                    project='nycu_pcs/TiT', target_jsonl=None):
    """Publish a new schema-2 snapshot; never overwrite an existing input."""
    from legacy_hqq.recover_baselines import DATASETS, MODELS, MODES
    import itertools

    destination = BASE / 'data' / validate_id(input_id)
    if destination.exists():
        raise FileExistsError(destination)
    serials = sorted(set(int(s) for s in serials))
    if not serials:
        raise ValueError('At least one serial is required')
    baseline_dir = BASE / 'data' / validate_id(baseline_input_id)
    baseline_manifest = read_manifest(baseline_dir)
    baseline_source = baseline_dir / 'selected_runs.jsonl'
    cohort_source = baseline_dir / 'cohort_receipt.json'
    cohort = json.loads(cohort_source.read_text(encoding='utf-8-sig'))
    if (cohort['status'] != 'complete'
            or sha256(baseline_source) != cohort['artifacts']['selected_runs.jsonl']):
        raise ValueError('Baseline cohort disagrees with its original receipt')
    filters = {'config.serial': {'$in': serials + [str(s) for s in serials]}}
    if target_jsonl is None:
        import wandb
        from wandb.apis.public.runs import _create_runs_query
        runs = wandb.Api(timeout=120).runs(project, filters=filters, per_page=500,
                                        order='+created_at')
        runs.QUERY = _create_runs_query(lazy=False)
        expected_count = len(runs)
        rows = []
        for run in runs:
            rows.append(dict(id=run.id, name=run.name, state=run.state, url=run.url,
                             created_at=run.created_at, config=dict(run.config),
                             summary=dict(run.summary)))
            if len(rows) % 500 == 0:
                print(f'Downloaded {len(rows)}/{expected_count}', flush=True)
        source_type = 'W&B full config/summary snapshot; no history, arrays or inference'
        source = f'https://wandb.ai/{project}'
    else:
        target_jsonl = Path(target_jsonl).resolve()
        rows = [json.loads(line) for line in target_jsonl.read_text(encoding='utf-8-sig').splitlines() if line]
        expected_count = len(rows)
        source_type, source = 'local full config/summary JSONL import', str(target_jsonl)
    if not rows or len(rows) != expected_count:
        raise ValueError('Empty or incomplete target acquisition')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate target run IDs')
    if any(int(r['config']['serial']) not in serials for r in rows):
        raise ValueError('Targets include an unrequested serial')
    # Invalid config values must not become explicit null (= activation off).
    def clean(value, in_config=False):
        import math
        if isinstance(value, float) and not math.isfinite(value):
            return 'invalid_nonfinite' if in_config else None
        if isinstance(value, dict):
            return {k: clean(v, in_config or k == 'config') for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v, in_config) for v in value]
        return value
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'snapshot'
        staging.mkdir()
        files = {}
        for name, original in {'baseline_selected_runs.jsonl': baseline_source,
                               'baseline_cohort_receipt.json': cohort_source,
                               'baseline_input_manifest.json': baseline_dir / 'source_manifest.json'}.items():
            expected = sha256(original)
            shutil.copy2(original, staging / name)
            if sha256(staging / name) != expected:
                raise ValueError(f'Baseline copy checksum mismatch: {name}')
            files[name] = dict(source=str(original), bytes=(staging/name).stat().st_size, sha256=expected)
        target = staging / 'target_runs.jsonl'
        target.write_text(''.join(json.dumps(clean(r), ensure_ascii=False, allow_nan=False) + '\n'
                                  for r in rows), encoding='utf-8')
        files[target.name] = dict(source=source, bytes=target.stat().st_size, sha256=sha256(target))
        if target_jsonl is not None:
            files[target.name]['original_sha256'] = sha256(target_jsonl)
        manifest = dict(schema_version=2, status='complete', input_id=input_id,
                        acquired_at=datetime.now(timezone.utc).isoformat(), timezone='UTC',
                        project=project, serials=serials, filters=filters, source_type=source_type,
                        total_runs=len(rows), run_ids=[r['id'] for r in rows],
                        state_counts=dict(collections.Counter(r['state'] for r in rows)),
                        expected_contexts=list(itertools.product(DATASETS, MODELS, MODES)),
                        baseline_input_id=baseline_input_id,
                        baseline_manifest_sha256=sha256(baseline_dir / 'source_manifest.json'),
                        baseline_precision=baseline_manifest.get('baseline_precision', 'unquantized_AMP'),
                        files=files, nonfinite_json_policy='nonfinite config -> invalid_nonfinite; summary -> null',
                        historical_receipts='preserved byte-for-byte; original paths are provenance')
        (staging / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        read_manifest(staging)
        staging.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-id', required=True)
    parser.add_argument('--serials', nargs='+', type=int, help='Acquire/import a generic schema-2 input')
    parser.add_argument('--project', default='nycu_pcs/TiT')
    parser.add_argument('--baseline-input-id', default='baseline_20261021')
    parser.add_argument('--target-jsonl', type=Path, help='Use a local export instead of downloading')
    parser.add_argument('--selected-file', type=Path, default=REPO / 'data/baseline_recovery/selected_20261021/selected_runs.jsonl')
    parser.add_argument('--target-file', type=Path, default=REPO / 'outputs/serial_20007/data/snapshot_20261021/serial_20007_runs.json')
    parser.add_argument('--reference-file', type=Path, default=REPO / 'outputs/serial_20007/results/main/acc_ratio_pairs.csv')
    args = parser.parse_args()
    if args.serials:
        print(acquire_serials(args.input_id, args.serials, args.baseline_input_id,
                             args.project, args.target_jsonl))
    elif args.target_jsonl:
        parser.error('--target-jsonl requires --serials')
    else:
        print(import_snapshot(args.input_id, args.selected_file, args.target_file, args.reference_file))


if __name__ == '__main__':
    main()
