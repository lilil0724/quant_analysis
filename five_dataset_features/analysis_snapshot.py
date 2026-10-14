"""Keep five-dataset analysis outputs tied to their verified inference inputs."""

import hashlib
import json
from pathlib import Path


SNAPSHOT_FILE = 'analysis_snapshot.json'
SCHEMA = 1
ANALYSIS_CODE = ('analyze.py', 'metrics.py', 'analysis_snapshot.py')
DATASET_OUTPUTS = (
    'summary.csv', 'layer_metrics.csv', 'precision_probe.csv',
    'fp_indicators.json', 'fp_pca.npz', 'cka_heatmap.png', 'cka_curves.png',
    'pca_distribution_correct_swap.png', 'activation_transition.png',
)
REPORT_OUTPUTS = (
    'REPORT.md', 'analysis_version.json', 'all_conditions.csv',
    'fp_indicators.csv', 'descriptive_spearman.csv',
    'fp_predictors_vs_loss.png',
)


class SnapshotError(RuntimeError):
    """An analysis output cannot be tied to the current verified inputs."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash_files(folder, names, action):
    hashes = {}
    for name in names:
        path = folder / name
        if not path.is_file():
            raise SnapshotError(f'Missing {path}. {action}')
        hashes[name] = sha256(path)
    return hashes


def _source_hashes(root, dataset, conditions, action):
    names = ['dataset.json'] + [f'{name}/manifest.json' for name in conditions]
    return _hash_files(root / dataset, names, action)


def _code_hashes(action):
    return _hash_files(Path(__file__).parent, ANALYSIS_CODE, action)


def _dataset_outputs(root, dataset, conditions, action):
    names = [*DATASET_OUTPUTS, *(f'coords_{name}.npy' for name in conditions)]
    return _hash_files(root / 'analysis' / dataset, names, action)


def _read_snapshot(path, action):
    if not path.is_file():
        raise SnapshotError(f'Missing {path}. {action}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise SnapshotError(f'Invalid {path}. {action}') from exc
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise SnapshotError(f'Unsupported {path}. {action}')
    return value


def _write_snapshot(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def _require_equal(label, recorded, current, action):
    if not isinstance(recorded, dict) or set(recorded) != set(current):
        raise SnapshotError(f'{label} coverage changed. {action}')
    for name, digest in current.items():
        if recorded[name] != digest:
            raise SnapshotError(f'{label} changed: {name}. {action}')


def source_identity(root, dataset, conditions):
    """Capture source manifests and analysis code before the dataset run."""
    root = Path(root)
    action = f'Rerun dataset {dataset}, then report and package.'
    return {
        'source_manifests': _source_hashes(root, dataset, conditions, action),
        'analysis_code': _code_hashes('Rerun all five datasets, then report and package.'),
    }


def record_dataset(root, dataset, conditions, initial_identity):
    """Record a completed dataset analysis only if its inputs stayed fixed."""
    root = Path(root)
    action = f'Rerun dataset {dataset}, then report and package.'
    sources = _source_hashes(root, dataset, conditions, action)
    _require_equal(f'{dataset} source manifests',
                   initial_identity['source_manifests'], sources, action)
    code_action = 'Rerun all five datasets, then report and package.'
    code = _code_hashes(code_action)
    _require_equal('analysis code', initial_identity['analysis_code'], code,
                   code_action)
    value = {
        'schema': SCHEMA,
        'dataset': dataset,
        'source_manifests': sources,
        'analysis_code': code,
        'derived_files': _dataset_outputs(root, dataset, conditions, action),
    }
    _write_snapshot(root / 'analysis' / dataset / SNAPSHOT_FILE, value)


def verify_dataset(root, dataset, conditions):
    """Verify source manifests, analysis code, and every derived dataset file."""
    root = Path(root)
    action = f'Rerun dataset {dataset}, then report and package.'
    path = root / 'analysis' / dataset / SNAPSHOT_FILE
    value = _read_snapshot(path, action)
    if value.get('dataset') != dataset:
        raise SnapshotError(f'{path} belongs to another dataset. {action}')
    _require_equal(f'{dataset} source manifests', value.get('source_manifests'),
                   _source_hashes(root, dataset, conditions, action), action)
    code_action = 'Rerun all five datasets, then report and package.'
    _require_equal('analysis code', value.get('analysis_code'),
                   _code_hashes(code_action), code_action)
    _require_equal(f'{dataset} derived files', value.get('derived_files'),
                   _dataset_outputs(root, dataset, conditions, action), action)
    return sha256(path)


def record_report(root, datasets, conditions):
    """Tie the aggregate report to the exact five dataset analyses it reads."""
    root = Path(root)
    receipts = {dataset: verify_dataset(root, dataset, conditions)
                for dataset in datasets}
    action = 'Rerun report, then package.'
    value = {
        'schema': SCHEMA,
        'datasets': receipts,
        'analysis_code': _code_hashes('Rerun all five datasets, then report and package.'),
        'derived_files': _hash_files(root / 'analysis', REPORT_OUTPUTS, action),
    }
    _write_snapshot(root / 'analysis' / SNAPSHOT_FILE, value)


def verify_report(root, datasets, conditions):
    """Reject mixed or stale analyses before a package can be built."""
    root = Path(root)
    action = 'Rerun report, then package.'
    receipts = {dataset: verify_dataset(root, dataset, conditions)
                for dataset in datasets}
    path = root / 'analysis' / SNAPSHOT_FILE
    value = _read_snapshot(path, action)
    _require_equal('dataset analyses', value.get('datasets'), receipts, action)
    code_action = 'Rerun all five datasets, then report and package.'
    _require_equal('analysis code', value.get('analysis_code'),
                   _code_hashes(code_action), code_action)
    _require_equal('report files', value.get('derived_files'),
                   _hash_files(root / 'analysis', REPORT_OUTPUTS, action), action)
    return sha256(path)
