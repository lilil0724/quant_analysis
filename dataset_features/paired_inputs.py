"""Read-only schema-2 adapter for the existing W3/G128 paired analyses.

Legacy readers keep their original validation. This module only maps the newer
per-condition exports to those readers' in-memory input contract; it never
creates a legacy manifest or modifies an inference export.
"""
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np

from .common import DATASETS, sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def source_map_identity(path):
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    # Validate the entire map even if a pilot uses only some of its contexts.
    _source_map(path)
    return dict(path=str(path), sha256=sha256(path))


def _source_map(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 1 or not isinstance(data.get('pairs'), list):
        raise ValueError('Source map requires schema=1 and a pairs list')
    result = {}
    for row in data['pairs']:
        if (not isinstance(row, dict) or set(row) != {'dataset', 'model', 'mode', 'serial', 'folder'}
                or any(not isinstance(row[k], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', row[k])
                       for k in ('dataset', 'model', 'mode'))
                or row['mode'] not in ('ft', 'fz', 'cal') or type(row['serial']) is not int or row['serial'] <= 0
                or row['dataset'] not in DATASETS
                or row['model'] not in ('vit_b16', 'swin_base_patch4_window7_224_in22k', 'beitv2_base_patch16_224')
                or not isinstance(row['folder'], str) or not row['folder']):
            raise ValueError(f'Invalid source-map entry: {row}')
        key = row['dataset'], row['model'], row['mode']
        if key in result:
            raise ValueError(f'Duplicate source-map context: {key}')
        folder = Path(row['folder']).expanduser()
        if not folder.is_absolute():
            folder = path.parent / folder
        result[key] = folder.resolve(), row['serial']
    return result


def select_source(folder, serial, dataset, model, mode, source_map=None):
    if source_map is None:
        return Path(folder), serial
    return _source_map(source_map).get((dataset, model, mode), (Path(folder), serial))


def safe_path(folder, name):
    part = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or part.is_absolute() or '..' in part.parts:
        raise ValueError(f'Unsafe source path: {name}')
    path = Path(folder).joinpath(*part.parts)
    if not path.resolve().is_relative_to(Path(folder).resolve()):
        raise ValueError(f'Source path escapes pair: {path}')
    return path


def is_schema2(folder):
    folder = Path(folder)
    if (folder / 'manifest.json').is_file():
        if (folder / 'fp/manifest.json').is_file():
            raise ValueError(f'Ambiguous legacy/schema-2 pair: {folder}')
        return False
    return (folder / 'fp/manifest.json').is_file()


def condition_roots(folder):
    """Select by recorded settings, rejecting two eligible W3 conditions."""
    folder = Path(folder)
    fp = safe_path(folder, 'fp')
    candidates = []
    for child in sorted(folder.iterdir()):
        if not child.is_dir() or child.name.startswith('.') or '.partial' in child.name or '.failed.' in child.name:
            continue
        manifest = safe_path(folder, child.name + '/manifest.json')
        if not manifest.is_file():
            continue
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        c = meta.get('condition', {})
        if (c.get('weight_bits') == 3 and c.get('weight_group_size') == 128
                and c.get('activation_bits', 'missing') is None and c.get('scope') == 'linear'):
            if c.get('name') != child.name:
                raise ValueError(f'Condition name/directory mismatch: {child}')
            candidates.append(child)
    if len(candidates) != 1:
        raise ValueError(f'{folder}: require exactly one weight-only Linear W3/G128 condition; found {len(candidates)}')
    return {'fp': fp, 'w3_g128': candidates[0]}


def pair_file(folder, condition, name):
    """Resolve an existing analysis's logical file without copying the arrays."""
    folder = Path(folder)
    if not is_schema2(folder):
        return safe_path(folder, condition + '/' + name)
    root = condition_roots(folder)[condition]
    relative = ('features/' + name if name.endswith('.npy') and name != 'logits.npy' else name)
    return safe_path(folder, root.relative_to(folder).as_posix() + '/' + relative)


def read_samples(folder):
    folder = Path(folder)
    path = pair_file(folder, 'fp', 'samples.json') if is_schema2(folder) else safe_path(folder, 'samples.json')
    return json.loads(path.read_text(encoding='utf-8'))


def _metadata(root, expected, hashes, folder):
    path = root / 'manifest.json'
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta.get('schema') != 2 or any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{root}: require complete schema-2 source with matching identity')
    source, execution = meta.get('metadata'), meta.get('execution')
    keys = ('checkpoint_sha256', 'data_sha256', 'cfg_sha256', 'code_sha256', 'hqq_source_sha256')
    if not isinstance(source, dict) or any(not re.fullmatch('[0-9a-fA-F]{64}', str(source.get(k, ''))) for k in keys):
        raise ValueError(f'{root}: missing source provenance hashes')
    types = {'seed': int, 'fp16': bool, 'hqq_compute_dtype': str, 'batch_size': int,
             'cal_ap_only': bool, 'dataset_in_memory': bool, 'reduce_op': str}
    if (not isinstance(execution, dict) or any(type(execution.get(k)) is not t for k, t in types.items())
            or execution['batch_size'] < 1 or execution['hqq_compute_dtype'] not in ('float16', 'bfloat16', 'float32')
            or 'reduce_pattern' not in execution or not isinstance(source.get('preprocessing'), dict)
            or not isinstance(source.get('checkpoint'), str) or not source['checkpoint']):
        raise ValueError(f'{root}: incomplete execution/preprocessing identity')
    preprocessing_keys = ('image_size', 'test_resize_size', 'test_resize_directly',
                          'test_square_resize_center_crop', 'custom_mean_std', 'mean', 'std',
                          'pre_resize_factor', 'resize_size', 'dataset_in_memory', 'cal_ap_only', 'aug_type')
    if (not set(preprocessing_keys) <= source['preprocessing'].keys()
            or execution['reduce_pattern'] is not None and not isinstance(execution['reduce_pattern'], str)
            or not execution['reduce_op']):
        raise ValueError(f'{root}: incomplete execution/preprocessing identity')
    if type(meta.get('n_samples')) is not int or meta['n_samples'] < 2 or not isinstance(meta.get('files'), dict):
        raise ValueError(f'{root}: invalid source sample count/checksum manifest')
    hashes[path.relative_to(folder).as_posix()] = sha256(path)
    return meta


def _checked_file(folder, root, meta, name, hashes):
    path = safe_path(folder, root.relative_to(folder).as_posix() + '/' + name)
    expected = meta['files'].get(name)
    if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f'Source checksum mismatch: {path}')
    hashes[path.relative_to(folder).as_posix()] = expected
    return path


def _samples(folder, root, meta, hashes):
    path = _checked_file(folder, root, meta, 'samples.json', hashes)
    samples = json.loads(path.read_text(encoding='utf-8'))
    ids, targets, n = samples.get('sample_ids'), samples.get('targets'), meta['n_samples']
    if (not isinstance(ids, list) or not isinstance(targets, list) or len(ids) != n or len(targets) != n
            or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != n
            or any(type(v) is not int or v < 0 for v in targets)):
        raise ValueError(f'{root}: invalid sample IDs/labels')
    metrics_path = _checked_file(folder, root, meta, 'metrics.json', hashes)
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if metrics.get('eval/n_samples') != n:
        raise ValueError(f'{root}: metric/manifest sample counts differ')
    return samples


def _features(folder, root, meta, selected, hashes):
    layers = meta.get('layers', {})
    names = [p[len('features/'):-4] for p in layers if p.startswith('features/') and p.endswith('.npy')]
    selected = names if selected == 'all' else list(selected)
    if not selected or meta.get('feature_dtype') not in ('float16', 'float32'):
        raise ValueError(f'{root}: required features are not exported')
    result = {}
    for layer in selected:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', layer) or layer in ('.', '..'):
            raise ValueError(f'Unsafe feature layer: {layer}')
        name = 'features/' + layer + '.npy'
        path = _checked_file(folder, root, meta, name, hashes)
        shape = layers.get(name)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            if (not isinstance(shape, list) or len(shape) != 2 or shape[0] != meta['n_samples']
                    or type(shape[1]) is not int or shape[1] < 2 or list(array.shape) != shape
                    or array.dtype != np.dtype(meta['feature_dtype'])
                    or not all(np.isfinite(array[i:i+1024]).all() for i in range(0, len(array), 1024))):
                raise ValueError(f'{root}: incompatible/nonfinite feature {name}')
            result[layer] = shape
        finally:
            array._mmap.close()
    return result


def _predictions(folder, root, meta, samples, hashes, margins):
    path = _checked_file(folder, root, meta, 'predictions.csv', hashes)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'row', 'sample_id', 'target', 'prediction', 'correct'} <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: missing prediction columns')
        rows = list(reader)
    n = meta['n_samples']
    if len(rows) != n:
        raise ValueError(f'{path}: incomplete prediction rows')
    for i, row in enumerate(rows):
        pred, correct = int(row['prediction']), int(row['correct'])
        if (int(row['row']) != i or row['sample_id'] != samples['sample_ids'][i]
                or int(row['target']) != samples['targets'][i] or pred < 0
                or correct != int(pred == samples['targets'][i])):
            raise ValueError(f'{path}: invalid sample/prediction pairing at row {i}')
    classes, ties = None, dict(tied_maxima=0, stored_vs_numpy_argmax_disagreements=0)
    if not margins:
        return rows, classes, ties
    zpath = _checked_file(folder, root, meta, 'logits.npy', hashes)
    logits = np.load(zpath, mmap_mode='r', allow_pickle=False)
    try:
        if (logits.ndim != 2 or logits.shape[0] != n or logits.shape[1] < 2 or logits.dtype != np.float32
                or max(samples['targets']) >= logits.shape[1]):
            raise ValueError(f'{zpath}: incompatible logit shape/dtype/labels')
        classes = logits.shape[1]
        for start in range(0, n, 1024):
            values = np.asarray(logits[start:start+1024], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f'{zpath}: nonfinite logits')
            top = np.sort(np.partition(values, -2, axis=1)[:, -2:], axis=1)
            spread, winner = values.std(axis=1), values.argmax(axis=1)
            for j, row in enumerate(rows[start:start+len(values)]):
                prediction, target = int(row['prediction']), int(row['target'])
                if prediction >= classes or values[j, prediction] != top[j, 1]:
                    raise ValueError(f'{path}: stored prediction is not maximal')
                ties['tied_maxima'] += int(np.count_nonzero(values[j] == top[j, 1]) > 1)
                ties['stored_vs_numpy_argmax_disagreements'] += int(prediction != winner[j])
                # New CSV stores subtraction in float32; the legacy analyses use float64.
                native = np.sort(np.partition(logits[start+j], -2)[-2:])
                if 'margin' not in row or not np.isclose(float(row['margin']), float(native[1]-native[0]), rtol=1e-6, atol=1e-7):
                    raise ValueError(f'{path}: inconsistent stored margin')
                row['margin'] = str(float(top[j, 1]-top[j, 0]))
                row['normalized_margin'] = str(float((top[j, 1]-top[j, 0])/max(spread[j], 1e-12)))
                other = values[j].copy()
                other[target] = -np.inf
                row['true_class_margin'] = str(float(values[j, target]-other.max()))
        return rows, classes, ties
    finally:
        logits._mmap.close()


def read_pair(folder, serial, dataset, model, mode, *, features=(), fp_only=False,
              predictions=False, margins=False):
    """Validate consumed sources and supply the old algorithms' input structure.

    ``features`` is ``'all'``, a sequence of layer names, or empty for logit-only
    stages. Derived margins use the old exporter's float64 population-std rule.
    All hashes refer to real input files, including both original manifests.
    """
    folder = Path(folder)
    roots, hashes = condition_roots(folder), {}
    expected = dict(status='complete', serial=serial, dataset=dataset, model=model, checkpoint_mode=mode)
    manifests = {c: _metadata(r, expected, hashes, folder) for c, r in roots.items()}
    fp, quant = manifests['fp'], manifests['w3_g128']
    if (fp.get('condition', {}).get('name') != 'fp'
            or fp['condition'].get('weight_bits', 'missing') is not None
            or fp['condition'].get('activation_bits', 'missing') is not None):
        raise ValueError(f'{folder}: baseline is not unquantized FP')
    for key in ('checkpoint', 'checkpoint_sha256', 'data_sha256', 'cfg_sha256', 'code_sha256', 'hqq_source_sha256', 'preprocessing'):
        if fp['metadata'].get(key) != quant['metadata'].get(key):
            raise ValueError(f'{folder}: FP/Q source identity mismatch: {key}')
    if fp['execution'] != quant['execution']:
        raise ValueError(f'{folder}: FP/Q execution identity mismatch')
    if fp['n_samples'] != quant['n_samples']:
        raise ValueError(f'{folder}: FP/Q sample counts differ')
    for c, meta in manifests.items():
        coverage = meta.get('coverage', {})
        weights, activations = coverage.get('weights'), coverage.get('activations')
        if not isinstance(weights, list) or activations != [] or (not weights if c != 'fp' else bool(weights)):
            raise ValueError(f'{roots[c]}: missing/incompatible weight-only coverage')
        if any(not isinstance(s.get('name'), str) or s.get('bits') != 3 or s.get('group_size') != 128
               or not isinstance(s.get('calls'), int) or s['calls'] < 1 for s in weights):
            raise ValueError(f'{roots[c]}: invalid quantization coverage')
        if len({s['name'] for s in weights}) != len(weights):
            raise ValueError(f'{roots[c]}: duplicate quantization sites')
        if any('dfsm' in s['name'].split('.') or 'head' in s['name'].split('.') for s in weights):
            raise ValueError(f'{roots[c]}: classifier is included in quantized coverage')
    samples = {c: _samples(folder, r, manifests[c], hashes) for c, r in roots.items()}
    if samples['fp'] != samples['w3_g128']:
        raise ValueError(f'{folder}: FP/Q sample IDs or targets differ')
    layers = {}
    if features:
        layers = _features(folder, roots['fp'], fp, features, hashes)
        if not fp_only:
            other = _features(folder, roots['w3_g128'], quant, features, hashes)
            if other != layers or fp.get('feature_dtype') != quant.get('feature_dtype'):
                raise ValueError(f'{folder}: FP/Q feature layers/shapes/dtypes differ')
    rows, ties, classes = {}, {}, None
    if predictions or margins:
        for c, r in roots.items():
            rows[c], width, ties[c] = _predictions(folder, r, manifests[c], samples[c], hashes, margins)
            if classes is not None and width != classes:
                raise ValueError(f'{folder}: FP/Q logit class counts differ')
            classes = width
    source, execution = fp['metadata'], fp['execution']
    meta = dict(**expected, source_schema=2, n_samples=fp['n_samples'], layers=layers,
                feature_dtype=fp.get('feature_dtype'), logit_dtype='float32' if margins else None,
                checkpoint=source['checkpoint'], checkpoint_sha256=source['checkpoint_sha256'],
                test_csv=None, test_csv_sha256=source.get('test_csv_sha256'),
                cfg_sha256=source['cfg_sha256'], code_sha256=source['code_sha256'],
                hqq_source_sha256=source['hqq_source_sha256'], data_sha256=source['data_sha256'],
                seed=execution['seed'], amp_fp16=execution['fp16'], preprocessing=source['preprocessing'],
                execution=execution,
                quantization=dict(weight_bits=3, weight_group_size=128, activation_bits=None,
                                  compute_dtype=execution['hqq_compute_dtype'], exclude=None,
                                  # Call counts depend on dataset size and CAL views, not the protocol.
                                  coverage=sorted([{k: s[k] for k in ('name', 'bits', 'group_size')}
                                                   for s in quant['coverage']['weights']], key=lambda s: s['name'])),
                files=hashes, samples_sha256=digest(dict(ids=samples['fp']['sample_ids'], targets=samples['fp']['targets'])))
    # Use portable paths in receipts on Windows as well as Linux.
    meta['source_manifests'] = {(r.relative_to(folder)/'manifest.json').as_posix():
                              hashes[(r.relative_to(folder)/'manifest.json').as_posix()] for r in roots.values()}
    return dict(meta=meta, targets=samples['fp']['targets'], ids=samples['fp']['sample_ids'],
                predictions=rows, ties=ties, num_classes=classes, hashes=hashes)
