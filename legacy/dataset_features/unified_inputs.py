"""Read schema-2 TGDA exports independently of legacy/paired schema-1 workflows.

CPU comparison: python -m five_dataset_features.unified_inputs --fp DIR --quant DIR --output DIR
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, name):
    part = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or part.is_absolute() or '..' in part.parts:
        raise ValueError(f'Unsafe source path: {name}')
    path = root.joinpath(*part.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Source path escapes condition: {name}')
    return path


def read_condition(folder):
    root = Path(folder).resolve()
    meta = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if meta.get('schema') != 2 or meta.get('status') != 'complete':
        raise ValueError('Require a complete schema-2 condition; smoke is not analysis input')
    files = meta.get('files', {})
    for name, digest in files.items():
        if sha256(safe_path(root, name)) != digest:
            raise ValueError(f'Source checksum mismatch: {name}')
    if not {'samples.json', 'metrics.json'} <= files.keys():
        raise ValueError('Export requires checksummed samples and metrics')
    metadata, execution = meta.get('metadata'), meta.get('execution')
    hashes = ('checkpoint_sha256', 'data_sha256', 'code_sha256', 'cfg_sha256', 'hqq_source_sha256')
    if not isinstance(metadata, dict) or any(
            not isinstance(metadata.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', metadata[key])
            for key in hashes):
        raise ValueError('Invalid source provenance: require all SHA-256 hashes')
    preprocessing = metadata.get('preprocessing')
    preprocessing_keys = ('image_size', 'test_resize_size', 'test_resize_directly',
                          'test_square_resize_center_crop', 'custom_mean_std', 'mean', 'std',
                          'pre_resize_factor', 'resize_size', 'dataset_in_memory', 'cal_ap_only', 'aug_type')
    if not isinstance(preprocessing, dict) or not set(preprocessing_keys) <= preprocessing.keys():
        raise ValueError('Incomplete preprocessing provenance')
    execution_types = {'seed': int, 'fp16': bool, 'hqq_compute_dtype': str, 'batch_size': int,
                       'cal_ap_only': bool, 'dataset_in_memory': bool, 'reduce_op': str}
    if not isinstance(execution, dict) or any(
            type(execution.get(key)) is not kind for key, kind in execution_types.items()):
        raise ValueError('Incomplete or invalid execution provenance')
    if (execution['batch_size'] < 1 or execution['hqq_compute_dtype'] not in ('float16', 'bfloat16', 'float32')
            or not execution['reduce_op'] or 'reduce_pattern' not in execution
            or execution['reduce_pattern'] is not None and not isinstance(execution['reduce_pattern'], str)):
        raise ValueError('Invalid execution provenance')
    samples = json.loads((root / 'samples.json').read_text(encoding='utf-8'))
    ids, targets, n = samples['sample_ids'], samples['targets'], meta['n_samples']
    if (type(n) is not int or n < 2 or len(ids) != n or len(targets) != n or len(set(ids)) != n
            or any(not isinstance(v, str) or not v for v in ids)
            or any(type(v) is not int or v < 0 for v in targets)):
        raise ValueError('Invalid sample identity/count')
    features = {}
    try:
        for name, shape in meta['layers'].items():
            if not name.startswith('features/'):
                continue
            if name not in files or not name.endswith('.npy'):
                raise ValueError('Feature file missing from checksums')
            array = np.load(safe_path(root, name), mmap_mode='r', allow_pickle=False)
            features[Path(name).stem] = array
            if (array.ndim != 2 or list(array.shape) != shape or array.shape[0] != n
                    or array.dtype != np.dtype(meta['feature_dtype'])
                    or not all(np.isfinite(array[i:i + 1024]).all() for i in range(0, n, 1024))):
                raise ValueError(f'Invalid feature shape/dtype/values: {name}')
        if not features:
            raise ValueError('No exported features; rerun TGDA with --save_features')
        return dict(folder=root, manifest=meta, sample_ids=ids, targets=np.asarray(targets), features=features,
                    metrics=json.loads((root / 'metrics.json').read_text(encoding='utf-8')))
    except BaseException:
        for array in features.values():
            array._mmap.close()
        raise


def read_pair(fp_folder, quant_folder):
    fp = read_condition(fp_folder)
    quant = None
    try:
        quant = read_condition(quant_folder)
        if fp['manifest']['condition']['name'] != 'fp':
            raise ValueError('Baseline must be the unquantized fp condition')
        left, right = fp['manifest'], quant['manifest']
        keys = ('checkpoint_sha256', 'data_sha256', 'code_sha256', 'cfg_sha256', 'hqq_source_sha256', 'preprocessing')
        if any(left['metadata'][k] != right['metadata'][k] for k in keys) or left['execution'] != right['execution']:
            raise ValueError('Pair source/execution identity mismatch')
        if fp['sample_ids'] != quant['sample_ids'] or not np.array_equal(fp['targets'], quant['targets']):
            raise ValueError('Pair sample/target mismatch')
        if set(fp['features']) != set(quant['features']) or any(
                fp['features'][k].shape != quant['features'][k].shape for k in fp['features']):
            raise ValueError('Pair representation shape/layers differ')
        return fp, quant
    except BaseException:
        for condition in (fp, quant):
            if condition is not None:
                for array in condition['features'].values():
                    array._mmap.close()
        raise


def main():
    from .metrics import centered_linear_cka, class_equal_cosine_distances
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fp', required=True, type=Path)
    parser.add_argument('--quant', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output)
                              for p in (args.fp, args.quant)):
        raise ValueError('Choose a fresh analysis output outside both source conditions')
    fp, quant = read_pair(args.fp, args.quant)
    try:
        rows = []
        for layer, x in fp['features'].items():
            y = quant['features'][layer]
            rows.append({'layer': layer, 'cka': centered_linear_cka(x, y),
                         **{'fp_' + k: v for k, v in class_equal_cosine_distances(x, fp['targets']).items()},
                         **{'quant_' + k: v for k, v in class_equal_cosine_distances(y, quant['targets']).items()}})
        output.mkdir(parents=True)
        with (output / 'layer_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / 'sources.json').write_text(json.dumps({
            'fp': str(fp['folder']), 'quant': str(quant['folder']),
            'manifest_sha256': {str(c['folder']): sha256(c['folder'] / 'manifest.json') for c in (fp, quant)},
            'fp_metrics': fp['metrics'], 'quant_metrics': quant['metrics']}, indent=2, allow_nan=False), encoding='utf-8')
    finally:
        for condition in (fp, quant):
            for array in condition['features'].values():
                array._mmap.close()


if __name__ == '__main__':
    main()
