"""Analyze existing TGDA serial/dataset/model/mode FP versus W3/g128 exports.

CPU analysis reuses metrics.py; no inference, checkpoint loading or GPU claim.
"""
import argparse
import hashlib
import itertools
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

from .analyze import DATASETS, read_csv, sha256, write_csv
from .metrics import (centered_linear_cka, class_equal_cosine_distances,
                      norm_quantiles, pca_fit, pca_project, prediction_metrics,
                      regularized_h_score, spearman_description)

MODELS = ('vit_b16', 'swin_base_patch4_window7_224_in22k', 'beitv2_base_patch16_224')
MODES = ('ft', 'fz', 'cal')
CONDITIONS = ('fp', 'w3_g128')
PAIR_FILES = ('summary.json', 'layer_metrics.csv', 'fp_pca.npz', 'pca.png')
GLOBAL_FILES = ('all_pairs.csv', 'coverage.csv', 'REPORT.md', 'analysis_manifest.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def inside(path, folder):
    return path == folder or folder in path.parents


def safe_file(folder, name):
    part = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or part.is_absolute() or '..' in part.parts:
        raise ValueError(f'Unsafe manifest path: {name}')
    path = folder.joinpath(*part.parts)
    if not inside(path.resolve(), folder.resolve()):
        raise ValueError(f'File link escapes source/output folder: {path}')
    return path


def finite_array(array):
    return all(np.isfinite(array[start:start + 1024]).all() for start in range(0, len(array), 1024))


def check_pair(folder, serial, dataset, model, mode):
    meta = json.loads(safe_file(folder, 'manifest.json').read_text(encoding='utf-8'))
    expected = dict(status='complete', serial=serial, dataset=dataset, model=model, checkpoint_mode=mode)
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{folder}: incomplete or incompatible manifest identity')
    n, layers = meta.get('n_samples'), meta.get('layers')
    if type(n) is not int or n < 2 or not isinstance(layers, dict) or 'final' not in layers:
        raise ValueError(f'{folder}: invalid sample/layer manifest')
    quant = meta.get('quantization', {})
    if any(quant.get(k, 'missing') != v for k, v in
           [('weight_bits', 3), ('weight_group_size', 128), ('activation_bits', None)]):
        raise ValueError(f'{folder}: expected W3/g128 with no activation quantization')
    if meta.get('feature_dtype') not in ('float16', 'float32') or meta.get('logit_dtype') != 'float32':
        raise ValueError(f'{folder}: invalid declared array dtype')
    dtype = np.dtype(meta['feature_dtype'])
    required = {'samples.json', 'layer_metrics.csv', 'summary.json'}
    for layer, shape in layers.items():
        if (not isinstance(layer, str) or '/' in layer or '\\' in layer or ':' in layer
                or layer in ('.', '..') or not isinstance(shape, list) or len(shape) != 2
                or shape[0] != n or type(shape[1]) is not int or shape[1] < 2):
            raise ValueError(f'{folder}: invalid feature layer/shape')
        required.update(f'{c}/{layer}.npy' for c in CONDITIONS)
    required.update(f'{c}/{name}' for c in CONDITIONS for name in ('predictions.csv', 'logits.npy'))
    files = meta.get('files')
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError(f'{folder}: required source files absent from checksums')
    for name, checksum in files.items():
        path = safe_file(folder, name)
        if not path.is_file() or sha256(path) != checksum:
            raise ValueError(f'{folder}: missing file or checksum mismatch: {name}')
    samples = json.loads((folder / 'samples.json').read_text(encoding='utf-8'))
    ids, targets = samples.get('sample_ids'), samples.get('targets')
    if (not isinstance(ids, list) or not isinstance(targets, list) or len(ids) != n
            or len(targets) != n or any(not isinstance(v, str) for v in ids)
            or len(set(ids)) != n or any(type(v) is not int or v < 0 for v in targets)
            or meta.get('samples_sha256') != digest({'ids': ids, 'targets': targets})):
        raise ValueError(f'{folder}: sample IDs, labels or metadata digest mismatch')
    predictions, classes = {}, None
    for condition in CONDITIONS:
        for layer, shape in layers.items():
            x = np.load(folder / condition / f'{layer}.npy', mmap_mode='r', allow_pickle=False)
            if list(x.shape) != shape or x.dtype != dtype or not finite_array(x):
                raise ValueError(f'{folder}: incompatible/nonfinite {condition}/{layer} features')
            del x
        logits = np.load(folder / condition / 'logits.npy', mmap_mode='r', allow_pickle=False)
        if (logits.ndim != 2 or logits.shape[0] != n or logits.shape[1] < 2
                or logits.dtype != np.float32 or max(targets) >= logits.shape[1]
                or not finite_array(logits) or classes not in (None, logits.shape[1])):
            raise ValueError(f'{folder}: invalid paired logits')
        classes = logits.shape[1]
        rows = read_csv(folder / condition / 'predictions.csv')
        if len(rows) != n:
            raise ValueError(f'{folder}: incomplete prediction rows')
        for start in range(0, n, 1024):
            argmax = logits[start:start + 1024].argmax(axis=1)
            for i, prediction in enumerate(argmax, start):
                row = rows[i]
                if (int(row['row']) != i or row['sample_id'] != ids[i]
                        or int(row['target']) != targets[i] or int(row['prediction']) != prediction
                        or int(row['correct']) != int(prediction == targets[i])
                        or any(not np.isfinite(float(row[key])) for key in
                               ('margin', 'normalized_margin', 'true_class_margin'))):
                    raise ValueError(f'{folder}: invalid sample pairing/prediction row {i}')
        predictions[condition] = rows
        del logits
    return meta, targets, predictions


def preflight(input_root, serial, datasets, models, modes, output_root=None):
    if type(serial) is not int or serial <= 0:
        raise ValueError('serial must be a positive integer')
    for values, allowed in [(datasets, DATASETS), (models, MODELS), (modes, MODES)]:
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f'Invalid or duplicate selection: {values}')
    root = Path(input_root).expanduser().resolve()
    serial_root = root / str(serial)
    out = (Path(output_root).expanduser() if output_root else serial_root / 'analysis').resolve()
    if inside(out, root) and not inside(out, serial_root / 'analysis'):
        raise ValueError('Output inside input root must use the selected serial/analysis subtree')
    scope = dict(serial=serial, datasets=list(datasets), models=list(models), modes=list(modes),
                 input_root=str(root))
    pairs = []
    for dataset, model, mode in itertools.product(datasets, models, modes):
        folder = serial_root / dataset / model / mode
        if not inside(folder.resolve(), serial_root) or inside(out, folder.resolve()) or inside(folder.resolve(), out):
            raise ValueError(f'Unsafe source/output overlap: {folder} / {out}')
        if inside(out, serial_root) and out.relative_to(serial_root).parts[0] in DATASETS:
            raise ValueError(f'Output is inside a source dataset folder: {out}')
        meta, targets, predictions = check_pair(folder, serial, dataset, model, mode)
        pairs.append(dict(folder=folder, dataset=dataset, model=model, mode=mode,
                          meta=meta, targets=targets, predictions=predictions))
    if out in (root, serial_root) or inside(root, out):
        raise ValueError('Output cannot replace the input root or serial folder')
    return out, scope, pairs


def source_provenance(pairs):
    return {f'{p["dataset"]}/{p["model"]}/{p["mode"]}':
            dict(manifest_sha256=sha256(p['folder'] / 'manifest.json'), files=p['meta']['files'])
            for p in pairs}


def code_digest():
    folder = Path(__file__).parent
    return digest({name: sha256(folder / name) for name in ('paired.py', 'metrics.py', 'analyze.py')})


def h_score_bounded(features, targets, temporary_root):
    # Existing H-score converts to float64. A disk-backed input avoids a full
    # float64 feature copy in RAM while retaining that shared metric definition.
    with tempfile.TemporaryDirectory(prefix='.hscore-', dir=temporary_root) as tmp:
        array = np.lib.format.open_memmap(Path(tmp) / 'features.npy', mode='w+',
                                         dtype='float64', shape=features.shape)
        for start in range(0, len(features), 1024):
            array[start:start + 1024] = features[start:start + 1024]
        value = regularized_h_score(array, targets)
        del array
    return value


def analyze_pair(pair, out, serial):
    folder = out / pair['dataset'] / pair['model'] / pair['mode']
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(serial=serial, dataset=pair['dataset'], model=pair['model'], mode=pair['mode'])
    rows, final_metrics = [], {}
    targets = np.asarray(pair['targets'])
    for layer in pair['meta']['layers']:
        fp = np.load(pair['folder'] / 'fp' / f'{layer}.npy', mmap_mode='r', allow_pickle=False)
        quant = np.load(pair['folder'] / 'w3_g128' / f'{layer}.npy', mmap_mode='r', allow_pickle=False)
        for condition, features in [('fp', fp), ('w3_g128', quant)]:
            row = clean(dict(**identity, condition=condition, layer=layer,
                             cka_vs_fp=centered_linear_cka(fp, features),
                             **class_equal_cosine_distances(features, targets), **norm_quantiles(features)))
            rows.append(row)
            if layer == 'final':
                final_metrics[condition] = row
        del fp, quant
    write_csv(folder / 'layer_metrics.csv', rows)
    summary = dict(**identity, **prediction_metrics(pair['predictions']['fp'], pair['predictions']['w3_g128']))
    fp = np.load(pair['folder'] / 'fp/final.npy', mmap_mode='r', allow_pickle=False)
    mean, components, explained = pca_fit(fp)
    np.savez(folder / 'fp_pca.npz', mean=mean, components=components, explained=explained)
    coords = {}
    for condition in CONDITIONS:
        features = np.load(pair['folder'] / condition / 'final.npy', mmap_mode='r', allow_pickle=False)
        coords[condition] = pca_project(features, mean, components)
        np.save(folder / f'coords_{condition}.npy', coords[condition])
        summary[f'{condition}_h_score_final'] = h_score_bounded(features, targets, folder)
        summary[f'{condition}_normalized_margin_q10'] = float(np.quantile(
            [float(row['normalized_margin']) for row in pair['predictions'][condition]], 0.1))
        for key in ('separability', 'norm_p99_p50'):
            summary[f'{condition}_{key}'] = final_metrics[condition][key]
        del features
    summary['final_cka'] = final_metrics['w3_g128']['cka_vs_fp']
    summary['feature_view'] = 'raw-image backbone features; CAL logits aggregate crop/flip views'
    write_json(folder / 'summary.json', summary)
    plot_pca(folder, identity, coords, targets, explained)
    del fp
    print(f'ANALYZED {pair["dataset"]}/{pair["model"]}/{pair["mode"]}', flush=True)
    return clean(summary)


def plot_pca(folder, identity, coords, targets, explained):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    # Saved coordinates cover all rows; plotting is bounded for very large tests.
    select = np.linspace(0, len(targets) - 1, min(len(targets), 20000), dtype=int)
    for ax, condition in zip(axes, CONDITIONS):
        ax.scatter(coords[condition][select, 0], coords[condition][select, 1],
                   c=targets[select], cmap='nipy_spectral', s=4, alpha=0.65)
        ax.set(title=f'{condition}: shared FP basis', xlabel=f'PC1 ({explained[0]:.1%})',
               ylabel=f'PC2 ({explained[1]:.1%})')
    fig.suptitle(f'{identity["dataset"]} / {identity["model"]} / {identity["mode"]}')
    fig.tight_layout()
    fig.savefig(folder / 'pca.png', dpi=150)
    plt.close(fig)


def output_names(pairs, full=False):
    names = list(GLOBAL_FILES)
    for pair in pairs:
        prefix = f'{pair["dataset"]}/{pair["model"]}/{pair["mode"]}'
        names.extend(f'{prefix}/{name}' for name in PAIR_FILES)
        if full:
            names.extend(f'{prefix}/coords_{c}.npy' for c in CONDITIONS)
    return names


def output_hashes(out, pairs):
    return {name: sha256(safe_file(out, name)) for name in output_names(pairs, full=True)
            if name != 'analysis_manifest.json'}


def verify_existing(out, scope, pairs):
    meta = json.loads((out / 'analysis_manifest.json').read_text(encoding='utf-8'))
    if (meta.get('status') != 'complete' or meta.get('scope') != scope
            or meta.get('sources') != source_provenance(pairs) or meta.get('code_sha256') != code_digest()):
        raise ValueError('Analysis scope/source/code provenance differs; rerun analyze')
    if meta.get('outputs') != output_hashes(out, pairs):
        raise ValueError('Existing analysis output checksums differ; rerun analyze')
    rows = [json.loads(safe_file(out, f'{p["dataset"]}/{p["model"]}/{p["mode"]}/summary.json')
                       .read_text(encoding='utf-8')) for p in pairs]
    for pair, row in zip(pairs, rows):
        if any(row.get(k) != v for k, v in dict(serial=scope['serial'], dataset=pair['dataset'],
                                              model=pair['model'], mode=pair['mode']).items()):
            raise ValueError('Per-pair analysis identity differs')
    return rows


def report(out, scope, rows):
    lines = [f'# W3/g128 paired analysis: serial {scope["serial"]}', '',
             'CPU metrics on existing complete paired test exports; no inference or GPU execution.',
             'CKA, class-equal cosine distances and norms cover every declared layer and sample.',
             'PCA fits FP final features once per pair and projects both conditions with that basis.',
             'CAL caveat: features come from the first raw-image backbone call; prediction logits retain crop/flip aggregation.',
             'Cross-dataset relationships are descriptive, confounded and not causal or significance tests.',
             'Dataset is the comparison unit within each model/mode; matrix cells are not pooled as independent datasets.', '']
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['mode'])].append(row)
    for (model, mode), group in grouped.items():
        lines += [f'## {model} / {mode}', '',
                  '| Dataset | N | FP acc | W3 acc | Relative loss | Final CKA | Damage | Rescue | Net R |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in group:
            values = [row[key] for key in ('dataset', 'n', 'fp_accuracy', 'accuracy', 'relative_loss',
                                           'final_cka', 'damage_count', 'rescue_count', 'net_damage_r')]
            lines.append('| ' + ' | '.join('NA' if v is None else f'{v:.5g}' if isinstance(v, float)
                                           else str(v) for v in values) + ' |')
        lines += ['', f'Dataset count: {len(group)}. Descriptive Spearman against relative loss:']
        for predictor in ('fp_separability', 'fp_h_score_final', 'fp_norm_p99_p50', 'fp_normalized_margin_q10'):
            usable = [r for r in group if r.get(predictor) is not None and r['relative_loss'] is not None]
            rho = spearman_description([r[predictor] for r in usable], [r['relative_loss'] for r in usable])
            lines.append(f'- {predictor}: rho={"NA" if rho is None else format(rho, ".5g")}; datasets={len(usable)}')
        lines.append('')
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def package(out, pairs):
    # Explicit whitelist; no features/logits, checkpoints, full coordinates or
    # arbitrary user files. Replace the prior ZIP only after a successful close.
    with tempfile.TemporaryDirectory(prefix='.package-', dir=out) as tmp:
        staged = Path(tmp) / 'results.zip'
        with zipfile.ZipFile(staged, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in output_names(pairs):
                archive.write(safe_file(out, name), name)
        staged.replace(out / 'w3_g128_results.zip')


def run(command, input_root, serial, datasets, models, modes, output_root=None):
    if command not in ('preflight', 'analyze', 'report', 'package', 'all'):
        raise ValueError(f'Unknown command: {command}')
    out, scope, pairs = preflight(input_root, serial, datasets, models, modes, output_root)
    sources = source_provenance(pairs)
    if command == 'preflight':
        print(f'PREFLIGHT complete: {len(pairs)} paired exports', flush=True)
        return out
    if command in ('analyze', 'all'):
        out.mkdir(parents=True, exist_ok=True)
        # Validate generated paths before touching any prior output.
        for name in output_names(pairs, full=True):
            safe_file(out, name)
        write_json(out / 'analysis_manifest.json', dict(status='running', scope=scope))
        rows = [analyze_pair(p, out, serial) for p in pairs]
        write_csv(out / 'all_pairs.csv', rows)
        write_csv(out / 'coverage.csv', [dict(serial=serial, dataset=p['dataset'], model=p['model'],
                  mode=p['mode'], n_samples=p['meta']['n_samples'], status='complete') for p in pairs])
        report(out, scope, rows)
        if source_provenance(pairs) != sources:
            raise ValueError('Source manifests changed during analysis')
        write_json(out / 'analysis_manifest.json', dict(status='complete', scope=scope, sources=sources,
                   code_sha256=code_digest(), outputs=output_hashes(out, pairs), execution='CPU'))
    else:
        rows = verify_existing(out, scope, pairs)
    if command == 'report':
        report(out, scope, rows)
    if command in ('all', 'package'):
        verify_existing(out, scope, pairs)
        package(out, pairs)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'analyze', 'report', 'package', 'all'))
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--serial', required=True, type=int)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    try:
        run(args.command, args.input_root, args.serial, args.datasets, args.models, args.modes, args.output_root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
