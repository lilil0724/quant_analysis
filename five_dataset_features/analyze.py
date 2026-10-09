"""Analyze copied full inference outputs locally and make a small results ZIP.

python -m five_dataset_features.analyze dataset --root C:/data/hqq --dataset cub
python -m five_dataset_features.analyze report --root C:/data/hqq
python -m five_dataset_features.analyze package --root C:/data/hqq
"""
import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import (centered_linear_cka, class_equal_cosine_distances,
                      norm_quantiles, pca_fit, pca_project, prediction_metrics,
                      regularized_h_score, spearman_description)


DATASETS = ('aircraft', 'cars', 'cub', 'dogs', 'flowers', 'food', 'inat17', 'moe', 'nabirds', 'pets', 'soyageing', 'soygene', 'soyglobal', 'soylocal', 'vegfru', 'cotton')
LAYERS, WIDTH = 13, 768


def grid():
    result = ['fp']
    for w in (3, 4):
        for g in (8, 128):
            stem = f'w{w}_g{g}'
            result += [stem, f'{stem}_a8', f'{stem}_a8_qkv',
                       f'{stem}_a4', f'{stem}_a4_qkv']
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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


def check_dataset(root, dataset):
    folder = root / dataset
    meta = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
    n = meta['n_samples']
    if meta['dataset'] != dataset or len(meta['sample_ids']) != n or len(meta['targets']) != n:
        raise RuntimeError(f'{dataset}: invalid dataset manifest')
    arrays, predictions = {}, {}
    for name in grid():
        path = folder / name
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'complete' or manifest['n_samples'] != n or manifest['dataset'] != dataset:
            raise RuntimeError(f'{dataset}/{name}: incomplete or incompatible manifest')
        condition = manifest['condition']
        if condition['name'] != name:
            raise RuntimeError(f'{dataset}/{name}: condition name mismatch')
        identity_data = {'schema': meta['schema'], 'checkpoint': meta['checkpoint_sha256'],
                         'data': meta['data_sha256'], 'code': meta['code_sha256'],
                         'hqq': meta['hqq_commit'], 'dtype': meta['feature_dtype'],
                         'execution': meta['execution'],
                         'condition': condition}
        encoded = json.dumps(identity_data, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False).encode('utf-8')
        if manifest['identity'] != hashlib.sha256(encoded).hexdigest():
            raise RuntimeError(f'{dataset}/{name}: identity mismatch')
        for filename, expected in manifest['files'].items():
            if sha256(path / filename) != expected:
                raise RuntimeError(f'{dataset}/{name}: checksum mismatch for {filename}')
        feature = np.load(path / 'features.npy', mmap_mode='r', allow_pickle=False)
        if feature.shape != (n, LAYERS, WIDTH) or str(feature.dtype) != meta['feature_dtype']:
            raise RuntimeError(f'{dataset}/{name}: feature shape {feature.shape}')
        rows = read_csv(path / 'predictions.csv')
        if len(rows) != n or [row['sample_id'] for row in rows] != meta['sample_ids']:
            raise RuntimeError(f'{dataset}/{name}: prediction/sample ID mismatch')
        if [int(row['target']) for row in rows] != meta['targets']:
            raise RuntimeError(f'{dataset}/{name}: target mismatch')
        arrays[name], predictions[name] = feature, rows
    return meta, arrays, predictions


def make_figures(out, dataset, layers, summary, coords, fp_rows, predictions, explained, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = grid()[1:]
    matrix = np.asarray([[next(row['cka'] for row in layers if row['condition'] == name and row['layer'] == layer)
                          for layer in range(LAYERS)] for name in names], float)
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(matrix, vmin=0, vmax=1, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xticks(range(LAYERS), [*range(1, 13), 'final'])
    ax.set_title(f'{dataset}: centered linear CKA vs FP')
    fig.colorbar(im, ax=ax, label='CKA')
    fig.tight_layout()
    fig.savefig(out / 'cka_heatmap.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in names:
        row = [item['cka'] for item in layers if item['condition'] == name]
        ax.plot(range(LAYERS), row, alpha=0.65, label=name)
    ax.set(xticks=range(LAYERS), xticklabels=[*range(1, 13), 'final'],
           ylabel='centered linear CKA', xlabel='feature position', title=f'{dataset}: CKA profiles')
    ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(out / 'cka_curves.png', dpi=150)
    plt.close(fig)
    # The worst condition is chosen descriptively. Every condition has saved
    # full-test coordinates locally and representative coordinates in ZIP.
    finite = [row for row in summary if row['condition'] != 'fp' and row['relative_loss'] is not None]
    worst = max(finite, key=lambda row: row['relative_loss'])['condition'] if finite else names[0]
    fp_coord, q_coord = coords['fp'], coords[worst]
    correct = np.array([int(row['correct']) for row in predictions[worst]], bool)
    swapped = np.array([a['prediction'] != b['prediction'] for a, b in zip(fp_rows, predictions[worst])])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(fp_coord[:, 0], fp_coord[:, 1], c=labels, cmap='nipy_spectral', s=3, alpha=0.65)
    axes[0].set_title('FP class distribution')
    axes[1].scatter(q_coord[:, 0], q_coord[:, 1], c=np.where(correct, '#238b45', '#cb181d'), s=3, alpha=0.65)
    axes[1].set_title(f'{worst}: correct (green) / wrong (red)')
    axes[2].scatter(q_coord[:, 0], q_coord[:, 1], c=np.where(swapped, '#d94801', '#969696'), s=3, alpha=0.65)
    axes[2].set_title(f'{worst}: swap (orange) / stable (gray)')
    for ax in axes:
        ax.set(xlabel=f'PC1 ({explained[0]:.1%})', ylabel=f'PC2 ({explained[1]:.1%})')
    fig.tight_layout()
    fig.savefig(out / 'pca_distribution_correct_swap.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    by_name = {row['condition']: row for row in summary}
    final = {row['condition']: row for row in layers if row['layer'] == 12}
    for w in (3, 4):
        for g in (8, 128):
            stem = f'w{w}_g{g}'
            for act in (8, 4):
                sequence = [stem, f'{stem}_a{act}', f'{stem}_a{act}_qkv']
                axes[0].plot(range(3), [by_name[n]['accuracy'] for n in sequence],
                             marker='o', alpha=0.7, label=f'W{w}/g{g}/A{act}')
                axes[1].plot(range(3), [final[n]['cka'] for n in sequence], marker='o', alpha=0.7)
                axes[2].plot(range(3), [final[n]['separability'] for n in sequence], marker='o', alpha=0.7)
    for ax, title in zip(axes, ('Accuracy', 'Final CKA vs FP', 'Final class-equal separation')):
        ax.set(xticks=range(3), xticklabels=['weight', 'Linear', 'Linear+QKV'], title=title)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'activation_transition.png', dpi=150)
    plt.close(fig)


def analyze_dataset(root, dataset):
    meta, arrays, predictions = check_dataset(root, dataset)
    output = root / 'analysis' / dataset
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(meta['targets'])
    fp = arrays['fp']
    mean, components, explained = pca_fit(fp[:, 12, :])
    np.savez(output / 'fp_pca.npz', mean=mean, components=components, explained=explained)
    fp_rows = predictions['fp']
    coords = {}
    summary, layers = [], []
    for name in grid():
        feature = arrays[name]
        coords[name] = pca_project(feature[:, 12, :], mean, components)
        np.save(output / f'coords_{name}.npy', coords[name])
        row = {'dataset': dataset, 'condition': name,
               **prediction_metrics(fp_rows, predictions[name])}
        row['h_score_final'] = regularized_h_score(feature[:, 12, :], labels)
        row['normalized_margin_q10'] = float(np.quantile(
            [float(item['normalized_margin']) for item in predictions[name]], 0.1))
        summary.append(row)
        for layer in range(LAYERS):
            x = feature[:, layer, :]
            layer_row = {'dataset': dataset, 'condition': name, 'layer': layer,
                         'cka': centered_linear_cka(fp[:, layer, :], x),
                         **class_equal_cosine_distances(x, labels),
                         **norm_quantiles(x),
                         'h_score_fp_only': regularized_h_score(x, labels) if name == 'fp' else None}
            layers.append(layer_row)
        print(f'ANALYZED {dataset}/{name}', flush=True)
    # Only the FP final features define the candidate predictors.
    fp_final = next(row for row in layers if row['condition'] == 'fp' and row['layer'] == 12)
    indicator = {'dataset': dataset, 'fp_accuracy': summary[0]['fp_accuracy'],
                 'fp_separability': fp_final['separability'],
                 'fp_d_inter': fp_final['d_inter'], 'fp_d_intra': fp_final['d_intra'],
                 'fp_h_score': summary[0]['h_score_final'],
                 'fp_norm_p50': fp_final['norm_p50'], 'fp_norm_p95': fp_final['norm_p95'],
                 'fp_norm_p99': fp_final['norm_p99'],
                 'fp_norm_p99_p50': fp_final['norm_p99_p50'],
                 'fp_normalized_margin_q10': summary[0]['normalized_margin_q10']}
    (output / 'fp_indicators.json').write_text(json.dumps(indicator, indent=2) + '\n', encoding='utf-8')
    write_csv(output / 'summary.csv', summary)
    write_csv(output / 'layer_metrics.csv', layers)
    # First real batch: measure serialization error in CKA and compare with A8.
    probe_rows = []
    fp_probe = np.load(root / dataset / 'fp' / 'precision_probe.npy', allow_pickle=False)
    for name in grid():
        q_probe = np.load(root / dataset / name / 'precision_probe.npy', allow_pickle=False)
        for layer in range(LAYERS):
            full = centered_linear_cka(fp_probe[:, layer, :], q_probe[:, layer, :])
            rounded = centered_linear_cka(fp_probe[:, layer, :].astype(np.float16),
                                          q_probe[:, layer, :].astype(np.float16))
            probe_rows.append({'dataset': dataset, 'condition': name, 'layer': layer,
                               'cka_fp32': full, 'cka_fp16': rounded,
                               'absolute_storage_delta': None if full is None or rounded is None else abs(full - rounded)})
    write_csv(output / 'precision_probe.csv', probe_rows)
    make_figures(output, dataset, layers, summary, coords, fp_rows, predictions, explained, labels)
    return summary


def aggregate_report(root):
    out = root / 'analysis'
    summaries, indicators = [], []
    for dataset in DATASETS:
        summaries.extend(read_csv(out / dataset / 'summary.csv'))
        indicators.append(json.loads((out / dataset / 'fp_indicators.json').read_text(encoding='utf-8')))
    write_csv(out / 'all_conditions.csv', summaries)
    write_csv(out / 'fp_indicators.csv', indicators)
    predictors = [key for key in indicators[0] if key != 'dataset']
    scopes = {'weight_only': lambda name: '_a' not in name,
              'linear_a8': lambda name: '_a8' in name and '_qkv' not in name,
              'linear_a4': lambda name: '_a4' in name and '_qkv' not in name,
              'qkv_a8': lambda name: '_a8_qkv' in name,
              'qkv_a4': lambda name: '_a4_qkv' in name}
    associations = []
    for scope, select in scopes.items():
        losses = []
        for dataset in DATASETS:
            values = [float(row['relative_loss']) for row in summaries
                      if row['dataset'] == dataset and row['condition'] != 'fp' and
                      select(row['condition']) and row['relative_loss'] not in ('', 'None')]
            losses.append(float(np.mean(values)) if values else np.nan)
        for predictor in predictors:
            x = [float(item[predictor]) if item[predictor] is not None else np.nan for item in indicators]
            associations.append({'scope': scope, 'predictor': predictor,
                                 'n_datasets': int((np.isfinite(x) & np.isfinite(losses)).sum()),
                                 'spearman_rho': spearman_description(x, losses)})
    write_csv(out / 'descriptive_spearman.csv', associations)
    report = ['# Five-dataset ViT-B/16 HQQ + activation fake quantization', '',
              'All primary statistics use matched full test sets. Five dataset-model pairs are the units of comparison; quantization conditions are repeated measurements.',
              'Spearman coefficients are descriptive candidate screening only. No significance threshold or predictive generalization claim is made.',
              'QKV and Linear activations use fake quantization; no integer kernel or speedup is claimed.', '',
              '## Files', '',
              '- `all_conditions.csv`: accuracy, relative loss, swap, damage, rescue and R for all 105 conditions.',
              '- `<dataset>/layer_metrics.csv`: 13-layer CKA, class-equal cosine distances, separation and norms.',
              '- `fp_indicators.csv`: pre-quantization predictors for the selected dataset-model pairs.',
              '- `descriptive_spearman.csv`: dataset-level associations by condition family.',
              '- `<dataset>/precision_probe.csv`: first real batch FP32 versus FP16 storage sensitivity.',
              '- `<dataset>/activation_transition.png`: weight-only to Linear to Linear+QKV comparisons.',
              '- Local `<dataset>/fp_pca.npz`: FP-only PCA basis; all condition coordinates share this projection.', '',
              'Undefined values remain empty in CSV, including relative loss at zero FP accuracy and R without swaps.', '']
    (out / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    version_files = [Path(__file__), Path(__file__).with_name('metrics.py'),
                     Path(__file__).with_name('smoke.py')]
    (out / 'analysis_version.json').write_text(json.dumps(
        {path.name: sha256(path) for path in version_files}, indent=2) + '\n',
        encoding='utf-8')
    make_cross_scatter(out, indicators, summaries, scopes)


def make_cross_scatter(out, indicators, summaries, scopes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    predictors = ('fp_separability', 'fp_h_score', 'fp_norm_p99_p50',
                  'fp_normalized_margin_q10', 'fp_accuracy')
    fig, axes = plt.subplots(len(predictors), len(scopes), figsize=(19, 16))
    for col, (scope, select) in enumerate(scopes.items()):
        for row, predictor in enumerate(predictors):
            ax = axes[row, col]
            for item in indicators:
                values = [float(record['relative_loss']) for record in summaries
                          if record['dataset'] == item['dataset'] and record['condition'] != 'fp'
                          and select(record['condition']) and record['relative_loss'] not in ('', 'None')]
                if values and item[predictor] is not None:
                    ax.scatter(item[predictor], np.mean(values))
                    ax.annotate(item['dataset'], (item[predictor], np.mean(values)), fontsize=6)
            ax.set(title=scope if row == 0 else '', xlabel=predictor,
                   ylabel='Mean relative loss' if col == 0 else '')
    fig.tight_layout()
    fig.savefig(out / 'fp_predictors_vs_loss.png', dpi=150)
    plt.close(fig)


def exemplar_indices(meta):
    by_class = defaultdict(list)
    for index, target in enumerate(meta['targets']):
        by_class[int(target)].append(index)
    classes = sorted(target for target, indices in by_class.items() if len(indices) >= 2)
    if not classes:
        raise RuntimeError('No class has two eligible exemplar samples')
    random = np.random.default_rng(100)
    chosen = random.choice(classes, size=min(10, len(classes)), replace=False)
    return sorted(int(index) for target in chosen for index in random.choice(by_class[int(target)], size=2, replace=False))


def checked_exemplars(root, dataset, meta):
    """Accept only the display images saved by the matching server inference."""
    source = root / dataset / 'exemplars'
    manifest_path = source / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'{dataset}: missing copied exemplar manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    indices = exemplar_indices(meta)
    names = [f'{index:06d}.jpg' for index in indices]
    if (manifest.get('status') != 'complete' or
            manifest.get('dataset') != dataset or
            manifest.get('data_sha256') != meta['data_sha256'] or
            manifest.get('seed') != 100 or
            manifest.get('indices') != indices or
            manifest.get('sample_ids') != [meta['sample_ids'][i] for i in indices] or
            not isinstance(manifest.get('files'), dict) or
            set(manifest['files']) != set(names)):
        raise RuntimeError(f'{dataset}: missing or incompatible exemplar manifest')
    for name in names:
        if not (source / name).is_file() or sha256(source / name) != manifest['files'][name]:
            raise RuntimeError(f'{dataset}: exemplar checksum mismatch: {name}')
    return indices, source


def package(root):
    out = root / 'analysis'
    if not (out / 'REPORT.md').is_file():
        raise RuntimeError('Run report after all selected dataset analyses')
    stage = Path(tempfile.mkdtemp(prefix='.download-package-', dir=out))
    try:
        for name in ('REPORT.md', 'analysis_version.json', 'all_conditions.csv', 'fp_indicators.csv',
                     'descriptive_spearman.csv', 'fp_predictors_vs_loss.png'):
            shutil.copy2(out / name, stage / name)
        for dataset in DATASETS:
            meta, arrays, predictions = check_dataset(root, dataset)
            source = out / dataset
            dest = stage / dataset
            dest.mkdir()
            for name in ('summary.csv', 'layer_metrics.csv', 'precision_probe.csv',
                         'fp_indicators.json', 'cka_heatmap.png', 'cka_curves.png',
                         'pca_distribution_correct_swap.png', 'activation_transition.png'):
                shutil.copy2(source / name, dest / name)
            shutil.copy2(root / dataset / 'dataset.json', dest / 'dataset.json')
            for name in grid():
                shutil.copy2(root / dataset / name / 'manifest.json', dest / f'manifest_{name}.json')
            indices, exemplar_source = checked_exemplars(root, dataset, meta)
            image_dir = dest / 'images'
            image_dir.mkdir()
            shutil.copy2(exemplar_source / 'manifest.json', dest / 'exemplar_manifest.json')
            for index in indices:
                name = f'{index:06d}.jpg'
                shutil.copy2(exemplar_source / name, image_dir / name)
            features = np.stack([np.asarray(arrays[name][indices, 12, :]) for name in grid()])
            np.save(dest / 'representative_final_features.npy', features)
            rows = []
            for name in grid():
                coord = np.load(source / f'coords_{name}.npy', mmap_mode='r')
                for index in indices:
                    rows.append({'condition': name, 'sample_id': meta['sample_ids'][index],
                                 'target': meta['targets'][index], 'index': index,
                                 'prediction': predictions[name][index]['prediction'],
                                 'correct': predictions[name][index]['correct'],
                                 'pc1': float(coord[index, 0]), 'pc2': float(coord[index, 1])})
            write_csv(dest / 'representative_predictions_and_coordinates.csv', rows)
        checksums = {}
        for path in stage.rglob('*'):
            if path.is_file():
                checksums[str(path.relative_to(stage)).replace('\\', '/')] = sha256(path)
        (stage / 'CHECKSUMS.json').write_text(json.dumps(checksums, indent=2) + '\n', encoding='utf-8')
        archive = out / 'five_dataset_hqq_results.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipout:
            for path in stage.rglob('*'):
                if path.is_file():
                    zipout.write(path, path.relative_to(stage))
        size = archive.stat().st_size
        print(f'Local package: {archive} ({size / 1e6:.2f} MB; target <= 50 MB)')
        return archive
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    global DATASETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('dataset', 'report', 'package'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS,
                        help='Report/package scope; defaults to all 16 datasets')
    args = parser.parse_args()
    if args.datasets:
        DATASETS = tuple(dict.fromkeys(args.datasets))
    if args.command == 'dataset':
        if not args.dataset:
            parser.error('--dataset is required for dataset analysis')
        analyze_dataset(args.root, args.dataset)
    elif args.command == 'report':
        aggregate_report(args.root)
    else:
        package(args.root)


if __name__ == '__main__':
    main()
