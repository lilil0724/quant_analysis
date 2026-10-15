"""Plot paired FP/W3 final features from a TGDA W3/g128 source serial.

Each dataset/model/mode uses the same selected sample rows for FP and W3 and
fits one joint t-SNE. Coordinates from separately fitted t-SNE runs must not
be compared. Class sampling keeps every sample from 10% of classes (5-20
classes, or all when fewer than 5 exist). Source arrays are read only.
"""

import argparse
import csv
import inspect
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from .analyze import DATASETS, sha256
from .paired import MODELS, MODES, digest

CONDITIONS = ('fp', 'w3_g128')


def class_sample_indices(targets, fraction, seed):
    """Choose classes reproducibly, then retain all their original rows."""
    labels = np.asarray(targets)
    if (labels.ndim != 1 or not len(labels) or not np.isfinite(fraction)
            or not 0 < fraction <= 1):
        raise ValueError('Expected nonempty 1D labels and class-fraction in (0, 1]')
    classes = np.unique(labels)
    count = min(len(classes), 20, max(5, math.ceil(len(classes) * fraction)))
    selected = np.random.default_rng(seed).choice(classes, size=count, replace=False)
    return np.flatnonzero(np.isin(labels, selected))


def sample_indices(targets, limit, seed):
    """Sample matched rows, covering each class when the budget permits."""
    labels = np.asarray(targets)
    if labels.ndim != 1 or not len(labels) or limit < 2:
        raise ValueError('Expected nonempty 1D labels and max-samples >= 2')
    if len(labels) <= limit:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(labels, return_inverse=True)
    order = np.lexsort((rng.random(len(labels)), inverse))
    starts = np.r_[0, np.flatnonzero(inverse[order][1:] != inverse[order][:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    if len(starts) > limit:
        # When classes outnumber the budget, preserve repeated examples from
        # selected classes rather than plotting only isolated singletons.
        groups = rng.choice(len(starts), size=max(1, limit // 2), replace=False)
        first = list(order[starts[groups]])
        first.extend(int(order[starts[group] + 1]) for group in groups
                     if ends[group] - starts[group] > 1)
        first = np.asarray(first, dtype=np.int64)
        available = np.zeros(len(labels), dtype=bool)
        for group in groups:
            available[order[starts[group]:ends[group]]] = True
    else:
        first = order[starts]
        available = np.ones(len(labels), dtype=bool)
    remainder = limit - len(first)
    if remainder:
        available[first] = False
        if np.count_nonzero(available) < remainder:
            available[:] = True
            available[first] = False
        extra = rng.choice(np.flatnonzero(available), size=remainder,
                           replace=False)
        first = np.concatenate((first, extra))
    return np.sort(first)


def _inside(path, parent):
    return path == parent or parent in path.parents


def _source_pair(folder, serial, dataset, model, mode):
    if not folder.is_dir():
        raise ValueError(f'Missing pair: {folder}')
    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Missing pair manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    identity = dict(status='complete', serial=serial, dataset=dataset,
                    model=model, checkpoint_mode=mode)
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError(f'Incomplete or mismatched pair: {folder}')
    n = manifest.get('n_samples')
    layers = manifest.get('layers')
    dtype_name = manifest.get('feature_dtype')
    quant = manifest.get('quantization', {})
    if (type(n) is not int or n < 2 or not isinstance(layers, dict)
            or 'final' not in layers or dtype_name not in ('float16', 'float32')
            or not isinstance(quant, dict)
            or any(quant.get(key, 'missing') != value for key, value in
                   (('weight_bits', 3), ('weight_group_size', 128),
                    ('activation_bits', None)))):
        raise ValueError(f'Invalid W3/g128 final-feature manifest: {folder}')
    shape = layers['final']
    if (not isinstance(shape, list) or len(shape) != 2 or shape[0] != n
            or type(shape[1]) is not int or shape[1] < 2):
        raise ValueError(f'Invalid final feature shape: {folder}')
    required = ('samples.json', 'fp/final.npy', 'w3_g128/final.npy')
    recorded = manifest.get('files')
    if not isinstance(recorded, dict) or any(name not in recorded for name in required):
        raise ValueError(f'Missing source checksums: {folder}')
    hashes = {}
    for name in required:
        path = folder / name
        if not path.is_file():
            raise ValueError(f'Missing source file: {path}')
        actual = sha256(path)
        if actual != recorded[name]:
            raise ValueError(f'Source checksum mismatch: {path}')
        hashes[name] = actual
    samples = json.loads((folder / 'samples.json').read_text(encoding='utf-8'))
    ids, targets = samples.get('sample_ids'), samples.get('targets')
    if (not isinstance(ids, list) or not isinstance(targets, list)
            or len(ids) != n or len(targets) != n
            or any(not isinstance(item, str) for item in ids)
            or len(set(ids)) != n
            or any(type(label) is not int or label < 0 for label in targets)):
        raise ValueError(f'Invalid paired sample IDs or targets: {folder}')
    if manifest.get('samples_sha256') != digest({'ids': ids, 'targets': targets}):
        raise ValueError(f'Sample identity digest mismatch: {folder}')
    for condition in CONDITIONS:
        arr = np.load(folder / condition / 'final.npy', mmap_mode='r', allow_pickle=False)
        if list(arr.shape) != shape or arr.dtype != np.dtype(dtype_name):
            raise ValueError(f'Invalid {condition} final features: {folder}')
        del arr
    return dict(folder=folder, manifest_sha256=sha256(manifest_path),
                source_sha256=hashes, targets=np.asarray(targets, dtype=np.int64),
                n=n, width=shape[1], dataset=dataset, model=model, mode=mode)


def _embed(fp, quant, perplexity, iterations, seed):
    from sklearn import __version__ as sklearn_version
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    joined = np.concatenate((fp, quant), axis=0).astype(np.float64)
    norms = np.linalg.norm(joined, axis=1)
    if not np.isfinite(joined).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Selected features must be finite and have nonzero norm')
    joined = (joined / norms[:, None]).astype(np.float32)
    pca_components = None
    if joined.shape[1] > 50:
        pca_components = min(50, joined.shape[0] - 1, joined.shape[1])
        joined = PCA(n_components=pca_components, svd_solver='randomized',
                     random_state=seed).fit_transform(joined)
    effective_perplexity = min(float(perplexity), (len(joined) - 1) / 3)
    tsne_args = dict(n_components=2, perplexity=effective_perplexity,
                     init='pca', learning_rate='auto', random_state=seed)
    iteration_key = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
    tsne_args[iteration_key] = iterations
    coords = TSNE(**tsne_args).fit_transform(joined)
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite t-SNE coordinates')
    return coords, effective_perplexity, pca_components, sklearn_version


def _plot(path, coords, labels, pair, chosen, effective_perplexity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    count = len(chosen)
    unique = np.unique(labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True,
                             layout='constrained')
    discrete = len(unique) <= 20
    if discrete:
        colors = plt.get_cmap('tab20').colors[:len(unique)]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(len(unique) + 1) - .5, len(unique))
        mapped = np.searchsorted(unique, labels)
    else:
        cmap, norm, mapped = 'viridis', None, labels
    for ax, name, offset in zip(axes, ('FP', 'W3/g128'), (0, count)):
        scatter = ax.scatter(coords[offset:offset + count, 0],
                             coords[offset:offset + count, 1], c=mapped,
                             cmap=cmap, norm=norm,
                             s=7, alpha=.65, rasterized=True)
        ax.set(title=name, xlabel='t-SNE 1', ylabel='t-SNE 2')
        ax.set_aspect('equal', adjustable='box')
    colorbar = fig.colorbar(scatter, ax=axes, label='class ID',
                            fraction=.027, pad=.025)
    if discrete:
        # Every selected class has its own labeled color, including population.
        _, populations = np.unique(labels, return_counts=True)
        colorbar.set_ticks(np.arange(len(unique)), labels=[
            f'{label} (n={population})' for label, population in zip(unique, populations)])
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"{pair['dataset']} / {pair['model']} / {pair['mode']}\n"
                 f"{count:,} paired samples of {pair['n']:,}; "
                 f"{len(unique):,} classes shown; perplexity {effective_perplexity:.1f}")
    fig.savefig(path, format='png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def _render_pair(pair, destination, chosen, max_samples, perplexity, iterations,
                 seed, overwrite, sampling, class_fraction):
    params = dict(sampling=sampling,
                  class_fraction=class_fraction if sampling == 'classes' else None,
                  min_classes=5 if sampling == 'classes' else None,
                  max_classes=20 if sampling == 'classes' else None,
                  max_samples=max_samples if sampling == 'samples' else None,
                  perplexity=perplexity,
                  iterations=iterations, seed=seed, normalization='per-sample L2',
                  fit='joint FP and W3, with shared PCA-50 when width > 50')
    code_sha256 = sha256(Path(__file__))
    method_path = destination / 'method.json'
    plot_path = destination / 'tsne.png'
    csv_path = destination / 'coordinates.csv'
    if method_path.is_file() and not overwrite:
        meta = json.loads(method_path.read_text(encoding='utf-8'))
        if (meta.get('parameters') == params
                and meta.get('code_sha256') == code_sha256
                and meta.get('manifest_sha256') == pair['manifest_sha256']
                and meta.get('source_sha256') == pair['source_sha256']
                and plot_path.is_file() and csv_path.is_file()
                and meta.get('output_sha256') ==
                    {'tsne.png': sha256(plot_path), 'coordinates.csv': sha256(csv_path)}):
            print(f"SKIP {pair['dataset']}/{pair['model']}/{pair['mode']}", flush=True)
            return
        raise ValueError(f'Existing t-SNE differs; use --overwrite: {destination}')
    if not overwrite and (plot_path.exists() or csv_path.exists()):
        raise ValueError(f'Incomplete t-SNE output; use --overwrite: {destination}')
    data = []
    for condition in CONDITIONS:
        arr = np.load(pair['folder'] / condition / 'final.npy', mmap_mode='r',
                      allow_pickle=False)
        data.append(np.asarray(arr[chosen], dtype=np.float32))
        del arr
    coords, effective_perplexity, pca_components, sklearn_version = _embed(
        *data, perplexity, iterations, seed)
    for name, expected_hash in pair['source_sha256'].items():
        if sha256(pair['folder'] / name) != expected_hash:
            raise ValueError(f'Source changed during t-SNE: {pair["folder"] / name}')
    destination.mkdir(parents=True, exist_ok=True)
    png_tmp = destination / '.tsne.png.tmp'
    csv_tmp = destination / '.coordinates.csv.tmp'
    method_tmp = destination / '.method.json.tmp'
    labels = pair['targets'][chosen]
    classes, populations = np.unique(labels, return_counts=True)
    _plot(png_tmp, coords, labels, pair, chosen, effective_perplexity)
    with csv_tmp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('row', 'target', 'fp_x', 'fp_y', 'w3_x', 'w3_y'))
        for j, original in enumerate(chosen):
            writer.writerow((int(original), int(labels[j]),
                             *map(float, coords[j]), *map(float, coords[j + len(chosen)])))
    meta = dict(status='complete', serial=int(pair['folder'].parents[2].name),
                dataset=pair['dataset'], model=pair['model'], mode=pair['mode'],
                total_samples=pair['n'], selected_samples=len(chosen),
                total_classes=int(len(np.unique(pair['targets']))),
                selected_classes=int(len(np.unique(labels))), fit_points=len(coords),
                selected_class_ids=classes.tolist(),
                class_sample_counts={str(label): int(n) for label, n in zip(classes, populations)},
                effective_perplexity=effective_perplexity,
                pca_components=pca_components, sklearn_version=sklearn_version,
                parameters=params, code_sha256=code_sha256,
                manifest_sha256=pair['manifest_sha256'],
                source_sha256=pair['source_sha256'],
                output_sha256={'tsne.png': sha256(png_tmp),
                               'coordinates.csv': sha256(csv_tmp)},
                interpretation='Exploratory 2D neighborhood view; axes and gaps are not accuracy or causal measures.')
    method_tmp.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    os.replace(png_tmp, plot_path)
    os.replace(csv_tmp, csv_path)
    os.replace(method_tmp, method_path)
    print(f"PLOTTED {pair['dataset']}/{pair['model']}/{pair['mode']} "
          f"({len(chosen)}/{pair['n']} samples)", flush=True)


def run(input_root, serial, datasets, models, modes, *, output_root=None,
        max_samples=2000, perplexity=30.0, iterations=750, seed=100,
        overwrite=False, stage='run', sampling='classes', class_fraction=.1):
    """Preflight the selected matrix, then generate or resume isolated outputs."""
    if type(serial) is not int or serial <= 0:
        raise ValueError('serial must be a positive integer')
    if (stage not in ('run', 'preflight') or sampling not in ('classes', 'samples')
            or (sampling == 'samples' and max_samples < 2)
            or not np.isfinite(class_fraction) or not 0 < class_fraction <= 1
            or not np.isfinite(perplexity) or perplexity <= 0
            or iterations < 250 or type(seed) is not int or seed < 0):
        raise ValueError('Invalid stage, sampling, class-fraction, max-samples, perplexity, iterations, or seed')
    for values, allowed, label in ((datasets, DATASETS, 'datasets'),
                                   (models, MODELS, 'models'),
                                   (modes, MODES, 'modes')):
        if not values or len(values) != len(set(values)) or not set(values) <= set(allowed):
            raise ValueError(f'Invalid or duplicate {label}: {values}')
    root = Path(input_root).expanduser().resolve()
    if root.name == str(serial):
        root = root.parent
    serial_root = root / str(serial)
    output_name = 'tsne_classes' if sampling == 'classes' else 'tsne'
    output = (Path(output_root).expanduser() if output_root else
              serial_root / 'analysis' / output_name).resolve()
    if (output in (root, serial_root) or _inside(root, output)
            or (_inside(output, root)
                and not _inside(output, serial_root / 'analysis'))):
        raise ValueError(f'Output cannot replace or overlap source pairs: {output}')
    pairs = []
    for dataset, model, mode in itertools.product(datasets, models, modes):
        folder = serial_root / dataset / model / mode
        if _inside(output, folder.resolve()) or _inside(folder.resolve(), output):
            raise ValueError(f'Output overlaps input pair: {folder}')
        pairs.append(_source_pair(folder, serial, dataset, model, mode))
    # Select before any output is written, so a bad subset cannot leave a
    # partially rendered matrix. Class IDs must agree across models/modes.
    selected_by_dataset = {}
    for pair in pairs:
        chosen = (class_sample_indices(pair['targets'], class_fraction, seed)
                  if sampling == 'classes' else
                  sample_indices(pair['targets'], max_samples, seed))
        if len(chosen) < 2:
            raise ValueError(f'Too few selected samples: {pair["folder"]}')
        pair['chosen'] = chosen
        classes = np.unique(pair['targets'][chosen]).tolist()
        if sampling == 'classes':
            previous = selected_by_dataset.setdefault(pair['dataset'], classes)
            if previous != classes:
                raise ValueError(f'Selected class IDs differ across pairs: {pair["dataset"]}')
        print(f'SELECTED {pair["dataset"]}/{pair["model"]}/{pair["mode"]}: '
              f'{len(classes)} classes, {len(chosen)}/{pair["n"]} samples; '
              f'{2 * len(chosen)} joint fit points', flush=True)
    print(f'PREFLIGHT complete: {len(pairs)} paired exports', flush=True)
    if stage == 'preflight':
        return output
    for pair in pairs:
        destination = output / pair['dataset'] / pair['model'] / pair['mode']
        _render_pair(pair, destination, pair['chosen'], max_samples, perplexity,
                     iterations, seed, overwrite, sampling, class_fraction)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'run'))
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--serial', required=True, type=int)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--sampling', choices=('classes', 'samples'), default='classes',
                        help='classes: 10%% of classes (5-20), all their samples; samples: legacy point budget')
    parser.add_argument('--class-fraction', type=float, default=.1,
                        help='Fraction of classes before clamping to 5-20 (default .1)')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Paired point budget for --sampling samples only; ignored in classes mode')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--iterations', type=int, default=750)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        run(args.input_root, args.serial, args.datasets, args.models, args.modes,
            output_root=args.output_root, max_samples=args.max_samples,
            perplexity=args.perplexity, iterations=args.iterations,
            seed=args.seed, overwrite=args.overwrite, stage=args.stage,
            sampling=args.sampling, class_fraction=args.class_fraction)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
