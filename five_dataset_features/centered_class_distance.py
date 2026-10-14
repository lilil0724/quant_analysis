"""Recompute FP final-feature geometry from existing paired FP/W3 exports."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

DATASETS = ('aircraft', 'cars', 'cub', 'dogs', 'flowers', 'food', 'inat17', 'moe',
            'nabirds', 'pets', 'soyageing', 'soygene', 'soyglobal', 'soylocal', 'vegfru', 'cotton')
MODELS = ('vit_b16', 'swin_base_patch4_window7_224_in22k', 'beitv2_base_patch16_224')
INPUT_FILES = ('fp/final.npy', 'samples.json', 'fp/predictions.csv', 'w3_g128/predictions.csv')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def distances_from_class_sums(sums, counts):
    eligible = counts >= 2
    intra = None
    if eligible.any():
        c = counts[eligible]
        intra = float(np.clip(np.mean(1 - (np.sum(sums[eligible] ** 2, axis=1) - c)
                                      / (c * (c - 1))), 0, 2))
    inter = None
    pairs = len(counts) * (len(counts) - 1) // 2
    if pairs:
        means = sums / counts[:, None]
        pair_sum = (np.sum(means.sum(axis=0) ** 2) - np.sum(means ** 2)) / 2
        inter = float(np.clip(1 - pair_sum / pairs, 0, 2))
    gap = None if inter is None or intra is None else inter - intra
    return dict(d_inter=inter, d_intra=intra, distance_gap=gap,
                distance_ratio=None if gap is None or intra <= 0 else inter / intra,
                normalized_separation=None if gap is None or inter + intra <= 0 else gap / (inter + intra))


def analyze_features(features, labels, chunk_size=2048):
    labels = np.asarray(labels)
    if (features.ndim != 2 or len(features) < 2 or features.shape[1] < 2
            or features.dtype.kind != 'f' or labels.shape != (len(features),)
            or labels.dtype.kind not in 'iu' or np.any(labels < 0) or chunk_size < 1):
        raise ValueError('Require floating [N,D] features, N/D >= 2, and paired integer labels')
    n, d = features.shape
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    mean = np.zeros(d, dtype=np.float64)
    energy = 0.
    for start in range(0, n, chunk_size):
        x = np.asarray(features[start:start + chunk_size], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite FP features')
        mean += x.sum(axis=0)
        energy += float(np.sum(x * x))
    mean /= n
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError('Invalid FP feature energy')
    raw_sums = np.zeros((len(classes), d), dtype=np.float64)
    centered_sums = np.zeros_like(raw_sums)
    zero_count, min_centered_norm = 0, float('inf')
    for start in range(0, n, chunk_size):
        x = np.asarray(features[start:start + chunk_size], dtype=np.float64)
        norms = np.linalg.norm(x, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError('Raw cosine features contain zero/nonfinite norms')
        rows = inverse[start:start + len(x)]
        np.add.at(raw_sums, rows, x / norms[:, None])
        centered = x - mean
        norms = np.linalg.norm(centered, axis=1)
        if not np.isfinite(norms).all():
            raise ValueError('Nonfinite centered feature norms')
        min_centered_norm = min(min_centered_norm, float(norms.min()))
        valid = norms > 0
        zero_count += int((~valid).sum())
        np.add.at(centered_sums, rows[valid], centered[valid] / norms[valid, None])
    raw = distances_from_class_sums(raw_sums, counts)
    centered = ({key: None for key in raw} if zero_count else distances_from_class_sums(centered_sums, counts))
    mean_energy = float(mean @ mean)
    row = dict(n_samples=n, n_classes=len(classes), feature_dim=d, source_dtype=str(features.dtype),
               n_intra_classes=int((counts >= 2).sum()), n_inter_class_pairs=len(classes)*(len(classes)-1)//2,
               mean_norm=float(np.sqrt(mean_energy)), rms_feature_norm=float(np.sqrt(energy/n)),
               mean_energy_fraction=float(np.clip(mean_energy / (energy/n), 0, 1)),
               centered_min_norm=min_centered_norm, centered_zero_norm_count=zero_count,
               centered_status='undefined_zero_norm' if zero_count else 'defined')
    row.update({'raw_' + key: value for key, value in raw.items()})
    row.update({'centered_' + key: value for key, value in centered.items()})
    return row, mean


def validate_pair(folder, serial, dataset, model, mode):
    manifest_path = folder/'manifest.json'
    meta = json.loads(manifest_path.read_text(encoding='utf-8'))
    identity = dict(status='complete', serial=serial, dataset=dataset, model=model, checkpoint_mode=mode)
    if any(meta.get(key) != value for key, value in identity.items()):
        raise ValueError(f'{folder}: incomplete/mismatched source identity')
    quant = meta.get('quantization', {})
    if any(quant.get(key, 'missing') != value for key, value in
           [('weight_bits', 3), ('weight_group_size', 128), ('activation_bits', None)]):
        raise ValueError(f'{folder}: expected matched weight-only W3/G128 export')
    hashes = {'manifest.json': sha256_file(manifest_path)}
    for name in INPUT_FILES:
        hashes[name] = sha256_file(folder/name)
        if hashes[name] != meta.get('files', {}).get(name):
            raise ValueError(f'{folder}: checksum mismatch: {name}')
    samples = json.loads((folder/'samples.json').read_text(encoding='utf-8'))
    ids, targets = samples.get('sample_ids'), samples.get('targets')
    n = meta.get('n_samples')
    if (type(n) is not int or n < 2 or not isinstance(ids, list) or not isinstance(targets, list)
            or len(ids) != n or len(targets) != n or any(type(v) is not str for v in ids)
            or len(set(ids)) != n or any(type(v) is not int or v < 0 for v in targets)
            or digest_json(dict(ids=ids, targets=targets)) != meta.get('samples_sha256')):
        raise ValueError(f'{folder}: invalid sample IDs/targets/digest')
    x = np.load(folder/'fp/final.npy', mmap_mode='r', allow_pickle=False)
    if (list(x.shape) != meta.get('layers', {}).get('final') or x.shape[0] != n
            or str(x.dtype) != meta.get('feature_dtype') or x.dtype not in (np.float16, np.float32)):
        raise ValueError(f'{folder}: feature shape/dtype differs from manifest')
    del x
    accuracies = []
    for condition in ('fp', 'w3_g128'):
        with (folder/condition/'predictions.csv').open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != n:
            raise ValueError(f'{folder}: incomplete {condition} predictions')
        correct = 0
        for i, row in enumerate(rows):
            pred, flag = int(row['prediction']), int(row['correct'])
            if (int(row['row']) != i or row['sample_id'] != ids[i] or int(row['target']) != targets[i]
                    or pred < 0 or flag != int(pred == targets[i])):
                raise ValueError(f'{folder}: unpaired/invalid {condition} prediction at row {i}')
            correct += flag
        accuracies.append(correct/n)
    return np.asarray(targets), hashes, accuracies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True, help='Parent of serial directories, or selected serial directory')
    parser.add_argument('--serial', type=int, default=20004)
    parser.add_argument('--output-root', type=Path, help='New directory; default SERIAL/analysis/centered_fp_distance')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--modes', nargs='+', choices=('ft', 'fz', 'cal'), default=['ft', 'fz', 'cal'])
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--dry-run', action='store_true', help='Print selection/paths only; does not validate source data')
    args = parser.parse_args(argv)
    if args.serial <= 0 or args.chunk_size <= 0:
        parser.error('serial and chunk-size must be positive')
    source = args.input_root.expanduser().resolve()
    if source.name != str(args.serial):
        source /= str(args.serial)
    out = (args.output_root.expanduser() if args.output_root else source/'analysis/centered_fp_distance').resolve()
    if out == source or source.is_relative_to(out) or any(out.is_relative_to(source/d) for d in DATASETS):
        raise ValueError('Output must not overlap the source dataset directories')
    selections = list(itertools.product(dict.fromkeys(args.datasets), dict.fromkeys(args.models), dict.fromkeys(args.modes)))
    config = dict(serial=args.serial, input_root=str(source), output_root=str(out), n_pairs=len(selections),
                  selection=[dict(dataset=d, model=m, mode=c) for d, m, c in selections],
                  centering='sample-weighted full FP test-set mean, before per-sample L2 normalization',
                  metric='class-pair-equal inter and class-equal distinct-pair intra cosine distance',
                  accumulation_dtype='float64', chunk_size=args.chunk_size,
                  source_scope='FP final backbone features; raw-image view, before CAL attention pooling',
                  validation='Only consumed source files are checked; other arrays/logits are not reread.',
                  code_sha256=sha256_file(Path(__file__)))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    partial = out.with_name(out.name + '.partial')
    if out.exists() or partial.exists():
        raise FileExistsError(f'Preserving existing output: {out}; choose another --output-root')
    pairs = []
    for dataset, model, mode in selections:
        folder = source/dataset/model/mode
        labels, hashes, accuracies = validate_pair(folder, args.serial, dataset, model, mode)
        pairs.append((folder, dataset, model, mode, labels, hashes, accuracies))
    partial.mkdir(parents=True)
    config.update(status='running', inputs={})
    try:
        rows = []
        for index, (folder, dataset, model, mode, labels, hashes, accuracies) in enumerate(pairs, 1):
            x = np.load(folder/'fp/final.npy', mmap_mode='r', allow_pickle=False)
            metrics, mean = analyze_features(x, labels, args.chunk_size)
            del x
            if any(sha256_file(folder/name) != digest for name, digest in hashes.items()):
                raise ValueError(f'{folder}: source changed during analysis')
            fp_acc, w3_acc = accuracies
            rows.append(dict(serial=args.serial, dataset=dataset, model=model, mode=mode,
                             fp_accuracy=fp_acc, w3_accuracy=w3_acc,
                             relative_loss_percent=None if fp_acc <= 0 else 100*(1-w3_acc/fp_acc), **metrics))
            mean_path = partial/'means'/dataset/model/mode/'fp_mean.npy'
            mean_path.parent.mkdir(parents=True)
            np.save(mean_path, mean)
            config['inputs'][f'{dataset}/{model}/{mode}'] = dict(source_folder=str(folder), sha256=hashes)
            print(f'[{index}/{len(pairs)}] {dataset}/{model}/{mode}: N={len(labels)} '
                  f'inter={metrics["raw_d_inter"]} -> {metrics["centered_d_inter"]}', flush=True)
        with (partial/'centered_class_distances.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        config.update(status='complete', output_sha256={p.relative_to(partial).as_posix(): sha256_file(p)
                      for p in partial.rglob('*') if p.is_file()})
        (partial/'run_config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf-8')
        if out.exists():
            raise FileExistsError(f'Output appeared during analysis: {out}')
        partial.rename(out)
        print(f'Complete: {out/"centered_class_distances.csv"}', flush=True)
    except BaseException as error:
        config.update(status='failed', error=str(error))
        (partial/'run_config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
