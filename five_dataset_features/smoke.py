"""Validate the six-condition CUB real-batch smoke output before full jobs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .analyze import sha256
from .metrics import (centered_linear_cka, class_equal_cosine_distances,
                      norm_quantiles, prediction_metrics, regularized_h_score)


CONDITIONS = ('fp', 'w3_g128', 'w3_g128_a8', 'w3_g128_a4',
              'w3_g128_a8_qkv', 'w3_g128_a4_qkv')
INCREMENTAL_PAIRS = (('w3_g128', 'w3_g128_a8'),
                     ('w3_g128_a8', 'w3_g128_a8_qkv'),
                     ('w3_g128', 'w3_g128_a4'),
                     ('w3_g128_a4', 'w3_g128_a4_qkv'))


def check_smoke(root):
    folder = root / 'smoke' / 'cub'
    if not (folder / 'dataset.json').is_file():
        folder = root / 'cub'
    meta = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
    n = meta['n_samples']
    if n < 2 or len(meta['sample_ids']) != n or not meta.get('full_n_samples'):
        raise RuntimeError('Invalid smoke dataset manifest')
    arrays, probes, predictions = {}, {}, {}
    timings = {}
    for name in CONDITIONS:
        path = folder / name
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] != 'complete' or manifest['n_samples'] != n:
            raise RuntimeError(f'Incomplete smoke condition: {name}')
        timings[name] = manifest['inference_seconds']
        if any(sha256(path / filename) != expected for filename, expected in manifest['files'].items()):
            raise RuntimeError(f'Checksum mismatch: {name}')
        arrays[name] = np.load(path / 'features.npy', mmap_mode='r', allow_pickle=False)
        probes[name] = np.load(path / 'precision_probe.npy', allow_pickle=False)
        if arrays[name].shape != (n, 13, 768) or probes[name].shape[1:] != (13, 768):
            raise RuntimeError(f'Feature shape mismatch: {name}')
        with open(path / 'predictions.csv', newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if [row['sample_id'] for row in rows] != meta['sample_ids']:
            raise RuntimeError(f'Sample order mismatch: {name}')
        if [int(row['target']) for row in rows] != meta['targets']:
            raise RuntimeError(f'Target mismatch: {name}')
        predictions[name] = rows
    fp_probe = probes['fp']
    labels = meta['targets'][:len(fp_probe)]
    results = []
    for name in CONDITIONS[1:]:
        for layer in range(13):
            fp32 = centered_linear_cka(fp_probe[:, layer, :], probes[name][:, layer, :])
            fp16 = centered_linear_cka(fp_probe[:, layer, :].astype(np.float16),
                                       probes[name][:, layer, :].astype(np.float16))
            storage_delta = None if fp32 is None or fp16 is None else abs(fp32 - fp16)
            change = None if fp32 is None else abs(1 - fp32)
            results.append({'condition': name, 'layer': layer, 'cka_fp32': fp32,
                            'cka_fp16': fp16, 'storage_delta': storage_delta,
                            'quant_change': change,
                            'resolved_10x': None if change is None or storage_delta is None else
                            change > 1e-10 and storage_delta * 10 < change})
    incremental = []
    for base, name in INCREMENTAL_PAIRS:
        for layer in range(13):
            x, y = probes[base][:, layer, :], probes[name][:, layer, :]
            fp32 = centered_linear_cka(x, y)
            fp16 = centered_linear_cka(x.astype(np.float16), y.astype(np.float16))
            storage_delta = None if fp32 is None or fp16 is None else abs(fp32 - fp16)
            change = None if fp32 is None else abs(1 - fp32)
            incremental.append({'base_condition': base, 'condition': name,
                                'layer': layer, 'cka_fp32': fp32,
                                'cka_fp16': fp16, 'storage_delta': storage_delta,
                                'activation_change': change,
                                'resolved_10x': None if change is None or storage_delta is None else
                                change > 1e-10 and storage_delta * 10 < change})
    fp_final = fp_probe[:, 12, :]
    quant_final = probes['w3_g128_a8'][:, 12, :]
    fp_sep32 = class_equal_cosine_distances(fp_final, labels)['separability']
    fp_sep16 = class_equal_cosine_distances(fp_final.astype(np.float16), labels)['separability']
    q_sep32 = class_equal_cosine_distances(quant_final, labels)['separability']
    q_sep16 = class_equal_cosine_distances(quant_final.astype(np.float16), labels)['separability']
    final_fp = arrays['fp'][:, 12, :]
    final_features = {}
    for name in CONDITIONS:
        final = arrays[name][:, 12, :]
        final_features[name] = {
            'cka_to_fp': centered_linear_cka(final_fp, final),
            **class_equal_cosine_distances(final, meta['targets']),
            **norm_quantiles(final),
            'h_score': regularized_h_score(final, meta['targets']),
            'normalized_margin_q10': float(np.quantile(
                [float(row['normalized_margin']) for row in predictions[name]], 0.1)),
        }
    report = {'dataset': 'cub', 'samples': n, 'conditions': list(CONDITIONS),
              'class_counts': meta['class_counts'],
              'prediction_metrics': {name: prediction_metrics(predictions['fp'], predictions[name])
                                     for name in CONDITIONS},
              'predicted_class_counts': {name: dict(Counter(row['prediction'] for row in predictions[name]))
                                         for name in CONDITIONS},
              'final_feature_metrics': final_features,
              'full_test_samples': meta.get('full_n_samples'),
              'smoke_inference_seconds': timings,
              'estimated_21_condition_inference_hours': None if not meta.get('full_n_samples') else
              float(np.mean(list(timings.values())) * 21 * meta['full_n_samples'] / n / 3600),
              'estimate_note': 'Linear sample scaling from six CUB conditions; excludes model/HQQ setup, other datasets and scheduler overhead.',
              'storage_dtype': meta['feature_dtype'],
              'final_fp_separation_fp32': fp_sep32, 'final_fp_separation_fp16': fp_sep16,
              'final_a8_separation_fp32': q_sep32, 'final_a8_separation_fp16': q_sep16,
              'final_fp_norm_p99_fp32': norm_quantiles(fp_final)['norm_p99'],
              'final_fp_norm_p99_fp16': norm_quantiles(fp_final.astype(np.float16))['norm_p99'],
              'cka_by_layer': results,
              'incremental_cka_by_layer': incremental}
    output = root / 'analysis' / 'smoke_precision.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Validated six CUB conditions; precision report: {output}')
    final_a8 = next(row for row in results if row['condition'] == 'w3_g128_a8' and row['layer'] == 12)
    print(f'Final A8 CKA change={final_a8["quant_change"]}; FP16 storage delta={final_a8["storage_delta"]}; resolved_10x={final_a8["resolved_10x"]}')
    incremental_a8 = next(row for row in incremental if row['condition'] == 'w3_g128_a8' and row['layer'] == 12)
    print(f'Incremental W3-to-A8 CKA change={incremental_a8["activation_change"]}; FP16 storage delta={incremental_a8["storage_delta"]}; resolved_10x={incremental_a8["resolved_10x"]}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    check_smoke(args.root)


if __name__ == '__main__':
    main()
