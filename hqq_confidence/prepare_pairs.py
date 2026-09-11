"""Validate and pair cached HQQ confidence artifacts by stable sample ID."""

import argparse
import os

import numpy as np
import pandas as pd

if __package__:
    from .common import (EXPECTED_CONDITIONS, PREPARED_DIR, SWEEP_COLUMNS,
                         as_bool, condition_config, sweep_id, write_run_config)
else:
    from common import (EXPECTED_CONDITIONS, PREPARED_DIR, SWEEP_COLUMNS,
                        as_bool, condition_config, sweep_id, write_run_config)


REQUIRED_ARRAYS = [
    'sample_id', 'target', 'logits', 'correct', 'top1_class', 'top2_class',
    'top1_logit', 'top2_logit', 'logit_margin', 'top1_probability',
    'top2_probability', 'true_class_probability', 'entropy',
    'normalized_entropy',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate complete 13-condition sweeps and pair samples.')
    parser.add_argument(
        '--manifest-file',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'selected_runs.csv'))
    parser.add_argument('--output-dir', default=PREPARED_DIR)
    parser.add_argument(
        '--allow-unknown-provenance', action='store_true',
        help='include legacy artifacts whose AMP or git provenance is missing')
    return parser.parse_args()


def require_columns(frame, columns):
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'Manifest is missing required columns: {", ".join(missing)}')


def _metadata(payload):
    result = {}
    for key in payload.files:
        if key.startswith('metadata_'):
            value = payload[key]
            result[key[9:]] = value.item() if value.ndim == 0 else value.tolist()
    return result


def _close(actual, expected, name, rtol=2e-4, atol=1e-6):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False):
        raise ValueError(f'Artifact field {name} does not match values recomputed from logits')


def load_predictions(path):
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(set(REQUIRED_ARRAYS).difference(payload.files))
        if missing:
            raise ValueError(f'{path} is missing arrays: {", ".join(missing)}')
        data = {key: np.asarray(payload[key]) for key in REQUIRED_ARRAYS}
        metadata = _metadata(payload)

    logits = data['logits'].astype(np.float64, copy=False)
    targets = data['target'].astype(np.int64, copy=False)
    if logits.ndim != 2 or logits.shape[0] != len(targets) or logits.shape[1] < 2:
        raise ValueError(f'{path} has incompatible logits and target shapes')
    if not np.isfinite(logits).all():
        raise ValueError(f'{path} contains non-finite logits')
    if targets.ndim != 1 or (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError(f'{path} contains invalid targets')
    sample_ids = data['sample_id'].astype(str)
    if sample_ids.ndim != 1 or len(sample_ids) != len(targets):
        raise ValueError(f'{path} has invalid sample IDs')
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(f'{path} contains duplicate sample IDs')

    rows = np.arange(len(logits))
    top1 = data['top1_class'].astype(np.int64, copy=False)
    top2 = data['top2_class'].astype(np.int64, copy=False)
    if (top1.shape != targets.shape or top2.shape != targets.shape
            or (top1 < 0).any() or (top1 >= logits.shape[1]).any()
            or (top2 < 0).any() or (top2 >= logits.shape[1]).any()
            or np.any(top1 == top2)):
        raise ValueError(f'{path} contains invalid top-1 or top-2 classes')
    top_values = np.sort(logits, axis=1)[:, -2:][:, ::-1]
    if (not np.array_equal(logits[rows, top1], top_values[:, 0])
            or not np.array_equal(logits[rows, top2], top_values[:, 1])):
        raise ValueError('Artifact top-1 or top-2 class does not match logits')
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    entropy = -(probabilities * log_probabilities).sum(axis=1)
    expected = {
        'top1_logit': logits[rows, top1], 'top2_logit': logits[rows, top2],
        'logit_margin': logits[rows, top1] - logits[rows, top2],
        'top1_probability': probabilities[rows, top1],
        'top2_probability': probabilities[rows, top2],
        'true_class_probability': probabilities[rows, targets],
        'entropy': entropy, 'normalized_entropy': entropy / np.log(logits.shape[1]),
        'correct': top1 == targets,
    }
    for name, values in expected.items():
        if name == 'correct':
            if not np.array_equal(data[name], values):
                raise ValueError(f'Artifact field {name} does not match logits')
        else:
            _close(data[name], values, name)
    data.update({
        'sample_id': sample_ids, 'target': targets, 'logits': logits,
        'correct': expected['correct'], '_probabilities': probabilities,
        '_log_probabilities': log_probabilities, '_metadata': metadata,
    })
    return data


def calibration_rows(data, bins=15):
    confidence = data['top1_probability'].astype(float)
    correct = data['correct'].astype(float)
    ordered = np.argsort(confidence, kind='stable')
    rows = []
    for index, selected in enumerate(np.array_split(ordered, min(bins, len(ordered))), 1):
        if not len(selected):
            continue
        rows.append({
            'bin': index, 'count': len(selected),
            'confidence_min': float(confidence[selected].min()),
            'confidence_max': float(confidence[selected].max()),
            'mean_confidence': float(confidence[selected].mean()),
            'accuracy': float(correct[selected].mean()),
            'ece_contribution': float(
                len(selected) / len(correct)
                * abs(confidence[selected].mean() - correct[selected].mean())),
        })
    return rows


def condition_metrics(data):
    rows = np.arange(len(data['target']))
    target = data['target']
    probs = data['_probabilities']
    true_probs = np.clip(probs[rows, target], 1e-12, 1)
    brier = (np.square(probs).sum(axis=1) - 2 * true_probs + 1).mean()
    reliability = calibration_rows(data)
    return {
        'n_samples': len(target), 'num_classes': data['logits'].shape[1],
        'accuracy': 100 * float(data['correct'].mean()),
        'nll_uncalibrated': float(-np.log(true_probs).mean()),
        'brier_uncalibrated': float(brier),
        'ece_uncalibrated': float(sum(row['ece_contribution'] for row in reliability)),
        'mean_top1_probability': float(data['top1_probability'].mean()),
        'mean_logit_margin': float(data['logit_margin'].mean()),
        'mean_normalized_entropy': float(data['normalized_entropy'].mean()),
    }, reliability


def _value_known(value):
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def provenance_status(group):
    reasons = []
    amp = [as_bool(value) for value in group.get('amp_fp16', pd.Series([None] * len(group)))]
    if any(value is False for value in amp):
        reasons.append('not_matched_amp_fp16')
    elif any(value is None for value in amp):
        reasons.append('unknown_amp_fp16')
    dtypes = {str(value) for value in group.get('compute_dtype', pd.Series(dtype=str))
              if _value_known(value)}
    if len(dtypes) > 1:
        reasons.append('compute_dtype_mismatch')
    elif not dtypes:
        reasons.append('unknown_compute_dtype')
    commits = {str(value) for value in group.get('git_commit', pd.Series(dtype=str))
               if _value_known(value)}
    if len(commits) > 1:
        reasons.append('git_commit_mismatch')
    elif not commits:
        reasons.append('unknown_git_commit')
    if 'debugging' in group and any(as_bool(value) is True for value in group['debugging']):
        reasons.append('debugging_run')
    counts = dict(zip(group['condition'], group['quantized_linear_count']))
    baseline_count = counts.get('fp_unquantized')
    if not _value_known(baseline_count):
        reasons.append('unknown_quantized_layer_count')
    elif float(baseline_count) != 0:
        reasons.append('baseline_has_quantized_layers')
    for condition in EXPECTED_CONDITIONS[1:]:
        count = counts.get(condition)
        if not _value_known(count):
            reasons.append('unknown_quantized_layer_count')
            break
        if float(count) <= 0:
            reasons.append('hqq_has_no_quantized_layers')
            break
    return sorted(set(reasons))


def _align(data, baseline):
    positions = {sample_id: index for index, sample_id in enumerate(data['sample_id'])}
    if set(positions) != set(baseline['sample_id']):
        raise ValueError('Sample ID sets differ within a sweep')
    order = np.asarray([positions[value] for value in baseline['sample_id']])
    aligned = {
        key: (value[order] if isinstance(value, np.ndarray)
              and value.ndim > 0 and len(value) == len(order) else value)
        for key, value in data.items()
    }
    if not np.array_equal(aligned['target'], baseline['target']):
        raise ValueError('Targets differ for matched sample IDs')
    if aligned['logits'].shape != baseline['logits'].shape:
        raise ValueError('N or C differs within a sweep')
    return aligned


def _save_pair(path, baseline, quantized):
    baseline_margin = baseline['logit_margin'].astype(float)
    std = baseline_margin.std()
    margin_z = ((baseline_margin - baseline_margin.mean()) / std
                if std else np.zeros_like(baseline_margin))
    margin_percentile = pd.Series(baseline_margin).rank(
        method='average', pct=True).to_numpy()
    damage = baseline['correct'] & ~quantized['correct']
    rescue = ~baseline['correct'] & quantized['correct']
    np.savez_compressed(
        path, sample_id=baseline['sample_id'], target=baseline['target'],
        baseline_correct=baseline['correct'], hqq_correct=quantized['correct'],
        baseline_top1_class=baseline['top1_class'], hqq_top1_class=quantized['top1_class'],
        baseline_logit_margin=baseline_margin,
        baseline_logit_margin_z=margin_z,
        baseline_margin_percentile=margin_percentile,
        baseline_top1_probability=baseline['top1_probability'],
        hqq_top1_probability=quantized['top1_probability'],
        delta_top1_probability=(quantized['top1_probability']
                                - baseline['top1_probability']),
        baseline_true_class_probability=baseline['true_class_probability'],
        hqq_true_class_probability=quantized['true_class_probability'],
        delta_true_class_probability=(quantized['true_class_probability']
                                      - baseline['true_class_probability']),
        baseline_normalized_entropy=baseline['normalized_entropy'],
        hqq_normalized_entropy=quantized['normalized_entropy'],
        delta_normalized_entropy=(quantized['normalized_entropy']
                                  - baseline['normalized_entropy']),
        hqq_logit_margin=quantized['logit_margin'],
        delta_logit_margin=quantized['logit_margin'] - baseline_margin,
        damage=damage, rescue=rescue,
        unchanged_correct=baseline['correct'] & quantized['correct'],
        unchanged_wrong=~baseline['correct'] & ~quantized['correct'],
        agreement=baseline['top1_class'] == quantized['top1_class'],
    )


def _validate_metadata(data, row):
    metadata = data['_metadata']
    expected = {
        'condition': row['condition'], 'dataset_name': row['dataset_name'],
        'model_name': row['model_name'],
        'checkpoint_sha256': row['checkpoint_sha256'], 'split': 'test',
    }
    for key, value in expected.items():
        if key in metadata and str(metadata[key]) != str(value):
            raise ValueError(f'Artifact metadata {key} does not match its manifest row')
    if 'compute_dtype' in metadata and _value_known(row.get('compute_dtype')):
        if str(metadata['compute_dtype']) != str(row['compute_dtype']):
            raise ValueError('Artifact compute dtype does not match its manifest row')
    if 'amp_fp16' in metadata and _value_known(row.get('amp_fp16')):
        if as_bool(metadata['amp_fp16']) != as_bool(row['amp_fp16']):
            raise ValueError('Artifact AMP setting does not match its manifest row')


def checkpoint_kind(path):
    normalized = str(path).replace('\\', '/')
    for kind in ('cal', 'ft', 'fz'):
        if f'/{kind}_ckpts/' in normalized:
            return kind
    return 'unknown'


def prepare(manifest_file, output_dir, allow_unknown_provenance=False):
    manifest = pd.read_csv(manifest_file)
    required = SWEEP_COLUMNS + [
        'run_id', 'condition', 'npz_path', 'summary_accuracy',
        'quantized_linear_count',
    ]
    require_columns(manifest, required)
    unknown = sorted(set(manifest['condition']).difference(EXPECTED_CONDITIONS))
    if unknown:
        raise ValueError(f'Manifest contains unknown conditions: {", ".join(unknown)}')
    identity = SWEEP_COLUMNS + ['condition']
    duplicates = manifest[manifest.duplicated(identity, keep=False)]
    if len(duplicates):
        raise ValueError(
            'Ambiguous duplicate runs; provide a manifest with one explicit run per condition')

    os.makedirs(output_dir, exist_ok=True)
    completeness_rows = []
    eligible_groups = []
    for key, group in manifest.groupby(SWEEP_COLUMNS, dropna=False, sort=True):
        observed = set(group['condition'])
        missing = sorted(set(EXPECTED_CONDITIONS).difference(observed))
        extra = sorted(observed.difference(EXPECTED_CONDITIONS))
        reasons = provenance_status(group)
        unknown_only = set(reasons).issubset({
            'unknown_amp_fp16', 'unknown_git_commit',
            'unknown_compute_dtype', 'unknown_quantized_layer_count',
        })
        provenance_ok = not reasons or (allow_unknown_provenance and unknown_only)
        complete = not missing and not extra and len(group) == len(EXPECTED_CONDITIONS)
        sid = sweep_id(group.iloc[0])
        completeness_rows.append({
            **dict(zip(SWEEP_COLUMNS, key)), 'sweep_id': sid,
            'observed_conditions': len(observed),
            'missing_conditions': ', '.join(missing),
            'extra_conditions': ', '.join(extra),
            'provenance_issues': ', '.join(reasons),
            'complete_conditions': complete, 'provenance_ok': provenance_ok,
            'eligible': complete and provenance_ok,
        })
        if complete and provenance_ok:
            eligible_groups.append((sid, group))
    completeness = pd.DataFrame(completeness_rows)
    completeness.to_csv(os.path.join(output_dir, 'completeness.csv'), index=False)
    completeness[[
        'sweep_id', *SWEEP_COLUMNS, 'provenance_issues', 'provenance_ok',
    ]].to_csv(os.path.join(output_dir, 'configuration_quality.csv'), index=False)
    if not eligible_groups:
        raise ValueError('No complete, provenance-compatible 13-condition sweeps remain')

    condition_rows = []
    transition_rows = []
    reliability_rows = []
    quality_rows = []
    pairs_root = os.path.join(output_dir, 'pairs')
    for sid, group in eligible_groups:
        by_condition = group.set_index('condition', drop=False)
        baseline_row = by_condition.loc['fp_unquantized']
        baseline = load_predictions(baseline_row['npz_path'])
        _validate_metadata(baseline, baseline_row)
        baseline_metrics, baseline_reliability = condition_metrics(baseline)
        condition_rows.append({
            'sweep_id': sid, **{key: baseline_row[key] for key in SWEEP_COLUMNS},
            'run_id': baseline_row['run_id'], 'condition': 'fp_unquantized',
            'ckpt_path': baseline_row.get('ckpt_path'),
            'ckpt_kind': checkpoint_kind(baseline_row.get('ckpt_path')),
            'hqq_nbits': -1, 'hqq_group_size': -1, **baseline_metrics,
        })
        for row in baseline_reliability:
            reliability_rows.append({'sweep_id': sid, 'condition': 'fp_unquantized', **row})
        baseline_summary = float(baseline_row['summary_accuracy'])
        baseline_match = bool(np.isclose(
            baseline_summary, baseline_metrics['accuracy'], rtol=0, atol=1e-4))
        quality_rows.append({
            'sweep_id': sid, 'condition': 'fp_unquantized',
            'run_id': baseline_row['run_id'],
            'summary_accuracy': baseline_summary,
            'recomputed_accuracy': baseline_metrics['accuracy'],
            'accuracy_matches': baseline_match, 'sample_ids_match': True,
            'targets_match': True, 'shape_matches': True,
        })
        if not baseline_match:
            raise ValueError('W&B accuracy does not match artifact for fp_unquantized')

        for condition in EXPECTED_CONDITIONS[1:]:
            quantized_row = by_condition.loc[condition]
            quantized = load_predictions(quantized_row['npz_path'])
            _validate_metadata(quantized, quantized_row)
            quantized = _align(quantized, baseline)
            metrics, reliability = condition_metrics(quantized)
            _, bits, group_size = condition_config(condition)
            for row in reliability:
                reliability_rows.append({'sweep_id': sid, 'condition': condition, **row})
            condition_rows.append({
                'sweep_id': sid,
                **{key: quantized_row[key] for key in SWEEP_COLUMNS},
                'run_id': quantized_row['run_id'], 'condition': condition,
                'ckpt_path': quantized_row.get('ckpt_path'),
                'ckpt_kind': checkpoint_kind(quantized_row.get('ckpt_path')),
                'hqq_nbits': bits, 'hqq_group_size': group_size, **metrics,
            })
            summary_accuracy = float(quantized_row['summary_accuracy'])
            quality_rows.append({
                'sweep_id': sid, 'condition': condition,
                'run_id': quantized_row['run_id'],
                'summary_accuracy': summary_accuracy,
                'recomputed_accuracy': metrics['accuracy'],
                'accuracy_matches': bool(np.isclose(
                    summary_accuracy, metrics['accuracy'], rtol=0, atol=1e-4)),
                'sample_ids_match': True, 'targets_match': True,
                'shape_matches': True,
            })
            if not quality_rows[-1]['accuracy_matches']:
                raise ValueError(f'W&B accuracy does not match artifact for {condition}')

            baseline_correct = baseline['correct']
            hqq_correct = quantized['correct']
            damage = baseline_correct & ~hqq_correct
            rescue = ~baseline_correct & hqq_correct
            drop = baseline_metrics['accuracy'] - metrics['accuracy']
            decomposed = 100 * (damage.sum() - rescue.sum()) / len(damage)
            if not np.isclose(drop, decomposed, rtol=0, atol=1e-10):
                raise ValueError('Accuracy drop does not match damage/rescue decomposition')
            pair_dir = os.path.join(pairs_root, sid)
            os.makedirs(pair_dir, exist_ok=True)
            pair_path = os.path.join(pair_dir, f'{condition}.npz')
            _save_pair(pair_path, baseline, quantized)
            transition_rows.append({
                'sweep_id': sid,
                **{key: quantized_row[key] for key in SWEEP_COLUMNS},
                'condition': condition, 'hqq_nbits': bits,
                'hqq_group_size': group_size,
                'ckpt_path': quantized_row.get('ckpt_path'),
                'ckpt_kind': checkpoint_kind(quantized_row.get('ckpt_path')),
                'baseline_accuracy': baseline_metrics['accuracy'],
                'hqq_top1': metrics['accuracy'],
                'accuracy_drop_points': drop,
                'accuracy_ratio': (metrics['accuracy'] / baseline_metrics['accuracy']
                                   if baseline_metrics['accuracy'] else np.nan),
                'damage_count': int(damage.sum()),
                'damage_rate': float(damage.sum() / baseline_correct.sum())
                if baseline_correct.sum() else np.nan,
                'rescue_count': int(rescue.sum()),
                'rescue_rate': float(rescue.sum() / (~baseline_correct).sum())
                if (~baseline_correct).sum() else np.nan,
                'unchanged_correct_count': int((baseline_correct & hqq_correct).sum()),
                'unchanged_wrong_count': int((~baseline_correct & ~hqq_correct).sum()),
                'prediction_agreement': float(
                    (baseline['top1_class'] == quantized['top1_class']).mean()),
                'mean_delta_top1_probability': float(
                    (quantized['top1_probability'] - baseline['top1_probability']).mean()),
                'mean_delta_true_class_probability': float(
                    (quantized['true_class_probability']
                     - baseline['true_class_probability']).mean()),
                'mean_delta_logit_margin': float(
                    (quantized['logit_margin'] - baseline['logit_margin']).mean()),
                'mean_delta_normalized_entropy': float(
                    (quantized['normalized_entropy']
                     - baseline['normalized_entropy']).mean()),
                'pair_file': os.path.relpath(pair_path, output_dir),
            })

    pd.DataFrame(condition_rows).to_csv(
        os.path.join(output_dir, 'condition_summary.csv'), index=False)
    pd.DataFrame(transition_rows).to_csv(
        os.path.join(output_dir, 'transition_summary.csv'), index=False)
    pd.DataFrame(reliability_rows).to_csv(
        os.path.join(output_dir, 'reliability_bins.csv'), index=False)
    pd.DataFrame(quality_rows).to_csv(
        os.path.join(output_dir, 'pairing_quality.csv'), index=False)
    return completeness


def main():
    args = parse_args()
    args.manifest_file = os.path.abspath(args.manifest_file)
    args.output_dir = os.path.abspath(args.output_dir)
    write_run_config(args.output_dir, args)
    completeness = prepare(
        args.manifest_file, args.output_dir, args.allow_unknown_provenance)
    print(f'Prepared {int(completeness["eligible"].sum())} complete sweeps')
    print(f'Wrote {args.output_dir}')


if __name__ == '__main__':
    main()
