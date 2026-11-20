"""Analyze whether unquantized confidence predicts HQQ classification damage."""

import argparse
import json
import os

import numpy as np
import pandas as pd

if __package__:
    from .common import PREPARED_DIR, STATS_DIR, write_run_config
else:
    from common import PREPARED_DIR, STATS_DIR, write_run_config


BOOTSTRAP_METRICS = [
    'accuracy_ratio', 'accuracy_drop_points', 'damage_rate', 'rescue_rate',
    'prediction_agreement', 'mean_delta_top1_probability',
    'mean_delta_true_class_probability', 'mean_delta_logit_margin',
    'mean_delta_normalized_entropy',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute checkpoint-clustered HQQ confidence statistics.')
    parser.add_argument('--input-dir', default=PREPARED_DIR)
    parser.add_argument('--results-dir', default=STATS_DIR)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def _rate_interval(frame, bootstrap_samples, rng):
    checkpoints = (frame.groupby('checkpoint_sha256', as_index=False)
                   .agg(damage_count=('damage_count', 'sum'),
                        n_samples=('n_samples', 'sum')))
    rates = (checkpoints['damage_count'] / checkpoints['n_samples']).to_numpy()
    estimate = rates.mean()
    if len(rates) == 1:
        return float(estimate), float(estimate)
    indexes = rng.integers(0, len(rates), size=(bootstrap_samples, len(rates)))
    draws = rates[indexes].mean(axis=1)
    return tuple(np.quantile(draws, [.025, .975]))


def confidence_bins(prepared_dir, transitions, bootstrap_samples, seed):
    rows = []
    for record in transitions.to_dict('records'):
        pair_path = os.path.join(prepared_dir, record['pair_file'])
        with np.load(pair_path, allow_pickle=False) as pair:
            baseline_correct = pair['baseline_correct'].astype(bool)
            percentiles = pair['baseline_margin_percentile'][baseline_correct]
            damage = pair['damage'][baseline_correct].astype(bool)
        deciles = np.clip(np.ceil(percentiles * 10).astype(int), 1, 10)
        for decile in sorted(np.unique(deciles)):
            selected = deciles == decile
            rows.append({
                'sweep_id': record['sweep_id'], 'condition': record['condition'],
                'checkpoint_sha256': record['checkpoint_sha256'],
                'hqq_nbits': record['hqq_nbits'],
                'hqq_group_size': record['hqq_group_size'],
                'confidence_decile': int(decile),
                'n_samples': int(selected.sum()),
                'damage_count': int(damage[selected].sum()),
                'damage_rate': float(damage[selected].mean()),
            })
    by_sweep = pd.DataFrame(rows)
    aggregate_rows = []
    rng = np.random.default_rng(seed)
    keys = ['condition', 'hqq_nbits', 'hqq_group_size', 'confidence_decile']
    for key, frame in by_sweep.groupby(keys, sort=True):
        low, high = _rate_interval(frame, bootstrap_samples, rng)
        aggregate_rows.append({
            **dict(zip(keys, key)),
            'n_checkpoints': frame['checkpoint_sha256'].nunique(),
            'n_sweeps': frame['sweep_id'].nunique(),
            'n_samples': int(frame['n_samples'].sum()),
            'damage_count': int(frame['damage_count'].sum()),
            'damage_rate': float(frame.groupby('checkpoint_sha256').apply(
                lambda rows: rows['damage_count'].sum() / rows['n_samples'].sum(),
                include_groups=False).mean()),
            'ci95_low': low, 'ci95_high': high,
        })
    return by_sweep, pd.DataFrame(aggregate_rows)


def _mean_interval(values, bootstrap_samples, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, estimate, estimate
    indexes = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    draws = values[indexes].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return estimate, float(low), float(high)


def condition_statistics(transitions, bootstrap_samples, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for (condition, bits, group_size), frame in transitions.groupby(
            ['condition', 'hqq_nbits', 'hqq_group_size'], sort=True):
        checkpoint_frame = frame.groupby('checkpoint_sha256', as_index=False)[
            BOOTSTRAP_METRICS].mean()
        row = {
            'condition': condition, 'hqq_nbits': bits,
            'hqq_group_size': group_size,
            'n_checkpoints': checkpoint_frame['checkpoint_sha256'].nunique(),
            'n_sweeps': frame['sweep_id'].nunique(),
        }
        for metric in BOOTSTRAP_METRICS:
            mean, low, high = _mean_interval(
                checkpoint_frame[metric], bootstrap_samples, rng)
            row[f'{metric}_mean'] = mean
            row[f'{metric}_ci95_low'] = low
            row[f'{metric}_ci95_high'] = high
        rows.append(row)
    return pd.DataFrame(rows)


def context_statistics(transitions, bootstrap_samples, seed):
    """Summarize conditions within each reported experimental context."""
    rows = []
    for offset, context_type in enumerate(
            ('dataset_name', 'model_name', 'ckpt_kind')):
        for context_value, frame in transitions.groupby(
                context_type, dropna=False, sort=True):
            summary = condition_statistics(
                frame, bootstrap_samples, seed + offset)
            summary.insert(0, 'context_value', context_value)
            summary.insert(0, 'context_type', context_type)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_report(results_dir, transitions, bins, statistics):
    complete_sweeps = transitions['sweep_id'].nunique()
    lines = [
        '# HQQ confidence fragility report', '',
        'This is an exploratory matched-sample analysis. The unquantized control '
        'uses matched AMP FP16 and is not an FP32 baseline.', '',
        f'- Complete sweeps: {complete_sweeps}',
        f'- HQQ condition pairs: {len(transitions)}',
        f'- Conditions: {transitions["condition"].nunique()}', '',
        '## Mean condition outcomes', '',
        statistics[[
            'condition', 'accuracy_ratio_mean', 'accuracy_drop_points_mean',
            'damage_rate_mean', 'rescue_rate_mean',
        ]].to_string(index=False), '',
        '## Confidence-bin analysis', '',
        f'- Prepared condition/decile cells: {len(bins)}',
        '- Damage is evaluated only among samples the unquantized model classified correctly.',
        '- Confidence is the within-checkpoint percentile of the unquantized top1-top2 margin.',
        '- Intervals resample checkpoints, not serial reruns or individual images.',
    ]
    with open(os.path.join(results_dir, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def analyze(prepared_dir, results_dir, bootstrap_samples=2000, seed=0):
    if bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be positive')
    os.makedirs(results_dir, exist_ok=True)
    transitions = pd.read_csv(os.path.join(prepared_dir, 'transition_summary.csv'))
    if transitions.empty:
        raise ValueError('No matched HQQ transitions are available')
    by_sweep, bins = confidence_bins(
        prepared_dir, transitions, bootstrap_samples, seed)
    statistics = condition_statistics(
        transitions, bootstrap_samples, seed + 1)
    contexts = context_statistics(
        transitions, bootstrap_samples, seed + 2)
    by_sweep.to_csv(
        os.path.join(results_dir, 'confidence_bins_by_sweep.csv'), index=False)
    bins.to_csv(os.path.join(results_dir, 'confidence_bins.csv'), index=False)
    statistics.to_csv(
        os.path.join(results_dir, 'condition_statistics.csv'), index=False)
    contexts.to_csv(
        os.path.join(results_dir, 'condition_statistics_by_context.csv'), index=False)
    metadata = {
        'method': 'checkpoint_cluster_bootstrap',
        'cluster': 'checkpoint_sha256; serial reruns are not independent clusters',
        'bootstrap_samples': bootstrap_samples, 'seed': seed,
        'confidence_predictor': 'within-sweep percentile of baseline top1-top2 logit margin',
        'primary_outcome': 'correct-to-wrong damage among baseline-correct samples',
        'interval': 'percentile 95%', 'temperature_scaling': False,
    }
    with open(os.path.join(results_dir, 'bootstrap_metadata.json'),
              'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    quality = pd.read_csv(os.path.join(prepared_dir, 'pairing_quality.csv'))
    with open(os.path.join(results_dir, 'data_quality.md'),
              'w', encoding='utf-8') as handle:
        handle.write(
            '# Data quality\n\n'
            f'- Matched condition pairs: {len(quality)}\n'
            f'- Accuracy checks passed: {int(quality["accuracy_matches"].sum())}/{len(quality)}\n'
            f'- Complete sweeps analyzed: {transitions["sweep_id"].nunique()}\n')
    _write_report(results_dir, transitions, bins, statistics)
    return bins


def main():
    args = parse_args()
    args.input_dir = os.path.abspath(args.input_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    write_run_config(args.results_dir, args)
    bins = analyze(
        args.input_dir, args.results_dir, args.bootstrap_samples, args.seed)
    print(f'Prepared {len(bins)} confidence-bin statistics')
    print(f'Wrote {args.results_dir}')


if __name__ == '__main__':
    main()
