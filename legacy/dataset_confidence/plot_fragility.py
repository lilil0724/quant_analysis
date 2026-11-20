"""Create deterministic figures for the matched HQQ confidence analysis."""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__:
    from .common import PLOTS_DIR, PREPARED_DIR, STATS_DIR, write_run_config
else:
    from common import PLOTS_DIR, PREPARED_DIR, STATS_DIR, write_run_config


EXPECTED_PLOTS = [
    'overview.png', 'confidence_damage_by_condition.png',
    'condition_heatmaps.png', 'baseline_vs_hqq_top1.png',
    'transition_decomposition.png', 'reliability_diagrams.png',
    'confidence_delta_distributions.png',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot HQQ confidence, damage, calibration, and accuracy results.')
    parser.add_argument('--prepared-dir', default=PREPARED_DIR)
    parser.add_argument('--stats-dir', default=STATS_DIR)
    parser.add_argument('--results-dir', default=PLOTS_DIR)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def _save(figure, path, dpi):
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    return path


def _condition_order(frame):
    return (frame[['condition', 'hqq_nbits', 'hqq_group_size']]
            .drop_duplicates().sort_values(['hqq_nbits', 'hqq_group_size']))


def plot_confidence_damage(bins, path, dpi):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, group_size in zip(axes, (8, 128)):
        panel = bins[bins['hqq_group_size'] == group_size]
        for bits, line in panel.groupby('hqq_nbits'):
            line = line.sort_values('confidence_decile')
            axis.plot(line['confidence_decile'], line['damage_rate'],
                      marker='o', label=f'{bits:g}-bit')
            axis.fill_between(
                line['confidence_decile'], line['ci95_low'], line['ci95_high'],
                alpha=.15)
        axis.set(title=f'group size {group_size}', xlabel='Baseline margin decile',
                 ylabel='Damage rate' if group_size == 8 else '')
        axis.set_xticks(range(1, 11))
    axes[-1].legend(title='HQQ')
    figure.suptitle('Correct-to-wrong damage by unquantized confidence')
    figure.tight_layout()
    return _save(figure, path, dpi)


def plot_heatmaps(transitions, path, dpi):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title, fmt in [
            (axes[0], 'accuracy_ratio', 'Mean accuracy ratio', '.3f'),
            (axes[1], 'damage_rate', 'Mean damage rate', '.3f')]:
        pivot = transitions.pivot_table(
            index='hqq_group_size', columns='hqq_nbits', values=metric,
            aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap='viridis_r' if metric == 'damage_rate'
                    else 'viridis', ax=axis)
        axis.set(title=title, xlabel='HQQ nbits', ylabel='Group size')
    figure.tight_layout()
    return _save(figure, path, dpi)


def plot_accuracy(transitions, path, dpi):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.scatterplot(
        data=transitions, x='baseline_accuracy', y='hqq_top1',
        hue='hqq_nbits', style='hqq_group_size', palette='viridis', ax=axes[0])
    limits = [min(transitions['baseline_accuracy'].min(), transitions['hqq_top1'].min()),
              max(transitions['baseline_accuracy'].max(), transitions['hqq_top1'].max())]
    axes[0].plot(limits, limits, color='black', linewidth=.8, linestyle='--')
    axes[0].set(title='Unquantized vs HQQ accuracy', xlabel='Unquantized accuracy',
                ylabel='HQQ accuracy')
    sns.scatterplot(
        data=transitions, x='baseline_accuracy', y='accuracy_drop_points',
        hue='hqq_nbits', style='hqq_group_size', palette='viridis',
        legend=False, ax=axes[1])
    axes[1].axhline(0, color='black', linewidth=.8)
    axes[1].set(title='Accuracy drop', xlabel='Unquantized accuracy',
                ylabel='Baseline − HQQ points')
    figure.tight_layout()
    return _save(figure, path, dpi)


def plot_transitions(transitions, path, dpi):
    frame = transitions.copy()
    frame['n_samples'] = (
        frame['damage_count'] + frame['rescue_count']
        + frame['unchanged_correct_count'] + frame['unchanged_wrong_count'])
    frame['damage_fraction'] = frame['damage_count'] / frame['n_samples']
    frame['rescue_fraction'] = frame['rescue_count'] / frame['n_samples']
    summary = frame.groupby('condition')[['damage_fraction', 'rescue_fraction']].mean()
    order = _condition_order(frame)['condition']
    summary = summary.loc[order]
    x = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(x, summary['damage_fraction'], label='correct → wrong')
    axis.bar(x, -summary['rescue_fraction'], label='wrong → correct')
    axis.axhline(0, color='black', linewidth=.8)
    axis.set(title='Prediction-transition decomposition', ylabel='Fraction of test samples',
             xticks=x, xticklabels=summary.index)
    axis.tick_params(axis='x', rotation=35)
    axis.legend()
    figure.tight_layout()
    return _save(figure, path, dpi)


def plot_reliability(reliability, condition_summary, path, dpi):
    frame = reliability.copy()
    frame['confidence_sum'] = frame['mean_confidence'] * frame['count']
    frame['correct_sum'] = frame['accuracy'] * frame['count']
    summary = frame.groupby(['condition', 'bin'], as_index=False).agg(
        count=('count', 'sum'), confidence_sum=('confidence_sum', 'sum'),
        correct_sum=('correct_sum', 'sum'))
    summary['mean_confidence'] = summary['confidence_sum'] / summary['count']
    summary['accuracy'] = summary['correct_sum'] / summary['count']
    figure, axis = plt.subplots(figsize=(7, 6))
    ece = condition_summary.groupby('condition')['ece_uncalibrated'].mean()
    for condition, line in summary.groupby('condition'):
        axis.plot(line['mean_confidence'], line['accuracy'], marker='o',
                  markersize=3,
                  label=f'{condition} (ECE={ece.get(condition, np.nan):.3f})')
    axis.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=.8)
    axis.set(title='Uncalibrated reliability', xlabel='Mean confidence',
             ylabel='Empirical accuracy', xlim=(0, 1), ylim=(0, 1))
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    return _save(figure, path, dpi)


def plot_delta_distributions(prepared_dir, transitions, path, dpi):
    edges = np.linspace(-1, 1, 51)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = {}
    for record in transitions.to_dict('records'):
        with np.load(os.path.join(prepared_dir, record['pair_file']),
                     allow_pickle=False) as pair:
            histogram, _ = np.histogram(pair['delta_top1_probability'], bins=edges)
        counts.setdefault(record['condition'], np.zeros(len(histogram), dtype=np.int64))
        counts[record['condition']] += histogram
    figure, axis = plt.subplots(figsize=(9, 4))
    for condition in _condition_order(transitions)['condition']:
        values = counts[condition]
        axis.plot(centers, values / values.sum(), label=condition)
    axis.axvline(0, color='black', linewidth=.8)
    axis.set(title='Paired top-1 confidence change',
             xlabel='HQQ − unquantized top-1 probability', ylabel='Sample fraction')
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    return _save(figure, path, dpi)


def plot_overview(bins, transitions, path, dpi):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    damage = transitions.groupby('hqq_nbits')['damage_rate'].mean().sort_index()
    axes[0, 0].plot(damage.index, damage.values, marker='o')
    axes[0, 0].set(title='Damage by bit width', xlabel='HQQ nbits', ylabel='Mean rate')
    ratio = transitions.pivot_table(index='hqq_group_size', columns='hqq_nbits',
                                    values='accuracy_ratio', aggfunc='mean')
    sns.heatmap(ratio, annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 1])
    axes[0, 1].set_title('Accuracy ratio')
    for group_size, line in bins.groupby('hqq_group_size'):
        curve = line.groupby('confidence_decile')['damage_rate'].mean().sort_index()
        axes[1, 0].plot(curve.index, curve.values, marker='o', label=f'g={group_size:g}')
    axes[1, 0].set(title='Confidence relationship', xlabel='Baseline margin decile',
                   ylabel='Damage rate')
    axes[1, 0].legend()
    drop = transitions.groupby('condition')['accuracy_drop_points'].mean()
    order = _condition_order(transitions)['condition']
    axes[1, 1].barh(order, drop.loc[order])
    axes[1, 1].set(title='Mean accuracy drop', xlabel='Baseline − HQQ points')
    figure.suptitle('HQQ confidence fragility overview')
    figure.tight_layout()
    return _save(figure, path, dpi)


def plot_all(prepared_dir, stats_dir, results_dir, dpi=300):
    if dpi < 1:
        raise ValueError('dpi must be positive')
    os.makedirs(results_dir, exist_ok=True)
    transitions = pd.read_csv(os.path.join(prepared_dir, 'transition_summary.csv'))
    reliability = pd.read_csv(os.path.join(prepared_dir, 'reliability_bins.csv'))
    condition_summary = pd.read_csv(os.path.join(prepared_dir, 'condition_summary.csv'))
    bins = pd.read_csv(os.path.join(stats_dir, 'confidence_bins.csv'))
    sns.set_theme(style='whitegrid')
    writers = [
        ('overview.png', lambda path: plot_overview(bins, transitions, path, dpi),
         'confidence_bins.csv; transition_summary.csv'),
        ('confidence_damage_by_condition.png',
         lambda path: plot_confidence_damage(bins, path, dpi), 'confidence_bins.csv'),
        ('condition_heatmaps.png', lambda path: plot_heatmaps(transitions, path, dpi),
         'transition_summary.csv'),
        ('baseline_vs_hqq_top1.png', lambda path: plot_accuracy(transitions, path, dpi),
         'transition_summary.csv'),
        ('transition_decomposition.png', lambda path: plot_transitions(transitions, path, dpi),
         'transition_summary.csv'),
        ('reliability_diagrams.png',
         lambda path: plot_reliability(reliability, condition_summary, path, dpi),
         'reliability_bins.csv; condition_summary.csv'),
        ('confidence_delta_distributions.png',
         lambda path: plot_delta_distributions(prepared_dir, transitions, path, dpi),
         'pairs/*.npz'),
    ]
    written = []
    manifest = []
    for filename, writer, source in writers:
        path = os.path.join(results_dir, filename)
        written.append(writer(path))
        manifest.append({'plot_file': filename, 'source': source})
    pd.DataFrame(manifest).to_csv(
        os.path.join(results_dir, 'plots_manifest.csv'), index=False)
    return written


def main():
    args = parse_args()
    args.prepared_dir = os.path.abspath(args.prepared_dir)
    args.stats_dir = os.path.abspath(args.stats_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    write_run_config(args.results_dir, args)
    written = plot_all(args.prepared_dir, args.stats_dir, args.results_dir, args.dpi)
    print(f'Wrote {len(written)} figures to {args.results_dir}')


if __name__ == '__main__':
    main()
