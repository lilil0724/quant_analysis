"""Test FP inter/intra-class distances against five-dataset weight-only loss.

Run from the repository root with::

    conda run -n opencode_env python -m five_dataset_features.distance_hypothesis

Uses the verified saved analysis snapshot; full inference arrays are not needed.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from five_dataset_features.geometry_robustness import (
    COLORS, CONDITIONS, DATASETS, ROOT, sha256, style,
    verify_aggregate, write_csv,
)


def save_slide(fig, out, name):
    """Preserve the 16:9 canvas when saving presentation figures."""
    fig.savefig(out / f'{name}.png', dpi=180, facecolor='white')
    fig.savefig(out / f'{name}.svg', facecolor='white')
    plt.close(fig)


def build_tables(fp, all_rows):
    """Return one FP row per dataset and one row per fixed quantization condition."""
    if len(fp) != len(DATASETS) or set(fp.dataset) != set(DATASETS):
        raise ValueError('Expected one FP indicator row for each dataset')
    fp = fp.set_index('dataset')
    selected = all_rows[all_rows.condition.isin(CONDITIONS)].copy()
    expected = pd.MultiIndex.from_product([DATASETS, CONDITIONS], names=['dataset', 'condition'])
    actual = pd.MultiIndex.from_frame(selected[['dataset', 'condition']])
    if len(selected) != len(expected) or not actual.is_unique or set(actual) != set(expected):
        raise ValueError('Expected exactly one row for each dataset and weight-only condition')
    if not np.allclose(selected.accuracy_drop,
                       selected.fp_accuracy - selected.accuracy, atol=1e-9):
        raise ValueError('Inconsistent absolute accuracy drops')
    if not np.allclose(selected.relative_loss,
                       selected.accuracy_drop / selected.fp_accuracy, atol=1e-9):
        raise ValueError('Inconsistent relative accuracy losses')
    rows = []
    condition_rows = []
    for dataset in DATASETS:
        base = fp.loc[dataset]
        group = selected[selected.dataset == dataset]
        if not np.allclose(group.fp_accuracy, base.fp_accuracy, atol=1e-9):
            raise ValueError(f'FP accuracy mismatch: {dataset}')
        inter = float(base.fp_d_inter)
        intra = float(base.fp_d_intra)
        if intra <= 0 or not np.isclose(inter - intra, base.fp_separability, atol=1e-9):
            raise ValueError(f'Invalid FP distance summary: {dataset}')
        rows.append({
            'dataset': dataset,
            'fp_d_inter': inter,
            'fp_d_intra': intra,
            'fp_distance_gap': inter - intra,
            'fp_distance_ratio': inter / intra,
            'fp_accuracy': float(base.fp_accuracy),
            'mean_relative_loss': float(group.relative_loss.mean()),
            'mean_accuracy_drop': float(group.accuracy_drop.mean()),
        })
        for condition in CONDITIONS:
            row = group[group.condition == condition].iloc[0]
            condition_rows.append({
                'dataset': dataset, 'condition': condition,
                'relative_loss': float(row.relative_loss),
                'accuracy_drop': float(row.accuracy_drop),
                'quantized_accuracy': float(row.accuracy),
            })
    return pd.DataFrame(rows), pd.DataFrame(condition_rows)


def correlations(summary, conditions):
    predictors = ('fp_distance_gap', 'fp_distance_ratio', 'fp_d_inter',
                  'fp_d_intra', 'fp_accuracy')
    outcomes = [('mean', 'relative_loss', summary.set_index('dataset').mean_relative_loss),
                ('mean', 'accuracy_drop', summary.set_index('dataset').mean_accuracy_drop)]
    for condition in CONDITIONS:
        group = conditions[conditions.condition == condition].set_index('dataset')
        for measure in ('relative_loss', 'accuracy_drop'):
            outcomes.append((condition, measure, group[measure]))
    indexed = summary.set_index('dataset')
    def rho(predictor, outcome):
        values = indexed[predictor]
        if values.nunique() < 2 or outcome.nunique() < 2:
            return float('nan')
        return float(values.corr(outcome, method='spearman'))

    return pd.DataFrame([
        {'predictor': predictor, 'condition': condition, 'loss_metric': measure,
         'spearman_rho': rho(predictor, outcome),
         'n_datasets': len(DATASETS)}
        for predictor in predictors
        for condition, measure, outcome in outcomes
    ])


def plot_distances(summary, out):
    ordered = summary.sort_values('fp_distance_gap')
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=.16, right=.91, top=.83, bottom=.18)
    for y, row in enumerate(ordered.itertuples()):
        color = COLORS[row.dataset]
        ax.plot([row.fp_d_intra, row.fp_d_inter], [y, y], color=color, lw=4, alpha=.8)
        ax.scatter([row.fp_d_intra], [y], s=170, marker='o', color=color, zorder=3,
                   edgecolor='white', linewidth=1.5)
        ax.scatter([row.fp_d_inter], [y], s=170, marker='D', color=color, zorder=3,
                   edgecolor='white', linewidth=1.5)
        ax.text(1.035, y, f'gap {row.fp_distance_gap:.3f}  |  ratio {row.fp_distance_ratio:.2f}',
                va='center', fontsize=11, color=color)
    ax.set(yticks=range(len(ordered)), yticklabels=ordered.dataset, xlim=(.15, 1.36),
           xlabel='Cosine distance in normalized FP final CLS features')
    ax.set_ylim(len(ordered) - .5, -.5)
    ax.grid(axis='x', alpha=.18)
    ax.text(.18, -.65, '● intra-class', fontsize=12, color='#394A55')
    ax.text(.45, -.65, '◆ inter-class', fontsize=12, color='#394A55')
    fig.suptitle('How close are inter- and intra-class distances?',
                 x=.16, ha='left', y=.94, fontsize=21, weight='bold')
    fig.text(.16, .07, 'Class-equal means | AMP-FP16 reference | smaller gap or ratio means closer distances',
             fontsize=11, color='#555555')
    save_slide(fig, out, '01_inter_intra_distances')


def plot_mean_losses(summary, corr, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    fig.subplots_adjust(left=.09, right=.96, top=.81, bottom=.19, wspace=.28)
    for ax, metric, label, title in (
        (axes[0], 'mean_relative_loss', 'Mean relative accuracy loss (%)',
         'Loss relative to FP accuracy'),
        (axes[1], 'mean_accuracy_drop', 'Mean accuracy drop (points)',
         'Absolute accuracy drop'),
    ):
        for row in summary.itertuples():
            ax.scatter(row.fp_distance_gap, 100 * getattr(row, metric), s=170,
                       color=COLORS[row.dataset], edgecolor='white', linewidth=1.3, zorder=3)
            ax.annotate(row.dataset, (row.fp_distance_gap, 100 * getattr(row, metric)),
                        xytext=(5, 7), textcoords='offset points', fontsize=10,
                        color=COLORS[row.dataset])
        measure = 'relative_loss' if metric == 'mean_relative_loss' else 'accuracy_drop'
        rho = corr[(corr.predictor == 'fp_distance_gap') &
                   (corr.condition == 'mean') & (corr.loss_metric == measure)].iloc[0].spearman_rho
        ax.set(xlabel='FP inter − intra cosine distance', ylabel=label,
               title=f'{title}  |  ρ = {rho:+.2f}')
        ax.grid(alpha=.2)
    fig.suptitle('Smaller FP gap generally accompanies larger weight-only loss',
                 x=.09, ha='left', y=.94, fontsize=20, weight='bold')
    fig.text(.09, .065, 'Mean of W3/W4 × G8/G128 per dataset | five dataset-checkpoint pairs | descriptive association',
             fontsize=11, color='#555555')
    save_slide(fig, out, '02_gap_vs_mean_loss')


def plot_fixed_conditions(summary, conditions, corr, out):
    fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5), sharex=True)
    fig.subplots_adjust(left=.09, right=.96, top=.83, bottom=.16, wspace=.25, hspace=.35)
    gap = summary.set_index('dataset').fp_distance_gap
    for ax, condition in zip(axes.flat, CONDITIONS):
        group = conditions[conditions.condition == condition].set_index('dataset')
        for dataset in DATASETS:
            ax.scatter(gap[dataset], 100 * group.loc[dataset, 'relative_loss'], s=100,
                       color=COLORS[dataset], edgecolor='white', linewidth=1, zorder=3)
            ax.annotate(dataset, (gap[dataset], 100 * group.loc[dataset, 'relative_loss']),
                        xytext=(4, 4), textcoords='offset points', fontsize=8,
                        color=COLORS[dataset])
        rho = corr[(corr.predictor == 'fp_distance_gap') &
                   (corr.condition == condition) &
                   (corr.loss_metric == 'relative_loss')].iloc[0].spearman_rho
        ax.set_title(f'{condition.upper().replace("_", " / ")}  |  ρ = {rho:+.2f}')
        ax.grid(alpha=.2)
    for ax in axes[1]:
        ax.set_xlabel('FP inter − intra cosine distance')
    for ax in axes[:, 0]:
        ax.set_ylabel('Relative accuracy loss (%)')
    fig.suptitle('Check the relationship under each fixed quantization setting',
                 x=.09, ha='left', y=.95, fontsize=20, weight='bold')
    fig.text(.09, .055, 'Each panel reuses the same five dataset-checkpoint pairs | AMP-FP16 reference',
             fontsize=11, color='#555555')
    save_slide(fig, out, '03_gap_vs_fixed_conditions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, default=ROOT / 'analysis')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'analysis' / 'distance_hypothesis')
    args = parser.parse_args()
    verify_aggregate(args.analysis_root)
    fp_path = args.analysis_root / 'fp_indicators.csv'
    all_path = args.analysis_root / 'all_conditions.csv'
    summary, conditions = build_tables(pd.read_csv(fp_path), pd.read_csv(all_path))
    corr = correlations(summary, conditions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'dataset_distances_and_loss.csv', summary.to_dict('records'))
    write_csv(args.output_dir / 'condition_losses.csv', conditions.to_dict('records'))
    write_csv(args.output_dir / 'spearman_by_condition.csv', corr.to_dict('records'))
    style()
    plot_distances(summary, args.output_dir)
    plot_mean_losses(summary, corr, args.output_dir)
    plot_fixed_conditions(summary, conditions, corr, args.output_dir)
    provenance = {
        'analysis_snapshot_sha256': sha256(args.analysis_root / 'analysis_snapshot.json'),
        'source_csv_sha256': {'fp_indicators.csv': sha256(fp_path),
                              'all_conditions.csv': sha256(all_path)},
        'source_paths': [str(fp_path.resolve()), str(all_path.resolve())],
        'reference': 'AMP-FP16',
        'weight_only_conditions': CONDITIONS,
    }
    (args.output_dir / 'run_config.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    get_rho = lambda condition, metric: corr[(corr.predictor == 'fp_distance_gap') &
                                            (corr.condition == condition) &
                                            (corr.loss_metric == metric)].iloc[0].spearman_rho
    lines = [
        '# Inter/intra-class distance hypothesis', '',
        'FP normalized final CLS features use class-equal mean cosine distances. '
        '`gap = D_inter - D_intra`; `ratio = D_inter / D_intra`. Smaller values mean closer distances.', '',
        'The five dataset-checkpoint pairs have different FP accuracies and one seed each. '
        'The mean over four conditions is one observation per dataset, not 20 independent samples.', '',
        '## Dataset summary', '',
        '| Dataset | D_inter | D_intra | Gap | Ratio | FP accuracy | Mean relative loss | Mean absolute drop |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    lines.extend(
        f'| {r.dataset} | {r.fp_d_inter:.3f} | {r.fp_d_intra:.3f} | '
        f'{r.fp_distance_gap:.3f} | {r.fp_distance_ratio:.2f} | '
        f'{100*r.fp_accuracy:.1f}% | {100*r.mean_relative_loss:.2f}% | '
        f'{100*r.mean_accuracy_drop:.2f} pp |'
        for r in summary.itertuples()
    )
    lines += ['',
        f'Mean weight-only relative-loss Spearman rho versus FP gap: {get_rho("mean", "relative_loss"):+.2f}; '
        f'absolute-drop rho: {get_rho("mean", "accuracy_drop"):+.2f}. '
        'Relative loss is the primary cross-baseline outcome; absolute drop is shown for sensitivity. '
        'Cotton has the smallest gap but less mean relative loss than both soybean datasets. '
        'Its absolute drop is nearly equal to soyageing. These observations are descriptive, not causal.', '',
        '## Fixed-condition relative-loss correlations', '',
        '| Condition | Spearman rho: FP gap vs relative loss |',
        '|---|---:|',
    ]
    lines.extend(f'| {condition} | {get_rho(condition, "relative_loss"):+.2f} |'
                 for condition in CONDITIONS)
    fp_accuracy_rho = corr[(corr.predictor == 'fp_accuracy') &
                           (corr.condition == 'mean') &
                           (corr.loss_metric == 'relative_loss')].iloc[0].spearman_rho
    lines += ['', 'The gap and ratio have the same dataset ordering in this sample. '
              'D_inter varies much less than D_intra, so the gap is driven mainly by within-class spread. '
              f'FP accuracy has rho {fp_accuracy_rho:+.2f} with mean relative loss; '
              'classifier logit margin may also confound a geometry-only explanation.', '',
              'The input CSVs and snapshot SHA-256 values are in `run_config.json`. '
              'Exact numeric outputs are in `dataset_distances_and_loss.csv`, '
              '`condition_losses.csv`, and `spearman_by_condition.csv`. '
              'The three figures are saved as PNG and SVG.', '']
    (args.output_dir / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
