"""Descriptive feature-geometry test and slide figures for the five-dataset run.

Run from the repository root with opencode_env::

    python five_dataset_features/geometry_robustness.py
    python five_dataset_features/geometry_robustness.py --input-root five_dataset_features/data

Aggregate figures use the recorded analysis snapshot. The optional sample test
needs the original full inference arrays, not the 2-D PCA coordinates or ZIP.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ('cotton', 'soyageing', 'soyglobal', 'cub', 'pets')
CONDITIONS = ('w3_g8', 'w3_g128', 'w4_g8', 'w4_g128')
COLORS = dict(zip(DATASETS, ('#008A70', '#D87626', '#3278AB', '#A4558C', '#8A7333')))
ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_aggregate(analysis):
    snapshot = json.loads((analysis / 'analysis_snapshot.json').read_text(encoding='utf-8'))
    for name, expected in snapshot['analysis_code'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Analysis code differs from snapshot: {name}')
    for name in ('all_conditions.csv', 'fp_indicators.csv'):
        if sha256(analysis / name) != snapshot['derived_files'][name]:
            raise ValueError(f'Aggregate output differs from snapshot: {name}')
    for dataset in DATASETS:
        local = analysis / dataset / 'analysis_snapshot.json'
        if sha256(local) != snapshot['datasets'][dataset]:
            raise ValueError(f'Dataset snapshot differs: {dataset}')
        item = json.loads(local.read_text(encoding='utf-8'))
        for name in ('summary.csv', 'layer_metrics.csv'):
            if sha256(analysis / dataset / name) != item['derived_files'][name]:
                raise ValueError(f'Dataset output differs from snapshot: {dataset}/{name}')
    return snapshot


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=180, facecolor='white', bbox_inches='tight')
    fig.savefig(out / f'{name}.svg', facecolor='white', bbox_inches='tight')
    plt.close(fig)


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def schematic(out):
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=.06, right=.96, top=.85, bottom=.13)
    a = np.array([[1.5, 3.25], [1.8, 4.2], [2.2, 2.75], [2.5, 3.8],
                  [2.9, 3.05], [3.1, 4.35], [3.4, 3.5]])
    b = np.array([[8.1, 3.0], [8.4, 4.0], [8.7, 2.55], [9.1, 3.5],
                  [9.4, 4.25], [9.7, 2.9], [10.0, 3.8]])
    ax.scatter(*a.T, s=120, c='#3278AB', edgecolor='white', linewidth=1.5, zorder=3)
    ax.scatter(*b.T, s=120, c='#D87626', edgecolor='white', linewidth=1.5, zorder=3)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    ax.scatter([ca[0], cb[0]], [ca[1], cb[1]], marker='X', s=270,
               c=['#07547F', '#A94F0C'], edgecolor='white', linewidth=1.3, zorder=4)
    ax.plot([6.0, 6.0], [1.3, 5.6], ls='--', lw=2, c='#666666')
    ax.text(6.1, 5.25, 'Illustrative boundary', color='#555555', fontsize=12)
    ax.annotate('', xy=cb, xytext=ca, arrowprops=dict(arrowstyle='<->', lw=2.2, color='#364850'))
    ax.text(6.0, 4.05, 'inter-class distance', ha='center', fontsize=13,
            bbox=dict(facecolor='white', edgecolor='none', pad=3))
    ax.annotate('', xy=a[5], xytext=ca, arrowprops=dict(arrowstyle='<->', lw=2.2, color='#3278AB'))
    ax.text(1.8, 4.7, 'intra-class distance', color='#07547F', fontsize=13)
    fp = np.array([5.1, 3.1]); q = np.array([6.55, 3.15])
    ax.scatter(*fp, s=190, c='#3278AB', edgecolor='white', linewidth=2, zorder=5)
    ax.scatter(*q, s=190, c='#C33438', edgecolor='white', linewidth=2, zorder=5)
    ax.annotate('', xy=q, xytext=fp, arrowprops=dict(arrowstyle='->', lw=3, color='#C33438'))
    ax.text(4.25, 2.56, 'FP feature', color='#07547F', fontsize=12)
    ax.text(6.35, 2.58, 'Quantized feature', color='#A12226', fontsize=12)
    ax.text(5.75, 2.05, 'quantization drift', ha='center', color='#A12226', fontsize=13)
    ax.set(xlim=(.6, 10.7), ylim=(1.2, 5.7))
    ax.axis('off')
    fig.suptitle('Feature geometry and quantization damage', x=.06, ha='left',
                 y=.94, fontsize=24, weight='bold')
    fig.text(.06, .07, 'Schematic only | Real tests use 768-D final CLS features and paired predictions',
             fontsize=12, color='#555555')
    save_figure(fig, out, '01_geometry_schematic')


def aggregate_analysis(analysis, out):
    verify_aggregate(analysis)
    fp = pd.read_csv(analysis / 'fp_indicators.csv').set_index('dataset')
    all_rows = pd.read_csv(analysis / 'all_conditions.csv')
    if len(all_rows) != 105 or set(all_rows.dataset) != set(DATASETS):
        raise ValueError('Expected five datasets and 105 conditions')
    rows = []
    for dataset in DATASETS:
        selected = all_rows[(all_rows.dataset == dataset) & all_rows.condition.isin(CONDITIONS)]
        if len(selected) != 4:
            raise ValueError(f'Missing weight-only conditions for {dataset}')
        fixed = all_rows[(all_rows.dataset == dataset) & (all_rows.condition == 'w3_g128')]
        layers = pd.read_csv(analysis / dataset / 'layer_metrics.csv')
        qfinal = layers[(layers.condition == 'w3_g128') & (layers.layer == 12)]
        if len(fixed) != 1 or len(qfinal) != 1:
            raise ValueError(f'Missing fixed-condition result for {dataset}')
        rows.append({
            'dataset': dataset, 'fp_accuracy': float(fp.loc[dataset, 'fp_accuracy']),
            'fp_d_intra': float(fp.loc[dataset, 'fp_d_intra']),
            'fp_d_inter': float(fp.loc[dataset, 'fp_d_inter']),
            'fp_separation': float(fp.loc[dataset, 'fp_separability']),
            'weight_only_mean_relative_loss': float(selected.relative_loss.mean()),
            'w3_g128_relative_loss': float(fixed.iloc[0].relative_loss),
            'w3_g128_damage_rate': float(fixed.iloc[0].damage_rate),
            'w3_g128_rescue_rate': float(fixed.iloc[0].rescue_rate),
            'w3_g128_separation': float(qfinal.iloc[0].separability),
        })
    write_csv(out / 'aggregate_geometry.csv', rows)
    frame = pd.DataFrame(rows)
    rho = frame.fp_separation.corr(frame.weight_only_mean_relative_loss, method='spearman')
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    fig.subplots_adjust(left=.09, right=.96, top=.82, bottom=.18, wspace=.28)
    for _, r in frame.iterrows():
        c = COLORS[r.dataset]
        axes[0].scatter(r.fp_separation, 100 * r.weight_only_mean_relative_loss,
                        s=150, c=c, edgecolor='white', linewidth=1.2, zorder=3)
        axes[0].annotate(r.dataset, (r.fp_separation, 100 * r.weight_only_mean_relative_loss),
                         xytext=(6, 5), textcoords='offset points', color=c, fontsize=11)
        axes[1].plot([0, 1], [r.fp_separation, r.w3_g128_separation],
                     marker='o', markersize=10, lw=2.2, c=c)
        axes[1].annotate(r.dataset, (1, r.w3_g128_separation),
                         xytext=(8, 0), textcoords='offset points', va='center',
                         fontsize=10, color=c)
    axes[0].set(xlabel='FP class-equal separation', ylabel='Mean weight-only relative loss (%)',
                title='Before quantization vs accuracy loss')
    axes[0].grid(alpha=.2)
    axes[1].set(xticks=[0, 1], xticklabels=['FP', 'W3 / G128'],
                ylabel='Class-equal separation', title='Feature separation after quantization')
    axes[1].grid(axis='y', alpha=.2)
    axes[1].set_xlim(-.08, 1.24)
    fig.suptitle('Five datasets: geometry is associated with weight-only loss',
                 x=.09, ha='left', y=.94, fontsize=21, weight='bold')
    fig.text(.09, .065, f'Spearman rho = {rho:+.2f} (5 pairs) | Descriptive only | AMP-FP16 reference | one seed each',
             fontsize=11, color='#555555')
    save_figure(fig, out, '02_five_dataset_geometry')
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=.09, right=.96, top=.82, bottom=.19)
    positions = np.arange(len(DATASETS))
    ax.bar(positions - .2, 100 * frame.w3_g128_damage_rate, .37,
           color='#C33438', label='FP correct -> quantized wrong')
    ax.bar(positions + .2, 100 * frame.w3_g128_rescue_rate, .37,
           color='#3278AB', label='FP wrong -> quantized correct')
    ax.set(xticks=positions, xticklabels=DATASETS, ylabel='Share of all test images (%)',
           title='W3 / G128: accuracy change is damage minus rescue')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
    fig.text(.09, .07, 'Paired predictions | Smaller feature separation alone does not determine accuracy',
             fontsize=11, color='#555555')
    save_figure(fig, out, '03_damage_and_rescue')
    return frame, rho


def _load_condition(folder, dataset, name, sample_ids, targets, n):
    path = folder / name
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['dataset'] != dataset or manifest['n_samples'] != n:
        raise ValueError(f'Incomplete or mismatched manifest: {dataset}/{name}')
    if manifest['condition']['name'] != name:
        raise ValueError(f'Condition name mismatch: {dataset}/{name}')
    for filename in ('features.npy', 'predictions.csv'):
        if sha256(path / filename) != manifest['files'][filename]:
            raise ValueError(f'Checksum mismatch: {dataset}/{name}/{filename}')
    feature = np.load(path / 'features.npy', mmap_mode='r', allow_pickle=False)
    if feature.shape != (n, 13, 768) or not np.issubdtype(feature.dtype, np.floating):
        raise ValueError(f'Invalid feature array: {dataset}/{name}')
    pred = pd.read_csv(path / 'predictions.csv')
    if len(pred) != n or pred.sample_id.tolist() != sample_ids or pred.target.tolist() != targets:
        raise ValueError(f'Prediction pairing failed: {dataset}/{name}')
    x = np.asarray(feature[:, 12, :], dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(x).all() or np.any(norms == 0):
        raise ValueError(f'Nonfinite or zero features: {dataset}/{name}')
    return x / norms[:, None], pred


def leave_one_out_gap(x, labels):
    """Euclidean nearest-rival gap on L2-normalized features; no self-centroid bias."""
    classes, index = np.unique(labels, return_inverse=True)
    counts = np.bincount(index)
    sums = np.zeros((len(classes), x.shape[1]), dtype=np.float64)
    np.add.at(sums, index, x)
    centers = sums / counts[:, None]
    own = np.full(len(x), np.nan, dtype=np.float64)
    eligible = counts[index] > 1
    own_center = (sums[index[eligible]] - x[eligible]) / (counts[index[eligible], None] - 1)
    own[eligible] = np.linalg.norm(x[eligible] - own_center, axis=1)
    rival = np.full(len(x), np.nan, dtype=np.float64)
    ncc = np.full(len(x), -1, dtype=np.int64)
    center_norm = np.sum(centers * centers, axis=1)
    for start in range(0, len(x), 1024):
        sl = slice(start, min(start + 1024, len(x)))
        squared = (np.sum(x[sl] * x[sl], axis=1)[:, None] + center_norm[None, :]
                   - 2 * x[sl] @ centers.T)
        distances = np.sqrt(np.maximum(squared, 0))
        local_index = index[sl]
        distances[np.arange(len(local_index)), local_index] = np.inf
        nearest = distances.argmin(axis=1)
        rival[sl] = distances[np.arange(len(local_index)), nearest]
        ncc[sl] = np.where(own[sl] <= rival[sl], labels[sl], classes[nearest])
    return rival - own, ncc


def binary_auc(outcome, score):
    """Tie-aware AUC of a score for a binary event; undefined for one class."""
    outcome = np.asarray(outcome, dtype=bool)
    score = np.asarray(score, dtype=float)
    positives = int(outcome.sum())
    negatives = len(outcome) - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    ranks = pd.Series(score).rank(method='average').to_numpy()
    return float((ranks[outcome].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def raw_analysis(input_root, out, analysis):
    summaries = []
    bin_rows = []
    stratified_rows = []
    quadrant_rows = []
    for dataset in DATASETS:
        folder = input_root / dataset
        meta_path = folder / 'dataset.json'
        if not meta_path.is_file():
            raise FileNotFoundError(f'Missing full inference data: {meta_path}')
        recorded = json.loads((analysis / dataset / 'analysis_snapshot.json').read_text(encoding='utf-8'))
        if sha256(meta_path) != recorded['source_manifests']['dataset.json']:
            raise ValueError(f'Full inference dataset.json differs from saved analysis: {dataset}')
        for condition in ('fp', *CONDITIONS):
            name = f'{condition}/manifest.json'
            if sha256(folder / name) != recorded['source_manifests'][name]:
                raise ValueError(f'Full inference manifest differs from saved analysis: {dataset}/{name}')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        n = meta['n_samples']
        sample_ids, targets = meta['sample_ids'], meta['targets']
        if meta['dataset'] != dataset or len(sample_ids) != n or len(targets) != n:
            raise ValueError(f'Invalid dataset metadata: {dataset}')
        fp_x, fp = _load_condition(folder, dataset, 'fp', sample_ids, targets, n)
        labels = np.asarray(targets)
        gap, ncc = leave_one_out_gap(fp_x, labels)
        fp_correct = fp.prediction.to_numpy() == labels
        usable = fp_correct & np.isfinite(gap)
        if usable.sum() < 20:
            raise ValueError(f'Too few baseline-correct samples with defined gap: {dataset}')
        margin = fp.normalized_margin.to_numpy(dtype=float)
        if not np.isfinite(margin).all():
            raise ValueError(f'Invalid baseline normalized margins: {dataset}')
        margin_bins = pd.qcut(pd.Series(margin[usable]), 5, labels=False,
                              duplicates='drop').to_numpy()
        usable_indices = np.flatnonzero(usable)
        cut = np.quantile(gap[usable], [.2, .8])
        groups = {'low 20%': usable & (gap <= cut[0]),
                  'high 20%': usable & (gap >= cut[1])}
        q10 = float(np.quantile(gap[usable], .1))
        agreement = float(np.mean(ncc[usable] == fp.prediction.to_numpy()[usable]))
        for condition in CONDITIONS:
            q_x, q = _load_condition(folder, dataset, condition, sample_ids, targets, n)
            q_correct = q.prediction.to_numpy() == labels
            damage = fp_correct & ~q_correct
            drift = np.linalg.norm(q_x - fp_x, axis=1)
            for group, mask in groups.items():
                bin_rows.append({'dataset': dataset, 'condition': condition, 'group': group,
                                 'n': int(mask.sum()), 'damage_count': int(damage[mask].sum()),
                                 'damage_rate_given_fp_correct': float(damage[mask].mean())})
            low, high = (bin_rows[-2]['damage_rate_given_fp_correct'],
                         bin_rows[-1]['damage_rate_given_fp_correct'])
            for margin_bin in np.unique(margin_bins):
                members = usable_indices[margin_bins == margin_bin]
                local_cut = np.quantile(gap[members], [.2, .8])
                for group, selected in (('low 20%', members[gap[members] <= local_cut[0]]),
                                        ('high 20%', members[gap[members] >= local_cut[1]])):
                    stratified_rows.append({
                        'dataset': dataset, 'condition': condition,
                        'margin_bin': int(margin_bin), 'group': group,
                        'n': len(selected), 'damage_count': int(damage[selected].sum()),
                    })
            recent = stratified_rows[-2 * len(np.unique(margin_bins)):]
            strat_low = [item for item in recent if item['group'] == 'low 20%']
            strat_high = [item for item in recent if item['group'] == 'high 20%']
            strat_low_rate = sum(item['damage_count'] for item in strat_low) / sum(item['n'] for item in strat_low)
            strat_high_rate = sum(item['damage_count'] for item in strat_high) / sum(item['n'] for item in strat_high)
            small_gap = gap <= np.median(gap[usable])
            high_drift = drift > np.median(drift[usable])
            for gap_group, gap_mask in (('Small gap', small_gap), ('Large gap', ~small_gap)):
                for drift_group, drift_mask in (('Low drift', ~high_drift), ('High drift', high_drift)):
                    mask = usable & gap_mask & drift_mask
                    quadrant_rows.append({
                        'dataset': dataset, 'condition': condition,
                        'gap_group': gap_group, 'drift_group': drift_group,
                        'n': int(mask.sum()), 'damage_count': int(damage[mask].sum()),
                        'damage_rate_given_fp_correct': float(damage[mask].mean()) if mask.any() else float('nan'),
                    })
            summaries.append({
                'dataset': dataset, 'condition': condition, 'n': n,
                'n_fp_correct_gap_defined': int(usable.sum()),
                'fp_gap_q10': q10, 'fp_head_nearest_centroid_agreement': agreement,
                'low20_damage_rate': low, 'high20_damage_rate': high,
                'low_minus_high_damage_rate': low - high,
                'margin_stratified_low20_damage_rate': strat_low_rate,
                'margin_stratified_high20_damage_rate': strat_high_rate,
                'margin_stratified_low_minus_high': strat_low_rate - strat_high_rate,
                'gap_damage_auc': binary_auc(damage[usable], -gap[usable]),
                'margin_damage_auc': binary_auc(damage[usable], -margin[usable]),
                'drift_damage_auc': binary_auc(damage[usable], drift[usable]),
                'gap_margin_spearman': float(pd.Series(gap[usable]).corr(
                    pd.Series(margin[usable]), method='spearman')),
                'median_normalized_feature_drift': float(np.median(drift)),
                'fraction_fp_correct_gap_below_2drift': float(np.mean(gap[usable] <= 2 * drift[usable])),
                'damage_count': int(damage.sum()),
                'rescue_count': int((~fp_correct & q_correct).sum()),
            })
    write_csv(out / 'sample_geometry_summary.csv', summaries)
    write_csv(out / 'sample_gap_bins.csv', bin_rows)
    write_csv(out / 'margin_stratified_gap_bins.csv', stratified_rows)
    write_csv(out / 'gap_drift_quadrants.csv', quadrant_rows)
    bins = pd.DataFrame(bin_rows)
    selected = bins[bins.condition == 'w3_g128']
    summary = pd.DataFrame(summaries)
    fixed = summary[summary.condition == 'w3_g128'].set_index('dataset')
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), sharey=True)
    fig.subplots_adjust(left=.09, right=.96, top=.78, bottom=.2, wspace=.13)
    x = np.arange(len(DATASETS))
    low = [float(selected[(selected.dataset == d) & (selected.group == 'low 20%')].iloc[0].damage_rate_given_fp_correct) for d in DATASETS]
    high = [float(selected[(selected.dataset == d) & (selected.group == 'high 20%')].iloc[0].damage_rate_given_fp_correct) for d in DATASETS]
    strat_low = [fixed.loc[d, 'margin_stratified_low20_damage_rate'] for d in DATASETS]
    strat_high = [fixed.loc[d, 'margin_stratified_high20_damage_rate'] for d in DATASETS]
    for ax, lo, hi, title in zip(axes, (low, strat_low), (high, strat_high),
                                 ('Raw gap quintiles', 'Within FP logit-margin quintiles')):
        ax.bar(x - .19, 100 * np.array(lo), .35, color='#C33438', label='Small gap')
        ax.bar(x + .19, 100 * np.array(hi), .35, color='#3278AB', label='Large gap')
        ax.set(xticks=x, xticklabels=DATASETS, title=title)
        ax.tick_params(axis='x', labelrotation=20)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('FP-correct to quantized-wrong rate (%)')
    axes[1].legend(frameon=False)
    fig.suptitle('W3 / G128: geometric gap and sample damage', x=.09, ha='left',
                 y=.92, fontsize=21, weight='bold')
    fig.text(.09, .07, 'Descriptive test | quintiles defined within each dataset | test-set leave-one-out class centroids',
             fontsize=11, color='#555555')
    save_figure(fig, out, '04_gap_and_damage')
    cells = pd.DataFrame(quadrant_rows)
    fig, axes = plt.subplots(1, 4, figsize=(13.33, 7.5))
    fig.subplots_adjust(left=.09, right=.91, top=.81, bottom=.18, wspace=.17)
    for panel, (ax, condition) in enumerate(zip(axes, CONDITIONS)):
        subset = cells[cells.condition == condition]
        matrix = np.full((2, 2), np.nan)
        for row_index, gap_group in enumerate(('Small gap', 'Large gap')):
            for col_index, drift_group in enumerate(('Low drift', 'High drift')):
                selected = subset[(subset.gap_group == gap_group) &
                                  (subset.drift_group == drift_group)]
                matrix[row_index, col_index] = 100 * selected.damage_rate_given_fp_correct.mean()
        im = ax.imshow(matrix, vmin=0, vmax=60, cmap='YlOrRd', aspect='auto')
        ax.set(xticks=[0, 1], xticklabels=['Low drift', 'High drift'],
               yticks=[0, 1],
               yticklabels=['Small gap', 'Large gap'] if panel == 0 else ['', ''],
               title=condition.upper().replace('_', ' / '))
        ax.tick_params(axis='x', labelrotation=30)
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                ax.text(col, row, f'{value:.1f}%' if np.isfinite(value) else 'n/a',
                        ha='center', va='center', weight='bold',
                        color='white' if value >= 35 else '#333333', fontsize=12)
    fig.colorbar(im, ax=axes, shrink=.78, pad=.018,
                 label='Dataset-equal mean C→W rate (%)')
    fig.suptitle('Damage by geometric gap and feature drift',
                 x=.09, ha='left', y=.93, fontsize=21, weight='bold')
    fig.text(.09, .055, 'Descriptive | median split within each dataset and condition | five datasets weighted equally',
             fontsize=11, color='#555555')
    save_figure(fig, out, '05_gap_and_drift')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, default=ROOT / 'analysis')
    parser.add_argument('--input-root', type=Path, help='Full five-dataset inference output root')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis' / 'geometry_robustness')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    schematic(args.output_dir)
    aggregate, rho = aggregate_analysis(args.analysis_root, args.output_dir)
    result = {'five_dataset_weight_only_separation_rho': float(rho),
              'aggregate_dataset_count': len(aggregate), 'sample_level_run': False,
              'analysis_snapshot_sha256': sha256(args.analysis_root / 'analysis_snapshot.json')}
    if args.input_root:
        rows = raw_analysis(args.input_root, args.output_dir, args.analysis_root)
        result['sample_level_run'] = True
        result['sample_level_conditions'] = len(rows)
        result['input_root'] = str(args.input_root.resolve())
        result['verified_source_manifests'] = {
            dataset: {
                key: value for key, value in json.loads(
                    (args.analysis_root / dataset / 'analysis_snapshot.json').read_text(
                        encoding='utf-8'))['source_manifests'].items()
                if key == 'dataset.json' or key.split('/')[0] in ('fp', *CONDITIONS)
            }
            for dataset in DATASETS
        }
    (args.output_dir / 'run_config.json').write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    lines = [
        '# Feature geometry and quantization robustness', '',
        '## Completed aggregate test', '',
        'Verified the saved five-dataset analysis snapshot and compared FP final-feature '
        'class-equal separation with mean relative accuracy loss across W3/W4 and G8/G128 '
        f'weight-only conditions. Descriptive Spearman rho = {rho:+.2f} across five '
        'dataset-checkpoint pairs. This is an association, not an independent or causal effect.', '',
        '`aggregate_geometry.csv` contains the exact inputs. `01_geometry_schematic` is '
        'illustrative; `02_five_dataset_geometry` and `03_damage_and_rescue` use saved '
        'full-test-set summaries. Each figure has PNG and editable SVG versions.', '',
        '## Sample-level nearest-rival test', '',
    ]
    if result['sample_level_run']:
        sample = pd.DataFrame(rows)
        positive = int((sample.low_minus_high_damage_rate > 0).sum())
        adjusted = int((sample.margin_stratified_low_minus_high > 0).sum())
        auc_gap = float(sample.gap_damage_auc.median())
        auc_margin = float(sample.margin_damage_auc.median())
        auc_drift = float(sample.drift_damage_auc.median())
        q10 = sample.drop_duplicates('dataset').set_index('dataset').fp_gap_q10
        gap_rho = float(q10.corr(aggregate.set_index('dataset').weight_only_mean_relative_loss,
                                 method='spearman'))
        accuracy_rho = float(q10.corr(aggregate.set_index('dataset').fp_accuracy,
                                     method='spearman'))
        margin_better = int((sample.margin_damage_auc > sample.gap_damage_auc).sum())
        lines += [
            f'Completed {len(rows)} dataset-condition comparisons. The low-gap group had '
            f'higher FP-correct-to-quantized-wrong rate in {positive}/{len(rows)} comparisons.',
            f'After grouping by baseline normalized logit-margin quintile, the low-gap '
            f'group had higher damage in {adjusted}/{len(rows)} comparisons. Median '
            f'within-condition damage AUC was {auc_gap:.3f} for geometric gap, '
            f'{auc_drift:.3f} for feature drift, and {auc_margin:.3f} for normalized '
            f'logit margin. The five-dataset FP gap Q10 '
            f'vs mean weight-only relative-loss rank correlation was {gap_rho:+.2f}; '
            f'its correlation with FP accuracy was {accuracy_rho:+.2f}. Logit margin '
            f'had higher damage AUC than geometric gap in {margin_better}/{len(rows)} '
            'condition comparisons.',
            '`sample_geometry_summary.csv`, `sample_gap_bins.csv`, '
            '`margin_stratified_gap_bins.csv`, and `gap_drift_quadrants.csv` give '
            'counts and rates. `04_gap_and_damage` plots the fixed W3/G128 '
            'comparison; `05_gap_and_drift` shows the dataset-equal quadrant rates.', '',
            '### W3/G128 by dataset', '',
            '| Dataset | FP-correct images | Head/centroid agreement | Raw low-high damage (pp) | Margin-stratified low-high (pp) | Gap AUC | Logit-margin AUC |',
            '|---|---:|---:|---:|---:|---:|---:|',
        ]
        for dataset in DATASETS:
            item = sample[(sample.dataset == dataset) &
                          (sample.condition == 'w3_g128')].iloc[0]
            lines.append(
                f"| {dataset} | {int(item.n_fp_correct_gap_defined)} | "
                f"{item.fp_head_nearest_centroid_agreement:.3f} | "
                f"{100 * item.low_minus_high_damage_rate:+.1f} | "
                f"{100 * item.margin_stratified_low_minus_high:+.1f} | "
                f"{item.gap_damage_auc:.3f} | {item.margin_damage_auc:.3f} |"
            )
        lines += ['', 'The 20 comparisons reuse the same five datasets and checkpoints; '
                  'they are not 20 independent replications. The very low head/centroid '
                  'agreement for cotton and soyglobal limits a decision-boundary '
                  'interpretation of their geometric gaps.', '']
    else:
        lines += [
            'Pending: the full `[N,13,768]` feature arrays and paired prediction CSVs '
            'are not present under this project. The 2-D PCA coordinates and compact ZIP '
            'cannot support this test. Pass the original inference root with `--input-root` '
            'to compute leave-one-out nearest-centroid gap, normalized feature drift, '
            'nearest-centroid/head agreement, and damage rates for four weight-only conditions.', '',
        ]
    lines += [
        '## Interpretation limits', '',
        'The five datasets have different classes, baseline accuracies, and dataset-specific '
        'checkpoints, with one seed each. The FP reference here is AMP-FP16. '
        'Class-equal separation is mean inter-class cosine distance minus mean intra-class '
        'cosine distance; it is not the classifier decision margin. The sample-level test '
        'uses Euclidean distances on L2-normalized final features and leave-one-out '
        'test-set centroids, so it remains descriptive. The saved source metadata also '
        'has an unresolved quant_zero/quant_scale provenance question; verify the effective '
        'runner settings before a formal paper claim.', '',
    ]
    (args.output_dir / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'verified_source_manifests'}, indent=2))


if __name__ == '__main__':
    main()
