"""Compare paired FP and weight-only HQQ final CLS features in one FP PCA basis.

Run from the repository root:
    conda run -n opencode_env python -m five_dataset_features.paired_feature_pca

The two fixed conditions are W3/G128 (primary) and W4/G128 (sensitivity).
All saved sample metrics use the full matched test sets. PCA is a visualization;
full-dimensional feature and prediction metrics carry the quantitative result.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .analyze import DATASETS, LAYERS, WIDTH
from .analysis_snapshot import ANALYSIS_CODE, sha256
from .metrics import class_equal_cosine_distances, prediction_metrics


MODULE = Path(__file__).resolve().parent
CONDITIONS = ('fp', 'w3_g128', 'w4_g128')
QUANTIZED = CONDITIONS[1:]
OUTCOMES = ('C_to_C', 'C_to_W', 'W_to_C', 'W_to_W')
COLORS = {'C_to_C': '#4876a2', 'C_to_W': '#c43c39',
          'W_to_C': '#29936b', 'W_to_W': '#866a9e'}


def read_predictions(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def check_sources(input_root, analysis_root, dataset):
    """Bridge the current data/analysis split without trusting stale paths."""
    source = input_root / dataset
    derived = analysis_root / dataset
    meta_path = source / 'dataset.json'
    snapshot_path = derived / 'analysis_snapshot.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    n = meta['n_samples']
    if (meta['dataset'] != dataset or snapshot['dataset'] != dataset or
            len(meta['sample_ids']) != n or len(meta['targets']) != n):
        raise ValueError(f'{dataset}: invalid dataset or snapshot identity')

    # The saved PCA basis was fitted by the original analysis. Verify its
    # source-manifest and code hashes after the repository layout migration.
    for name, expected in snapshot['source_manifests'].items():
        if sha256(source / name) != expected:
            raise ValueError(f'{dataset}: snapshot source changed: {name}')
    for name in ANALYSIS_CODE:
        if sha256(MODULE / name) != snapshot['analysis_code'][name]:
            raise ValueError(f'{dataset}: original analysis code changed: {name}')
    for name in ('fp_pca.npz', 'summary.csv', 'layer_metrics.csv'):
        if sha256(derived / name) != snapshot['derived_files'][name]:
            raise ValueError(f'{dataset}: saved analysis changed: {name}')

    arrays, predictions, hashes = {}, {}, {
        'dataset.json': sha256(meta_path),
        'analysis_snapshot.json': sha256(snapshot_path),
        'fp_pca.npz': sha256(derived / 'fp_pca.npz'),
    }
    for condition in CONDITIONS:
        folder = source / condition
        manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        spec = manifest['condition']
        if (manifest['status'] != 'complete' or manifest['dataset'] != dataset or
                manifest['n_samples'] != n or spec['name'] != condition):
            raise ValueError(f'{dataset}/{condition}: incomplete or incompatible manifest')
        if condition == 'fp':
            if spec['weight_bits'] is not None or spec['act_bits'] is not None:
                raise ValueError(f'{dataset}/fp: expected unquantized condition')
        elif (spec['weight_bits'] != int(condition[1]) or
              spec['weight_group_size'] != 128 or spec['act_bits'] is not None or
              spec['qkv']):
            raise ValueError(f'{dataset}/{condition}: expected weight-only W3/W4 G128')
        identity = {'schema': meta['schema'], 'checkpoint': meta['checkpoint_sha256'],
                    'data': meta['data_sha256'], 'code': meta['code_sha256'],
                    'hqq': meta['hqq_commit'], 'dtype': meta['feature_dtype'],
                    'execution': meta['execution'], 'condition': spec}
        encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False).encode('utf-8')
        if hashlib.sha256(encoded).hexdigest() != manifest['identity']:
            raise ValueError(f'{dataset}/{condition}: manifest identity mismatch')
        hashes[f'{condition}/manifest.json'] = sha256(manifest_path)
        for filename, expected in manifest['files'].items():
            path = folder / filename
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f'{dataset}/{condition}: checksum mismatch: {filename}')
            hashes[f'{condition}/{filename}'] = actual
        feature = np.load(folder / 'features.npy', mmap_mode='r', allow_pickle=False)
        if feature.shape != (n, LAYERS, WIDTH) or str(feature.dtype) != meta['feature_dtype']:
            raise ValueError(f'{dataset}/{condition}: invalid feature shape or dtype')
        rows = read_predictions(folder / 'predictions.csv')
        if ([row['sample_id'] for row in rows] != meta['sample_ids'] or
                [int(row['target']) for row in rows] != meta['targets'] or
                any(int(row['correct']) != (int(row['prediction']) == int(row['target']))
                    for row in rows)):
            raise ValueError(f'{dataset}/{condition}: prediction/sample pairing mismatch')
        arrays[condition], predictions[condition] = feature, rows

    with np.load(derived / 'fp_pca.npz', allow_pickle=False) as basis:
        mean = basis['mean'].copy()
        components = basis['components'].copy()
        explained = basis['explained'].copy()
    if (mean.shape != (WIDTH,) or components.shape != (WIDTH, 2) or
            explained.shape != (2,) or not np.isfinite(mean).all() or
            not np.isfinite(components).all() or not np.isfinite(explained).all()):
        raise ValueError(f'{dataset}: invalid saved FP PCA basis')
    np.testing.assert_allclose(components.T @ components, np.eye(2), atol=1e-8)
    return meta, arrays, predictions, mean, components, explained, hashes


def binary_auc(outcome, score):
    """Tie-aware AUC; undefined if either outcome class is absent."""
    outcome = np.asarray(outcome, dtype=bool)
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all():
        raise ValueError('AUC scores must be finite')
    positives = int(outcome.sum())
    negatives = len(outcome) - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    ranks = pd.Series(score).rank(method='average').to_numpy()
    return float((ranks[outcome].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def paired_metrics(fp_final, q_final, components, fp_rows, q_rows):
    """Use matched images, with 2D drift checked against the full 768D drift."""
    fp = np.asarray(fp_final, dtype=np.float64)
    q = np.asarray(q_final, dtype=np.float64)
    if fp.shape != q.shape or fp.ndim != 2 or fp.shape[1] != WIDTH:
        raise ValueError('Expected matched [N,768] feature arrays')
    if not np.isfinite(fp).all() or not np.isfinite(q).all():
        raise ValueError('Non-finite feature value')
    if len(fp_rows) != len(fp) or len(q_rows) != len(fp):
        raise ValueError('Prediction/feature length mismatch')
    delta = q - fp
    drift = np.linalg.norm(delta, axis=1)
    fp_norm = np.linalg.norm(fp, axis=1)
    if np.any(fp_norm <= 0):
        raise ValueError('Zero FP feature norm')
    relative_drift = drift / fp_norm
    projected = delta @ components
    drift_2d = np.linalg.norm(projected, axis=1)
    energy = np.square(drift).sum()
    captured = float(np.square(drift_2d).sum() / energy) if energy > 0 else float('nan')
    fp_correct = np.fromiter((int(row['correct']) for row in fp_rows), bool, len(fp))
    q_correct = np.fromiter((int(row['correct']) for row in q_rows), bool, len(fp))
    outcome = np.select(
        [fp_correct & q_correct, fp_correct & ~q_correct,
         ~fp_correct & q_correct, ~fp_correct & ~q_correct],
        OUTCOMES, default='unknown')
    margin = np.fromiter((float(row['normalized_margin']) for row in fp_rows),
                         np.float64, len(fp))
    if not np.isfinite(margin).all():
        raise ValueError('Non-finite FP normalized logit margin')
    damage = fp_correct & ~q_correct
    auc_drift = binary_auc(damage[fp_correct], relative_drift[fp_correct])
    auc_margin = binary_auc(damage[fp_correct], -margin[fp_correct])
    details = dict(drift=drift, relative_drift=relative_drift,
                   drift_2d=drift_2d, fp_correct=fp_correct,
                   q_correct=q_correct, outcome=outcome, margin=margin)
    summary = dict(
        n_samples=len(fp), n_fp_correct=int(fp_correct.sum()),
        n_damage=int(damage.sum()), n_rescue=int((~fp_correct & q_correct).sum()),
        median_full_drift=float(np.median(drift)),
        median_relative_drift=float(np.median(relative_drift)),
        median_projected_drift=float(np.median(drift_2d)),
        drift_energy_captured_2d=captured,
        median_relative_drift_damage=(float(np.median(relative_drift[damage]))
                                      if damage.any() else float('nan')),
        median_relative_drift_stable_correct=(float(np.median(relative_drift[fp_correct & q_correct]))
                                              if (fp_correct & q_correct).any() else float('nan')),
        drift_damage_auc=auc_drift, fp_margin_damage_auc=auc_margin)
    return details, summary


def pick_arrows(outcomes, dataset, condition, count_per_group=25):
    seed = int(hashlib.sha256(f'{dataset}/{condition}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    chosen = []
    for group in OUTCOMES:
        indices = np.flatnonzero(outcomes == group)
        if len(indices):
            chosen.extend(rng.choice(indices, min(count_per_group, len(indices)), replace=False))
    return np.asarray(sorted(chosen), dtype=int)


def make_dataset_plot(output, dataset, meta, coords, outcomes, explained):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, sharey=True)
    labels = np.asarray(meta['targets'])
    fp = coords['fp']
    joined = np.concatenate([coords[name] for name in CONDITIONS])
    low = joined.min(axis=0)
    high = joined.max(axis=0)
    pad = np.maximum((high - low) * .06, 1e-6)
    xlim, ylim = (low[0] - pad[0], high[0] + pad[0]), (low[1] - pad[1], high[1] + pad[1])

    counts = pd.Series(labels).value_counts()
    selected = counts.head(8).index.tolist()
    axes[0].scatter(fp[:, 0], fp[:, 1], s=2, c='#d0d0d0', alpha=.35,
                    rasterized=True)
    cmap = plt.get_cmap('tab10')
    for index, label in enumerate(selected):
        subset = labels == label
        axes[0].scatter(fp[subset, 0], fp[subset, 1], s=5,
                        color=cmap(index), alpha=.6, rasterized=True,
                        label=f'class {label}')
    axes[0].set_title('FP final CLS; eight largest classes colored')
    axes[0].legend(loc='upper right', fontsize=7, ncol=2, markerscale=2)

    for ax, condition in zip(axes[1:], QUANTIZED):
        q = coords[condition]
        group = outcomes[condition]
        ax.scatter(fp[:, 0], fp[:, 1], s=2, c='#aaaaaa', alpha=.17,
                   rasterized=True)
        chosen = pick_arrows(group, dataset, condition)
        for outcome in OUTCOMES:
            part = chosen[group[chosen] == outcome]
            if not len(part):
                continue
            start = fp[part]
            motion = q[part] - start
            ax.quiver(start[:, 0], start[:, 1], motion[:, 0], motion[:, 1],
                      angles='xy', scale_units='xy', scale=1, color=COLORS[outcome],
                      width=.003, headwidth=3.4, headlength=4.2, alpha=.85,
                      label=f'{outcome} (shown {len(part)})')
        ax.set_title(f'{condition}: paired FP → HQQ (true-scale arrows)')
        ax.legend(loc='upper right', fontsize=7)
    for ax in axes:
        ax.set(xlim=xlim, ylim=ylim,
               xlabel=f'FP PCA PC1 ({explained[0]:.1%} FP variance)',
               ylabel=f'FP PCA PC2 ({explained[1]:.1%} FP variance)')
        ax.set_aspect('equal', adjustable='box')
    fig.suptitle(f'{dataset}: matched final CLS features; shared axes across all panels', y=1.02)
    fig.tight_layout()
    path = output / 'plots' / f'{dataset}_paired_pca.png'
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def make_overview(output, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = dict(zip(DATASETS, plt.get_cmap('tab10').colors[:len(DATASETS)]))
    markers = {'w3_g128': 'o', 'w4_g128': '^'}
    for dataset, frame in summary.groupby('dataset'):
        ordered = frame.set_index('condition').loc[list(QUANTIZED)]
        axes[0].plot(ordered.median_relative_drift, ordered.relative_loss,
                     color=colors[dataset], alpha=.55, lw=1)
        for condition, row in ordered.iterrows():
            for ax, x, y in [
                    (axes[0], row.median_relative_drift, row.relative_loss),
                    (axes[1], row.pca_fp_variance_2d, row.drift_energy_captured_2d)]:
                ax.scatter(x, y, color=colors[dataset], marker=markers[condition],
                           s=70, edgecolor='black', linewidth=.4,
                           label=f'{dataset} / {condition}')
    axes[0].set(xlabel='Median ||Q − FP|| / ||FP|| in 768D',
                ylabel='Relative accuracy loss',
                title='Fixed-condition feature drift and accuracy loss')
    axes[1].plot([0, 1], [0, 1], '--', color='#777777', lw=.8, label='Equal fractions')
    axes[1].set(xlabel='FP variance captured by PC1+PC2',
                ylabel='Quantization drift energy captured in 2D',
                title='How much does the PCA picture show?')
    axes[1].set_xlim(0, max(.4, summary.pca_fp_variance_2d.max() * 1.12))
    axes[1].set_ylim(0, min(1, max(.4, summary.drift_energy_captured_2d.max() * 1.12)))
    axes[0].grid(alpha=.2)
    axes[1].grid(alpha=.2)
    handles = [Line2D([0], [0], color=colors[name], marker='o', lw=1,
                      label=name) for name in DATASETS]
    handles += [Line2D([0], [0], color='#555555', marker='o', linestyle='None',
                       label='W3/G128'),
                Line2D([0], [0], color='#555555', marker='^', linestyle='None',
                       label='W4/G128')]
    fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=8,
               bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=(0, .07, 1, 1))
    path = output / 'plots' / 'five_dataset_overview.png'
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def cross_dataset_ranks(summary):
    rows = []
    predictors = ['fp_accuracy', 'fp_separability', 'median_relative_drift',
                  'drift_energy_captured_2d', 'separability_change']
    for condition, frame in summary.groupby('condition'):
        for predictor in predictors:
            paired = frame[[predictor, 'relative_loss']].dropna()
            rho = (float(spearmanr(paired[predictor], paired.relative_loss).statistic)
                   if len(paired) >= 3 and paired[predictor].nunique() > 1 and
                   paired.relative_loss.nunique() > 1 else float('nan'))
            rows.append(dict(condition=condition, predictor=predictor,
                             n_datasets=len(paired), spearman_rho=rho))
    return pd.DataFrame(rows)


def write_report(output, summary, ranks, input_root):
    variance_range = (summary.pca_fp_variance_2d.min(),
                      summary.pca_fp_variance_2d.max())
    capture_range = (summary.drift_energy_captured_2d.min(),
                     summary.drift_energy_captured_2d.max())
    capture_lower_count = int((summary.drift_energy_captured_2d <
                               summary.pca_fp_variance_2d).sum())
    lines = [
        '# Paired final CLS feature PCA：五個 dataset 的固定條件分析', '',
        '## 問題與方法', '',
        '同一張測試影像分別通過 AMP-FP16 reference 與 weight-only HQQ 模型，'
        '取 classifier 前 final CLS feature（768D）。每個 dataset 用自己的 FP feature '
        '所擬合的 PCA 基底，將 FP、W3/G128、W4/G128 投到同一 2D 平面；'
        '五個 dataset 的 PCA 座標軸彼此不同。圖中的 FP、W3、W4 共用同一組 X/Y 範圍。', '',
        '每條箭頭是同一影像的 FP → HQQ 投影位移，長度沒有放大；每種 outcome '
        '最多顯示 25 條，但表格使用完整測試集。C→W 是原本正確、量化後錯誤；'
        'W→C 是被救回。', '',
        '## 完整測試集摘要', '',
        '| Dataset | Condition | FP acc. | Relative loss | Damage / rescue | FP PC1+PC2 | '
        '2D drift energy | Median relative 768D drift | Drift AUC | FP margin AUC |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary.itertuples():
        lines.append(f'| {row.dataset} | {row.condition} | {row.fp_accuracy:.1%} | '
                     f'{row.relative_loss:.1%} | {row.n_damage} / {row.n_rescue} | '
                     f'{row.pca_fp_variance_2d:.1%} | '
                     f'{row.drift_energy_captured_2d:.1%} | '
                     f'{row.median_relative_drift:.3f} | '
                     f'{row.drift_damage_auc:.2f} | {row.fp_margin_damage_auc:.2f} |')
    lines += ['', '![Five-dataset overview](plots/five_dataset_overview.png)', '',
              '`2D drift energy` = 所有影像在 FP PCA 前兩軸的位移平方和，除以完整 '
              '768D 位移平方和。它與 `FP PC1+PC2` 解釋的 FP feature 變異是兩個不同量。'
              'Drift AUC 在 FP 原本正確的影像中，以量化後的相對 feature 位移區分 C→W；'
              'FP margin AUC 使用量化前 normalized logit margin。AUC 是條件內描述值。', '',
              f'目前五個 dataset 的 FP 前兩個 PC 共解釋 **{variance_range[0]:.1%}–'
              f'{variance_range[1]:.1%}** 的 FP feature 變異；十個 dataset × condition '
              f'配對的 2D drift energy 只涵蓋 **{capture_range[0]:.1%}–'
              f'{capture_range[1]:.1%}** 的完整位移。'
              f'在 **{capture_lower_count}/{len(summary)}** 個配對中，'
              '2D 位移涵蓋比例低於 FP feature 變異涵蓋比例。', '',
              '## 各 dataset 的配對 PCA 圖', '']
    for dataset in DATASETS:
        lines.extend([f'### {dataset}', '',
                      f'![{dataset} paired PCA](plots/{dataset}_paired_pca.png)', ''])
    lines += ['## 跨 dataset 的描述性排序關聯', '',
              '| Condition | FP accuracy | FP separation | 768D drift | 2D drift energy | Separation change |',
              '|---|---:|---:|---:|---:|---:|']
    for condition, frame in ranks.groupby('condition'):
        lookup = frame.set_index('predictor').spearman_rho
        lines.append(f'| {condition} | {lookup["fp_accuracy"]:+.2f} | '
                     f'{lookup["fp_separability"]:+.2f} | '
                     f'{lookup["median_relative_drift"]:+.2f} | '
                     f'{lookup["drift_energy_captured_2d"]:+.2f} | '
                     f'{lookup["separability_change"]:+.2f} |')
    lines += ['', '每一欄的 Spearman ρ 僅比較五個 dataset–checkpoint 配對；'
              'W3 與 W4 重複使用相同的五個配對，不能當成十個獨立樣本。'
              '完整 768D feature 位移是量化後才觀察到的 response，不能當成量化前預測因子。', '',
              '## 解讀界線', '',
              '- PCA 尋找 FP feature 最大變異方向，不保證涵蓋分類決策方向。'
              '小的 2D 位移可能伴隨 768D 中重要的決策變化。',
              '- 各 dataset 有自己的類別、checkpoint 與單一訓練 seed；FP baseline '
              '也不同。跨 dataset 關聯不能識別 dataset 單獨造成的效果。',
              '- 這裡的 FP reference 是 AMP-FP16，不能與 15-dataset legacy '
              'FP32 baseline 直接合併。',
              '- 類別分離度是 normalized feature 的 class-equal cosine distance gap，'
              '不是 classifier 的 logit margin。', '',
              '## 重現與來源', '',
              '```powershell',
              'conda run -n opencode_env python -m five_dataset_features.paired_feature_pca',
              'conda run -n opencode_env python -m five_dataset_features.verify_paired_feature_pca',
              '```', '',
              f'原始資料：`{input_root}`。每個選定條件的 manifest、feature、'
              'prediction SHA-256，以及程式與輸出雜湊都記錄在 `run_config.json`。'
              '`samples.csv` 保留逐樣本配對結果；`summary.csv` 與 `cross_dataset_spearman.csv` '
              '保留彙總值。', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=MODULE / 'data')
    parser.add_argument('--analysis-root', type=Path, default=MODULE / 'analysis')
    parser.add_argument('--output-dir', type=Path,
                        default=MODULE / 'analysis/paired_feature_pca')
    args = parser.parse_args()
    input_root, analysis_root, output = (args.input_root.resolve(),
                                         args.analysis_root.resolve(),
                                         args.output_dir.resolve())
    output.mkdir(parents=True, exist_ok=True)
    (output / 'plots').mkdir(exist_ok=True)
    summaries, sample_tables, source_hashes = [], [], {}
    for dataset in DATASETS:
        meta, arrays, predictions, mean, components, explained, hashes = check_sources(
            input_root, analysis_root, dataset)
        source_hashes[dataset] = hashes
        fp = np.asarray(arrays['fp'][:, 12, :], dtype=np.float64)
        fp_geometry = class_equal_cosine_distances(fp, meta['targets'])
        coords = {'fp': (fp - mean) @ components}
        outcomes = {}
        for condition in QUANTIZED:
            q = np.asarray(arrays[condition][:, 12, :], dtype=np.float64)
            coords[condition] = (q - mean) @ components
            details, paired = paired_metrics(fp, q, components,
                                             predictions['fp'], predictions[condition])
            outcomes[condition] = details['outcome']
            accuracy = prediction_metrics(predictions['fp'], predictions[condition])
            q_geometry = class_equal_cosine_distances(q, meta['targets'])
            summaries.append(dict(dataset=dataset, condition=condition,
                                  pca_fp_variance_2d=float(explained.sum()),
                                  fp_separability=fp_geometry['separability'],
                                  q_separability=q_geometry['separability'],
                                  separability_change=(q_geometry['separability'] - fp_geometry['separability']
                                                       if fp_geometry['separability'] is not None and
                                                       q_geometry['separability'] is not None else float('nan')),
                                  **accuracy, **paired))
            sample_tables.append(pd.DataFrame({
                'dataset': dataset, 'condition': condition,
                'sample_id': meta['sample_ids'], 'target': meta['targets'],
                'fp_correct': details['fp_correct'], 'q_correct': details['q_correct'],
                'outcome': details['outcome'],
                'fp_normalized_margin': details['margin'],
                'full_drift': details['drift'],
                'relative_full_drift': details['relative_drift'],
                'projected_2d_drift': details['drift_2d'],
                'fp_pc1': coords['fp'][:, 0], 'fp_pc2': coords['fp'][:, 1],
                'q_pc1': coords[condition][:, 0], 'q_pc2': coords[condition][:, 1],
            }))
        make_dataset_plot(output, dataset, meta, coords, outcomes, explained)
        print(f'ANALYZED {dataset}: {len(fp)} paired images x 2 conditions', flush=True)

    summary = pd.DataFrame(summaries)
    samples = pd.concat(sample_tables, ignore_index=True)
    ranks = cross_dataset_ranks(summary)
    summary.to_csv(output / 'summary.csv', index=False)
    samples.to_csv(output / 'samples.csv', index=False)
    ranks.to_csv(output / 'cross_dataset_spearman.csv', index=False)
    make_overview(output, summary)
    write_report(output, summary, ranks, input_root)
    config = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  input_root=str(input_root), analysis_root=str(analysis_root),
                  output_dir=str(output), datasets=list(DATASETS),
                  conditions=list(CONDITIONS), final_feature_index=12,
                  fp_reference='AMP-FP16', inference='descriptive_five_dataset_checkpoint_pairs',
                  source_sha256=source_hashes,
                  code_sha256={name: sha256(MODULE / name) for name in
                               [*ANALYSIS_CODE, 'paired_feature_pca.py']},
                  output_sha256={path.relative_to(output).as_posix(): sha256(path)
                                 for path in sorted(output.rglob('*')) if path.is_file() and
                                 path.name not in ('run_config.json', 'verification.json')})
    (output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Saved {len(summary)} dataset-condition summaries and {len(samples)} '
          f'paired sample rows: {output / "REPORT.md"}')


if __name__ == '__main__':
    main()
