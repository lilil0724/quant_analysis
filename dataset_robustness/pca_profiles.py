"""Exploratory PCA of matched dataset quantization-loss profiles.

Run from the repository root:
    conda run -n opencode_env python -m dataset_robustness.pca_profiles

Each fit has one row per dataset and one column per bit/group condition. Fits
stay separate by model and checkpoint kind. PCA is descriptive, not a test of a
dataset-only causal effect.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from dataset_robustness.analyze import prepare_cells, short_model


ROOT = Path(__file__).resolve().parent
REFERENCE = ('vit_b16', 'ft')
CONTEXT = ['model_name', 'ckpt_kind']
SETTING = ['hqq_nbits', 'hqq_group_size']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_source(main_results):
    """Rebuild cells from the exact source and analyzer used for the main run."""
    config = json.loads((main_results / 'run_config.json').read_text(encoding='utf-8'))
    source = Path(config['input_file'])
    if sha256(source) != config['input_sha256']:
        raise ValueError('Main-run source hash changed; rerun the main analysis first.')
    analyzer = ROOT / 'analyze.py'
    if sha256(analyzer) != config['script_sha256']:
        raise ValueError('Main-run analyzer hash changed; rerun the main analysis first.')
    cells, excluded = prepare_cells(pd.read_csv(source))
    if len(cells) != config['n_valid_cells'] or len(excluded) != config['n_excluded_cells']:
        raise ValueError('Rebuilt cell counts differ from the main-run manifest.')
    return cells, excluded, config, source


def complete_matrix(frame, bits, groups, metric):
    """Use a fixed condition grid; return omissions for a coverage audit."""
    columns = pd.MultiIndex.from_product([bits, groups], names=SETTING)
    value = (1.0 - frame['accuracy_ratio'] if metric == 'relative_loss'
             else frame['accuracy_drop_points'])
    wide = frame.assign(pca_value=value).pivot(
        index='dataset_name', columns=SETTING, values='pca_value')
    wide = wide.reindex(columns=columns).sort_index()
    included = wide.notna().all(axis=1)
    coverage = pd.DataFrame({
        'dataset_name': wide.index,
        'observed_conditions': wide.notna().sum(axis=1).to_numpy(),
        'expected_conditions': len(columns),
        'included': included.to_numpy(),
    })
    return wide.loc[included], coverage


def centered_pca(wide):
    """SVD PCA with a deterministic sign convention for displayed axes."""
    if wide.shape[0] < 3 or wide.shape[1] < 2:
        raise ValueError('PCA needs at least 3 complete datasets and 2 conditions.')
    matrix = wide.to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    total = np.square(singular).sum()
    if total <= 0:
        raise ValueError('All dataset profiles are identical after centering.')
    scores = u[:, :2] * singular[:2]
    loadings = vt[:2].T.copy()
    for component in range(2):
        anchor = np.argmax(np.abs(loadings[:, component]))
        if loadings[anchor, component] < 0:
            scores[:, component] *= -1
            loadings[:, component] *= -1
    explained = np.square(singular) / total
    return scores, loadings, explained


def fit_all(cells, primary_bits, groups):
    scopes = {'primary': primary_bits,
              'without_2bit': tuple(bit for bit in primary_bits if bit != 2)}
    summaries, scores_rows, loading_rows, variance_rows, coverage_rows = [], [], [], [], []
    matrices = {}
    for (model, kind), frame in cells.groupby(CONTEXT, sort=True):
        baseline = frame.groupby('dataset_name').fp32_top1.first()
        for scope, bits in scopes.items():
            for metric in ('relative_loss', 'absolute_drop_points'):
                wide, coverage = complete_matrix(frame, bits, groups, metric)
                identity = dict(model_name=model, ckpt_kind=kind, scope=scope, metric=metric)
                coverage_rows.append(coverage.assign(**identity))
                if wide.shape[0] < 3:
                    raise ValueError(f'Insufficient complete datasets for {identity}.')
                score, loading, explained = centered_pca(wide)
                matrices[(model, kind, scope, metric)] = wide
                summaries.append(dict(**identity, n_datasets=len(wide), n_conditions=wide.shape[1],
                                      pc1_variance_fraction=explained[0],
                                      pc2_variance_fraction=explained[1],
                                      pc1_pc2_variance_fraction=explained[:2].sum(),
                                      baseline_min=baseline.loc[wide.index].min(),
                                      baseline_max=baseline.loc[wide.index].max()))
                for dataset, (pc1, pc2) in zip(wide.index, score):
                    scores_rows.append(dict(**identity, dataset_name=dataset,
                                            pc1=pc1, pc2=pc2,
                                            fp32_top1=baseline.loc[dataset],
                                            mean_condition_loss=wide.loc[dataset].mean()))
                for (bit, group), (pc1, pc2) in zip(wide.columns, loading):
                    loading_rows.append(dict(**identity, hqq_nbits=bit, hqq_group_size=group,
                                             pc1_loading=pc1, pc2_loading=pc2))
                for number, fraction in enumerate(explained, start=1):
                    variance_rows.append(dict(**identity, component=number,
                                              variance_fraction=fraction,
                                              cumulative_fraction=explained[:number].sum()))
    return (pd.DataFrame(summaries), pd.DataFrame(scores_rows),
            pd.DataFrame(loading_rows), pd.DataFrame(variance_rows),
            pd.concat(coverage_rows, ignore_index=True), matrices)


def distance_sensitivity(matrices):
    """Compare full-profile dataset-distance order without aligning PCA axes."""
    rows = []
    for model, kind in sorted({key[:2] for key in matrices}):
        reference = matrices[(model, kind, 'primary', 'relative_loss')]
        for scope, metric in [('primary', 'absolute_drop_points'),
                              ('without_2bit', 'relative_loss')]:
            alternative = matrices[(model, kind, scope, metric)]
            common = reference.index.intersection(alternative.index)
            if len(common) < 3:
                rho = np.nan
            else:
                a = pdist(reference.loc[common].to_numpy())
                b = pdist(alternative.loc[common].to_numpy())
                rho = (float(spearmanr(a, b).statistic)
                       if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan)
            rows.append(dict(model_name=model, ckpt_kind=kind, variant_scope=scope,
                             variant_metric=metric, n_datasets=len(common),
                             n_dataset_pairs=len(common) * (len(common) - 1) // 2,
                             pair_distance_spearman_rho=rho))
    return pd.DataFrame(rows)


def save_plots(output, summary, scores, loadings, variance):
    plot_dir = output / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    manifest = []

    def save(fig, name, source, description):
        path = plot_dir / name
        fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        manifest.append(dict(file=f'plots/{name}', source_table=source,
                             description=description))

    def score_plot(model, kind, scope, metric, name):
        part = scores[(scores.model_name == model) & (scores.ckpt_kind == kind) &
                      (scores.scope == scope) & (scores.metric == metric)]
        details = summary[(summary.model_name == model) & (summary.ckpt_kind == kind) &
                          (summary.scope == scope) & (summary.metric == metric)].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 7))
        dots = ax.scatter(part.pc1, part.pc2, c=part.fp32_top1, cmap='viridis',
                          s=75, edgecolors='black', linewidths=.4)
        ax.axhline(0, color='#aaaaaa', lw=.7)
        ax.axvline(0, color='#aaaaaa', lw=.7)
        metric_label = ('Relative accuracy loss' if metric == 'relative_loss'
                        else 'Accuracy drop (percentage points)')
        scope_label = ('2/3/4-bit' if scope == 'primary' else '3/4-bit')
        ax.set(xlabel=f'PC1 ({details.pc1_variance_fraction:.1%} variance)',
               ylabel=f'PC2 ({details.pc2_variance_fraction:.1%} variance)',
               title=f'{short_model(model)} / {kind} | {metric_label} | {scope_label}\n'
                     f'{details.n_datasets} datasets, {details.n_conditions} quantization conditions')
        fig.colorbar(dots, ax=ax, label='FP32 top-1 accuracy (%)')
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed = []
        offsets = [(6, 4), (6, -8), (-6, 4), (-6, -8),
                   (10, 17), (10, -19), (-10, 17), (-10, -19),
                   (22, 27), (22, -29), (-22, 27), (-22, -29),
                   (30, 40), (-30, 40), (30, -42), (-30, -42)]
        for row in part.itertuples():
            for dx, dy in offsets:
                annotation = ax.annotate(
                    row.dataset_name, (row.pc1, row.pc2), xytext=(dx, dy),
                    textcoords='offset points', fontsize=8,
                    ha='left' if dx > 0 else 'right',
                    va='bottom' if dy > 0 else 'top',
                    arrowprops=(dict(arrowstyle='-', color='#999999', lw=.5)
                                if abs(dx) > 10 or abs(dy) > 19 else None))
                box = annotation.get_window_extent(renderer).expanded(1.03, 1.14)
                inside = (box.x0 >= ax.bbox.x0 and box.x1 <= ax.bbox.x1 and
                          box.y0 >= ax.bbox.y0 and box.y1 <= ax.bbox.y1)
                if inside and not any(box.overlaps(other) for other in placed):
                    placed.append(box)
                    break
                annotation.remove()
            else:
                ax.annotate(row.dataset_name, (row.pc1, row.pc2), xytext=(6, 4),
                            textcoords='offset points', fontsize=8)
        save(fig, name, 'scores.csv', 'Dataset PCA scores; color encodes matched FP32 baseline')

    model, kind = REFERENCE
    for scope, metric, name in [
            ('primary', 'relative_loss', 'vit_ft_primary_relative_loss.png'),
            ('primary', 'absolute_drop_points', 'vit_ft_primary_absolute_drop.png'),
            ('without_2bit', 'relative_loss', 'vit_ft_without_2bit_relative_loss.png')]:
        score_plot(model, kind, scope, metric, name)

    chosen = loadings[(loadings.model_name == model) & (loadings.ckpt_kind == kind) &
                      (loadings.scope == 'primary') & (loadings.metric == 'relative_loss')]
    labels = [f'{bit:g}b / g{group:g}' for bit, group in
              chosen[['hqq_nbits', 'hqq_group_size']].itertuples(index=False, name=None)]
    values = chosen[['pc1_loading', 'pc2_loading']].to_numpy().T
    fig, ax = plt.subplots(figsize=(11, 3.7))
    scale = np.abs(values).max()
    image = ax.imshow(values, cmap='RdBu_r', vmin=-scale, vmax=scale, aspect='auto')
    ax.set(yticks=[0, 1], yticklabels=['PC1', 'PC2'], xticks=np.arange(len(labels)),
           xticklabels=labels, title='ViT / ft: condition loadings for relative loss')
    ax.tick_params(axis='x', rotation=55)
    for row in range(2):
        for col in range(len(labels)):
            ax.text(col, row, f'{values[row, col]:+.2f}', ha='center', va='center', fontsize=7)
    fig.colorbar(image, ax=ax, label='PCA loading')
    fig.tight_layout()
    save(fig, 'vit_ft_primary_relative_loss_loadings.png', 'loadings.csv',
         'PC1 and PC2 weights by bit width and group size')

    chosen = variance[(variance.model_name == model) & (variance.ckpt_kind == kind) &
                      (variance.scope == 'primary') & (variance.metric == 'relative_loss')]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(chosen.component, 100 * chosen.variance_fraction, color='#4b806a')
    ax.plot(chosen.component, 100 * chosen.cumulative_fraction, 'o-', color='#314f77',
            label='Cumulative')
    ax.set(xlabel='Principal component', ylabel='Variance explained (%)',
           title='ViT / ft: relative-loss PCA variance', xticks=chosen.component)
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    save(fig, 'vit_ft_primary_relative_loss_scree.png', 'variance.csv',
         'Per-component and cumulative explained variance')
    return pd.DataFrame(manifest)


def write_report(output, summary, scores, loadings, sensitivity, source, config):
    reference = summary[(summary.model_name == REFERENCE[0]) &
                        (summary.ckpt_kind == REFERENCE[1])]
    primary = reference[(reference.scope == 'primary') &
                        (reference.metric == 'relative_loss')].iloc[0]
    ref_scores = scores[(scores.model_name == REFERENCE[0]) &
                        (scores.ckpt_kind == REFERENCE[1]) &
                        (scores.scope == 'primary') &
                        (scores.metric == 'relative_loss')]
    baseline_rho = float(spearmanr(ref_scores.fp32_top1, ref_scores.pc1).statistic)
    ref_loadings = loadings[(loadings.model_name == REFERENCE[0]) &
                            (loadings.ckpt_kind == REFERENCE[1]) &
                            (loadings.scope == 'primary') &
                            (loadings.metric == 'relative_loss')]
    top_pc1 = ref_loadings.loc[ref_loadings.pc1_loading.abs().idxmax()]
    top_pc2 = ref_loadings.loc[ref_loadings.pc2_loading.abs().idxmax()]
    ref_sensitivity = sensitivity[(sensitivity.model_name == REFERENCE[0]) &
                                  (sensitivity.ckpt_kind == REFERENCE[1])]
    ref_drop_rho = ref_sensitivity[
        ref_sensitivity.variant_metric == 'absolute_drop_points'].iloc[0].pair_distance_spearman_rho
    ref_no2_rho = ref_sensitivity[
        ref_sensitivity.variant_scope == 'without_2bit'].iloc[0].pair_distance_spearman_rho
    rows = [
        '# Dataset quantization robustness：PCA 第一階段', '',
        '## 問題與分析單位', '',
        '在固定 model 與 checkpoint kind 的背景內，每列是一個 dataset-specific checkpoint，'
        '每欄是一組 bit width × group size。不同量化條件是同一 checkpoint 的重複量測，'
        '不是獨立訓練 seed。這個 PCA 描述退化曲線的差異，不能識別 dataset 單獨造成的效果。', '',
        '主要數值是 `1 - accuracy_ratio`（relative loss）。Ratio 大於 1 時保留負 loss；'
        '`accuracy_drop_points = fp32_top1 - hqq_top1` 另行分析。沒有截斷或補值。'
        'PCA 對每個條件欄置中，但不除以其標準差，因此變異較大的條件可能主導 PC。', '',
        '## ViT / ft 主要結果', '',
        f'主要矩陣有 **{int(primary.n_datasets)} datasets × '
        f'{int(primary.n_conditions)} conditions**。PC1 解釋 '
        f'**{primary.pc1_variance_fraction:.1%}**、PC2 解釋 '
        f'**{primary.pc2_variance_fraction:.1%}** 的 dataset 間曲線變異，'
        f'合計 **{primary.pc1_pc2_variance_fraction:.1%}**。'
        f'PC1 與 FP32 baseline 的 Spearman ρ = **{baseline_rho:.2f}**（15 個 dataset 的描述值），'
        '因此 PC1 上的分離不能直接解釋成 dataset 固有的量化耐受度。', '',
        f'PC1 絕對值最大的 loading 是 **{top_pc1.hqq_nbits:g}-bit / '
        f'group {top_pc1.hqq_group_size:g}**（{top_pc1.pc1_loading:+.2f}）；'
        f'PC2 是 **{top_pc2.hqq_nbits:g}-bit / group '
        f'{top_pc2.hqq_group_size:g}**（{top_pc2.pc2_loading:+.2f}）。'
        '解讀 PC 時須同看 loadings 與原始退化曲線；PC 座標軸的正負方向沒有固有意義。', '',
        '![ViT / ft primary PCA](plots/vit_ft_primary_relative_loss.png)', '',
        '![ViT / ft PCA loadings](plots/vit_ft_primary_relative_loss_loadings.png)', '',
        '![ViT / ft PCA explained variance](plots/vit_ft_primary_relative_loss_scree.png)', '',
        '## 指標、bit 範圍與模型背景敏感度', '',
        'Absolute drop、僅 3/4-bit 的 relative loss，以及其他八個 model / checkpoint kind '
        '各自重新擬合 PCA。不同 fit 的 PC 座標軸不能直接疊在一起比較。'
        '`distance_sensitivity.csv` 比較完整條件曲線中所有 dataset pair 距離的排序；'
        'dataset pair 彼此相依，這些 rho 只是描述值，沒有 p-value。'
        f'ViT / ft 的 ratio 對 absolute drop 距離排序 ρ = **{ref_drop_rho:.2f}**；'
        f'去掉 2-bit 後 ρ = **{ref_no2_rho:.2f}**。', '',
        '| Model / checkpoint | Relative loss PC1+PC2 | Absolute drop PC1+PC2 | 3/4-bit PC1+PC2 | '
        'Distance rho: drop | Distance rho: 3/4-bit |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for (model, kind), group in summary.groupby(CONTEXT, sort=True):
        def evr(scope, metric):
            return group[(group.scope == scope) & (group.metric == metric)].iloc[0].pc1_pc2_variance_fraction
        sg = sensitivity[(sensitivity.model_name == model) & (sensitivity.ckpt_kind == kind)]
        drop = sg[(sg.variant_scope == 'primary') &
                  (sg.variant_metric == 'absolute_drop_points')].iloc[0].pair_distance_spearman_rho
        no2 = sg[(sg.variant_scope == 'without_2bit') &
                 (sg.variant_metric == 'relative_loss')].iloc[0].pair_distance_spearman_rho
        rows.append(f'| {short_model(model)} / {kind} | '
                    f'{evr("primary", "relative_loss"):.1%} | '
                    f'{evr("primary", "absolute_drop_points"):.1%} | '
                    f'{evr("without_2bit", "relative_loss"):.1%} | '
                    f'{drop:.2f} | {no2:.2f} |')
    rows += ['', '![ViT / ft absolute-drop PCA](plots/vit_ft_primary_absolute_drop.png)', '',
             '![ViT / ft 3/4-bit PCA](plots/vit_ft_without_2bit_relative_loss.png)', '',
             '## 解讀界線', '',
             '- Dataset、訓練結果與 dataset-specific checkpoint 仍混雜；每個 context '
             '只有一個訓練 seed。',
             '- 低 FP32 baseline 會使 relative loss 對小幅 absolute change 敏感。'
             '須一起查看 `fp32_top1`、absolute-drop PCA 與主分析的 baseline audit。',
             '- PCA 最大化變異，不是 robustness 預測，也不會自動分離預先定義的 dataset 類別。'
             '圖中的群聚僅供探索。', '',
             '## 重現方式', '',
             '在 repository 根目錄執行：', '',
             '```powershell',
             'conda run -n opencode_env python -m dataset_robustness.pca_profiles',
             'conda run -n opencode_env python -m dataset_robustness.verify_pca',
             '```', '',
             f'來源：`{source}`。輸入 SHA-256：`{config["input_sha256"]}`。'
             '`run_config.json` 記錄程式與輸出雜湊；`coverage.csv` 保留完整性稽核；'
             '`scores.csv`、`loadings.csv`、`variance.csv` 保留數值。', '']
    (output / 'report.md').write_text('\n'.join(rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-results', type=Path, default=ROOT / 'results/main')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/pca_phase1')
    args = parser.parse_args()
    main_results = args.main_results.resolve()
    output = args.output_dir.resolve()
    cells, excluded, source_config, source = load_verified_source(main_results)
    bits = tuple(float(value) for value in source_config['primary_bits'])
    groups = tuple(int(value) for value in source_config['group_sizes'])
    if 2 not in bits or len(bits) < 3:
        raise ValueError('The main run must include 2-bit plus at least two other primary bits.')
    summary, scores, loadings, variance, coverage, matrices = fit_all(cells, bits, groups)
    if REFERENCE not in {key[:2] for key in matrices}:
        raise ValueError('Reference ViT / ft context is absent.')
    sensitivity = distance_sensitivity(matrices)
    output.mkdir(parents=True, exist_ok=True)
    tables = dict(pca_summary=summary, scores=scores, loadings=loadings,
                  variance=variance, coverage=coverage,
                  distance_sensitivity=sensitivity)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    manifest = save_plots(output, summary, scores, loadings, variance)
    manifest.to_csv(output / 'plots_manifest.csv', index=False)
    write_report(output, summary, scores, loadings, sensitivity, source, source_config)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  main_results=str(main_results), main_run_config_sha256=sha256(main_results / 'run_config.json'),
                  source=str(source), source_sha256=sha256(source),
                  main_analyzer_sha256=sha256(ROOT / 'analyze.py'),
                  pca_script_sha256=sha256(Path(__file__)),
                  primary_bits=bits, group_sizes=groups,
                  n_source_cells=len(cells), n_source_exclusions=len(excluded),
                  n_contexts=len({key[:2] for key in matrices}), n_fits=len(summary),
                  reference_context={'model_name': REFERENCE[0], 'ckpt_kind': REFERENCE[1]},
                  preprocessing='column_centered_no_variance_scaling_no_imputation',
                  inference='descriptive_only_no_p_values_no_seed_CIs',
                  output_sha256={path.name: sha256(path) for path in sorted(output.glob('*.csv'))},
                  report_sha256=sha256(output / 'report.md'),
                  figure_sha256={path.name: sha256(path) for path in sorted((output / 'plots').glob('*.png'))})
    (output / 'run_config.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Fitted {len(summary)} PCA panels across {result["n_contexts"]} contexts. '
          f'Report: {output / "report.md"}')


if __name__ == '__main__':
    main()
