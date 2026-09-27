"""Compare dataset accuracy retention without assuming a universal ranking.

Run with: python -m dataset_robustness.analyze --help
The legacy summary owns checkpoint/seed pairing. This module never pools
different checkpoint paths or treats quantization settings as replicates.
"""

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ['model_name', 'ckpt_kind']
SETTING = ['hqq_nbits', 'hqq_group_size']
KEY = CONTEXT + ['dataset_name'] + SETTING
TOL = 1e-12  # Numerical equality only, not a practical equivalence margin.


def prepare_cells(summary):
    """Retain audit exclusions and reject ambiguous checkpoint aggregation."""
    required = KEY + ['ckpt_path', 'baseline_matched', 'n_seeds',
                      'matched_seed_count', 'fp32_top1', 'hqq_top1', 'accuracy_ratio']
    missing = set(required) - set(summary.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    data = summary.copy()
    for col in SETTING + ['n_seeds', 'matched_seed_count', 'fp32_top1',
                          'hqq_top1', 'accuracy_ratio']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    reasons = pd.Series('', index=data.index)

    def flag(mask, reason):
        reasons.loc[mask] += reason + ';'

    flag(~data.baseline_matched.astype(str).str.lower().isin(['true', '1']),
         'unmatched_baseline')
    flag(data.n_seeds.isna() | (data.n_seeds < 1) |
         (data.n_seeds != data.matched_seed_count), 'invalid_seed_coverage')
    numeric = SETTING + ['fp32_top1', 'hqq_top1', 'accuracy_ratio']
    flag(~np.isfinite(data[numeric]).all(axis=1), 'nonfinite_value')
    flag((data.fp32_top1 <= 0) | (data.fp32_top1 > 100) |
         (data.hqq_top1 < 0) | (data.hqq_top1 > 100), 'invalid_accuracy')
    flag((data.hqq_nbits <= 0) | (data.hqq_group_size <= 0), 'invalid_setting')
    flag(data[CONTEXT + ['dataset_name', 'ckpt_path']].isna().any(axis=1),
         'missing_identity')
    data['exclusion_reason'] = reasons.str.rstrip(';')
    excluded = data.loc[reasons.ne('')].copy()
    cells = data.loc[reasons.eq('')].drop(columns='exclusion_reason').copy()
    if cells.empty:
        raise ValueError('No valid matched cells remain.')
    if cells.duplicated(KEY).any():
        raise ValueError('Multiple rows/checkpoints in a dataset/model/kind/setting; '
                         'select an explicit checkpoint cohort rather than pooling.')
    identities = cells.groupby(CONTEXT + ['dataset_name']).ckpt_path.nunique()
    if (identities != 1).any():
        raise ValueError('Checkpoint identity changes across quantization settings.')
    for _, group in cells.groupby(CONTEXT + ['dataset_name']):
        if group.fp32_top1.max() - group.fp32_top1.min() > 1e-8:
            raise ValueError('Baseline accuracy changes across settings in a context.')
    # Mean(seed-wise ratios) need not equal ratio(mean accuracies) with >1 seed.
    single = cells.n_seeds.eq(1)
    if not np.allclose(cells.loc[single, 'accuracy_ratio'],
                       cells.loc[single, 'hqq_top1'] / cells.loc[single, 'fp32_top1'],
                       rtol=1e-7, atol=1e-10):
        raise ValueError('Single-seed ratio is inconsistent with its accuracies.')
    cells['accuracy_drop_points'] = cells.fp32_top1 - cells.hqq_top1
    cells['condition'] = [f'b{b:g}_g{g:g}' for b, g in cells[SETTING].to_numpy()]
    cells['context'] = cells.model_name + ' | ' + cells.ckpt_kind
    cells['ratio_rank'] = cells.groupby(CONTEXT + SETTING).accuracy_ratio.rank(
        method='average', ascending=False)
    cells['drop_rank'] = cells.groupby(CONTEXT + SETTING).accuracy_drop_points.rank(
        method='average', ascending=True)
    return cells.sort_values(KEY), excluded


def rank_agreement(left, right):
    """Spearman coefficient only; no unearned replicate-based p-values."""
    paired = pd.concat([left.rename('left'), right.rename('right')], axis=1).dropna()
    n = len(paired)
    if n < 3:
        return n, np.nan, 'fewer_than_3_datasets'
    if paired.left.nunique() < 2 or paired.right.nunique() < 2:
        return n, np.nan, 'constant_ranking'
    rho = paired.left.rank().corr(paired.right.rank())
    return n, float(rho), 'ok'


def common_panel(frame, bits, groups):
    """Fixed complete dataset cohort for all comparisons inside one scope."""
    columns = pd.MultiIndex.from_product([bits, groups], names=SETTING)
    wide = frame.pivot(index='dataset_name', columns=SETTING, values='accuracy_ratio')
    wide = wide.reindex(columns=columns)
    return wide.dropna(axis=0), wide


def analyze(cells, primary_bits=(2., 3., 4.)):
    groups = sorted(cells.hqq_group_size.unique())
    scopes = {'primary': list(primary_bits), 'without_2bit': [3., 4.],
              'full_grid': sorted(cells.hqq_nbits.unique())}
    profiles, coverage, within, orderings = [], [], [], []
    for scope, bits in scopes.items():
        for context, frame in cells.groupby(CONTEXT, sort=True):
            identity = dict(zip(CONTEXT, context))
            complete, wide = common_panel(frame, bits, groups)
            for dataset in wide.index:
                coverage.append(dict(scope=scope, **identity, dataset_name=dataset,
                                     expected_conditions=len(wide.columns),
                                     observed_conditions=int(wide.loc[dataset].notna().sum()),
                                     included=dataset in complete.index))
            if complete.empty:
                continue
            ranks = complete.rank(axis=0, ascending=False, method='average')
            baseline = frame.groupby('dataset_name').fp32_top1.first()
            selected = frame[frame.hqq_nbits.isin(bits)].set_index('dataset_name')
            for dataset in complete.index:
                rr, vv = ranks.loc[dataset], complete.loc[dataset]
                drops = selected.loc[[dataset], 'accuracy_drop_points']
                profiles.append(dict(scope=scope, **identity, dataset_name=dataset,
                                     n_datasets=len(complete), n_conditions=len(complete.columns),
                                     baseline_accuracy=baseline.loc[dataset],
                                     ratio_median=vv.median(), ratio_min=vv.min(), ratio_max=vv.max(),
                                     drop_median_points=drops.median(),
                                     rank_median=rr.median(), rank_best=rr.min(), rank_worst=rr.max(),
                                     rank_q25=rr.quantile(.25), rank_q75=rr.quantile(.75)))
            for a, b in itertools.combinations(complete.columns, 2):
                n, rho, status = rank_agreement(complete[a], complete[b])
                within.append(dict(scope=scope, **identity, bit_a=a[0], group_a=a[1],
                                   bit_b=b[0], group_b=b[1], n_datasets=n, rho=rho, status=status,
                                   rank_shift_median=(ranks[a] - ranks[b]).abs().median()))
            for a, b in itertools.combinations(complete.index, 2):
                delta = complete.loc[a] - complete.loc[b]
                wins, losses = int((delta > TOL).sum()), int((delta < -TOL).sum())
                orderings.append(dict(scope=scope, **identity, dataset_a=a, dataset_b=b,
                                      a_higher=wins, b_higher=losses,
                                      ties=int(len(delta) - wins - losses),
                                      n_conditions=len(delta), order_reverses=bool(wins and losses),
                                      median_ratio_difference=delta.median(),
                                      min_ratio_difference=delta.min(), max_ratio_difference=delta.max(),
                                      median_absolute_difference=delta.abs().median()))
    cross, metrics = [], []
    contexts = sorted(cells.context.unique())
    for setting, frame in cells.groupby(SETTING, sort=True):
        settings = dict(zip(SETTING, setting))
        panel = frame.pivot(index='dataset_name', columns='context', values='accuracy_ratio')
        panel = panel.reindex(columns=contexts).dropna()
        for a, b in itertools.combinations(contexts, 2):
            n, rho, status = rank_agreement(panel[a], panel[b])
            cross.append(dict(**settings, context_a=a, context_b=b, n_datasets=n,
                              rho=rho, status=status, primary=setting[0] in primary_bits,
                              datasets=';'.join(panel.index)))
        for context, group in frame.groupby(CONTEXT):
            n, rho, status = rank_agreement(group.accuracy_ratio, -group.accuracy_drop_points)
            metrics.append(dict(**settings, **dict(zip(CONTEXT, context)), n_datasets=n,
                                ratio_vs_drop_rho=rho, status=status,
                                baseline_min=group.fp32_top1.min(),
                                baseline_max=group.fp32_top1.max()))
    tables = {'dataset_profiles': pd.DataFrame(profiles), 'coverage': pd.DataFrame(coverage),
              'within_context_agreement': pd.DataFrame(within),
              'pairwise_order_stability': pd.DataFrame(orderings),
              'cross_context_agreement': pd.DataFrame(cross),
              'metric_agreement': pd.DataFrame(metrics)}
    if tables['dataset_profiles'].empty:
        raise ValueError('No complete cohorts available in any requested scope.')
    # Baseline context counts are not multiplied by the number of HQQ settings.
    tables['baseline_audit'] = cells.groupby(CONTEXT + ['dataset_name', 'ckpt_path'], as_index=False).agg(
        baseline_accuracy=('fp32_top1', 'first'), min_seeds=('n_seeds', 'min'),
        max_seeds=('n_seeds', 'max'), conditions=('condition', 'size'))
    p = tables['dataset_profiles']
    comparison = p.pivot(index=CONTEXT + ['dataset_name'], columns='scope',
                         values=['rank_median', 'ratio_median', 'n_datasets', 'n_conditions'])
    comparison.columns = ['__'.join(c) for c in comparison.columns]
    tables['scope_sensitivity'] = comparison.reset_index()
    tables['condition_overview'] = cells.groupby(SETTING, as_index=False).agg(
        n_cells=('accuracy_ratio', 'size'), ratio_median=('accuracy_ratio', 'median'),
        ratio_min=('accuracy_ratio', 'min'), ratio_max=('accuracy_ratio', 'max'),
        drop_median_points=('accuracy_drop_points', 'median'))
    # Dataset-specific portability; these are rank ranges, not an overall ranking.
    portable = []
    for dataset, frame in p[p.scope.eq('primary')].groupby('dataset_name'):
        portable.append(dict(dataset_name=dataset, n_contexts=len(frame),
                             best_context_median_rank=frame.rank_median.min(),
                             worst_context_median_rank=frame.rank_median.max(),
                             context_median_rank_span=frame.rank_median.max() - frame.rank_median.min()))
    tables['dataset_portability'] = pd.DataFrame(portable)
    return tables


def short_model(value):
    return value.split('_')[0].replace('beitv2', 'BEiTv2').replace('vit', 'ViT').replace('swin', 'Swin')


def make_plots(cells, tables, output, primary_bits):
    plot_dir = output / 'plots'
    plot_dir.mkdir(exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=.85)
    datasets = sorted(cells.dataset_name.unique())
    colors = dict(zip(datasets, sns.color_palette('tab20', len(datasets))))
    manifest = []

    def save(fig, name, source, description):
        path = plot_dir / name
        fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        manifest.append(dict(file=f'plots/{name}', source_table=source, description=description))

    groups = sorted(cells.hqq_group_size.unique())
    for context, frame in cells.groupby(CONTEXT):
        model, kind = context
        title = f'{short_model(model)} / {kind}'
        slug = f'{short_model(model).lower()}_{kind}'
        for scope, bits in [('primary', primary_bits), ('full_grid', sorted(cells.hqq_nbits.unique()))]:
            fig, axes = plt.subplots(1, len(groups), figsize=(16, 4.8), squeeze=False, sharey=True)
            selected = frame[frame.hqq_nbits.isin(bits)]
            for axis, group_size in zip(axes[0], groups):
                for dataset, line in selected[selected.hqq_group_size.eq(group_size)].groupby('dataset_name'):
                    line = line.sort_values('hqq_nbits')
                    axis.plot(line.hqq_nbits, line.accuracy_ratio, '.-', color=colors[dataset],
                              lw=1.2, ms=4, label=dataset)
                axis.axhline(1, color='black', lw=.7, ls='--')
                axis.set(title=f'group size {group_size:g}', xlabel='HQQ bits', xticks=bits,
                         ylim=(0, max(1.05, selected.accuracy_ratio.max() * 1.025)))
                axis.set_xticks(bits, [f'{b:g}' for b in bits],
                                rotation=45 if len(bits) > 4 else 0)
            axes[0, 0].set_ylabel('Accuracy retention ratio')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=8, bbox_to_anchor=(.5, -.04))
            fig.suptitle(f'{title}: {scope.replace("_", " ")} | one curve per dataset')
            fig.tight_layout(rect=(0, .09, 1, .94))
            save(fig, f'curves_{slug}_{scope}.png', 'cells.csv', title + ' retention curves')
        selected = frame[frame.hqq_nbits.isin(primary_bits)]
        ratio = selected.pivot(index='dataset_name', columns=SETTING, values='accuracy_ratio')
        ratio = ratio.reindex(index=datasets).sort_index(axis=1)
        complete, _ = common_panel(frame, primary_bits, groups)
        rank = complete.rank(axis=0, ascending=False, method='average').reindex(index=datasets)
        baseline = frame.groupby('dataset_name').fp32_top1.first()
        rowlabels = [f'{d}  ({baseline.get(d, np.nan):.1f}%)' for d in datasets]
        collabels = [f'{b:g}b\ng{g:g}' for b, g in ratio.columns]
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for axis, matrix, cmap, fmt, vmax, label in [
            (axes[0], ratio, 'viridis', '.2f', max(1.0, ratio.max().max()), 'Retention ratio'),
            (axes[1], rank, 'viridis_r', '.1f', len(datasets), 'Rank (1 = highest ratio)')]:
            sns.heatmap(matrix, ax=axis, cmap=cmap, annot=True, fmt=fmt, vmin=0 if axis is axes[0] else 1,
                        vmax=vmax, xticklabels=collabels, yticklabels=rowlabels, cbar_kws={'label': label},
                        annot_kws={'size': 7})
            axis.set(xlabel='Quantization condition', ylabel='Dataset (baseline accuracy)', title=label)
            axis.tick_params(axis='y', rotation=0)
            axis.tick_params(axis='x', rotation=0)
        fig.suptitle(f'{title}: primary conditions | alphabetical dataset order')
        fig.tight_layout(rect=(0, 0, 1, .96))
        save(fig, f'heatmap_{slug}.png', 'cells.csv', title + ' ratios and ranks with baseline accuracy')
    primary = tables['dataset_profiles'].query('scope == "primary"').copy()
    primary['context_label'] = primary.model_name.map(short_model) + '/' + primary.ckpt_kind
    fig, ax = plt.subplots(figsize=(11, 8))
    matrix = primary.pivot(index='dataset_name', columns='context_label', values='rank_median')
    sns.heatmap(matrix.sort_index(), annot=True, fmt='.1f', cmap='viridis_r', vmin=1,
                vmax=len(datasets), ax=ax, cbar_kws={'label': 'Median condition rank (1 = highest)'})
    ax.set(title='Dataset rank profiles by model/checkpoint\nMedian over 2/3/4-bit and observed group sizes; no universal ranking',
           xlabel='Model / checkpoint kind', ylabel='Dataset')
    ax.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    save(fig, 'dataset_rank_profiles.png', 'dataset_profiles.csv', 'Context-specific median ranks, alphabetical rows')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(cells.fp32_top1, cells.accuracy_ratio,
               c=cells.hqq_nbits, cmap='viridis', s=9, alpha=.35)
    ax.axhline(1, color='black', ls='--', lw=.7)
    ax.set(xlabel='Baseline accuracy (%)', ylabel='Accuracy retention ratio',
           title='Baseline audit: all matched HQQ cells (color = bit width)')
    fig.colorbar(ax.collections[0], ax=ax, label='HQQ bits')
    fig.tight_layout()
    save(fig, 'baseline_ratio_audit.png', 'cells.csv', 'Baseline accuracy versus retention, no excluded low-baseline contexts')
    return pd.DataFrame(manifest)


def markdown_table(frame, digits=3):
    def render(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.{digits}f}' if np.isfinite(value) else 'NA'
        return str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(render(v) for v in row) + ' |'
                 for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def make_report(cells, excluded, tables, output, quality, config):
    within = tables['within_context_agreement']
    primary_within = within[within.scope.eq('primary')]
    cross = tables['cross_context_agreement']
    primary_cross = cross[cross.primary]
    ordering = tables['pairwise_order_stability']
    primary_order = ordering[ordering.scope.eq('primary')]
    profiles = tables['dataset_profiles']
    context_summary = primary_within.groupby(CONTEXT).agg(
        rho_median=('rho', 'median'), rho_min=('rho', 'min'),
        undefined_comparisons=('rho', lambda s: s.isna().sum())).reset_index()
    flips = primary_order.groupby(CONTEXT).order_reverses.mean().rename('pair_reversal_fraction').reset_index()
    context_summary = context_summary.merge(flips, on=CONTEXT)
    context_summary['model_name'] = context_summary.model_name.map(short_model)
    portability = tables['dataset_portability'].sort_values('context_median_rank_span', ascending=False)
    baselines = tables['baseline_audit'].sort_values('baseline_accuracy')
    low = baselines.head(10)[['dataset_name', 'model_name', 'ckpt_kind', 'baseline_accuracy']].copy()
    low['model_name'] = low.model_name.map(short_model)
    metric = tables['metric_agreement']
    primary_metric = metric[metric.hqq_nbits.isin(config['primary_bits'])]
    sensitivity = within.groupby('scope').rho.agg(['median', 'min', 'count']).reset_index()
    scope_flips = ordering.groupby('scope').order_reverses.mean().rename('pair_reversal_fraction').reset_index()
    sensitivity = sensitivity.merge(scope_flips, on='scope')
    excluded_cohorts = int((~tables['coverage'].included).sum())
    nonfixed = quality.loc[~quality.fixed.astype(str).str.lower().eq('true'), 'column'].tolist() if not quality.empty else []
    unknown = quality.loc[quality.n_unique.eq(0), 'column'].tolist() if not quality.empty else []
    leads = []
    for context, group in profiles[profiles.scope.eq('primary')].groupby(CONTEXT):
        minimum = group.rank_median.min()
        leaders = group[group.rank_median.eq(minimum)]
        leads.append(dict(model=short_model(context[0]), checkpoint=context[1],
                          datasets='、'.join(leaders.dataset_name), median_rank=minimum,
                          full_rank_range='; '.join(f'{r.dataset_name}: {r.rank_best:g}–{r.rank_worst:g}'
                                                   for r in leaders.itertuples())))
    interpretation = (
        '目前結果較支持「在固定模型背景下存在相對穩定的 dataset 差異，但跨背景的可移植性較弱」。'
        if primary_cross.rho.median() < primary_within.rho.median() else
        '跨背景與跨量化條件的一致性應分開判讀，不能以單一係數概括 dataset robustness。')
    interpretation += (' 存在先後順序翻轉，因此不給出適用所有條件的 dataset 總排名。'
                       if primary_order.order_reverses.any() else
                       '本次未觀察到先後順序翻轉，但單一 seed 仍不足以證明可重現性。')
    lines = [
        '# Dataset quantization robustness：排名穩定性分析', '',
        '這份報告回答：在目前 HQQ 實驗中，哪些 dataset 保留較多 accuracy，且相對優勢是否隨量化條件或模型背景改變？', '',
        '## 主要結果', '',
        f'- 有效條件 **{len(cells):,}** 筆；排除 {len(excluded)} 筆；涵蓋 **{cells.dataset_name.nunique()} datasets、'
        f'{cells.model_name.nunique()} models、{cells.ckpt_kind.nunique()} checkpoint 類型**。',
        f'- 每條件 seed 數範圍：{cells.n_seeds.min():g}–{cells.n_seeds.max():g}。量化條件不是獨立訓練重複。',
        f'- 主要 2／3／4-bit 範圍內，固定 model／checkpoint 的條件間排名 Spearman ρ 中位數為 '
        f'**{primary_within.rho.median():.3f}**，最小值 {primary_within.rho.min():.3f}。',
        f'- 固定量化條件、改變 model／checkpoint 時，排名 ρ 中位數為 **{primary_cross.rho.median():.3f}**，'
        f'最小值 {primary_cross.rho.min():.3f}。',
        f'- 各 model／checkpoint 內的 dataset 配對中，**{primary_order.order_reverses.mean():.1%}** '
        '至少發生一次相對順序翻轉。分母是「dataset pair × model/checkpoint」，不是獨立樣本。',
        '- 以上係數及比例均為描述性摘要；未設定「穩定」顯著性門檻，未強制生成全域總排名。', '',
        '**解讀：** ' + interpretation, '',
        '## 各模型背景的穩定程度', '', markdown_table(context_summary), '',
        '`pair_reversal_fraction` 高代表更多 dataset 配對隨 bit/group 改變先後順序；'
        '它不衡量翻轉的幅度，應同時查看 pairwise_order_stability.csv 中的 ratio 差距。', '',
        '## 哪些 dataset 在各背景下相對領先？', '',
        '下表僅列各背景中「主要條件 rank 中位數最小」的 dataset（同分並列）。'
        '這是等權條件的描述，並非宣稱它在每一條件都最佳。完整名次範圍揭露其變動。', '',
        markdown_table(pd.DataFrame(leads)), '',
        '![Dataset rank profiles](plots/dataset_rank_profiles.png)', '',
        '各 dataset 跨背景的中位名次範圍如下；依 dataset 名稱排列，不建立總排名。', '',
        markdown_table(portability.sort_values('dataset_name')), '',
        '## bit 範圍敏感度', '', markdown_table(sensitivity), '',
        'primary = 2/3/4-bit；without_2bit = 3/4-bit；full_grid = 所有觀測 bit。'
        '每一範圍內，各 bit × group size 等權；範圍間條件數不同，翻轉比例不宜解讀為相同次數的重複檢定。'
        '極低 bit 的大幅退化與高 bit 接近 baseline 時的微小差異，皆可能造成排名翻轉；'
        '因此 full_grid 的較低一致性不等於所有實用量化條件都不穩定。'
        'scope_sensitivity.csv 保留每個 dataset 的中位名次及 ratio，供逐一比較。', '',
        '## Baseline 與指標敏感度', '',
        f'主要條件內，ratio 排名與「掉分越少越好」排名的一致性 ρ 中位數為 '
        f'{primary_metric.ratio_vs_drop_rho.median():.3f}，最小值 {primary_metric.ratio_vs_drop_rho.min():.3f}。', '',
        'Baseline 最低的 10 個 context（供稽核，並非排除門檻）：', '', markdown_table(low), '',
        '低 baseline 會放大 ratio 對 accuracy 變動的反應。所有有效 context 均保留；'
        'dataset 整體低 baseline 與個別 model/checkpoint 異常必須分開判讀。'
        '同時檢查 baseline、HQQ accuracy 與掉分百分點，不能把 ratio 排名直接當成部署 accuracy 排名。', '',
        '![Baseline audit](plots/baseline_ratio_audit.png)', '',
        '## 方法與納入規則', '',
        '- Legacy summary 先在同 dataset、normalized model、完整 checkpoint 路徑與 seed 內配對。'
        '跨 serial 重複不是獨立 seed；本分析使用 summary 的配對 ratio。',
        '- 主要 outcome 為 HQQ / FP32 accuracy ratio；輔助 outcome 為 FP32 − HQQ 的百分點差。'
        'Ratio > 1 不截斷，負掉分照實保留。',
        '- 同 dataset/model/checkpoint 類型不允許多 checkpoint 路徑混合；遇到歧義會停止。',
        '- 每個 model/checkpoint 與 bit 範圍使用所有指定條件皆存在的固定 dataset cohort。'
        '跨背景比較則在同量化條件下使用所有背景共同具備的 dataset，避免比較對象隨配對改變。',
        f'- coverage.csv 中被完整 cohort 規則排除的 dataset/context/scope 紀錄：{excluded_cohorts}。',
        '- Rank 1 為最高 ratio，同值採 average rank。Pairwise 比較的 1e-12 僅是浮點相等容差，'
        '不是實質差異門檻；不把微小差距自動解讀為有意義優勢。',
        '- Spearman ρ 為配對 dataset 的秩相關。少於 3 個 dataset 或常數排名時記為 NA，'
        '不填成 0、不產生 p-value。所有中位數均不加樣本數權重。', '',
        '## 設定品質與驗證缺口', '',
        f'- configuration_quality.csv 非固定欄位：{", ".join(nonfixed) or "無"}；'
        f'完全缺值欄位：{", ".join(unknown) or "無"}。缺值不代表設定已驗證固定。',
        '- 目前只有單一 seed 的 context 無法估計跨訓練 seed 不確定性；32 個量化條件不能當成 32 個 seed。',
        '- 這是 HQQ、既有模型與 dataset-specific checkpoint 下的關聯性比較；'
        '不能分離 dataset 與訓練結果，也不能推廣為 dataset 固有性質或所有 PTQ 方法的結論。',
        '- 未新增 W&B 下載。既有 summary 無逐樣本預測，因此這份報告不估計 paired-image uncertainty 或 damage/rescue。',
        '- 後續優先補多個獨立訓練 seed；對低 baseline checkpoint 核對訓練及評估設定；'
        '若要推廣到其他量化方法，再做同 checkpoint 的跨方法驗證。', '',
        '## 可重現性與檔案', '',
        f'- 輸入：`{config["input_file"]}`',
        f'- SHA-256：`{config["input_sha256"]}`',
        '- `cells.csv`：全部有效條件、baseline、掉分及逐條件排名。',
        '- `dataset_profiles.csv`：每背景與分析範圍內的 ratio 與排名分布。',
        '- `within_context_agreement.csv` / `cross_context_agreement.csv`：排名一致性。',
        '- `pairwise_order_stability.csv`：dataset 配對勝負、翻轉與實際差距。',
        '- `metric_agreement.csv` / `scope_sensitivity.csv`：指標與 bit 範圍敏感度。',
        '- `baseline_audit.csv` / `coverage.csv` / `excluded_cells.csv`：資料稽核。',
        '- `plots_manifest.csv`：所有圖片及其來源表；`run_config.json`：參數與來源指紋。',
        '- `legacy/`：從原始 CSV 重跑的 summary、correlation、XLSX 與原有圖表。', '',
    ]
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path,
                        default=ROOT / 'results_all/dataset_robustness/legacy/quant_summary.csv')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results_all/dataset_robustness')
    parser.add_argument('--primary-bits', type=float, nargs='+', default=[2., 3., 4.])
    args = parser.parse_args()
    if sorted(set(args.primary_bits)) != [2., 3., 4.]:
        parser.error('This agreed analysis contract requires primary bits 2 3 4.')
    args.input_file = args.input_file.resolve()
    output = args.results_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cells, excluded = prepare_cells(pd.read_csv(args.input_file))
    tables = analyze(cells, args.primary_bits)
    cells.to_csv(output / 'cells.csv', index=False)
    excluded.to_csv(output / 'excluded_cells.csv', index=False)
    for name, frame in tables.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    manifest = make_plots(cells, tables, output, args.primary_bits)
    manifest.to_csv(output / 'plots_manifest.csv', index=False)
    config = dict(input_file=str(args.input_file), results_dir=str(output),
                  input_sha256=hashlib.sha256(args.input_file.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  created_at=datetime.now(timezone.utc).isoformat(), primary_bits=args.primary_bits,
                  group_sizes=sorted(cells.hqq_group_size.unique().tolist()),
                  n_valid_cells=len(cells), n_excluded_cells=len(excluded),
                  inference='descriptive_only_no_p_values_no_seed_CIs',
                  weighting='equal_conditions_within_context_no_global_dataset_ranking')
    quality_file = args.input_file.parent / 'configuration_quality.csv'
    quality = pd.read_csv(quality_file) if quality_file.exists() else pd.DataFrame()
    config['configuration_quality_available'] = quality_file.exists()
    make_report(cells, excluded, tables, output, quality, config)
    (output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Analyzed {len(cells)} matched cells; {len(manifest)} figures. Report: {output / "report.md"}')


if __name__ == '__main__':
    main()
