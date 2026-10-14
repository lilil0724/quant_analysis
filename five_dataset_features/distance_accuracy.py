"""Compare FP-to-quantized distance drops with matched accuracy losses.

conda run -n opencode_env python -m five_dataset_features.distance_accuracy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import DATASETS, grid, write_csv
from .analysis_snapshot import sha256
from .metrics import spearman_description
from .quantized_class_distance import ROOT, STEMS, SUFFIXES, load_rows


PREDICTORS = ('gap_drop_fraction', 'inter_drop_fraction', 'intra_drop_fraction',
              'distance_ratio_drop_fraction')
FAMILIES = ('Weight only', 'Linear A8', 'Linear A8 + QKV', 'Linear A4', 'Linear A4 + QKV')


def build_table(rows):
    frame = pd.DataFrame(rows)
    expected = {(d, c) for d in DATASETS for c in grid()}
    actual = set(zip(frame.dataset, frame.condition))
    if len(frame) != len(expected) or actual != expected:
        raise ValueError('Expected 105 unique matched dataset-condition rows')
    frame = frame[frame.condition != 'fp'].copy()
    def drop(quantized, fp):
        return 1 - quantized / fp.where(fp > 0)
    frame['gap_drop_fraction'] = drop(frame.distance_gap, frame.fp_distance_gap)
    frame['inter_drop_fraction'] = drop(frame.d_inter, frame.fp_d_inter)
    frame['intra_drop_fraction'] = drop(frame.d_intra, frame.fp_d_intra)
    fp_ratio = frame.fp_d_inter / frame.fp_d_intra.where(frame.fp_d_intra > 0)
    q_ratio = frame.d_inter / frame.d_intra.where(frame.d_intra > 0)
    frame['distance_ratio_drop_fraction'] = drop(q_ratio, fp_ratio)
    frame['accuracy_drop_pp'] = 100 * frame.accuracy_drop
    frame['weight_group'] = frame.condition.str.extract(r'^(w[34]_g(?:8|128))')[0]
    suffix = frame.condition.str.replace(r'^w[34]_g(?:128|8)', '', regex=True)
    frame['activation_family'] = suffix.map(dict(zip(SUFFIXES, FAMILIES)))
    if frame.activation_family.isna().any():
        raise ValueError('Unknown activation family')
    return frame


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    rho = spearman_description(x, y)
    pearson = (float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 and
               np.ptp(x) > 0 and np.ptp(y) > 0 else None)
    return len(x), rho, pearson


def correlation_tables(frame):
    fixed, within = [], []
    for condition in grid()[1:]:
        group = frame[frame.condition == condition]
        for predictor in PREDICTORS:
            for outcome in ('relative_loss', 'accuracy_drop', 'accuracy'):
                n, rho, pearson = correlation(group[predictor], group[outcome])
                loo = [correlation(group.loc[group.dataset != dataset, predictor],
                                   group.loc[group.dataset != dataset, outcome])[1]
                       for dataset in DATASETS]
                finite = [value for value in loo if value is not None]
                fixed.append({'condition': condition, 'predictor': predictor, 'outcome': outcome,
                              'n_dataset_checkpoint_pairs': n, 'spearman_rho': rho,
                              'pearson_r': pearson,
                              'leave_one_dataset_out_rho_min': min(finite) if finite else None,
                              'leave_one_dataset_out_rho_max': max(finite) if finite else None})
    for dataset in DATASETS:
        group = frame[frame.dataset == dataset]
        scopes = {'all_20': group,
                  'exclude_A4_12': group[~group.activation_family.str.contains('A4')],
                  'weight_only_4': group[group.activation_family == 'Weight only']}
        for scope, selected in scopes.items():
            for predictor in PREDICTORS:
                for outcome in ('relative_loss', 'accuracy_drop'):
                    n, rho, pearson = correlation(selected[predictor], selected[outcome])
                    within.append({'dataset': dataset, 'scope': scope, 'predictor': predictor,
                                   'outcome': outcome, 'n_repeated_conditions': n,
                                   'spearman_rho': rho, 'pearson_r': pearson})
    return pd.DataFrame(fixed), pd.DataFrame(within)


def fmt(value):
    return 'undefined' if pd.isna(value) else f'{value:+.2f}'


def report(frame, fixed, within):
    primary = fixed[(fixed.predictor == 'gap_drop_fraction') & (fixed.outcome == 'relative_loss')]
    lines = ['# FP → quantized distance drop 與 accuracy', '',
             '## 定義與資料', '',
             '- gap = D_inter − D_intra，距離使用最後一層 768D CLS 的 class-equal cosine distance。',
             '- gap drop fraction = (FP gap − Q gap) / FP gap。',
             '- accuracy ratio = Q accuracy / FP accuracy；relative accuracy loss = 1 − accuracy ratio。',
             '- absolute accuracy drop = FP accuracy − Q accuracy；表格轉成 percentage points（pp）。',
             '- inter/intra/距離 ratio 的下降比例亦為 1 − Q/FP；intra drop 為負表示類內距離增加。',
             '- 所有 FP 參考皆為同資料集、同 checkpoint、同測試樣本的 AMP-FP16 未量化推論。',
             '- 來源為已驗證 snapshot 的保存分析，不重新驗證原始 inference arrays。', '',
             '## W3/G128 weight-only', '',
             '| Dataset | Gap drop (%) | FP acc (%) | Q acc (%) | Acc retained (%) | Relative acc loss (%) | Absolute drop (pp) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in frame[frame.condition == 'w3_g128'].itertuples():
        values = [100 * getattr(row, key) for key in ('gap_drop_fraction', 'fp_accuracy',
                  'accuracy', 'accuracy_ratio', 'relative_loss', 'accuracy_drop')]
        lines.append('| ' + row.dataset + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
    selected = primary[primary.condition == 'w3_g128'].iloc[0]
    lines += ['', f'固定 W3/G128 的 gap drop vs relative accuracy loss：'
              f'Spearman ρ = {fmt(selected.spearman_rho)}；Pearson r = {fmt(selected.pearson_r)}。'
              f'逐次排除一個資料集的 ρ 範圍 '
              f'{fmt(selected.leave_one_dataset_out_rho_min)} 至 {fmt(selected.leave_one_dataset_out_rho_max)}。', '',
              'W3/G128 的距離指標敏感度（同樣對 relative accuracy loss）：', '',
              '| Distance drop metric | Spearman ρ | Pearson r |',
              '|---|---:|---:|']
    for row in fixed[(fixed.condition == 'w3_g128') & (fixed.outcome == 'relative_loss')].itertuples():
        lines.append(f'| {row.predictor} | {fmt(row.spearman_rho)} | {fmt(row.pearson_r)} |')
    lines += ['', 'gap 與 inter/intra ratio 衡量不同的幾何變化；若相關方向不同，不能挑選單一指標宣稱普遍關係。', '',
              '## 固定條件，跨資料集比較', '',
              '每列只有 5 個 dataset-checkpoint pairs。未把 100 個條件混成獨立樣本，也未進行顯著性或因果判斷。', '',
              '| Condition | Gap drop vs relative loss ρ | Gap drop vs absolute drop ρ | Leave-one-dataset-out relative-loss ρ |',
              '|---|---:|---:|---|']
    for row in primary.itertuples():
        absolute = fixed[(fixed.condition == row.condition) &
                         (fixed.predictor == 'gap_drop_fraction') & (fixed.outcome == 'accuracy_drop')].iloc[0]
        lines.append(f'| {row.condition} | {fmt(row.spearman_rho)} | {fmt(absolute.spearman_rho)} | '
                     f'{fmt(row.leave_one_dataset_out_rho_min)} to {fmt(row.leave_one_dataset_out_rho_max)} |')
    lines += ['', f'固定條件的 relative-loss ρ 範圍為 {fmt(primary.spearman_rho.min())} '
              f'至 {fmt(primary.spearman_rho.max())}。相關方向會隨量化條件改變，'
              '不能由 gap drop 比例建立穩定的跨資料集 relative accuracy loss 排名。', '']
    a4 = fixed[(fixed.condition.str.contains('_a4')) & (fixed.predictor == 'gap_drop_fraction') &
               (fixed.outcome == 'accuracy_drop')]
    lines += [f'A4 的 {len(a4)} 個固定條件中，gap drop vs absolute accuracy drop 的 ρ 範圍為 '
              f'{fmt(a4.spearman_rho.min())} 至 {fmt(a4.spearman_rho.max())}。'
              '這與 relative-loss 結果應分開陳述；absolute drop 與各資料集 FP accuracy 及其下降空間有關，'
              '不能將此排序關係解讀為 gap drop 對 accuracy 的普遍因果效果。', '',
              '## 同一資料集，跨量化條件比較', '',
              '| Dataset | All 20 ρ | Exclude A4 (12) ρ | Weight only (4) ρ |',
              '|---|---:|---:|---:|']
    for dataset in DATASETS:
        part = within[(within.dataset == dataset) & (within.predictor == 'gap_drop_fraction') &
                      (within.outcome == 'relative_loss')].set_index('scope')
        lines.append('| ' + dataset + ' | ' + ' | '.join(fmt(part.loc[scope, 'spearman_rho'])
                     for scope in ('all_20', 'exclude_A4_12', 'weight_only_4')) + ' |')
    lines += ['', '同一資料集的 FP accuracy 固定，所以 relative loss 與 absolute drop 的 Spearman 排名相同。'
              '20 個條件共享同一個 checkpoint/測試集，是重複條件而非獨立模型重複實驗。'
              '排除 A4 是檢查相關是否只由嚴重量化主導；weight-only 是額外敏感度檢查。', '',
              '## 判讀', '',
              'gap 下降與 accuracy 下降的比例沒有一對一關係。'
              '原始 FP accuracy 差異會影響相對 loss，因此應同時查看 absolute drop。'
              'gap 描述 ground-truth 類別的平均表徵分離，未直接衡量分類頭的決策邊界。', '',
              '所有 inter、intra、distance ratio 下降比例，以及與原始 Q accuracy 的 Pearson/Spearman '
              '關係均保存在 CSV 中。跨資料集 raw Q accuracy 的關係也包含 FP 基線差異，不等同量化耐受性。', '']
    return '\n'.join(lines)


def plots(frame, fixed, within, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plot_presentation import COLORS

    def save(fig, name):
        for ext in ('png', 'svg'):
            fig.savefig(output / f'{name}.{ext}', dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.subplots_adjust(left=.08, right=.96, top=.83, bottom=.15, hspace=.35, wspace=.25)
    for ax, condition in zip(axes.flat, STEMS):
        selected = frame[frame.condition == condition]
        for row in selected.itertuples():
            x, y = 100 * row.gap_drop_fraction, 100 * row.relative_loss
            ax.scatter(x, y, color=COLORS[row.dataset], s=90)
            ax.annotate(row.dataset, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)
        rho = fixed[(fixed.condition == condition) & (fixed.predictor == 'gap_drop_fraction') &
                    (fixed.outcome == 'relative_loss')].iloc[0].spearman_rho
        ax.set_title(f'{condition.upper().replace("_", " / ")} | Spearman rho = {fmt(rho)}')
        ax.margins(.22)
        ax.set_xlabel('Gap drop from FP (%)')
        ax.set_ylabel('Relative accuracy loss (%)')
        ax.grid(alpha=.2)
    fig.suptitle('Weight-only: gap drop versus accuracy loss at fixed conditions', fontsize=21, y=.95)
    fig.text(.08, .045, '5 dataset-checkpoint pairs per panel | Matched AMP-FP16 reference | '
             'Descriptive associations; different checkpoints and baselines', fontsize=11)
    save(fig, 'fixed_weight_only')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.97, top=.83, bottom=.16, hspace=.36, wspace=.22)
    markers = ('o', '^', 'v', 's', 'D')
    for ax, dataset in zip(axes.flat, DATASETS):
        selected = frame[frame.dataset == dataset]
        for family, marker in zip(FAMILIES, markers):
            part = selected[selected.activation_family == family]
            ax.scatter(100 * part.gap_drop_fraction, 100 * part.relative_loss,
                       color=COLORS[dataset], marker=marker, s=65, alpha=.8)
        part = within[(within.dataset == dataset) & (within.predictor == 'gap_drop_fraction') &
                      (within.outcome == 'relative_loss')].set_index('scope')
        ax.set_title(f'{dataset} | all rho = {fmt(part.loc["all_20", "spearman_rho"])}\n'
                     f'exclude A4 rho = {fmt(part.loc["exclude_A4_12", "spearman_rho"])}', fontsize=12)
        ax.grid(alpha=.2)
    axes[1, 2].axis('off')
    handles = [plt.Line2D([], [], linestyle='', marker=marker, color='#555555', label=family)
               for family, marker in zip(FAMILIES, markers)]
    axes[1, 2].legend(handles=handles, loc='center', frameon=False, fontsize=12)
    fig.suptitle('Within each dataset: repeated quantization conditions', fontsize=22, y=.95)
    fig.supxlabel('Gap drop from FP (%)', y=.09)
    fig.supylabel('Relative accuracy loss (%)', x=.015)
    fig.text(.08, .025, '20 repeated conditions per dataset | Same checkpoint and test set | '
             'No independent-seed replication or causal interpretation', fontsize=11)
    save(fig, 'within_dataset')


def run(analysis):
    inputs = [analysis / name for name in ('analysis_snapshot.json', 'all_conditions.csv', 'fp_indicators.csv')]
    inputs += [analysis / dataset / name for dataset in DATASETS
               for name in ('analysis_snapshot.json', 'summary.csv', 'layer_metrics.csv')]
    source_hashes = {str(path.resolve()): sha256(path) for path in inputs}
    frame = build_table(load_rows(analysis))
    fixed, within = correlation_tables(frame)
    output = analysis / 'distance_accuracy'
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [('distance_accuracy.csv', frame), ('fixed_condition_correlations.csv', fixed),
                        ('within_dataset_correlations.csv', within)]:
        write_csv(output / name, table.replace({np.nan: None}).to_dict('records'))
        pd.testing.assert_frame_equal(pd.read_csv(output / name), table.reset_index(drop=True),
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    (output / 'REPORT.md').write_text(report(frame, fixed, within), encoding='utf-8')
    plots(frame, fixed, within, output)
    load_rows(analysis)
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise ValueError('Inputs changed during analysis')
    filenames = ('distance_accuracy.csv', 'fixed_condition_correlations.csv',
                 'within_dataset_correlations.csv', 'REPORT.md', 'fixed_weight_only.png',
                 'fixed_weight_only.svg', 'within_dataset.png', 'within_dataset.svg')
    code_files = (Path(__file__), ROOT / 'quantized_class_distance.py', ROOT / 'geometry_robustness.py',
                  ROOT / 'metrics.py', ROOT / 'plot_presentation.py', ROOT / 'analyze.py', ROOT / 'analysis_snapshot.py')
    config = {'schema': 1, 'source_sha256': source_hashes,
              'code_sha256': {str(path.resolve()): sha256(path) for path in code_files},
              'reference': 'matched AMP-FP16 unquantized', 'feature': 'final normalized CLS, layer 12, 768D',
              'raw_arrays_revalidated': False, 'statistical_scope': 'descriptive fixed-condition and repeated-condition correlations; no p-values',
              'verification': {'passed': True, 'quantized_rows': len(frame),
                               'saved_csv_tables_checked': 3},
              'output_sha256': {name: sha256(output / name) for name in filenames}}
    (output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(frame)} matched quantized rows and 3 CSV tables: {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, default=ROOT / 'analysis')
    args = parser.parse_args()
    run(args.analysis_root)


if __name__ == '__main__':
    main()
