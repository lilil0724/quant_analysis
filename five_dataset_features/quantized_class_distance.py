"""Summarize verified final-CLS cosine distances for all quantized conditions.

conda run -n opencode_env python -m five_dataset_features.quantized_class_distance
"""

import argparse
import json
import math
from pathlib import Path

from .analyze import DATASETS, grid, read_csv, write_csv
from .analysis_snapshot import sha256
from .geometry_robustness import verify_aggregate


ROOT = Path(__file__).resolve().parent
STEMS = ('w3_g8', 'w3_g128', 'w4_g8', 'w4_g128')
SUFFIXES = ('', '_a8', '_a8_qkv', '_a4', '_a4_qkv')


def build_rows(layer_rows, summaries):
    """Pair each final-layer distance with the same dataset's FP and accuracy."""
    expected = {(dataset, condition) for dataset in DATASETS for condition in grid()}
    summary = {(row['dataset'], row['condition']): row for row in summaries}
    if len(summaries) != len(expected) or set(summary) != expected:
        raise ValueError('Expected exactly 105 unique summary rows')
    layers = {(row['dataset'], row['condition'], int(row['layer'])): row
              for row in layer_rows}
    expected_layers = {(*key, layer) for key in expected for layer in range(13)}
    if len(layer_rows) != len(expected_layers) or set(layers) != expected_layers:
        raise ValueError('Expected exactly 13 unique layers for each condition')

    rows = []
    for dataset in DATASETS:
        fp = layers[dataset, 'fp', 12]
        fp_inter, fp_intra = float(fp['d_inter']), float(fp['d_intra'])
        fp_gap = fp_inter - fp_intra
        for condition in grid():
            layer = layers[dataset, condition, 12]
            inter, intra = float(layer['d_inter']), float(layer['d_intra'])
            gap = inter - intra
            if not all(math.isfinite(value) and -1e-9 <= value <= 2 + 1e-9
                       for value in (inter, intra)):
                raise ValueError(f'{dataset}/{condition}: invalid cosine distances')
            if not math.isclose(gap, float(layer['separability']), abs_tol=1e-10):
                raise ValueError(f'{dataset}/{condition}: inconsistent separation')
            for field in ('n_intra_classes', 'n_inter_class_pairs'):
                if int(layer[field]) <= 0 or layer[field] != fp[field]:
                    raise ValueError(f'{dataset}/{condition}: class coverage mismatch')
            outcome = summary[dataset, condition]
            base = summary[dataset, 'fp']
            fpa, accuracy = float(base['accuracy']), float(outcome['accuracy'])
            drop = fpa - accuracy
            if not (0 < fpa <= 1 and 0 <= accuracy <= 1) or not all((
                    math.isclose(float(outcome['fp_accuracy']), fpa, abs_tol=1e-10),
                    math.isclose(float(outcome['accuracy_drop']), drop, abs_tol=1e-10),
                    math.isclose(float(outcome['relative_loss']), drop / fpa, abs_tol=1e-10),
                    int(outcome['n']) == int(base['n']))):
                raise ValueError(f'{dataset}/{condition}: inconsistent accuracy baseline')
            rows.append({
                'dataset': dataset, 'condition': condition, 'layer': 12,
                'n_samples': int(outcome['n']),
                'n_intra_classes': int(layer['n_intra_classes']),
                'n_inter_class_pairs': int(layer['n_inter_class_pairs']),
                'd_inter': inter, 'd_intra': intra, 'distance_gap': gap,
                'distance_ratio': None if intra <= 0 else inter / intra,
                'fp_d_inter': fp_inter, 'fp_d_intra': fp_intra,
                'fp_distance_gap': fp_gap,
                'delta_inter': inter - fp_inter, 'delta_intra': intra - fp_intra,
                'delta_gap': gap - fp_gap,
                'gap_retention': None if fp_gap <= 0 else gap / fp_gap,
                'fp_accuracy': fpa, 'accuracy': accuracy,
                'accuracy_ratio': accuracy / fpa,
                'relative_loss': drop / fpa, 'accuracy_drop': drop,
            })
    return rows


def load_rows(analysis):
    verify_aggregate(analysis)
    layers = [row for dataset in DATASETS
              for row in read_csv(analysis / dataset / 'layer_metrics.csv')]
    summaries = read_csv(analysis / 'all_conditions.csv')
    # The dataset summaries must agree with the aggregate used for accuracy.
    for dataset in DATASETS:
        if read_csv(analysis / dataset / 'summary.csv') != [
                row for row in summaries if row['dataset'] == dataset]:
            raise ValueError(f'{dataset}: summary differs from aggregate')
    return build_rows(layers, summaries)


def report(rows):
    by_key = {(row['dataset'], row['condition']): row for row in rows}
    lines = ['# Quantized inter/intra-class distance', '',
             '使用已驗證 snapshot 的完整測試集距離，整理 5 個資料集 × '
             '21 個條件（FP + 20 個 quantized），不是 PCA 投影距離。', '',
             '- 特徵：最後一層 normalized CLS（layer 12，768D）；參考為同資料集、同 checkpoint 的 AMP-FP16 未量化推論。',
             '- 距離：樣本特徵 L2 normalize 後的 cosine distance，依 ground-truth class 分組。',
             '- intra：各類別內不同樣本配對的平均距離，再對至少兩個樣本的類別等權平均。',
             '- inter：每對不同類別的跨類樣本平均距離，再對所有類別配對等權平均；不是 normalized centroid distance。',
             '- gap = inter − intra；ratio = inter / intra；delta = quantized − FP。gap 越大表示此平均距離定義下的類別分離越大。',
             '- gap retention = Q gap / FP gap，與 accuracy retention 分開；CSV accuracy/drop 使用 0–1 單位。',
             '- 本階段驗證保存 CSV、分析程式與 snapshot hashes，未重新讀取或重新驗證全部原始 inference arrays。', '',
             '## W3/G128 weight-only', '',
             '| Dataset | FP inter | FP intra | FP gap | Q inter | Q intra | Q gap | Δgap | Gap retained |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        row = by_key[dataset, 'w3_g128']
        values = [row[key] for key in ('fp_d_inter', 'fp_d_intra', 'fp_distance_gap',
                                      'd_inter', 'd_intra', 'distance_gap', 'delta_gap')]
        lines.append('| ' + dataset + ' | ' + ' | '.join(f'{value:.6f}' for value in values)
                     + f" | {100 * row['gap_retention']:.2f}% |")
    for stem in STEMS:
        lines += ['', f'## {stem}: final CLS gap', '',
                  '| Dataset | FP | Weight only | + A8 | + A8 + QKV A8 | + A4 | + A4 + QKV A4 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for dataset in DATASETS:
            conditions = ('fp', *(stem + suffix for suffix in SUFFIXES))
            lines.append('| ' + dataset + ' | ' + ' | '.join(
                f"{by_key[dataset, condition]['distance_gap']:.6f}" for condition in conditions) + ' |')
    lines += ['', '## 判讀界線', '',
              'gap 縮小可能來自 inter 下降、intra 上升或兩者一起改變，應一起讀取三個值。'
              '此指標描述平均表徵分離，不能單獨判斷分類準確率或決策邊界。'
              '跨資料集各有不同 checkpoint、類別數與單一 seed；量化條件不是獨立模型重複試驗。', '',
              '完整距離、ratio、FP 差值、accuracy ratio、relative loss、absolute drop '
              '見 `final_class_distances.csv`（105 列）。', '']
    return '\n'.join(lines)


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .plot_presentation import COLORS

    by_key = {(row['dataset'], row['condition']): row for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.97, top=.84, bottom=.15, hspace=.32, wspace=.22)
    for ax, stem in zip(axes.flat, STEMS):
        for dataset in DATASETS:
            conditions = ('fp', *(stem + suffix for suffix in SUFFIXES))
            values = [by_key[dataset, condition]['distance_gap'] for condition in conditions]
            ax.plot(range(6), values, marker='o', color=COLORS[dataset], label=dataset)
        ax.set_title(stem.upper().replace('_', ' / '))
        ax.set_xticks(range(6), ['FP', 'W only', 'A8', 'A8 + QKV', 'A4', 'A4 + QKV'])
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
    for ax in axes[:, 0]:
        ax.set_ylabel('Inter - intra cosine distance')
    fig.suptitle('Final CLS class separation after quantization', fontsize=22, y=.95)
    fig.text(.5, .895, 'A8/A4 = Linear input activations; QKV uses the same activation bit width',
             ha='center', fontsize=12)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.5, .065), ncol=5, frameon=False)
    fig.text(.08, .025, 'Class-equal cosine distances | AMP-FP16 reference | Full test set | '
             'Five dataset-checkpoint pairs, one seed each', fontsize=11)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'quantized_distance_gap.{extension}', dpi=180)
    plt.close(fig)


def run(analysis, output):
    rows = load_rows(analysis)
    inputs = [analysis / 'analysis_snapshot.json', analysis / 'all_conditions.csv',
              analysis / 'fp_indicators.csv']
    inputs += [analysis / dataset / name for dataset in DATASETS
               for name in ('analysis_snapshot.json', 'summary.csv', 'layer_metrics.csv')]
    source_hashes = {str(path.resolve()): sha256(path) for path in inputs}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'final_class_distances.csv', rows)
    (output / 'REPORT.md').write_text(report(rows), encoding='utf-8')
    plot(rows, output)
    # Reload the saved table and independently check each field against validated inputs.
    saved = read_csv(output / 'final_class_distances.csv')
    expected = load_rows(analysis)
    if len(saved) != len(expected):
        raise ValueError('Saved table row count mismatch')
    for actual, target in zip(saved, expected):
        for key, value in target.items():
            if isinstance(value, (int, float)):
                valid = math.isclose(float(actual[key]), value, rel_tol=1e-12, abs_tol=1e-12)
            else:
                valid = actual[key] == ('' if value is None else value)
            if not valid:
                raise ValueError(f'Saved table mismatch: {target["dataset"]}/{target["condition"]}/{key}')
    if any(sha256(Path(path)) != digest for path, digest in source_hashes.items()):
        raise ValueError('Inputs changed during calculation')
    config = {
        'schema': 1, 'source_sha256': source_hashes,
        'code_sha256': {str(path.resolve()): sha256(path) for path in (
            Path(__file__), ROOT / 'geometry_robustness.py', ROOT / 'plot_presentation.py')},
        'reference': 'matched AMP-FP16 unquantized', 'feature': 'final normalized CLS, layer 12, 768D',
        'distance': 'class-equal mean pairwise cosine distance, ground-truth classes',
        'raw_arrays_revalidated': False,
        'verification': {'passed': True, 'saved_rows_checked': len(saved), 'quantized_rows': 100},
        'output_sha256': {name: sha256(output / name) for name in (
            'final_class_distances.csv', 'REPORT.md', 'quantized_distance_gap.png',
            'quantized_distance_gap.svg')},
    }
    (output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(saved)} rows (100 quantized + 5 FP): {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, default=ROOT / 'analysis')
    args = parser.parse_args()
    run(args.analysis_root, args.analysis_root / 'quantized_class_distance')


if __name__ == '__main__':
    main()
