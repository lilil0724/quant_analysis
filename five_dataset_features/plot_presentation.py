"""Slide-ready, fixed W/G CKA figures from the saved five-dataset analysis.

Run from the parent of ``five_dataset_features``::

    python -m five_dataset_features.plot_presentation --root five_dataset_features

This reads only the existing analysis CSVs. It checks their recorded snapshot
hashes, but cannot revalidate inference files that are not present locally.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from .analysis_snapshot import sha256


DATASETS = ('cotton', 'soyageing', 'soyglobal', 'cub', 'pets')
COLORS = {
    'cotton': '#009E73',
    'soyageing': '#D55E00',
    'soyglobal': '#0072B2',
    'cub': '#B34E8A',
    'pets': '#9A6700',
}
STAGES = (('', 'Weight only'), ('_a8', '+ Linear A8'), ('_a4', '+ Linear A4'))
QKV_STAGES = (('', 'Weight only'),
              ('_a8_qkv', 'Linear A8 + QKV A8'),
              ('_a4_qkv', 'Linear A4 + QKV A4'))


def _read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _require_snapshot_hash(path, snapshot, key):
    recorded = json.loads(snapshot.read_text(encoding='utf-8'))['derived_files']
    if recorded.get(key) != sha256(path):
        raise ValueError(f'{path}: missing or mismatched snapshot hash')


def load_metrics(root, weight_bits=3, group_size=128):
    """Load exact fixed-condition rows and reject incomplete/mixed outputs."""
    root = Path(root)
    analysis = root / 'analysis'
    stem = f'w{weight_bits}_g{group_size}'
    conditions = tuple(dict.fromkeys(stem + suffix
                                     for suffix, _ in (*STAGES, *QKV_STAGES)))
    report_snapshot = json.loads((analysis / 'analysis_snapshot.json').read_text(encoding='utf-8'))
    code_dir = Path(__file__).resolve().parent
    if any(sha256(code_dir / name) != digest
           for name, digest in report_snapshot['analysis_code'].items()):
        raise ValueError('Analysis code differs from the recorded report snapshot')
    _require_snapshot_hash(analysis / 'all_conditions.csv',
                           analysis / 'analysis_snapshot.json', 'all_conditions.csv')
    aggregate = {(row['dataset'], row['condition']): row
                 for row in _read_csv(analysis / 'all_conditions.csv')}
    if len(aggregate) != len(DATASETS) * 21:
        raise ValueError('Aggregate analysis does not contain all 105 conditions')

    summaries, layers = {}, {}
    for dataset in DATASETS:
        folder = analysis / dataset
        snapshot = folder / 'analysis_snapshot.json'
        if report_snapshot['datasets'].get(dataset) != sha256(snapshot):
            raise ValueError(f'{dataset}: analysis snapshot differs from aggregate report')
        for name in ('summary.csv', 'layer_metrics.csv'):
            _require_snapshot_hash(folder / name, snapshot, name)
        summary_rows = _read_csv(folder / 'summary.csv')
        layer_rows = _read_csv(folder / 'layer_metrics.csv')
        by_condition = {row['condition']: row for row in summary_rows}
        if len(by_condition) != 21 or len(summary_rows) != 21:
            raise ValueError(f'{dataset}: expected 21 unique summary conditions')
        if any(row['dataset'] != dataset for row in summary_rows + layer_rows):
            raise ValueError(f'{dataset}: dataset identity mismatch')
        for condition in ('fp', *conditions):
            row = by_condition.get(condition)
            if row is None or row != aggregate.get((dataset, condition)):
                raise ValueError(f'{dataset}/{condition}: missing or mismatched summary')
            summaries[dataset, condition] = row
            selected = [item for item in layer_rows if item['condition'] == condition]
            by_layer = {int(item['layer']): item for item in selected}
            if len(selected) != 13 or set(by_layer) != set(range(13)):
                raise ValueError(f'{dataset}/{condition}: expected 13 unique layers')
            layers[dataset, condition] = by_layer
            if condition != 'fp':
                cka = [float(by_layer[layer]['cka']) for layer in range(13)]
                if any(not math.isfinite(value) or not -1e-6 <= value <= 1.000001
                       for value in cka):
                    raise ValueError(f'{dataset}/{condition}: invalid CKA')
                fp_accuracy = float(row['fp_accuracy'])
                accuracy = float(row['accuracy'])
                relative_loss = float(row['relative_loss'])
                if fp_accuracy <= 0 or not math.isclose(
                        relative_loss, 1 - accuracy / fp_accuracy, abs_tol=1e-9):
                    raise ValueError(f'{dataset}/{condition}: inconsistent relative loss')
    return stem, summaries, layers


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.facecolor': 'white',
    })
    return plt


def _save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', dpi=200, facecolor='white')
    fig.savefig(output / f'{name}.svg', facecolor='white')


def plot_layer_profiles(output, stem, layers, *, qkv=False):
    plt = _style()
    stages = QKV_STAGES if qkv else STAGES
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.30, wspace=0.13)
    for ax, (suffix, title) in zip(axes, stages):
        for dataset in DATASETS:
            values = [float(layers[dataset, stem + suffix][layer]['cka'])
                      for layer in range(13)]
            ax.plot(range(13), values, marker='o', markersize=4.5, linewidth=2.5,
                    color=COLORS[dataset], label=dataset)
        ax.set_title(title, pad=14, weight='bold')
        ax.set_xlim(-0.3, 12.3)
        ax.set_ylim(0, 1.04)
        ax.set_xticks(range(13), [*map(str, range(1, 13)), 'F'], fontsize=10)
        ax.grid(axis='y', color='#D8DEE7', linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Centered linear CKA versus FP')
    title = (f'{stem.upper().replace("_", " / ")}: layerwise CKA with QKV'
             if qkv else
             f'Where does {stem.upper().replace("_", " / ")} change the representation?')
    fig.suptitle(title,
                 x=0.07, y=0.94, ha='left', fontsize=22, weight='bold')
    fig.text(0.07, 0.875,
             'Same test images and checkpoint within each dataset; five dataset–checkpoint pairs',
             fontsize=13, color='#465467')
    fig.supxlabel('Transformer block output  /  F = final normalized CLS',
                  y=0.215, fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.133),
               frameon=False, fontsize=13)
    activation_note = ('A8/A4 apply to both Linear and QKV activations.' if qkv else
                       'A8/A4 apply to Linear activations only.')
    fig.text(0.07, 0.048,
             f'FP = AMP-FP16. {activation_note} Fake quantization; '
             'dataset sample sizes and class counts differ.',
             fontsize=10.5, color='#596677')
    _save(fig, output, f'{stem}_layer_cka_qkv' if qkv else f'{stem}_layer_cka')
    plt.close(fig)


def plot_cka_and_outcomes(output, stem, summaries, layers):
    plt = _style()
    condition = stem + '_a4'
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes((0.075, 0.23, 0.43, 0.51))
    ax.set_xlim(0, 0.83)
    losses = [float(summaries[dataset, condition]['relative_loss']) * 100
              for dataset in DATASETS]
    ax.set_ylim(max(-5, min(losses) - 8), min(105, max(losses) + 8))
    ax.set_xlabel('Final CLS CKA versus FP')
    ax.set_ylabel('Relative accuracy loss (%)')
    ax.grid(color='#E0E5EC', linewidth=0.8)
    ax.set_axisbelow(True)
    offsets = {
        'cotton': (8, -13), 'soyageing': (-83, 9), 'soyglobal': (-80, -16),
        'cub': (9, -2), 'pets': (8, -2),
    }
    table_rows = []
    for dataset in DATASETS:
        row = summaries[dataset, condition]
        fp = summaries[dataset, 'fp']
        final = layers[dataset, condition][12]
        fp_final = layers[dataset, 'fp'][12]
        cka = float(final['cka'])
        loss = float(row['relative_loss']) * 100
        ax.scatter(cka, loss, s=170, color=COLORS[dataset], edgecolor='white',
                   linewidth=1.6, zorder=3)
        ax.annotate(dataset, (cka, loss), xytext=offsets[dataset],
                    textcoords='offset points', fontsize=11, color=COLORS[dataset],
                    weight='bold')
        table_rows.append([
            dataset,
            f'{float(fp["accuracy"])*100:.1f} -> {float(row["accuracy"])*100:.1f}',
            f'{cka:.3f}',
            f'{float(fp_final["separability"]):.3f} -> {float(final["separability"]):.3f}',
            f'{float(row["swap_rate"])*100:.1f}',
            f'{int(row["damage_count"]):,} / {int(row["rescue_count"]):,}',
        ])

    table_ax = fig.add_axes((0.53, 0.225, 0.44, 0.51))
    table_ax.axis('off')
    table_ax.set_title('Classification diagnostics', loc='left', pad=14,
                       fontsize=17, weight='bold')
    table = table_ax.table(
        cellText=table_rows,
        colLabels=['Dataset', 'FP -> A4\nacc. %', 'Final\nCKA',
                   'Final separation\nFP -> A4', 'Swap\n%', 'Damage /\nrescue'],
        colWidths=[0.17, 0.17, 0.12, 0.25, 0.10, 0.19],
        cellLoc='center', bbox=[0, 0.03, 1, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor('#DCE2EA')
        cell.set_linewidth(0.65)
        if row_index == 0:
            cell.set_facecolor('#203A56')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#F4F7FA' if row_index % 2 else 'white')
            if col_index == 0:
                cell.get_text().set_color(COLORS[DATASETS[row_index - 1]])
                cell.get_text().set_weight('bold')

    fig.suptitle(f'{stem.upper().replace("_", " / ")} + Linear A4: CKA and accuracy loss',
                 x=0.075, y=0.94, ha='left', fontsize=22, weight='bold')
    fig.text(0.075, 0.875,
             'Five dataset–checkpoint pairs; one fixed quantization condition per dataset',
             fontsize=13, color='#465467')
    fig.text(0.075, 0.135,
             'Relative loss = 100 x (1 - A4 accuracy / FP accuracy).  '
             'Separation uses class-equal cosine distances.',
             fontsize=10.5, color='#596677')
    fig.text(0.075, 0.092,
             'FP = AMP-FP16; damage = FP correct to A4 wrong, rescue = FP wrong to A4 correct. '
             'Counts reflect different test-set sizes.',
             fontsize=10.5, color='#596677')
    fig.text(0.075, 0.049,
             'Descriptive association only: datasets use different checkpoints, class counts, and sample sizes.',
             fontsize=10.5, color='#596677')
    _save(fig, output, f'{stem}_a4_cka_vs_loss')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Five-dataset input root containing analysis/')
    parser.add_argument('--output-dir', type=Path, help='Defaults to ROOT/analysis/slides_w3_g128')
    parser.add_argument('--weight-bits', type=int, default=3)
    parser.add_argument('--group-size', type=int, default=128)
    args = parser.parse_args()
    stem, summaries, layers = load_metrics(args.root, args.weight_bits, args.group_size)
    output = args.output_dir or args.root / 'analysis' / f'slides_{stem}'
    plot_layer_profiles(output, stem, layers)
    plot_layer_profiles(output, stem, layers, qkv=True)
    plot_cka_and_outcomes(output, stem, summaries, layers)
    print(f'Wrote three slide figures (PNG and SVG) to {output.resolve()}')


if __name__ == '__main__':
    main()
