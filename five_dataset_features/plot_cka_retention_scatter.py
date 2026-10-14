"""Slide-ready scatter of final CKA and accuracy retention at W3/G128.

Run from the repository root::

    python -m five_dataset_features.plot_cka_retention_scatter
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analysis_snapshot import sha256
from .plot_presentation import DATASETS, load_metrics


STAGES = (('', 'Weight-only', '#247A70', 'o'),
          ('_a8_qkv', 'Linear A8 + QKV A8', '#376EAD', '^'),
          ('_a4_qkv', 'Linear A4 + QKV A4', '#D1774E', 's'))
LABEL_OFFSETS = {
    ('', 'pets'): (-46, -20), ('', 'cub'): (10, -12),
    ('', 'cotton'): (-60, 14), ('', 'soyglobal'): (10, 2),
    ('', 'soyageing'): (-80, -16),
    ('_a4_qkv', 'pets'): (10, 9), ('_a4_qkv', 'cub'): (10, 8),
    ('_a4_qkv', 'cotton'): (10, -6), ('_a4_qkv', 'soyglobal'): (10, 8),
    ('_a4_qkv', 'soyageing'): (10, -15),
}


def plot(root, output):
    stem, summaries, layers = load_metrics(root, 3, 128)
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    ax = fig.add_axes((0.10, 0.25, 0.84, 0.53))
    ax.scatter([1], [1], s=170, marker='D', facecolor='white',
               edgecolor='#879D94', linewidth=2.5, label='FP reference', zorder=4)
    for suffix, label, color, marker in STAGES:
        x = [float(layers[dataset, stem + suffix][12]['cka']) for dataset in DATASETS]
        y = [float(summaries[dataset, stem + suffix]['accuracy']) /
             float(summaries[dataset, 'fp']['accuracy']) for dataset in DATASETS]
        if suffix == '_a8_qkv':
            ax.scatter(x, y, s=360, marker=marker, facecolor='none', edgecolor=color,
                       linewidth=2.8, label=label, zorder=5)
        else:
            ax.scatter(x, y, s=200, marker=marker, color=color, edgecolor='white',
                       linewidth=2, label=label, zorder=4)
        for dataset, cka, retention in zip(DATASETS, x, y):
            if suffix == '_a8_qkv':
                continue
            ax.annotate(dataset, (cka, retention), xytext=LABEL_OFFSETS[suffix, dataset],
                        textcoords='offset points', fontsize=12, weight='bold',
                        color='#263F38', zorder=5)

    ax.set_xlim(0, 1.06)
    ax.set_ylim(0, 1.07)
    ax.set_xticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.grid(color='#DCE7E1', linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel('Final CLS CKA(FP, quantized)', fontsize=17,
                  color='#20453E', labelpad=17)
    ax.set_ylabel('Accuracy retention (quantized / FP)', fontsize=17,
                  color='#20453E', labelpad=17)
    ax.tick_params(axis='both', labelsize=12, length=0, pad=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#B9CBC2')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.865),
               ncol=4, frameon=False, fontsize=13)
    fig.text(0.075, 0.945, 'CKA vs accuracy retention', fontsize=28,
             weight='bold', color='#173A35')
    fig.text(0.075, 0.895,
             'Five dataset–checkpoint pairs  |  fixed W3/G128  |  full test sets',
             fontsize=14, color='#4A625C')
    fig.text(0.075, 0.10,
             'Upper right means closer FP features and higher retained accuracy; '
             'the two measures need not rank datasets alike.',
             fontsize=14, weight='bold', color='#173A35')
    fig.text(0.075, 0.055,
             'FP = AMP-FP16. Each dataset uses one fine-tuned checkpoint and seed; '
             'quantized conditions are paired within dataset.',
             fontsize=10.5, color='#61736D')

    output.mkdir(parents=True, exist_ok=True)
    name = f'{stem}_cka_retention_scatter'
    fig.savefig(output / f'{name}.png', dpi=200, facecolor='white')
    fig.savefig(output / f'{name}.svg', facecolor='white')
    plt.close(fig)

    analysis = Path(root) / 'analysis'
    files = [analysis / 'analysis_snapshot.json']
    files += [analysis / dataset / filename for dataset in DATASETS
              for filename in ('summary.csv', 'layer_metrics.csv')]
    source = {'condition_stem': stem,
              'conditions': [stem + suffix for suffix, _, _, _ in STAGES],
              'input_sha256': {str(path.relative_to(root)): sha256(path) for path in files},
              'plotter_sha256': sha256(Path(__file__))}
    (output / f'{name}_sources.json').write_text(
        json.dumps(source, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote slide scatter (PNG and SVG) to {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root / 'analysis' / 'slides_w3_g128'
    plot(args.root, output)


if __name__ == '__main__':
    main()
