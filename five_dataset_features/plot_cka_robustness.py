"""Slide-ready comparison of FP-referenced CKA and accuracy retention.

Run from the repository root::

    python -m five_dataset_features.plot_cka_robustness
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .analysis_snapshot import sha256
from .plot_presentation import DATASETS, load_metrics


STAGES = (('', 'Weight-only', '#247A70'),
          ('_a4_qkv', 'Linear A4 + QKV A4', '#D1774E'))


def plot(root, output):
    stem, summaries, layers = load_metrics(root, 3, 128)
    datasets = sorted(DATASETS, key=lambda name: -float(
        summaries[name, stem]['accuracy']) / float(summaries[name, 'fp']['accuracy']))
    cka = {suffix: [float(layers[dataset, stem + suffix][12]['cka'])
                    for dataset in datasets] for suffix, _, _ in STAGES}
    retained = {suffix: [float(summaries[dataset, stem + suffix]['accuracy']) /
                         float(summaries[dataset, 'fp']['accuracy'])
                         for dataset in datasets] for suffix, _, _ in STAGES}

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    axes = (fig.add_axes((0.08, 0.27, 0.4, 0.45)),
            fig.add_axes((0.57, 0.27, 0.4, 0.45)))
    for ax, values, title, ylabel in (
            (axes[0], cka, 'Representation similarity', 'CKA(FP, quantized)'),
            (axes[1], retained, 'Accuracy retention', 'Quantized / FP accuracy')):
        ax.axhline(1, color='#9BAAA5', linestyle=(0, (5, 4)), linewidth=2,
                   label='FP reference', zorder=1)
        for suffix, label, color in STAGES:
            ax.plot(range(len(datasets)), values[suffix], color=color,
                    linewidth=3.5, marker='o', markersize=10, label=label, zorder=3)
        ax.set_xlim(-0.15, len(datasets) - 0.85)
        ax.set_ylim(0, 1.06)
        ax.set_xticks(range(len(datasets)), datasets, fontsize=12)
        ax.set_yticks((0, 0.25, 0.5, 0.75, 1), ('0', '.25', '.50', '.75', '1.0'),
                      fontsize=11)
        ax.set_title(title, fontsize=19, weight='bold', color='#173A35', pad=22)
        ax.set_ylabel(ylabel, fontsize=13, color='#2E4D46', labelpad=12)
        ax.grid(axis='y', color='#DCE7E1', linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0, pad=10)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#C2D0C9')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.845),
               ncol=3, frameon=False, fontsize=14, handlelength=2.8)
    fig.text(0.075, 0.945, 'CKA vs quantization robustness',
             fontsize=28, weight='bold', color='#173A35')
    fig.text(0.075, 0.892,
             'Five dataset–checkpoint pairs  |  fixed W3/G128  |  full test sets',
             fontsize=14, color='#4A625C')
    fig.text(0.075, 0.13,
             'Pets retains the most accuracy in both settings; CUB retains 96% in weight-only despite lower CKA.',
             fontsize=15, weight='bold', color='#173A35')
    fig.text(0.075, 0.075,
             'FP = AMP-FP16. Lines connect separate dataset–checkpoint pairs to guide the eye; '
             'each uses one checkpoint and seed. FP self-CKA = 1 for all.',
             fontsize=10.5, color='#61736D')

    output.mkdir(parents=True, exist_ok=True)
    name = f'{stem}_cka_robustness'
    fig.savefig(output / f'{name}.png', dpi=200, facecolor='white')
    fig.savefig(output / f'{name}.svg', facecolor='white')
    plt.close(fig)

    analysis = Path(root) / 'analysis'
    files = [analysis / 'analysis_snapshot.json']
    files += [analysis / dataset / filename for dataset in DATASETS
              for filename in ('summary.csv', 'layer_metrics.csv')]
    source = {'condition_stem': stem, 'conditions': [stem + suffix for suffix, _, _ in STAGES],
              'dataset_order': datasets,
              'input_sha256': {str(path.relative_to(root)): sha256(path) for path in files},
              'plotter_sha256': sha256(Path(__file__))}
    (output / f'{name}_sources.json').write_text(
        json.dumps(source, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote slide figure (PNG and SVG) to {output.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root / 'analysis' / 'slides_w3_g128'
    plot(args.root, output)


if __name__ == '__main__':
    main()
