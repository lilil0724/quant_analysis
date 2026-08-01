"""Plot pre-cleanup HQQ accuracy against quantization bit width.

Reads the raw WandB export because ``results_all/quant_old/quant.csv`` has
already aggregated away ``ckpt_path``. Each panel fixes (dataset, model);
colour identifies the checkpoint training mode and line style identifies the
HQQ group size.
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CKPT_KINDS = ('cal', 'ft', 'fz')
PLOT_CKPT_KINDS = ('ft', 'fz', 'cal')
CKPT_PATTERN = re.compile(r'/(cal|ft|fz)_ckpts/')
CKPT_COLORS = {
    'cal': '#0072B2',
    'ft': '#D55E00',
    'fz': '#009E73',
}
GROUP_STYLES = ('-', '--', '-.', ':')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot pre-cleanup HQQ Top-1 accuracy against nbits.')
    parser.add_argument(
        '--input-file', default=os.path.join('data', 'backbones_quant.csv'),
        help='raw CSV written by download_save_wandb_data_old.py')
    parser.add_argument(
        '--results-dir', default=os.path.join(BASE_DIR, 'results_all', 'quant_old', 'plots'),
        help='directory for the generated figure')
    parser.add_argument(
        '--output-name',
        help='output folder name under results_all')
    parser.add_argument(
        '--output-file', default='quant_acc_by_bits_group',
        help='output filename stem')
    parser.add_argument('--save-format', choices=('png', 'pdf'), default='png')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--subset-datasets', nargs='+')
    parser.add_argument('--subset-models', nargs='+')
    parser.add_argument('--fig-size', nargs=2, type=float, default=(15, 11))
    return parser.parse_args()


def checkpoint_kind(paths):
    kinds = paths.astype(str).str.extract(CKPT_PATTERN)[0]
    invalid = paths[kinds.isna()].unique()
    if len(invalid):
        examples = ', '.join(map(str, invalid[:3]))
        raise ValueError(
            'Could not classify ckpt_path as cal, ft, or fz from its directory: '
            f'{examples}')
    return kinds


def load_plot_df(args):
    df = pd.read_csv(args.input_file)
    required = {
        'dataset_name', 'model_name', 'ckpt_path', 'hqq', 'hqq_nbits', 'seed',
        'hqq_group_size', 'hqq_top1', 'serial', 'top1',
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f'{args.input_file} is not a pre-cleanup HQQ export; missing columns: '
            f'{", ".join(missing)}')

    hqq = df['hqq'].astype(str).str.lower().isin(('true', '1'))
    hqq_df = df.loc[hqq & df['hqq_top1'].notna()].copy()
    fp32_df = df.loc[(~hqq) & df['serial'].eq(4999) & df['top1'].notna()].copy()

    if args.subset_datasets:
        hqq_df = hqq_df[hqq_df['dataset_name'].isin(args.subset_datasets)]
        fp32_df = fp32_df[fp32_df['dataset_name'].isin(args.subset_datasets)]
    if args.subset_models:
        hqq_df = hqq_df[hqq_df['model_name'].isin(args.subset_models)]
        fp32_df = fp32_df[fp32_df['model_name'].isin(args.subset_models)]
    if hqq_df.empty:
        raise ValueError('No HQQ rows remain after filtering.')

    hqq_df['ckpt_kind'] = checkpoint_kind(hqq_df['ckpt_path'])
    fp32_df['ckpt_kind'] = checkpoint_kind(fp32_df['ckpt_path'])
    hqq_df['hqq_nbits'] = pd.to_numeric(hqq_df['hqq_nbits'], errors='raise')
    hqq_df['hqq_group_size'] = pd.to_numeric(hqq_df['hqq_group_size'], errors='raise')

    # One point per line and bit width, averaged over the repeated seed runs.
    keys = [
        'dataset_name', 'model_name', 'ckpt_kind', 'hqq_group_size',
        'hqq_nbits',
    ]
    hqq_agg = hqq_df.groupby(keys, as_index=False).agg(
        acc=('hqq_top1', 'mean'),
        seed_count=('seed', 'nunique'),
    )
    baseline_keys = ['dataset_name', 'model_name', 'ckpt_kind']
    baselines = fp32_df.groupby(baseline_keys, as_index=False).agg(
        fp32_acc=('top1', 'mean'),
    )
    expected = hqq_agg[baseline_keys].drop_duplicates()
    missing = expected.merge(baselines, on=baseline_keys, how='left')
    missing = missing[missing['fp32_acc'].isna()][baseline_keys]
    return hqq_agg, baselines, missing


def bit_label(value):
    return str(int(value)) if float(value).is_integer() else str(value)


def make_plot(args, df, baselines, ckpt_kind):
    df = df[df['ckpt_kind'] == ckpt_kind]
    baselines = baselines[baselines['ckpt_kind'] == ckpt_kind]
    datasets = sorted(df['dataset_name'].unique())
    models = sorted(df['model_name'].unique())
    group_sizes = sorted(df['hqq_group_size'].unique())
    nbits = sorted(df['hqq_nbits'].unique())
    group_styles = {
        group_size: GROUP_STYLES[i % len(GROUP_STYLES)]
        for i, group_size in enumerate(group_sizes)
    }

    sns.set_theme(style='whitegrid', context='notebook', font_scale=1.05)
    fig, axes = plt.subplots(
        len(datasets), len(models), squeeze=False,
        figsize=tuple(args.fig_size), sharex=True,
    )

    for row, dataset in enumerate(datasets):
        for col, model in enumerate(models):
            ax = axes[row, col]
            panel = df[(df['dataset_name'] == dataset) & (df['model_name'] == model)]
            for group_size in group_sizes:
                line = panel[
                    panel['hqq_group_size'] == group_size
                ].sort_values('hqq_nbits')
                if not line.empty:
                    ax.plot(
                        line['hqq_nbits'], line['acc'], marker='o',
                        color=CKPT_COLORS[ckpt_kind],
                        linestyle=group_styles[group_size], linewidth=1.8,
                        markersize=4,
                    )

            panel_baselines = baselines[
                (baselines['dataset_name'] == dataset)
                & (baselines['model_name'] == model)
            ].sort_values('fp32_acc')
            thresholds = [0.9 * value for value in panel_baselines['fp32_acc']]
            span = max(panel['acc'].max(), max(thresholds, default=0)) - min(
                panel['acc'].min(), min(thresholds, default=0))
            label_gap = max(1.5, 0.04 * span)
            label_y = []
            for threshold in thresholds:
                label_y.append(max(threshold, (label_y[-1] + label_gap) if label_y else threshold))

            for (_, baseline), threshold, text_y in zip(
                    panel_baselines.iterrows(), thresholds, label_y):
                ax.axhline(
                    threshold, color='#CC3311', linestyle='--', linewidth=1.2,
                    alpha=0.9, zorder=0,
                )
                ax.annotate(
                    '90% FP32',
                    xy=(nbits[-1], threshold), xytext=(nbits[-1] - 0.05, text_y),
                    textcoords='data', ha='right', va='center',
                    color='#CC3311', fontsize=7,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.7},
                    arrowprops={'arrowstyle': '-', 'color': '#CC3311', 'lw': 0.6},
                )

            if label_y:
                padding = max(2, 0.03 * span)
                ax.set_ylim(
                    max(0, min(panel['acc'].min(), min(thresholds)) - padding),
                    max(panel['acc'].max(), max(label_y)) + padding,
                )

            ax.set_title(f'{dataset} | {model}', fontsize=10)
            ax.set_xticks(nbits, [bit_label(bit) for bit in nbits])
            ax.grid(axis='y', linewidth=0.5)
            if col == 0:
                ax.set_ylabel('Top-1 Accuracy (%)')
            if row == len(datasets) - 1:
                ax.set_xlabel('HQQ weight bits')

    legend_handles = [
        Patch(color=CKPT_COLORS[ckpt_kind], label=f'checkpoint: {ckpt_kind}'),
    ] + [
        Line2D([0], [0], color='black', linestyle=group_styles[group_size],
               linewidth=1.8, label=f'group size: {bit_label(group_size)}')
        for group_size in group_sizes
    ] + [
        Line2D([0], [0], color='#CC3311', linestyle='--', linewidth=1.2,
               label='90% of 4999 FP32'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.01), frameon=False,
    )
    fig.suptitle(
        f'Pre-cleanup HQQ accuracy by bit width and group size ({ckpt_kind} checkpoints)',
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))

    os.makedirs(args.results_dir, exist_ok=True)
    output = os.path.join(
        args.results_dir, f'{args.output_file}_{ckpt_kind}.{args.save_format}')
    fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    return output


def main():
    args = parse_args()
    if args.output_name:
        args.results_dir = os.path.join(BASE_DIR, 'results_all', args.output_name, 'plots')
    os.makedirs(args.results_dir, exist_ok=True)
    config = vars(args).copy()
    config['started_at'] = datetime.now(timezone.utc).isoformat()
    try:
        config['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip()
        config['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        config['git_commit'] = None
        config['git_dirty'] = None
    with open(os.path.join(args.results_dir, 'run_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, default=str)
    df, baselines, missing = load_plot_df(args)
    for _, row in missing.iterrows():
        print(
            'Warning: no valid serial 4999 FP32 baseline for '
            f"{row['dataset_name']} | {row['model_name']} | {row['ckpt_kind']}; "
            'skipped its 90% reference line.')
    for ckpt_kind in PLOT_CKPT_KINDS:
        ckpt_df = df[df['ckpt_kind'] == ckpt_kind]
        if ckpt_df.empty:
            print(f'Warning: no HQQ rows for checkpoint kind {ckpt_kind}; skipped figure.')
            continue
        output = make_plot(args, df, baselines, ckpt_kind)
        print(
            f'Saved {output} with {len(ckpt_df)} averaged points across '
            f'{ckpt_df["dataset_name"].nunique()} datasets, '
            f'{ckpt_df["model_name"].nunique()} models, and '
            f'{ckpt_df["hqq_group_size"].nunique()} group sizes.')


if __name__ == '__main__':
    main()
