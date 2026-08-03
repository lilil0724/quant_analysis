"""Generate all PNG figures for the formal HQQ accuracy analysis."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORR_DIR = os.path.join(BASE_DIR, 'results_all', 'quant', 'corr')
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, 'results_all', 'quant', 'plots')
DEFAULT_OUTPUT = os.path.join(DEFAULT_RESULTS_DIR, 'quant_accuracy_overview.png')


def parse_args():
    parser = argparse.ArgumentParser(description='Create HQQ accuracy PNG figures.')
    parser.add_argument(
        '--input-file', default=os.path.join(DEFAULT_CORR_DIR, 'prepared_accuracy_cells.csv'),
        help='full path of prepared_accuracy_cells.csv produced by corr_quant.py')
    parser.add_argument(
        '--output-file', default=DEFAULT_OUTPUT,
        help='full path of the primary overview PNG')
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help='directory for all generated PNGs and run_config.json')
    parser.add_argument('--corr-dir', default=DEFAULT_CORR_DIR)
    parser.add_argument(
        '--ablation-file',
        help='full path of factor_ablation_accuracy.csv; defaults to <corr-dir>/factor_ablation_accuracy.csv')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def write_run_config(path, args):
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
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2)


def save_figure(figure, path, dpi):
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def heatmaps(cells, results_dir, dpi):
    for checkpoint in sorted(cells['ckpt_kind'].unique()):
        pivot = cells[cells['ckpt_kind'] == checkpoint].pivot_table(
            index='hqq_group_size', columns='hqq_nbits',
            values='accuracy_ratio', aggfunc='median')
        figure, axis = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1, ax=axis)
        axis.set(
            title=f'Median accuracy ratio: {checkpoint} checkpoints',
            xlabel='HQQ nbits', ylabel='HQQ group size')
        save_figure(figure, os.path.join(results_dir, f'accuracy_heatmap_{checkpoint}.png'), dpi)

    for column, prefix, label in [
            ('model_name', 'accuracy_heatmap_model', 'model'),
            ('dataset_name', 'accuracy_heatmap_dataset', 'dataset')]:
        for value in sorted(cells[column].unique()):
            pivot = cells[cells[column] == value].pivot_table(
                index='hqq_group_size', columns='hqq_nbits',
                values='accuracy_ratio', aggfunc='median')
            figure, axis = plt.subplots(figsize=(7, 4))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1, ax=axis)
            axis.set(
                title=f'Median accuracy ratio: {label} {value}',
                xlabel='HQQ nbits', ylabel='HQQ group size')
            save_figure(figure, os.path.join(results_dir, f'{prefix}_{value}.png'), dpi)


def interaction_lines(cells, grouping, title, path, dpi):
    figure, axis = plt.subplots(figsize=(7, 4))
    for value, frame in cells.groupby(grouping):
        summary = frame.groupby('hqq_nbits')['accuracy_ratio'].median().sort_index()
        axis.plot(summary.index, summary.values, marker='o', label=str(value))
    axis.axhline(1, color='black', linewidth=.7)
    axis.set(title=title, xlabel='HQQ nbits', ylabel='Accuracy ratio')
    axis.legend(title=grouping)
    save_figure(figure, path, dpi)


def factor_contribution(ablation, path, dpi):
    figure, axis = plt.subplots(figsize=(7, 4))
    ordered = ablation.sort_values('delta_r2')
    axis.barh(ordered['term_group'], ordered['delta_r2'])
    axis.set(
        title='Descriptive factor contribution: accuracy ratio',
        xlabel='Ablation delta R2')
    save_figure(figure, path, dpi)


def checkpoint_accuracy_lines(cells, results_dir, dpi):
    for checkpoint, frame in cells.groupby('ckpt_kind'):
        datasets = sorted(frame['dataset_name'].unique())
        models = sorted(frame['model_name'].unique())
        figure, axes = plt.subplots(
            len(datasets), len(models), squeeze=False,
            figsize=(4 * len(models), 3 * len(datasets)), sharex=True)
        for row, dataset in enumerate(datasets):
            for column, model in enumerate(models):
                axis = axes[row, column]
                panel = frame[
                    (frame['dataset_name'] == dataset)
                    & (frame['model_name'] == model)]
                for group_size, line in panel.groupby('hqq_group_size'):
                    line = line.groupby('hqq_nbits', as_index=False)['hqq_top1'].mean().sort_values('hqq_nbits')
                    axis.plot(line['hqq_nbits'], line['hqq_top1'], marker='o', label=f'g={group_size}')
                baseline = panel['fp32_top1'].dropna().median()
                if pd.notna(baseline):
                    axis.axhline(.9 * baseline, color='#CC3311', linestyle='--', linewidth=1, label='90% FP32')
                axis.set_title(f'{dataset} | {model}', fontsize=9)
                if row == len(datasets) - 1:
                    axis.set_xlabel('HQQ nbits')
                if column == 0:
                    axis.set_ylabel('Top-1 accuracy')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), frameon=False)
        figure.suptitle(f'HQQ Top-1 accuracy by bit width and group size ({checkpoint} checkpoints)')
        figure.tight_layout(rect=(0, .06, 1, .96))
        save_figure(figure, os.path.join(results_dir, f'quant_acc_by_bits_group_{checkpoint}.png'), dpi)


def overview(cells, ablation, output_file, dpi):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    nbits = cells.groupby('hqq_nbits')['accuracy_ratio'].median().sort_index()
    axes[0, 0].plot(nbits.index, nbits.values, marker='o')
    axes[0, 0].set(title='Accuracy retention by bit width', xlabel='HQQ nbits', ylabel='Median ratio')
    pivot = cells.pivot_table(index='hqq_group_size', columns='hqq_nbits', values='accuracy_ratio', aggfunc='median')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title('Nbits x group size')
    checkpoint = cells.groupby('ckpt_kind')['accuracy_ratio'].median().sort_values()
    axes[1, 0].barh(checkpoint.index.astype(str), checkpoint.values)
    axes[1, 0].set(title='Checkpoint relationship', xlabel='Median ratio')
    ordered = ablation.sort_values('delta_r2')
    axes[1, 1].barh(ordered['term_group'], ordered['delta_r2'])
    axes[1, 1].set(title='Factor contribution', xlabel='Ablation delta R2')
    figure.tight_layout()
    save_figure(figure, output_file, dpi)


def main():
    args = parse_args()
    args.results_dir = os.path.abspath(args.results_dir)
    args.output_file = os.path.abspath(args.output_file)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_run_config(os.path.join(args.results_dir, 'run_config.json'), args)

    cells = pd.read_csv(args.input_file)
    ablation_file = args.ablation_file or os.path.join(args.corr_dir, 'factor_ablation_accuracy.csv')
    ablation = pd.read_csv(ablation_file)

    sns.set_theme(style='whitegrid')
    heatmaps(cells, args.results_dir, args.dpi)
    interaction_lines(
        cells, 'hqq_group_size', 'Accuracy ratio by nbits and group size',
        os.path.join(args.results_dir, 'interaction_nbits_group_accuracy.png'), args.dpi)
    interaction_lines(
        cells, 'ckpt_kind', 'Accuracy ratio by nbits and checkpoint',
        os.path.join(args.results_dir, 'interaction_nbits_context_accuracy.png'), args.dpi)
    factor_contribution(
        ablation, os.path.join(args.results_dir, 'factor_contribution_accuracy.png'), args.dpi)
    checkpoint_accuracy_lines(cells, args.results_dir, args.dpi)
    overview(cells, ablation, args.output_file, args.dpi)
    print(f'Wrote PNG figures to {args.results_dir}')
    print(f'Primary overview: {args.output_file}')


if __name__ == '__main__':
    main()
