"""Draw the paper's figures from the build/ tables into figures/."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from registry import (BITS, BUILD, DATASET_LABELS, GRID_DATASETS, GROUP_LABELS, GROUP_SIZES, INPUTS, MODEL_KEYS,
                      MODEL_LABELS, MODE_LABELS, MODELS, MODES, PAPER, SETTINGS, SOY_COTTON, SEED)

FIGURES = PAPER / 'figures'
COLUMN, FULL = 3.3, 6.9
COLORS = {'other': '#3494BA', 'sc': '#D9622B'}
MODE_COLORS = {'ft': '#1b6ca8', 'fz': '#7a7a7a', 'cal': '#c2185b'}
MODEL_COLORS = {'vit': '#1b6ca8', 'swin': '#e08214', 'beit': '#5e3c99'}
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'legend.fontsize': 6,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6, 'font.family': 'serif',
                     'pdf.fonttype': 42, 'svg.hashsalt': 'paper', 'axes.linewidth': 0.6})
BIT_LABELS = ['1', '1.58', '2', '3', '4', '5', '6', '8']


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02, metadata={'CreationDate': None})
    plt.close(fig)


def retention_curves():
    table = pd.read_csv(BUILD / 'grid_datasets_g128.csv', index_col=0)
    table.columns = table.columns.astype(float)
    x = np.arange(len(BITS))
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    for dataset in GRID_DATASETS:
        group = 'sc' if dataset in SOY_COTTON else 'other'
        ax.plot(x, table.loc[dataset, BITS], color=COLORS[group], lw=0.7, alpha=0.55)
    for group in ['other', 'sc']:
        members = [d for d in GRID_DATASETS if (d in SOY_COTTON) == (group == 'sc')]
        ax.plot(x, table.loc[members, BITS].mean(), color=COLORS[group], lw=2,
                label=f'{GROUP_LABELS[group]} ({len(members)} datasets)')
    ax.set_xticks(x, BIT_LABELS)
    ax.set_xlabel('Weight bit width (group size 128)')
    ax.set_ylabel('Accuracy retention $a_Q/a_{FP}$')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc='lower right', frameon=False)
    save(fig, 'retention_curves')


def retention_bits():
    groups = pd.read_csv(BUILD / 'grid_groups.csv')
    x = np.arange(len(BITS))
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.6), sharey=True)
    for ax, gs in zip(axes, [128, 8]):
        cells = groups[groups.gs == gs]
        for group in ['other', 'sc']:
            per_bit = cells.groupby('bits')[group].agg(['median', 'min', 'max']).reindex(BITS)
            ax.fill_between(x, per_bit['min'], per_bit['max'], color=COLORS[group], alpha=0.2, lw=0)
            ax.plot(x, per_bit['median'], 'o-', color=COLORS[group], ms=2, lw=1.2, label=GROUP_LABELS[group])
        ax.set_title(f'Group size {gs}')
        ax.set_xticks(x, BIT_LABELS)
        ax.set_xlabel('Weight bit width')
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel('Group retention')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='upper left', frameon=False)
    save(fig, 'retention_bits')


def gap_heatmaps():
    groups = pd.read_csv(BUILD / 'grid_groups.csv')
    fig, axes = plt.subplots(3, 3, figsize=(FULL, 4.6), sharex=True, sharey=True)
    for ax, (model, mode) in zip(axes.flat, SETTINGS):
        cells = groups[(groups.model == model) & (groups['mode'] == mode)]
        matrix = cells.pivot(index='bits', columns='gs', values='gap').reindex(index=BITS, columns=GROUP_SIZES)
        image = ax.imshow(matrix, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
        for (i, j), value in np.ndenumerate(matrix.to_numpy()):
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=5,
                    color='white' if abs(value) > 0.4 else 'black')
        ax.set_title(f'{MODEL_LABELS[model]} {MODE_LABELS[mode]}')
        ax.set_xticks(range(len(GROUP_SIZES)), GROUP_SIZES)
        ax.set_yticks(range(len(BITS)), BIT_LABELS)
    for ax in axes[-1]:
        ax.set_xlabel('Group size')
    for ax in axes[:, 0]:
        ax.set_ylabel('Bit width')
    bar = fig.colorbar(image, ax=axes, shrink=0.6, pad=0.02)
    bar.set_label('Retention gap (FG $-$ UFG)')
    save(fig, 'gap_heatmaps')


def ranking_agreement():
    agreement = pd.read_csv(BUILD / 'difficulty_agreement.csv', index_col=[0, 1], header=[0, 1])
    labels = [f'{MODEL_LABELS[m]} {MODE_LABELS[k]}' for m, k in SETTINGS]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.6))
    image = ax.imshow(agreement.to_numpy(), cmap='viridis', vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(agreement.to_numpy()):
        if i != j:
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=5,
                    color='black' if value > 0.6 else 'white')
    ax.set_xticks(range(9), labels, rotation=60, ha='right')
    ax.set_yticks(range(9), labels)
    fig.colorbar(image, ax=ax, shrink=0.8, label='Spearman $\\rho$ of difficulty scores')
    save(fig, 'ranking_agreement')


def layer_cka():
    curves = pd.read_csv(BUILD / 'layer_cka.csv')
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        for mode in MODES:
            for group, style in [('other', '-'), ('sc', '--')]:
                line = curves[(curves.model == model) & (curves['mode'] == mode) & (curves.group == group)]
                ax.plot(line.depth, line.cka_vs_fp, style, color=MODE_COLORS[mode], lw=1,
                        label=f'{MODE_LABELS[mode]}, {GROUP_LABELS[group]}')
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel('Relative depth (block / final)')
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel('Median CKA, FP vs. W3/G128')
    axes[0].set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=6, frameon=False)
    save(fig, 'layer_cka')


def margin_distribution():
    balance = pd.read_csv(INPUTS / 'margin_balance.csv')
    balance['model'] = balance.model_name.map(MODEL_KEYS)
    cells = balance[(balance.model == 'vit') & (balance.checkpoint_mode == 'ft') & (balance.bin_width == 0.05)]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.6))
    for group, key in [('other', 'other'), ('sc', 'soy_cotton')]:
        part = cells[cells.group == key]
        ax.step(part.bin_low, part.raw_fraction, where='post', color=COLORS[group], lw=0.9,
                label=f'{GROUP_LABELS[group]}, raw')
    target = cells[cells.group == 'other']
    ax.fill_between(target.bin_low, target.target_fraction, step='post', color='0.6', alpha=0.45,
                    label='Matched target (both groups)')
    ax.set_xlim(0, 8)
    ax.set_xlabel('Baseline normalized decision margin $m(x)$ (bin width 0.05; clipped at 8)')
    ax.set_ylabel('Fraction of mass')
    ax.legend(frameon=False)
    save(fig, 'margin_distribution')


def swap_by_decile_placeholder():
    """Placeholder: per-sample quantized outcomes are not exported, so these values are synthetic."""
    deciles = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(COLUMN, 1.6))
    for group, start in [('other', 0.55), ('sc', 0.75)]:
        ax.plot(deciles, start * np.exp(-0.28 * (deciles - 1)), 'o-', color=COLORS[group], ms=2.5, lw=1,
                label=GROUP_LABELS[group])
    ax.set_xticks(deciles)
    ax.set_xlabel('Baseline margin decile (1 = lowest)')
    ax.set_ylabel('Swap rate at W3/G128')
    ax.legend(frameon=False)
    ax.text(0.5, 0.5, 'PLACEHOLDER', transform=ax.transAxes, fontsize=26, color='red', alpha=0.3,
            ha='center', va='center', rotation=20)
    save(fig, 'swap_by_decile_placeholder')


def tsne_example():
    coordinates = pd.read_csv(INPUTS / 'tsne_cub_vit_b16_ft.csv')
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.7), sharex=True, sharey=True)
    for ax, (prefix, title) in zip(axes, [('fp', 'FP baseline'), ('w3', 'W3/G128')]):
        ax.scatter(coordinates[f'{prefix}_x'], coordinates[f'{prefix}_y'], c=coordinates.target, cmap='tab20',
                   s=1.5, lw=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, 'tsne_cub_vit_ft')


def retention_box():
    cells = pd.read_csv(BUILD / 'accuracy_retention.csv')
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(COLUMN, 1.7))
    for i, group in enumerate(['sc', 'other']):
        values = cells[cells.group == group].retention.to_numpy()
        ax.boxplot(values, positions=[i], widths=0.5, orientation='horizontal', showfliers=False,
                   medianprops={'color': COLORS[group], 'lw': 1.2}, boxprops={'lw': 0.7},
                   whiskerprops={'lw': 0.7}, capprops={'lw': 0.7})
        ax.scatter(values, i + rng.uniform(-0.18, 0.18, len(values)), s=5, color=COLORS[group], alpha=0.6, lw=0)
    ax.set_yticks([0, 1], [f"{GROUP_LABELS[g]} ({(cells.group == g).sum()} contexts)" for g in ['sc', 'other']])
    ax.set_xlabel('Accuracy retention $a_Q/a_{FP}$ at W3/G128')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3, lw=0.4)
    save(fig, 'retention_box')


def accuracy_retention():
    from compute import PAIRS
    cells = pd.read_csv(BUILD / 'accuracy_retention.csv')
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    for model in MODELS:
        for group, marker in [('other', 'o'), ('sc', '^')]:
            part = cells[(cells.model == model) & (cells.group == group)]
            ax.scatter(part.fp32_top1, part.retention, s=7, marker=marker, color=MODEL_COLORS[model], alpha=0.7,
                       lw=0, label=f'{MODEL_LABELS[model]}, {GROUP_LABELS[group]}')
    for tag, (model, mode, low, high) in PAIRS.items():
        pair = cells[(cells.model == model) & (cells['mode'] == mode) & cells.dataset.isin([low, high])]
        pair = pair.set_index('dataset').loc[[low, high]]
        ax.plot(pair.fp32_top1, pair.retention, 'k-', lw=0.7)
        ax.scatter(pair.fp32_top1, pair.retention, s=14, facecolors='none', edgecolors='k', lw=0.7)
        ax.annotate(f'{MODEL_LABELS[model]} {MODE_LABELS[mode]}: {DATASET_LABELS[low]} vs. {DATASET_LABELS[high]}',
                    (pair.fp32_top1.iloc[0], pair.retention.iloc[0]), xytext=(-4, -2), textcoords='offset points',
                    ha='right', va='top', fontsize=5.5)
    ax.set_xlabel('Baseline top-1 (%)')
    ax.set_ylabel('Retention at W3/G128')
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    save(fig, 'accuracy_retention')


def main():
    FIGURES.mkdir(exist_ok=True)
    np.random.seed(SEED)
    retention_box()
    retention_curves()
    retention_bits()
    gap_heatmaps()
    ranking_agreement()
    layer_cka()
    margin_distribution()
    swap_by_decile_placeholder()
    tsne_example()
    accuracy_retention()


if __name__ == '__main__':
    main()
