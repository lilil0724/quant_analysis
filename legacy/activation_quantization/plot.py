"""Accuracy curves and paired A6-reference loss atlases for serial 20008."""
import itertools
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .contracts import DATASETS, MODELS, MODES
from .grid import WEIGHTS, ACTIVATIONS
from .storage import sha256

KEYS = ['dataset', 'model', 'mode', 'w', 'a']
LABELS = dict(zip(MODELS, ['ViT-B/16', 'Swin-B', 'BEiTv2-B']))
COLORS = {3: '#2374ab', 4: '#d17a22', 5: '#22856a'}


def prepare_views(frame):
    """Validate full coverage and match each point to its own checkpoint/W A6."""
    expected = set(itertools.product(DATASETS, MODELS, MODES, WEIGHTS, ACTIVATIONS))
    if frame.duplicated(KEYS).any() or set(frame[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError('Plots require the complete unique 3024-cell serial-20008 grid')
    if not np.isfinite(frame.top1).all() or not frame.top1.between(0, 100).all():
        raise ValueError('Plot accuracy must be finite and between 0 and 100 percent')
    identity = ['checkpoint', 'seed', 'classifier']
    if (frame.groupby(['dataset', 'model', 'mode', 'w'])[identity].nunique(dropna=False) != 1).any().any():
        raise ValueError('Activation settings must share checkpoint, seed and classifier')
    pair_keys = ['dataset', 'model', 'mode', 'w', *identity]
    reference = frame[frame.a.eq(6)][pair_keys + ['top1', 'run_id']].rename(
        columns={'top1': 'a6_top1', 'run_id': 'a6_run_id'})
    losses = frame[KEYS + identity + ['top1', 'run_id']].merge(
        reference, on=pair_keys, how='left', validate='many_to_one')
    if losses.a6_top1.isna().any():
        raise ValueError('A6 reference is missing for an activation comparison')
    losses['loss_vs_a6_pp'] = losses.a6_top1 - losses.top1
    losses = losses.sort_values(KEYS).reset_index(drop=True)
    if not losses.loc[losses.a.eq(6), 'loss_vs_a6_pp'].eq(0).all():
        raise ValueError('A6 self-reference loss must be exactly zero')
    means = frame.groupby(['model', 'mode', 'w', 'a'], as_index=False).agg(
        mean_top1=('top1', 'mean'), n_datasets=('dataset', 'nunique'))
    if not means.n_datasets.eq(16).all():
        raise ValueError('Every curve mean must cover the same 16 datasets')
    return means, losses


def _format_axis(axis):
    axis.set_xticks(range(len(ACTIVATIONS)), ['1', '1.58', '2', '3', '4', '5', '6'])
    axis.tick_params(labelsize=10)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)


def _accuracy_curves(frame, means):
    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True, sharey=True,
                             layout='constrained')
    x = np.arange(len(ACTIVATIONS))
    for row, model in enumerate(MODELS):
        for col, mode in enumerate(MODES):
            axis = axes[row, col]
            selected = frame[frame.model.eq(model) & frame['mode'].eq(mode)]
            for weight in WEIGHTS:
                trajectories = selected[selected.w.eq(weight)].pivot(
                    index='dataset', columns='a', values='top1').reindex(index=DATASETS, columns=ACTIVATIONS)
                for values in trajectories.to_numpy():
                    axis.plot(x, values, color=COLORS[weight], alpha=.12, linewidth=.8, zorder=1)
                average = means[means.model.eq(model) & means['mode'].eq(mode) & means.w.eq(weight)]
                values = average.set_index('a').reindex(ACTIVATIONS).mean_top1
                axis.plot(x, values, color=COLORS[weight], linewidth=2.7, marker='o',
                          markersize=4.5, zorder=3)
            axis.set_title(f'{LABELS[model]} | {mode.upper()}', fontsize=13, pad=8)
            axis.set_ylim(0, 100)
            axis.set_yticks([0, 20, 40, 60, 80, 100])
            axis.grid(axis='y', color='#dce2e8', linewidth=.65, zorder=0)
            axis.tick_params(labelbottom=True)
            _format_axis(axis)
    handles = [Line2D([0], [0], color=COLORS[w], linewidth=2.7, marker='o', label=f'W{w}') for w in WEIGHTS]
    axes[0, 0].legend(handles=handles, loc='upper left', ncols=3, frameon=False, fontsize=10)
    fig.suptitle('Serial 20008 | Accuracy versus activation precision', fontsize=20, fontweight='bold')
    fig.supxlabel('Activation bits (A) — discrete evaluated settings', fontsize=12)
    fig.supylabel('Top-1 accuracy (%)', fontsize=12)
    fig.text(.5, -.025, 'Thin lines: individual datasets. Bold lines: equal-dataset mean (16 per panel/weight). '
             'All runs use quantized weights and activations.', ha='center', fontsize=10, color='#444444')
    return fig


def _loss_heatmap(losses, weight, span):
    fig, axes = plt.subplots(3, 3, figsize=(20, 22), layout='constrained')
    cmap = plt.get_cmap('RdBu_r')
    norm = TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span)
    for row, model in enumerate(MODELS):
        for col, mode in enumerate(MODES):
            axis = axes[row, col]
            selected = losses[losses.w.eq(weight) & losses.model.eq(model) & losses['mode'].eq(mode)]
            matrix = selected.pivot(index='dataset', columns='a', values='loss_vs_a6_pp').reindex(
                index=DATASETS, columns=ACTIVATIONS).to_numpy()
            plotted = axis.imshow(matrix, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
            axis.set_title(f'{LABELS[model]} | {mode.upper()}', fontsize=14, pad=9)
            axis.set_xticks(range(7), ['1', '1.58', '2', '3', '4', '5', '6'])
            axis.set_yticks(range(16), DATASETS)
            axis.tick_params(axis='both', length=0, labelsize=10)
            axis.set_xticks(np.arange(-.5, 7, 1), minor=True)
            axis.set_yticks(np.arange(-.5, 16, 1), minor=True)
            axis.grid(which='minor', color='white', linewidth=.7)
            axis.tick_params(which='minor', length=0)
            for (i, j), value in np.ndenumerate(matrix):
                rgb = cmap(norm(value))[:3]
                light = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]
                axis.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=8,
                          color='white' if light < .48 else '#222222')
            for spine in axis.spines.values():
                spine.set_visible(False)
    bar = fig.colorbar(plotted, ax=axes, shrink=.66, pad=.018, fraction=.025)
    bar.set_label('Accuracy loss relative to the same W / checkpoint at A6 (percentage points)', fontsize=12)
    bar.set_ticks([-span, -span / 2, 0, span / 2, span])
    fig.suptitle(f'Serial 20008 | W{weight} activation sensitivity by dataset', fontsize=21, fontweight='bold')
    fig.supxlabel('Activation bits (A)', fontsize=13)
    fig.text(.5, -.015, 'Loss = Acc(W, A6) − Acc(W, A). Red: below A6. Blue: above A6. '
             'Same dataset order and color scale in every W3/W4/W5 panel. A6 is still quantized.',
             ha='center', fontsize=11, color='#444444')
    return fig


def render(frame, output, report_dir):
    means, losses = prepare_views(frame)
    tables, figures = output / 'tables', output / 'figures'
    figures.mkdir()
    means.to_csv(tables / 'accuracy_curve_means.csv', index=False)
    losses.to_csv(tables / 'activation_loss_vs_a6.csv', index=False)
    span = max(1, math.ceil(float(losses.loss_vs_a6_pp.abs().max()) / 10) * 10)
    names = ['accuracy_vs_activation'] + [f'activation_loss_vs_a6_w{w}' for w in WEIGHTS]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.facecolor': 'white', 'figure.facecolor': 'white'}):
        with PdfPages(figures / 'activation_observation_atlas.pdf') as pdf:
            for name, weight in zip(names, [None, *WEIGHTS]):
                fig = _accuracy_curves(frame, means) if weight is None else _loss_heatmap(losses, weight, span)
                try:
                    fig.savefig(figures / f'{name}.png', dpi=170, bbox_inches='tight')
                    fig.savefig(figures / f'{name}.svg', bbox_inches='tight')
                    pdf.savefig(fig, bbox_inches='tight')
                finally:
                    plt.close(fig)
    report_figures = report_dir / 'figures'
    report_figures.mkdir()
    import shutil
    for name in names:
        source, copied = figures / f'{name}.png', report_figures / f'{name}.png'
        shutil.copy2(source, copied)
        if sha256(source) != sha256(copied):
            raise ValueError('Report figure hash differs from result figure')
    verify_tables(frame, tables)
    return dict(figure_stems=names, panels_per_figure=9, reference_activation_bits=6,
                heatmap_color_limits_pp=[-span, span], negative_loss_cells=int(losses.loss_vs_a6_pp.lt(0).sum()),
                a6_reference_rows=432, paired_rows=len(losses), curve_mean_rows=len(means),
                datasets_per_mean=16, pdf_pages=4, report_figure_hashes_verified=True)


def verify_tables(frame, tables):
    expected_means, expected_losses = prepare_views(frame)
    means = pd.read_csv(tables / 'accuracy_curve_means.csv')
    losses = pd.read_csv(tables / 'activation_loss_vs_a6.csv')
    pd.testing.assert_frame_equal(means, expected_means, check_dtype=False, atol=1e-10, rtol=0)
    pd.testing.assert_frame_equal(losses, expected_losses, check_dtype=False, atol=1e-10, rtol=0)
    return dict(curve_mean_rows=len(means), paired_loss_rows=len(losses), formulas_verified=True)
