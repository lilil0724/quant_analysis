"""All-dataset unquantized-AMP accuracy-ratio summary and detailed heatmaps."""
import math
import shutil

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .contracts import DATASETS, MODELS, MODES
from .grid import ACTIVATIONS, WEIGHTS
from .group_curves import prepare as prepare_groups
from .plot import LABELS
from .storage import sha256


def prepare(frame, mapping):
    _, _, coverage = prepare_groups(frame, mapping)
    columns = ['dataset', 'model', 'mode', 'w', 'a', 'top1', 'baseline_top1',
               'accuracy_ratio', 'baseline_status', 'run_id', 'baseline_run_id']
    raw = frame[columns].copy()
    raw['baseline_run_id'] = raw.baseline_run_id.fillna('')
    if raw.loc[raw.baseline_status.ne('matched'), 'accuracy_ratio'].notna().any():
        raise ValueError('Unmatched baseline cells must have N/A accuracy ratios')
    raw['granularity_group'] = raw.dataset.map(mapping.set_index('dataset_name').granularity_group)
    strata = ['model', 'mode', 'w']
    allowed = set(coverage.loc[coverage.used, strata].itertuples(index=False, name=None))
    raw['shared_stratum'] = [key in allowed for key in raw[strata].itertuples(index=False, name=None)]
    raw = raw.sort_values(['dataset', *strata, 'a']).reset_index(drop=True)
    selected = raw[raw.shared_stratum & raw.baseline_status.eq('matched')]
    summary = selected.groupby(['dataset', 'granularity_group', 'a'], as_index=False).agg(
        mean_acc_ratio=('accuracy_ratio', 'mean'), n_contexts=('accuracy_ratio', 'size'))
    summary['available_shared_strata'] = len(allowed)
    if len(summary) != 112 or not summary.n_contexts.gt(0).all():
        raise ValueError('Dataset summary must cover 16 datasets times seven activation settings')
    if summary.groupby('dataset').n_contexts.nunique().ne(1).any():
        raise ValueError('Dataset summary coverage changes across activation bits')
    return raw, summary


def _heatmap(matrix, labels, title, norm, summary=False):
    fig, axes = plt.subplots(1, 1, figsize=(14, 9), layout='constrained') if summary else plt.subplots(
        3, 3, figsize=(20, 22), layout='constrained')
    cmap = plt.get_cmap('RdBu').with_extremes(bad='#d9dfe3')
    panels = [(axes, matrix, 'Dataset means within shared comparison strata')] if summary else [
        (axes[i, j], matrix[(model, mode)], f'{LABELS[model]} | {mode.upper()}')
        for i, model in enumerate(MODELS) for j, mode in enumerate(MODES)]
    for axis, values, subtitle in panels:
        plotted = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                              aspect='auto', interpolation='nearest')
        axis.set_title(subtitle, fontsize=12 if summary else 14, pad=10)
        axis.set_xticks(range(7), ['1', '1.58', '2', '3', '4', '5', '6'])
        axis.set_yticks(range(16), labels)
        axis.tick_params(length=0, labelsize=10 if summary else 9)
        axis.set_xticks(np.arange(-.5, 7, 1), minor=True)
        axis.set_yticks(np.arange(-.5, 16, 1), minor=True)
        axis.grid(which='minor', color='white', linewidth=.7)
        axis.tick_params(which='minor', length=0)
        axis.axhline(10.5, color='#3f4850', linewidth=1.5)
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                rgb = cmap(norm(value))[:3]
                light = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]
                label, color = f'{value:.3f}', 'white' if light < .48 else '#222222'
            else:
                label, color = 'N/A', '#65717a'
            axis.text(j, i, label, ha='center', va='center', color=color, fontsize=10 if summary else 7.8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(plotted, ax=axes, shrink=.82 if summary else .66, pad=.02, fraction=.03)
    bar.set_label('acc_ratio = quantized Top-1 / matched unquantized AMP Top-1', fontsize=11)
    bar.set_ticks([0, .5, 1, norm.vmax])
    fig.suptitle(title, fontsize=18 if summary else 21, fontweight='bold')
    fig.supxlabel('Activation bits (A)', fontsize=12)
    caption = ('Row n: available shared model/mode/W contexts; counts are fixed across A. '
               'Dataset means and equal-stratum group means use different aggregation weights.' if summary else
               'Gray N/A: missing or ambiguous baseline. Each cell is one checkpoint/W/A evaluation.')
    fig.text(.5, -.045 if summary else -.015,
             'Ratio 1 = matched unquantized AMP; red < 1, blue > 1. Values above 1 are not clipped.\n'
             'FGIR = other 11 datasets (above divider); UFGIR = soy + cotton (below divider).\n' + caption,
             ha='center', fontsize=9 if summary else 10, color='#444444')
    return fig


def render(frame, mapping_path, output, report_dir):
    mapping = pd.read_csv(mapping_path)
    raw, summary = prepare(frame, mapping)
    raw.to_csv(output / 'tables/baseline_ratio_cells.csv', index=False)
    summary.to_csv(output / 'tables/dataset_mean_baseline_ratio.csv', index=False)
    group_map = mapping.set_index('dataset_name').granularity_group.to_dict()
    order = [d for group in ('FGIR', 'UFGIR') for d in DATASETS if group_map[d] == group]
    upper = max(1.1, math.ceil(float(raw.accuracy_ratio.max()) * 10) / 10)
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=upper)
    matrix = summary.pivot(index='dataset', columns='a', values='mean_acc_ratio').reindex(
        index=order, columns=ACTIVATIONS).to_numpy()
    counts = summary.groupby('dataset').n_contexts.first()
    shared_count = int(summary.available_shared_strata.iloc[0])
    labels = [f'{d} | {group_map[d]} | n={counts[d]}/{shared_count}' for d in order]
    names = ['dataset_mean_baseline_acc_ratio'] + [f'baseline_acc_ratio_w{w}' for w in WEIGHTS]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        with PdfPages(output / 'figures/baseline_acc_ratio_heatmaps.pdf') as pdf:
            for name, weight in zip(names, [None, *WEIGHTS]):
                if weight is None:
                    fig = _heatmap(matrix, labels, 'Serial 20008 | All-dataset mean baseline acc_ratio', norm, True)
                else:
                    panels = {}
                    for model in MODELS:
                        for mode in MODES:
                            chosen = raw[raw.model.eq(model) & raw['mode'].eq(mode) & raw.w.eq(weight)]
                            panels[(model, mode)] = chosen.pivot(index='dataset', columns='a', values='accuracy_ratio').reindex(
                                index=order, columns=ACTIVATIONS).to_numpy()
                    fig = _heatmap(panels, order, f'Serial 20008 | W{weight} baseline acc_ratio (all datasets)', norm)
                try:
                    for extension in ('png', 'svg'):
                        fig.savefig(output / 'figures' / f'{name}.{extension}', dpi=170, bbox_inches='tight')
                    pdf.savefig(fig, bbox_inches='tight')
                finally:
                    plt.close(fig)
                src, dst = output / 'figures' / f'{name}.png', report_dir / 'figures' / f'{name}.png'
                shutil.copy2(src, dst)
                if sha256(src) != sha256(dst):
                    raise ValueError('Baseline ratio report image hash mismatch')
    verify(frame, mapping_path, output / 'tables')
    return dict(figure_stems=names, reference='uniquely matched unquantized AMP',
        ratio_cells=len(raw), matched_cells=int(raw.accuracy_ratio.notna().sum()),
        na_cells=int(raw.accuracy_ratio.isna().sum()), summary_cells=len(summary),
        ratios_above_one=int(raw.accuracy_ratio.gt(1).sum()), color_limits=[0, 1, upper],
        summary_context_counts={d: int(counts[d]) for d in order}, pdf_pages=4)


def verify(frame, mapping_path, tables):
    raw, summary = prepare(frame, pd.read_csv(mapping_path))
    for name, expected in [('baseline_ratio_cells', raw), ('dataset_mean_baseline_ratio', summary)]:
        saved = pd.read_csv(tables / f'{name}.csv')
        if 'baseline_run_id' in saved:
            saved['baseline_run_id'] = saved.baseline_run_id.fillna('')
        pd.testing.assert_frame_equal(saved, expected, check_dtype=False, atol=1e-10, rtol=0)
    return dict(status='passed', ratio_cells=len(raw), dataset_mean_cells=len(summary))
