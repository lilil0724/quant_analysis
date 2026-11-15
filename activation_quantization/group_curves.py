"""Two activation accuracy-ratio curves using the user's soy+cotton grouping."""
import itertools
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contracts import DATASETS, MODELS, MODES
from .grid import WEIGHTS, ACTIVATIONS
from .storage import sha256

UFGIR = {'soyageing', 'soygene', 'soyglobal', 'soylocal', 'cotton'}
GROUPS = ('FGIR', 'UFGIR')
STEM = 'fgir_ufgir_mean_acc_ratio'


def prepare(frame, mapping):
    if mapping.dataset_name.duplicated().any() or set(mapping.dataset_name) != set(DATASETS):
        raise ValueError('Grouping must map all 16 datasets exactly once')
    expected = {name: 'UFGIR' if name in UFGIR else 'FGIR' for name in DATASETS}
    if mapping.set_index('dataset_name').granularity_group.to_dict() != expected:
        raise ValueError('Grouping differs from the user-defined soy+cotton UFGIR / other FGIR contract')
    keys = ['dataset', 'model', 'mode', 'w', 'a']
    grid = set(itertools.product(DATASETS, MODELS, MODES, WEIGHTS, ACTIVATIONS))
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != grid:
        raise ValueError('Group curves require the complete unique 3024-cell grid')
    matched = frame.baseline_status.eq('matched')
    valid = frame.loc[matched].copy()
    if (not np.isfinite(valid.accuracy_ratio).all() or not valid.baseline_top1.gt(0).all()
            or not np.allclose(valid.accuracy_ratio, valid.top1 / valid.baseline_top1, atol=1e-10, rtol=0)):
        raise ValueError('Invalid matched accuracy ratios')
    valid['granularity_group'] = valid.dataset.map(expected)
    coverage, summaries = [], []
    for model, mode, weight in itertools.product(MODELS, MODES, WEIGHTS):
        selected = valid[valid.model.eq(model) & valid['mode'].eq(mode) & valid.w.eq(weight)]
        members = {}
        for group in GROUPS:
            sets = [frozenset(selected[selected.granularity_group.eq(group) & selected.a.eq(a)].dataset)
                    for a in ACTIVATIONS]
            if any(s != sets[0] for s in sets[1:]):
                raise ValueError('Matched dataset membership changes across activation bits')
            members[group] = sets[0]
        used = all(len(members[g]) >= 2 for g in GROUPS)
        coverage.append(dict(model=model, mode=mode, w=weight, used=used,
            n_fgir=len(members['FGIR']), n_ufgir=len(members['UFGIR']),
            fgir_members=';'.join(sorted(members['FGIR'])),
            ufgir_members=';'.join(sorted(members['UFGIR'])),
            reason='shared_stratum' if used else 'fewer_than_two_datasets_in_one_group'))
        if used:
            summary = selected.groupby(['granularity_group', 'a'], as_index=False).agg(
                mean_acc_ratio=('accuracy_ratio', 'mean'), n_datasets=('dataset', 'nunique'))
            summary = summary.assign(model=model, mode=mode, w=weight)
            summaries.append(summary)
    if not summaries:
        raise ValueError('No shared strata with at least two datasets per group')
    strata = pd.concat(summaries, ignore_index=True).sort_values(
        ['granularity_group', 'a', 'model', 'mode', 'w']).reset_index(drop=True)
    curves = strata.groupby(['granularity_group', 'a'], as_index=False).agg(
        mean_acc_ratio=('mean_acc_ratio', 'mean'), n_strata=('mean_acc_ratio', 'size'),
        n_dataset_cells=('n_datasets', 'sum'))
    if len(curves) != 14 or curves.n_strata.nunique() != 1:
        raise ValueError('Both curves must use the same fixed strata across all seven A settings')
    return curves, strata, pd.DataFrame(coverage)


def render(frame, mapping_path, output, report_dir):
    mapping = pd.read_csv(mapping_path)
    curves, strata, coverage = prepare(frame, mapping)
    tables, figures = output / 'tables', output / 'figures'
    for name, values in [('granularity_ratio_curves', curves), ('granularity_ratio_strata', strata),
                         ('granularity_ratio_coverage', coverage)]:
        values.to_csv(tables / f'{name}.csv', index=False)
    used = int(coverage.used.sum())
    colors = {'FGIR': '#2374ab', 'UFGIR': '#d27026'}
    with plt.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, axis = plt.subplots(figsize=(12, 7.5), layout='constrained')
        for group in GROUPS:
            values = curves[curves.granularity_group.eq(group)].set_index('a').reindex(ACTIVATIONS)
            axis.plot(range(7), values.mean_acc_ratio, color=colors[group], linewidth=3,
                      marker='o', markersize=7, label=f'{group} ({11 if group == "FGIR" else 5} datasets mapped)')
        axis.axhline(1, color='#7b858c', linewidth=1, linestyle='--', zorder=0)
        axis.set_xticks(range(7), ['1', '1.58', '2', '3', '4', '5', '6'])
        axis.set_ylim(0, max(1.06, float(curves.mean_acc_ratio.max()) * 1.07))
        axis.set_xlabel('Activation bits (A) — discrete evaluated settings', fontsize=12)
        axis.set_ylabel('Mean acc_ratio (quantized / unquantized AMP)', fontsize=12)
        axis.grid(axis='y', color='#dde3e8', linewidth=.7)
        axis.tick_params(labelsize=11)
        axis.legend(loc='upper left', fontsize=12, frameon=False)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
        fig.suptitle('Serial 20008 | Activation precision: FGIR vs UFGIR', fontsize=19, fontweight='bold')
        axis.set_title(f'Equal mean over {used} shared model / mode / W strata; W3, W4 and W5 included',
                       fontsize=11, color='#444444', pad=12)
        fig.text(.5, -.055, 'UFGIR = soyageing + soygene + soyglobal + soylocal + cotton; FGIR = all other datasets.\n'
                 'Within each stratum: mean of available uniquely matched dataset ratios. Dataset counts vary by stratum.\n'
                 'At least 2 datasets per group required; no missing-baseline imputation or confidence intervals.',
                 ha='center', fontsize=9.5, color='#444444')
        try:
            for extension in ('png', 'svg', 'pdf'):
                fig.savefig(figures / f'{STEM}.{extension}', dpi=180, bbox_inches='tight')
        finally:
            plt.close(fig)
    source = figures / f'{STEM}.png'
    copied = report_dir / 'figures' / source.name
    shutil.copy2(source, copied)
    if sha256(source) != sha256(copied):
        raise ValueError('Group ratio report image hash mismatch')
    verify(frame, mapping_path, tables)
    return dict(figure_stem=STEM, group_sizes={'FGIR': 11, 'UFGIR': 5},
        grouping='user-defined soy+cotton vs all other datasets', mapping_sha256=sha256(mapping_path),
        shared_strata=used, excluded_strata=int((~coverage.used).sum()),
        fgir_datasets_per_used_stratum=[int(coverage[coverage.used].n_fgir.min()), int(coverage[coverage.used].n_fgir.max())],
        ufgir_datasets_per_used_stratum=[int(coverage[coverage.used].n_ufgir.min()), int(coverage[coverage.used].n_ufgir.max())],
        curve_points=len(curves), aggregation='dataset mean within group/stratum, then equal mean of shared strata',
        reference='uniquely matched unquantized AMP; accuracy_ratio=quant_top1/baseline_top1')


def verify(frame, mapping_path, tables):
    expected = prepare(frame, pd.read_csv(mapping_path))
    for name, values in zip(['granularity_ratio_curves', 'granularity_ratio_strata', 'granularity_ratio_coverage'], expected):
        saved = pd.read_csv(tables / f'{name}.csv', keep_default_na=False)
        pd.testing.assert_frame_equal(saved, values, check_dtype=False, atol=1e-10, rtol=0)
    return dict(status='passed', points=14, mean_ratios_verified=True)
