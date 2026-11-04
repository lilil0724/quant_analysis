"""Condition heatmaps and W/A comparisons with an explicitly shared population."""
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from .quant import CONTEXT, encoded


def save(fig, directory, stem):
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(directory / f'{stem}.{extension}', dpi=160)
    plt.close(fig)


def draw_matrix(axis, matrix, columns, rows, norm, cmap, number_format='.3f'):
    values = matrix.to_numpy(dtype=float)
    im = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect='auto')
    axis.set_xticks(range(len(columns)), columns, rotation=30, ha='right')
    axis.set_yticks(range(len(rows)), rows)
    for i, j in np.ndindex(values.shape):
        value = values[i, j]
        rgb = cmap(norm(value))[:3] if np.isfinite(value) else (1, 1, 1)
        light = .2126*rgb[0] + .7152*rgb[1] + .0722*rgb[2]
        axis.text(j, i, f'{value:{number_format}}' if np.isfinite(value) else 'N/A', ha='center', va='center',
                  fontsize=7, color='white' if light < .5 else '#374151')
    return im


def plot_conditions(frame, expected_contexts, output):
    figures = output / 'figures'
    tables = output / 'tables'
    cmap = plt.get_cmap('RdBu').with_extremes(bad='#e5e7eb')
    ratios = frame.loc[frame.accepted, 'accuracy_ratio']
    upper = max(1.05, float(ratios.max())) if len(ratios) else 1.05
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=upper)
    datasets = sorted({key[0] for key in expected_contexts} | set(frame.dataset))
    models = sorted({key[1] for key in expected_contexts} | set(frame.model))
    modes = sorted({key[2] for key in expected_contexts} | set(frame['mode']))
    references = []
    for condition_id, group in frame.groupby('condition_id', sort=True):
        accepted = group.loc[group.accepted]
        # Each displayed context averages its accepted per-seed/evaluation ratios.
        series = accepted.groupby(CONTEXT).accuracy_ratio.mean()
        columns = pd.MultiIndex.from_product([models, modes], names=['model', 'mode'])
        wide = series.unstack(['model', 'mode']).reindex(index=datasets, columns=columns)
        stem = f'condition_{condition_id}'
        wide.to_csv(tables / f'{stem}_heatmap.csv', lineterminator='\n')
        fig, axes = plt.subplots(1, len(models), figsize=(5.5*len(models), max(6, .44*len(datasets)+2)),
                                 sharey=True, squeeze=False, layout='constrained')
        for axis, model in zip(axes[0], models):
            im = draw_matrix(axis, wide[model], [m.upper() for m in modes], datasets, norm, cmap)
            axis.set_title(model.replace('_patch16_224', '').replace('_patch4_window7_224_in22k', ''))
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Quantized accuracy / matched unquantized AMP accuracy', shrink=.8)
        serials = ', '.join(sorted(set(group.serial.map(str))))
        fig.suptitle(f"Serial {serials}: {group.condition_label.iloc[0]}\nCondition {condition_id}", fontsize=11)
        fig.supxlabel(f'{len(accepted)}/{len(group)} runs accepted; context means across accepted seeds/evaluations')
        save(fig, figures, stem)
        references.append(dict(stem=stem, caption=group.condition_label.iloc[0], kind='condition'))
    return references, norm, cmap


def plot_weight_activation(frame, output, norm, cmap):
    """Hold all non-bit settings fixed, then intersect full evaluation identities."""
    groups = {}
    for condition_id, rows in frame.loc[frame.accepted].groupby('condition_id', sort=True):
        settings = json.loads(rows.condition_json.iloc[0])
        if any(not isinstance(settings[name], (int, float))
               for name in ('hqq_nbits', 'hqq_act_bits')):
            continue
        settings.pop('hqq_nbits')
        settings.pop('hqq_act_bits')
        groups.setdefault(encoded(settings), []).append((condition_id, rows))
    refs, audit = [], []
    for settings_json, conditions in groups.items():
        if len(conditions) < 2:
            continue
        # Include baseline ID and every recorded evaluation field, not just context.
        populations = []
        for _, rows in conditions:
            keys = []
            for row in rows.itertuples():
                identity = json.loads(row.identity)
                identity.pop('condition')
                keys.append(encoded([identity, row.baseline_run_id]))
            populations.append(set(keys))
        common = set.intersection(*populations)
        tag = hashlib.sha256(settings_json.encode()).hexdigest()[:12]
        audit.append(dict(comparison_id=tag, settings_json=settings_json,
                          conditions=len(conditions), common_evaluations=len(common),
                          condition_ids=';'.join(c for c,_ in conditions),
                          population_counts=encoded([len(p) for p in populations]),
                          common_population_json=encoded(sorted(common))))
        if not common:
            continue
        selected = []
        for _, rows in conditions:
            for row in rows.to_dict('records'):
                identity = json.loads(row['identity'])
                identity.pop('condition')
                if encoded([identity, row['baseline_run_id']]) in common:
                    selected.append(row)
        selected = pd.DataFrame(selected)
        # Seed/evaluation means first; datasets then receive equal weight.
        context_means = selected.groupby(['hqq_nbits', 'hqq_act_bits', *CONTEXT], as_index=False).accuracy_ratio.mean()
        context_means.to_csv(output / 'tables' / f'wa_{tag}_context_means.csv', index=False, lineterminator='\n')
        models, modes = sorted(selected.model.unique()), sorted(selected['mode'].unique())
        weights = sorted(selected.hqq_nbits.unique())
        activations = sorted(selected.hqq_act_bits.unique())
        fig, axes = plt.subplots(len(models), len(modes), squeeze=False, layout='constrained',
                                 figsize=(5*len(modes), 3.4*len(models)))
        for i, model in enumerate(models):
            for j, mode in enumerate(modes):
                subset = context_means.loc[context_means.model.eq(model) & context_means['mode'].eq(mode)]
                wide = subset.pivot_table(index='hqq_nbits', columns='hqq_act_bits', values='accuracy_ratio', aggfunc='mean')
                wide = wide.reindex(index=weights, columns=activations)
                im = draw_matrix(axes[i,j], wide, [f'A{a:g}' for a in activations], [f'W{w:g}' for w in weights], norm, cmap)
                axes[i,j].set_title(f'{model.split("_base")[0]} / {mode.upper()}')
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Mean matched accuracy ratio')
        fig.suptitle(f'W/A comparison: identical non-bit settings and {len(common)} shared evaluations\nComparison {tag}', fontsize=11)
        fig.supxlabel('Equal dataset weighting; missing W/A combinations remain N/A')
        stem = f'wa_{tag}'
        save(fig, output / 'figures', stem)
        refs.append(dict(stem=stem, caption=f'W/A comparison {tag} ({len(common)} shared evaluations)', kind='comparison'))
    pd.DataFrame(audit, columns=['comparison_id', 'settings_json', 'conditions', 'common_evaluations',
                                'condition_ids', 'population_counts', 'common_population_json']).to_csv(
        output / 'audit/comparison_populations.csv', index=False, lineterminator='\n')
    return refs


# Display order/groups follow the user's reference figure, not the canonical
# granularity module's primary-versus-boundary research grouping.
DATASET_DISPLAY_GROUPS = {
    'FGIR': ['aircraft', 'cars', 'cub', 'dogs', 'flowers', 'food', 'inat17',
             'moe', 'nabirds', 'pets', 'vegfru'],
    'UFGIR': ['cotton', 'soyageing', 'soygene', 'soyglobal', 'soylocal'],
}


def dataset_activation_summaries(frame, expected_contexts, value_column='accuracy_ratio'):
    """One fixed-weight summary per non-activation setting, using shared identities."""
    groups = {}
    for cid, rows in frame.loc[frame.accepted].groupby('condition_id', sort=True):
        settings = json.loads(rows.condition_json.iloc[0])
        if any(not isinstance(settings[name], (int, float))
               for name in ('hqq_nbits', 'hqq_act_bits')):
            continue
        settings.pop('hqq_act_bits')
        groups.setdefault(encoded(settings), []).append((cid, rows))
    expected = pd.DataFrame(expected_contexts, columns=CONTEXT).drop_duplicates()
    summaries = []
    for settings_json, conditions in groups.items():
        populations, indexed = [], []
        for _, rows in conditions:
            rows = rows.copy()
            identities = []
            for row in rows.itertuples():
                identity = json.loads(row.identity)
                identity.pop('condition')
                identities.append(encoded([identity, row.baseline_run_id]))
            rows['comparison_identity'] = identities
            indexed.append(rows)
            populations.append(set(identities))
        common = set.intersection(*populations)
        if not common:
            continue
        selected = pd.concat([rows.loc[rows.comparison_identity.isin(common)] for rows in indexed], ignore_index=True)
        # Equal model/mode weighting after averaging accepted seeds/evaluations.
        contexts = selected.groupby(['hqq_act_bits', *CONTEXT], as_index=False)[value_column].mean()
        means = contexts.pivot_table(index='dataset', columns='hqq_act_bits', values=value_column, aggfunc='mean')
        counts = contexts.groupby(['dataset', 'hqq_act_bits']).size().unstack('hqq_act_bits', fill_value=0)
        if not counts.eq(counts.iloc[:, 0], axis=0).all().all():
            raise ValueError('Dataset activation plot does not have a shared context population')
        summaries.append(dict(settings_json=settings_json, settings=json.loads(settings_json),
                              comparison_id=hashlib.sha256(settings_json.encode()).hexdigest()[:12],
                              contexts=contexts, means=means, counts=counts,
                              expected_counts=expected.groupby('dataset').size(),
                              common_population=sorted(common),
                              population_counts=[len(p) for p in populations],
                              serials=sorted(set(selected.serial.map(str)))))
    return summaries


def plot_dataset_activation(frame, expected_contexts, output):
    """Reference-style dataset x activation heatmaps, separately for each W."""
    cmap = plt.get_cmap('RdBu').with_extremes(bad='#e5e7eb')
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=1.6)
    refs, audits = [], []
    for summary in dataset_activation_summaries(frame, expected_contexts):
        means = summary['means']
        labels = {dataset: group for group, datasets in DATASET_DISPLAY_GROUPS.items() for dataset in datasets}
        order = [dataset for datasets in DATASET_DISPLAY_GROUPS.values() for dataset in datasets if dataset in means.index]
        order += sorted(set(means.index) - set(order))
        activations = sorted(means.columns)
        means = means.reindex(index=order, columns=activations)
        weight = summary['settings']['hqq_nbits']
        tag = summary['comparison_id']
        stem = f'dataset_activation_w{weight:g}_{tag}'
        means.to_csv(output/'tables'/f'{stem}_means.csv', lineterminator='\n')
        summary['contexts'].to_csv(output/'tables'/f'{stem}_contexts.csv', index=False, lineterminator='\n')
        row_labels = [f"{dataset} | {labels.get(dataset, 'Other')} | "
                      f"n={int(summary['counts'].loc[dataset].iloc[0])}/{int(summary['expected_counts'].get(dataset, 0))}"
                      for dataset in order]
        fig, axis = plt.subplots(figsize=(12.8, 8.4), layout='constrained')
        im = draw_matrix(axis, means, [f'{a:g}' for a in activations], row_labels, norm, cmap)
        axis.tick_params(axis='both', length=0, labelsize=9)
        axis.set_xticks(range(len(activations)), [f'{a:g}' for a in activations], rotation=0, ha='center')
        axis.set_xticks(np.arange(-.5, len(activations), 1), minor=True)
        axis.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        axis.grid(which='minor', color='white', linewidth=.6, alpha=.75)
        axis.tick_params(which='minor', length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for i in range(1, len(order)):
            if labels.get(order[i]) != labels.get(order[i-1]):
                axis.axhline(i-.5, color='#444444', linewidth=1.1)
        axis.set_xlabel('Activation bits (A)', fontsize=11)
        axis.set_title('Dataset means within shared model / mode comparison strata', fontsize=11, pad=9)
        fig.suptitle(f"Serial {', '.join(summary['serials'])} | W{weight:g} | Dataset mean baseline acc_ratio",
                     fontsize=16, fontweight='bold')
        fig.colorbar(im, ax=axis, shrink=.82, fraction=.035, pad=.02, ticks=[0, .5, 1, 1.6],
                     label='acc_ratio = quantized Top-1 / matched unquantized AMP Top-1')
        save(fig, output/'figures', stem)
        refs.append(dict(stem=stem, caption=f'W{weight:g}: dataset mean accuracy ratio versus activation bits', kind='dataset_activation'))
        audits.append(dict(comparison_id=tag, weight_bits=weight, settings_json=summary['settings_json'],
                           shared_evaluations=len(summary['common_population']),
                           population_counts=encoded(summary['population_counts']),
                           common_population_json=encoded(summary['common_population'])))
    pd.DataFrame(audits).to_csv(output/'audit/dataset_activation_populations.csv', index=False, lineterminator='\n')
    return refs


def plot_weight_delta(frame, expected_contexts, output, delta_metric='top1_pp'):
    metrics = {'top1_pp': ('delta_top1_pp', 'Top-1', 'percentage points', '+.2f'),
               'acc_ratio': ('delta_accuracy_ratio', 'acc_ratio', 'dimensionless', '+.3f'),
               'acc_ratio_quotient': ('wa_over_weight_acc_ratio', 'acc_ratio quotient', 'dimensionless', '.3f')}
    value_column, metric_label, units, number_format = metrics[delta_metric]
    is_quotient = delta_metric == 'acc_ratio_quotient'
    selected = frame.assign(accepted=frame.weight_ratio_accepted if is_quotient else frame.weight_delta_accepted,
                            identity=frame.delta_identity)
    summaries = dataset_activation_summaries(selected, expected_contexts, value_column)
    cmap = plt.get_cmap('RdBu').with_extremes(bad='#e5e7eb')
    # Ratio differences have no universal +/-1 bound; use one scale across all W.
    largest = max((float(s['means'].abs().max().max()) for s in summaries), default=0)
    limit = (max(1.6, float(np.ceil(largest*10)/10)) if is_quotient else
             100 if delta_metric == 'top1_pp' else max(1., float(np.ceil(largest*10)/10)))
    norm = TwoSlopeNorm(vmin=0 if is_quotient else -limit, vcenter=1 if is_quotient else 0, vmax=limit)
    refs, audit = [], []
    for summary in summaries:
        means = summary['means']
        groups = {d: group for group, datasets in DATASET_DISPLAY_GROUPS.items() for d in datasets}
        order = [d for datasets in DATASET_DISPLAY_GROUPS.values() for d in datasets if d in means.index]
        order += sorted(set(means.index)-set(order))
        activations = sorted(means.columns)
        means = means.reindex(index=order, columns=activations)
        weight = summary['settings']['hqq_nbits']
        prefix = ('weight_acc_ratio_quotient' if is_quotient else
                  'weight_delta' if delta_metric == 'top1_pp' else 'weight_delta_acc_ratio')
        stem = f'{prefix}_w{weight:g}_{summary["comparison_id"]}'
        means.to_csv(output/'tables'/f'{stem}_means.csv', lineterminator='\n')
        summary['contexts'].to_csv(output/'tables'/f'{stem}_contexts.csv', index=False, lineterminator='\n')
        labels = [f"{d} | {groups.get(d, 'Other')} | n={int(summary['counts'].loc[d].iloc[0])}/"
                  f"{int(summary['expected_counts'].get(d, 0))}" for d in order]
        fig, axis = plt.subplots(figsize=(12.8, 8.4), layout='constrained')
        im = draw_matrix(axis, means, [f'{a:g}' for a in activations], labels, norm, cmap, number_format)
        axis.tick_params(axis='both', length=0, labelsize=9)
        axis.set_xticks(range(len(activations)), [f'{a:g}' for a in activations], rotation=0, ha='center')
        axis.set_xticks(np.arange(-.5, len(activations), 1), minor=True)
        axis.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        axis.grid(which='minor', color='white', linewidth=.6, alpha=.75)
        axis.tick_params(which='minor', length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for i in range(1, len(order)):
            if groups.get(order[i]) != groups.get(order[i-1]):
                axis.axhline(i-.5, color='#444444', linewidth=1.1)
        axis.set_xlabel('Activation bits (A)', fontsize=11)
        axis.set_title('W+A: extended, all heads | W-only: legacy linear, head excluded\n'
                       'Shared model/mode pairs; scope/head differences included', fontsize=10, pad=9)
        title = 'acc_ratio(W+A) / acc_ratio(W-only)' if is_quotient else f'Delta {metric_label} (W+A minus W-only)'
        fig.suptitle(f"Serial {', '.join(summary['serials'])} | W{weight:g} | {title}",
                     fontsize=16, fontweight='bold')
        fig.colorbar(im, ax=axis, shrink=.82, fraction=.035, pad=.02,
                     ticks=[0, .5, 1, limit] if is_quotient else [-limit, -limit/2, 0, limit/2, limit],
                     label='acc_ratio(W+A) / acc_ratio(W-only); 1 = equal' if is_quotient else
                     f'Delta {metric_label} ({units}); negative = W+A lower')
        save(fig, output/'figures', stem)
        caption = (f'W{weight:g}: acc_ratio(W+A) / acc_ratio(W-only)' if is_quotient else
                   f'W{weight:g}: descriptive W+A minus W-only {metric_label} delta ({units})')
        refs.append(dict(stem=stem, caption=caption,
                         kind='weight_delta', delta_metric=delta_metric, color_limit=limit))
        audit.append(dict(weight_bits=weight, comparison_id=summary['comparison_id'],
                          settings_json=summary['settings_json'], delta_metric=delta_metric, units=units,
                          shared_evaluations=len(summary['common_population']),
                          population_counts=encoded(summary['population_counts']),
                          common_population_json=encoded(summary['common_population'])))
    pd.DataFrame(audit).to_csv(output/'audit/weight_delta_populations.csv', index=False, lineterminator='\n')
    return refs
