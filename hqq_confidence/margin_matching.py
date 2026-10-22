"""Baseline-only overlap standardization, restored from serial 100004.

Accuracy/damage/rescue are fractions. Damage/rescue here are unconditional
weighted masses, unlike the conditional rates in the older fragility workflow.
"""
import re

import numpy as np
import pandas as pd

WIDTHS = (.025, .05, .10)
METRICS = ('baseline_accuracy', 'quant_accuracy', 'accuracy_ratio', 'absolute_drop',
           'damage_rate', 'rescue_rate', 'effective_n')


def normalized_margin(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite N x C logits with C >= 2')
    std = values.std(axis=1, ddof=0)
    if np.any(std <= 0):
        raise ValueError('Class-logit standard deviation must be positive')
    top = np.partition(values, -2, axis=1)[:, -2:]
    return (top[:, 1] - top[:, 0]) / std


def match_margin_distributions(margins, groups, datasets, bin_width):
    margins, groups, datasets = map(np.asarray, (margins, groups, datasets))
    if margins.ndim != 1 or not len(margins) or groups.shape != margins.shape or datasets.shape != margins.shape:
        raise ValueError('Expected equally sized sample vectors')
    if not np.isfinite(margins).all() or np.any(margins < 0) or not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError('Margins must be finite/nonnegative and bin width positive')
    names = np.unique(groups)
    if len(names) != 2:
        raise ValueError('Expected exactly two nonempty groups')
    edges = np.arange(int(np.floor(margins.max() / bin_width)) + 2) * bin_width
    bins = np.searchsorted(edges, margins, side='right') - 1
    base = np.zeros(len(margins), dtype=float)
    for group in names:
        members = np.unique(datasets[groups == group])
        for dataset in members:
            mask = (groups == group) & (datasets == dataset)
            base[mask] = 1 / (len(members) * mask.sum())
    for dataset in np.unique(datasets):
        if len(np.unique(groups[datasets == dataset])) != 1:
            raise ValueError('A dataset cannot belong to both groups')
    hist = {g: np.bincount(bins[groups == g], weights=base[groups == g], minlength=len(edges) - 1) for g in names}
    overlap_mass = np.minimum(hist[names[0]], hist[names[1]])
    overlap = float(overlap_mass.sum())
    if overlap <= 0:
        raise ValueError('No margin-bin overlap between groups')
    target = overlap_mass / overlap
    weights = np.zeros_like(base)
    matched = {}
    for group in names:
        mask = groups == group
        factors = np.divide(target, hist[group], out=np.zeros_like(target), where=hist[group] > 0)
        weights[mask] = base[mask] * factors[bins[mask]]
        matched[group] = np.bincount(bins[mask], weights=weights[mask], minlength=len(target))
        if not np.allclose(matched[group], target, rtol=0, atol=1e-12):
            raise RuntimeError('Matched margin proportions failed the balance check')
    return dict(weights=weights, base_weights=base, bin_ids=bins, edges=edges,
                target=target, original_hist=hist, matched_hist=matched, overlap=overlap)


def weighted_accuracy(baseline, quantized, weights):
    baseline, quantized = np.asarray(baseline, dtype=bool), np.asarray(quantized, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if baseline.shape != quantized.shape or baseline.shape != weights.shape or weights.ndim != 1:
        raise ValueError('Correctness and weights must be equal sample vectors')
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, with positive mass')
    weights = weights / weights.sum()
    fp, quant = float(weights @ baseline), float(weights @ quantized)
    damage = float(weights @ (baseline & ~quantized))
    rescue = float(weights @ (~baseline & quantized))
    if not np.isclose(quant, fp - damage + rescue, rtol=0, atol=1e-12):
        raise RuntimeError('Accuracy decomposition failed')
    return dict(baseline_accuracy=fp, quant_accuracy=quant,
                accuracy_ratio=quant / fp if fp > 0 else np.nan,
                absolute_drop=fp - quant, damage_rate=damage, rescue_rate=rescue,
                effective_n=float(1 / (weights @ weights)))


def weighted_ks(margin, group, weights):
    _, inverse = np.unique(margin, return_inverse=True)
    cdfs = [np.cumsum(np.bincount(inverse[group == g], weights=weights[group == g],
                                 minlength=inverse.max() + 1)) for g in np.unique(group)]
    return float(np.max(np.abs(cdfs[0] - cdfs[1])))


def compute_tables(samples, outcomes, conditions):
    """One model/mode at a time; no overlap is an explicit undefined estimate."""
    margin = samples.margin.to_numpy()
    groups, datasets = samples.group.to_numpy(), samples.dataset.to_numpy()
    baseline = samples.baseline_correct.to_numpy(dtype=bool)
    outcomes = np.asarray(outcomes, dtype=bool)
    if outcomes.shape != (len(samples), len(conditions)) or len(set(conditions)) != len(conditions):
        raise ValueError('Expected unique conditions and N x conditions correctness')
    names = np.unique(groups)
    if len(names) != 2 or 'other' not in names:
        raise ValueError('Expected an experimental group and other')
    focus = next(g for g in names if g != 'other')
    rows = {name: [] for name in ('group_metrics', 'balance', 'diagnostics', 'dataset_weights', 'dataset_metrics', 'gaps')}
    archive = {k: samples[k].to_numpy() for k in ('sample_id', 'target', 'baseline_correct', 'margin', 'dataset', 'group')}
    fits = {}
    # Base weights do not depend on binning and remain defined without overlap.
    base = np.zeros(len(samples))
    for group in names:
        members = np.unique(datasets[groups == group])
        for dataset in members:
            mask = datasets == dataset
            base[mask] = 1 / (len(members) * mask.sum())
    archive['base_weights'] = base
    for width in WIDTHS:
        try:
            fit = match_margin_distributions(margin, groups, datasets, width)
        except ValueError as error:
            if str(error) != 'No margin-bin overlap between groups':
                raise
            fit = None
        fits[width] = fit
        archive[f'weights_{width:g}'] = fit['weights'] if fit else np.zeros(len(samples))
        for group in names:
            mask = groups == group
            w = archive[f'weights_{width:g}'][mask]
            support = fit['target'][fit['bin_ids']] > 0 if fit else np.zeros(len(samples), bool)
            rows['diagnostics'].append(dict(bin_width=width, group=group,
                status='ok' if fit else 'no_common_support', n=int(mask.sum()),
                effective_n=float(w.sum()**2 / (w @ w)) if w.sum() else 0.,
                overlap_coefficient=fit['overlap'] if fit else 0.,
                supported_prior_mass=float(base[mask & support].sum()),
                raw_weighted_ks=weighted_ks(margin, groups, base),
                matched_weighted_ks=weighted_ks(margin, groups, fit['weights']) if fit else np.nan,
                max_weight_multiplier=float(np.max(w / base[mask])),
                raw_margin_mean=float(base[mask] @ margin[mask]),
                matched_margin_mean=float(w @ margin[mask]) if fit else np.nan))
            if fit:
                for i in range(len(fit['target'])):
                    rows['balance'].append(dict(bin_width=width, group=group, bin_low=fit['edges'][i], bin_high=fit['edges'][i+1],
                        raw_fraction=fit['original_hist'][group][i], target_fraction=fit['target'][i],
                        matched_fraction=fit['matched_hist'][group][i], balance_error=fit['matched_hist'][group][i]-fit['target'][i]))
            for dataset in sorted(set(datasets[mask])):
                dm = datasets == dataset
                dw = archive[f'weights_{width:g}'][dm]
                rows['dataset_weights'].append(dict(bin_width=width, group=group, dataset=dataset,
                    n=int(dm.sum()), initial_group_mass=float(base[dm].sum()), matched_group_mass=float(dw.sum()),
                    matched_effective_n=float(dw.sum()**2/(dw @ dw)) if dw.sum() else 0.))
        for index, condition in enumerate(conditions):
            match = re.fullmatch(r'hqq_n([0-9.]+)_g([0-9]+)', condition)
            if not match:
                raise ValueError(f'Unsupported condition {condition}')
            bits, size = float(match[1]), int(match[2])
            ratios = {}
            for mode, weights in (('raw', base), ('matched', archive[f'weights_{width:g}'])):
                for group in names:
                    mask = groups == group
                    if weights[mask].sum():
                        values = weighted_accuracy(baseline[mask], outcomes[mask, index], weights[mask])
                    else:
                        values = {k: np.nan for k in METRICS}
                    ratios[group, mode] = values['accuracy_ratio']
                    rows['group_metrics'].append(dict(bin_width=width, group=group, weighting=mode, condition=condition,
                        bits=bits, group_size=size, status='no_common_support' if mode == 'matched' and not fit else
                        ('zero_baseline_accuracy' if not np.isfinite(values['accuracy_ratio']) else 'ok'), **values))
                if width == .05:
                    for dataset in sorted(set(datasets)):
                        dm = datasets == dataset
                        values = weighted_accuracy(baseline[dm], outcomes[dm, index], weights[dm]) if weights[dm].sum() else {k:np.nan for k in METRICS}
                        rows['dataset_metrics'].append(dict(dataset=dataset, group=groups[dm][0], weighting=mode,
                            condition=condition, bits=bits, group_size=size, group_mass=float(weights[dm].sum()),
                            status='zero_matched_mass' if not weights[dm].sum() else
                            ('zero_baseline_accuracy' if not np.isfinite(values['accuracy_ratio']) else 'ok'), **values))
            raw = ratios['other','raw'] - ratios[focus,'raw']
            matched = ratios['other','matched'] - ratios[focus,'matched']
            reduction = abs(raw)-abs(matched)
            rows['gaps'].append(dict(bin_width=width, condition=condition, bits=bits, group_size=size,
                raw=raw, matched=matched, signed_gap_change=matched-raw, absolute_gap_reduction=reduction,
                absolute_gap_reduction_fraction=reduction/abs(raw) if abs(raw)>1e-12 else np.nan))
    tables = {k: pd.DataFrame(v) for k,v in rows.items()}
    # Preserve a readable schema when every setting lacks common support.
    if tables['balance'].empty:
        tables['balance'] = pd.DataFrame(columns=['bin_width','group','bin_low','bin_high','raw_fraction','target_fraction','matched_fraction','balance_error'])
    return tables, fits, archive
