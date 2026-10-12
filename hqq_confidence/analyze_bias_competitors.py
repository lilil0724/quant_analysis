"""Fixed retrospective cross-fit bias and cross-dataset competitor diagnostics.

Run from the repository root with ``python -m hqq_confidence.analyze_bias_competitors``.
Requires the previous raw-logit audit config and its unchanged source artifacts.
No inference, checkpoint modification, or test-set hyperparameter selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from docs.research.analyze_raw_logit_mechanism_20261010 import (
    BASE, CONDITIONS, SOY, sha256_file, validate_raw,
)

SEED = 100004
FOLDS = 5
SHUFFLES = 5
THRESHOLDS = np.array([.25, .5, 1.])
KNOTS = np.array([.05, .25, .5, 1., 2., 4., 8.])
MARGIN_BINS = np.array([0., .25, .5, 1., 2., 4., 8., np.inf])


def assign_folds(target, n_folds=FOLDS, seed=SEED):
    """Label-stratified shuffled round-robin folds, fixed across conditions."""
    rng = np.random.default_rng(seed)
    target = np.asarray(target)
    folds = np.empty(len(target), dtype=np.int8)
    for label in np.unique(target):
        indices = np.flatnonzero(target == label)
        rng.shuffle(indices)
        folds[indices] = (np.arange(len(indices)) + rng.integers(n_folds)) % n_folds
    if len(target) < n_folds or len(np.unique(folds)) != n_folds:
        raise ValueError('Every fold must contain samples')
    return folds


def crossfit_bias(fp, quant, folds, n_folds=FOLDS, chunk=1024):
    """Estimate centered class bias using only the complement of each fold."""
    n, c = fp.shape
    if quant.shape != fp.shape or len(folds) != n:
        raise ValueError('Unpaired logits/folds')
    sums = np.zeros((n_folds, c), dtype=np.float64)
    sizes = np.bincount(folds, minlength=n_folds)
    if np.any(sizes == 0) or np.any(sizes == n):
        raise ValueError('Empty training or heldout fold')
    for start in range(0, n, chunk):
        end = min(n, start+chunk)
        delta = quant[start:end].astype(np.float64) - fp[start:end]
        delta -= delta.mean(axis=1, keepdims=True)
        for fold in range(n_folds):
            sums[fold] += delta[folds[start:end] == fold].sum(axis=0)
    return (sums.sum(axis=0)-sums)/(n-sizes)[:, None]


def predict_preserving_ties(logits, original_winner):
    """Prefer the stored quant winner if tied; otherwise choose lowest index."""
    original_winner = np.asarray(original_winner, dtype=int)
    winner = logits.argmax(axis=1)
    keep = logits[np.arange(len(logits)), original_winner] == logits.max(axis=1)
    return np.where(keep, original_winner, winner)


def competitor_features(logits, top1, top2):
    """FP normalized top-2 margin and counts of other nearby classes."""
    z = np.asarray(logits, dtype=np.float64)
    top1, top2 = np.asarray(top1), np.asarray(top2)
    std = z.std(axis=1)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError('Nonpositive or nonfinite class-logit standard deviation')
    ii = np.arange(len(z))
    gap = (z[ii, top1, None]-z)/std[:, None]
    gap[ii, top1] = np.inf
    counts = np.stack([(gap <= t).sum(axis=1) for t in THRESHOLDS], axis=1)
    margin = (z[ii, top1]-z[ii, top2])/std
    return margin, counts


def winner_rank_bounds(fp, quant_winner):
    z = np.asarray(fp)
    value = z[np.arange(len(z)), quant_winner, None]
    return 1+(z > value).sum(axis=1), (z >= value).sum(axis=1)


def model_features(margin, counts, num_classes):
    t = np.log1p(margin)
    basic = np.column_stack([t, np.maximum(t[:, None]-np.log1p(KNOTS), 0),
                             np.full(len(t), np.log(num_classes))])
    return np.column_stack([basic, np.log1p(counts)])


def train_standardize(train, test, weights):
    mean = np.average(train, axis=0, weights=weights)
    scale = np.sqrt(np.average((train-mean)**2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1.
    return (train-mean)/scale, (test-mean)/scale, mean, scale


def pair_order(ids, targets, quant_ids, quant_targets):
    index = pd.Index(quant_ids)
    if not index.is_unique or len(index) != len(ids):
        raise ValueError('Invalid quantized sample IDs')
    order = index.get_indexer(ids)
    if np.any(order < 0) or not np.array_equal(quant_targets[order], targets):
        raise ValueError('Samples or targets do not match')
    return order


def protocol(reference, manifest, output):
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        serial=SEED, reference_config=str(reference), reference_sha256=sha256_file(reference),
        manifest=str(manifest), manifest_sha256=sha256_file(manifest), output=str(output),
        script_sha256=sha256_file(__file__),
        scope='14 datasets, old ViT-B/16 FT, one checkpoint/seed per dataset, four HQQ weight conditions',
        baseline='unquantized weights with AMP FP16; not pure FP32',
        bias=dict(folds=FOLDS, seed=SEED, assignment='class-stratified shuffled round-robin',
                  estimate='mean training-fold quant-FP logits after removing per-sample class mean',
                  correction='heldout quant logits minus training-fold class bias, strength fixed at 1',
                  tie_rule='prefer stored quant winner if it remains maximal, else lowest class index',
                  shuffles=SHUFFLES, shuffle_seed=SEED+11,
                  shuffle_control='permute class indices of estimated bias; same permutation across folds',
                  primary='dataset-equal mean corrected-minus-original micro accuracy, percentage points',
                  limitation='cross-fitting on existing test samples; retrospective diagnostic, not independent calibration'),
        competitors=dict(thresholds=THRESHOLDS.tolist(), normalization='FP class-logit std ddof=0',
                         count='other classes within threshold of stored FP top1',
                         rank='lower=1+number strictly greater, upper=number greater or equal',
                         outcomes=['damage among FP-correct samples', 'top1 swap among all samples'],
                         comparator='log1p normalized top2 margin plus fixed hinge basis and log class count',
                         margin_knots=KNOTS.tolist(), extended='comparator plus log1p counts at three thresholds',
                         swap_control='FP correctness indicator added to both models',
                         validation='leave-one-dataset-out; exploratory leave-soybean-family-out in both directions',
                         fitting='train-only weighted standardization; logistic lbfgs L2 C=100 max_iter=1000 tol=1e-7',
                         weights='each training dataset has total sample weight 1',
                         primary='heldout Brier score, then dataset-equal aggregation',
                         secondary=['log loss', 'AUROC'],
                         no_tuning=True),
        inference='no p-values from image-level or repeated-condition pseudo-replication; descriptive diagnostic only',
    )


def analyze_pairs(reference_config, output, chunk):
    manifest_path = Path(reference_config['manifest'])
    if sha256_file(manifest_path) != reference_config['manifest_sha256']:
        raise ValueError('Manifest hash changed')
    df = pd.read_csv(manifest_path)
    chosen = df[(df.serial == SEED) & df.dataset_name.isin(reference_config['datasets'])
                & df.condition.isin([BASE, *CONDITIONS])].copy()
    if len(chosen) != 70 or chosen.duplicated(['dataset_name', 'condition']).any():
        raise ValueError('Expected exactly 70 unique artifacts')
    expected = {r['path']: r['sha256'] for r in reference_config['current_raw_npz_sha256']}
    audit = []
    # Complete source-identity audit before any results or model fitting.
    for row in chosen.itertuples():
        actual = sha256_file(row.npz_path)
        if actual != expected.get(row.npz_path):
            raise ValueError(f'Artifact hash changed: {row.npz_path}')
        audit.append(dict(dataset_name=row.dataset_name, condition=row.condition,
                          path=row.npz_path, sha256=actual, reference_hash_match=True))
    pd.DataFrame(audit).to_csv(output/'source_audit.csv', index=False)
    bias_rows, ranks, bins, vectors, fold_audit = [], [], [], [], []
    cache, data = {}, {}
    for dataset in sorted(chosen.dataset_name.unique()):
        part = chosen[chosen.dataset_name == dataset].set_index('condition', drop=False)
        if set(part.index) != {BASE, *CONDITIONS}:
            raise ValueError('Incomplete condition set')
        for key in ('checkpoint_sha256', 'seed', 'model_name', 'git_commit', 'amp_fp16', 'compute_dtype', 'ckpt_path'):
            if part[key].nunique(dropna=False) != 1:
                raise ValueError(f'Identity mismatch: {dataset} {key}')
        configs = [json.loads(v) for v in part.config_json]
        for key in ('num_images_train', 'num_classes', 'num_images_test', 'train_trainval', 'eval_split', 'hqq_exclude'):
            if len({json.dumps(cfg.get(key), sort_keys=True) for cfg in configs}) != 1:
                raise ValueError(f'Protocol mismatch: {dataset} {key}')
        if configs[0].get('eval_split') != 'test' or not bool(part.loc[BASE].amp_fp16):
            raise ValueError('Unexpected baseline/split protocol')
        with np.load(part.loc[BASE].npz_path, allow_pickle=False) as ba:
            fp, fp_acc, tie_disagree, tie_count = validate_raw(ba, part.loc[BASE], chunk)
            ids, target = ba['sample_id'], ba['target']
            top1, top2 = ba['top1_class'], ba['top2_class']
            fp_correct = ba['correct'].astype(bool)
            n, c = fp.shape
            folds = assign_folds(target)
            margin, counts = np.empty(n), np.empty((n, 3), dtype=np.int32)
            for start in range(0, n, chunk):
                end = min(n, start+chunk)
                margin[start:end], counts[start:end] = competitor_features(fp[start:end], top1[start:end], top2[start:end])
            features = model_features(margin, counts, c)
            data[dataset] = dict(features=features, fp_correct=fp_correct, labels={})
            cache[f'{dataset}_features'] = features
            cache[f'{dataset}_fp_correct'] = fp_correct
            cache[f'{dataset}_folds'] = folds
            cache[f'{dataset}_target'] = target
            cache[f'{dataset}_sample_id'] = ids
            class_sizes = np.bincount(target, minlength=c)
            for fold in range(FOLDS):
                held = folds == fold
                train_sizes = np.bincount(target[~held], minlength=c)
                fold_audit.append(dict(dataset_name=dataset, fold=fold, n_test=int(held.sum()),
                                       n_train=int((~held).sum()), classes_missing_train=int((train_sizes == 0).sum()),
                                       n_samples=n, num_classes=c, fp_tied_maxima=tie_count,
                                       fp_argmax_disagreements=tie_disagree))
            for condition in CONDITIONS:
                with np.load(part.loc[condition].npz_path, allow_pickle=False) as qa:
                    quant, q_acc, q_tie_disagree, q_tie_count = validate_raw(qa, part.loc[condition], chunk)
                    order = pair_order(ids, target, qa['sample_id'], qa['target'])
                    # Avoid copying an entire large matrix when source ordering already matches.
                    if not np.array_equal(order, np.arange(n)):
                        quant = quant[order]
                    qpred = qa['top1_class'][order]
                    q_correct = qpred == target
                    means = crossfit_bias(fp, quant, folds, FOLDS, chunk)
                    rng = np.random.default_rng(SEED+11)
                    shuffled = np.stack([means[:, rng.permutation(c)] for _ in range(SHUFFLES)])
                    pred = np.empty(n, dtype=np.int32)
                    control_correct = np.empty((n, SHUFFLES), dtype=bool)
                    lower, upper = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int32)
                    for start in range(0, n, chunk):
                        end = min(n, start+chunk)
                        z = quant[start:end].astype(np.float64)
                        fold = folds[start:end]
                        pred[start:end] = predict_preserving_ties(z-means[fold], qpred[start:end])
                        for shuffle in range(SHUFFLES):
                            control_correct[start:end, shuffle] = predict_preserving_ties(z-shuffled[shuffle, fold], qpred[start:end]) == target[start:end]
                        lower[start:end], upper[start:end] = winner_rank_bounds(fp[start:end], qpred[start:end])
                    corrected = pred == target
                    swap = qpred != top1
                    damage = fp_correct & ~q_correct
                    rescue = ~fp_correct & q_correct
                    corrected_damage = fp_correct & ~corrected
                    corrected_rescue = ~fp_correct & corrected
                    macro_q = np.mean(np.bincount(target, weights=q_correct, minlength=c)[class_sizes > 0]/class_sizes[class_sizes > 0])
                    macro_fix = np.mean(np.bincount(target, weights=corrected, minlength=c)[class_sizes > 0]/class_sizes[class_sizes > 0])
                    shuffle_acc = 100*control_correct.mean(axis=0)
                    bias_rows.append(dict(dataset_name=dataset, condition=condition, n_samples=n, num_classes=c,
                        fp_acc_pct=100*fp_acc, quant_acc_pct=100*q_acc, corrected_acc_pct=100*corrected.mean(),
                        gain_pp=100*(corrected.mean()-q_acc), macro_gain_pp=100*(macro_fix-macro_q),
                        shuffle_mean_gain_pp=shuffle_acc.mean()-100*q_acc,
                        shuffle_min_gain_pp=shuffle_acc.min()-100*q_acc, shuffle_max_gain_pp=shuffle_acc.max()-100*q_acc,
                        specificity_pp=100*corrected.mean()-shuffle_acc.mean(),
                        improved_n=int((corrected & ~q_correct).sum()), harmed_n=int((~corrected & q_correct).sum()),
                        damage_n=int(damage.sum()), corrected_damage_n=int(corrected_damage.sum()),
                        rescue_n=int(rescue.sum()), corrected_rescue_n=int(corrected_rescue.sum()),
                        fp_correct_n=int(fp_correct.sum()), quant_correct_n=int(q_correct.sum()), corrected_correct_n=int(corrected.sum()),
                        bias_rms=float(np.sqrt(np.mean(means**2))), q_tied_maxima=q_tie_count,
                        q_argmax_disagreements=q_tie_disagree))
                    for fold in range(FOLDS):
                        for cls in range(c):
                            vectors.append(dict(dataset_name=dataset, condition=condition, fold=fold, class_index=cls, bias=float(means[fold, cls])))
                    rank_row = dict(dataset_name=dataset, condition=condition, swaps_n=int(swap.sum()), damage_n=int(damage.sum()))
                    for label, mask in [('swap', swap), ('damage', damage)]:
                        beyond = mask & (lower >= 3)
                        uncertain = mask & (lower < 3) & (upper >= 3)
                        rank_row[f'{label}_definitely_beyond_top2_n'] = int(beyond.sum())
                        rank_row[f'{label}_definitely_beyond_top2_pct'] = 100*beyond.sum()/mask.sum() if mask.any() else np.nan
                        rank_row[f'{label}_rank2_3_tie_ambiguous_n'] = int(uncertain.sum())
                        rank_row[f'{label}_rank_lower_median'] = float(np.median(lower[mask])) if mask.any() else np.nan
                    ranks.append(rank_row)
                    bin_ids = np.digitize(margin, MARGIN_BINS[1:-1])
                    for correct_flag in (False, True):
                        for bin_id in range(len(MARGIN_BINS)-1):
                            mask = (fp_correct == correct_flag) & (bin_ids == bin_id)
                            if not mask.any():
                                continue
                            bins.append(dict(dataset_name=dataset, condition=condition, fp_correct=correct_flag,
                                bin_lower=MARGIN_BINS[bin_id], bin_upper=MARGIN_BINS[bin_id+1], n=int(mask.sum()),
                                swap_rate=float(swap[mask].mean()), damage_rate=float(damage[mask].mean()),
                                competitors_025_mean=float(counts[mask, 0].mean()),
                                competitors_050_mean=float(counts[mask, 1].mean()), competitors_100_mean=float(counts[mask, 2].mean())))
                    data[dataset]['labels'][condition] = dict(swap=swap, damage=damage)
                    cache[f'{dataset}_{condition}_swap'] = swap
                    cache[f'{dataset}_{condition}_damage'] = damage
                    cache[f'{dataset}_{condition}_qpred'] = qpred
                    cache[f'{dataset}_{condition}_corrected_pred'] = pred
                    print(f'paired {dataset} {condition}: bias gain {bias_rows[-1]["gain_pp"]:+.4f} pp', flush=True)
    for name, rows in [('bias_results', bias_rows), ('winner_ranks', ranks), ('margin_bins', bins),
                       ('bias_vectors', vectors), ('fold_audit', fold_audit)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv', index=False)
    np.savez_compressed(output/'derived_features_and_predictions.npz', **cache)
    return data


def predict_dataset_split(data, condition, outcome, train_names, test_names, expanded):
    blocks, labels, weights = [], [], []
    basic_cols = 2+len(KNOTS)
    for name in train_names:
        item = data[name]
        mask = item['fp_correct'] if outcome == 'damage' else np.ones(len(item['fp_correct']), dtype=bool)
        x = item['features'][mask]
        if not expanded:
            x = x[:, :basic_cols]
        if outcome == 'swap':
            x = np.column_stack([x, item['fp_correct'][mask]])
        blocks.append(x)
        labels.append(item['labels'][condition][outcome][mask])
        weights.append(np.full(len(x), 1/len(x)))
    train_x, train_y, sample_weights = np.concatenate(blocks), np.concatenate(labels), np.concatenate(weights)
    if np.unique(train_y).size != 2:
        raise ValueError('Training outcome has a single class')
    test_blocks = []
    for name in test_names:
        item = data[name]
        mask = item['fp_correct'] if outcome == 'damage' else np.ones(len(item['fp_correct']), dtype=bool)
        x = item['features'][mask]
        if not expanded:
            x = x[:, :basic_cols]
        if outcome == 'swap':
            x = np.column_stack([x, item['fp_correct'][mask]])
        test_blocks.append(x)
    test_x = np.concatenate(test_blocks)
    train_x, test_x, mean, scale = train_standardize(train_x, test_x, sample_weights)
    model = LogisticRegression(C=100., max_iter=1000, tol=1e-7, solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train_x, train_y, sample_weight=sample_weights)
    predictions = model.predict_proba(test_x)[:, 1]
    coefficients = dict(intercept=float(model.intercept_[0]), coef=model.coef_[0].tolist(),
                        standardizer_mean=mean.tolist(), standardizer_scale=scale.tolist(),
                        iterations=int(model.n_iter_[0]), train_datasets=list(train_names), test_datasets=list(test_names))
    offset, results = 0, []
    for name, x in zip(test_names, test_blocks):
        p = predictions[offset:offset+len(x)]
        offset += len(x)
        item = data[name]
        mask = item['fp_correct'] if outcome == 'damage' else np.ones(len(item['fp_correct']), dtype=bool)
        y = item['labels'][condition][outcome][mask]
        results.append(dict(dataset_name=name, condition=condition, outcome=outcome,
                            model='margin_plus_competitors' if expanded else 'margin_only',
                            n_test=len(y), event_rate=float(y.mean()), brier=brier_score_loss(y, p),
                            log_loss=log_loss(y, p, labels=[False, True]),
                            auroc=roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan,
                            n_train=len(train_y), n_train_datasets=len(train_names), iterations=int(model.n_iter_[0])))
    return results, coefficients


def predictive_tests(data, output):
    names = sorted(data)
    rows, fits = [], []
    splits = [('lodo', [n for n in names if n != held], [held]) for held in names]
    splits += [('nonsoy_to_soy', [n for n in names if n not in SOY], [n for n in names if n in SOY]),
               ('soy_to_nonsoy', [n for n in names if n in SOY], [n for n in names if n not in SOY])]
    with threadpool_limits(limits=1):
        for condition in CONDITIONS:
            for outcome in ('damage', 'swap'):
                for scope, train, test in splits:
                    for expanded in (False, True):
                        result, fit = predict_dataset_split(data, condition, outcome, train, test, expanded)
                        for row in result:
                            row['scope'] = scope
                        rows.extend(result)
                        fits.append(dict(scope=scope, condition=condition, outcome=outcome, expanded=expanded, **fit))
                    print(f'prediction {condition} {outcome} {scope} heldout={",".join(test)}', flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'heldout_prediction_scores.csv', index=False)
    (output/'model_fits.json').write_text(json.dumps(fits, indent=2), encoding='utf-8')
    keys = ['scope', 'dataset_name', 'condition', 'outcome']
    basic = scores[scores.model == 'margin_only'].set_index(keys)
    extended = scores[scores.model == 'margin_plus_competitors'].set_index(keys)
    comparisons = basic[['n_test', 'event_rate']].copy()
    for metric in ('brier', 'log_loss', 'auroc'):
        comparisons[f'margin_{metric}'] = basic[metric]
        comparisons[f'competitors_{metric}'] = extended[metric]
        comparisons[f'delta_{metric}'] = extended[metric]-basic[metric]
    comparisons.reset_index().to_csv(output/'heldout_prediction_comparisons.csv', index=False)


def write_report(output):
    bias = pd.read_csv(output/'bias_results.csv')
    ranks = pd.read_csv(output/'winner_ranks.csv')
    pred = pd.read_csv(output/'heldout_prediction_comparisons.csv')
    summaries = []
    lines = ['# Bias and competitor tests — serial 100004', '',
        'Executed locally on existing paired test logits: 14 datasets, 56 pairs, ViT-B/16 FT. '
        'Baseline is unquantized weights with AMP FP16. No new inference or training was run.', '',
        '## Test 1: five-fold class bias correction', '',
        'Estimate centered quant-FP class bias on four folds, subtract it on the fifth. '
        'Strength is fixed at 1. Five class-index permutation controls retain bias magnitudes. '
        'Cross-fitting uses existing test data and diagnoses a mechanism; this is not an independently calibrated benchmark.', '',
        '| Condition | Mean gain (pp) | Improved datasets | Shuffle gain (pp) | Gain over shuffle (pp) |',
        '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        b = bias[bias.condition == condition]
        lines.append(f'| {condition} | {b.gain_pp.mean():+.4f} | {(b.gain_pp > 0).sum()}/14 | {b.shuffle_mean_gain_pp.mean():+.4f} | {b.specificity_pp.mean():+.4f} |')
        summaries.append(dict(test='bias', condition=condition, mean_gain_pp=b.gain_pp.mean(),
                              improved_datasets=int((b.gain_pp > 0).sum()), shuffle_mean_gain_pp=b.shuffle_mean_gain_pp.mean(),
                              specificity_pp=b.specificity_pp.mean()))
    lines += ['', '## Test 2: competitors beyond the top-two margin', '',
        'The primary outcome is quantization damage among baseline-correct images. '
        'Each dataset is held out in turn. Both fixed logistic models use a nonlinear margin basis and log class count; '
        'the expanded model adds counts of competitors within 0.25, 0.5, and 1 FP logit standard deviations. '
        'Standardization is fitted on training datasets only, each of which has total weight 1. '
        'Negative Brier/log-loss changes and positive AUROC changes favor the added competitors.', '',
        '| Condition | Outcome | Brier change | Log-loss change | AUROC change | Brier improved datasets |',
        '|---|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for outcome in ('damage', 'swap'):
            p = pred[(pred.scope == 'lodo') & (pred.condition == condition) & (pred.outcome == outcome)]
            lines.append(f'| {condition} | {outcome} | {p.delta_brier.mean():+.6f} | {p.delta_log_loss.mean():+.6f} | {p.delta_auroc.mean():+.6f} | {(p.delta_brier < 0).sum()}/14 |')
            summaries.append(dict(test='competitors', condition=condition, outcome=outcome,
                                  delta_brier=p.delta_brier.mean(), delta_log_loss=p.delta_log_loss.mean(),
                                  delta_auroc=p.delta_auroc.mean(), improved_datasets=int((p.delta_brier < 0).sum())))
    lines += ['', 'Tie-aware ranks give direct descriptive evidence of competitors beyond the top two:', '',
              '| Condition | Swaps definitely from FP rank >=3 (%) | Damage definitely from FP rank >=3 (%) |',
              '|---|---:|---:|']
    for condition in CONDITIONS:
        r = ranks[ranks.condition == condition]
        lines.append(f'| {condition} | {r.swap_definitely_beyond_top2_pct.mean():.3f} | {r.damage_definitely_beyond_top2_pct.mean():.3f} |')
    lines += ['', '## Exploratory soybean-family holdout', '',
              '| Direction | Outcome | Brier change, equal dataset/condition weight | Improved cells |',
              '|---|---|---:|---:|']
    for scope in ('nonsoy_to_soy', 'soy_to_nonsoy'):
        for outcome in ('damage', 'swap'):
            p = pred[(pred.scope == scope) & (pred.outcome == outcome)]
            lines.append(f'| {scope} | {outcome} | {p.delta_brier.mean():+.6f} | {(p.delta_brier < 0).sum()}/{len(p)} |')
    lines += ['', '## Interpretation boundaries', '',
        '- All aggregate tables weight datasets equally; four conditions reuse the same datasets/images.',
        '- One trained checkpoint and seed per dataset. Dataset properties and checkpoint training are confounded.',
        '- Bias folds use target labels for stratification; bias magnitudes use only paired logits, not correctness labels.',
        '- All-sample swap models include known FP correctness in both comparators as an analysis control; they are not label-free deployment predictors.',
        '- A quant winner tied across FP ranks 2 and 3 is not counted as definitely beyond top two.',
        '- No hyperparameters, thresholds, or models were selected from these results. Competitor counts cover a fixed nearby spectrum, not every possible full-logit feature.',
        '- Source hashes and checkpoint/split/sample identity were audited; these results do not transfer automatically to BEiTv2 or serial 20004.',
        '', '## Outputs and reproduction', '',
        '`protocol.json` was written before source-result analysis; `source_audit.csv` records all 70 artifact hashes. '
        '`fold_audit.csv`, `bias_vectors.csv`, and `bias_results.csv` retain the cross-fit evidence. '
        '`winner_ranks.csv` and `margin_bins.csv` give descriptive diagnostics. '
        '`heldout_prediction_comparisons.csv` and `model_fits.json` retain all heldout model results and coefficients. '
        '`derived_features_and_predictions.npz` permits accuracy/count verification without reloading all logits.', '',
        '```powershell',
        'python -B -m hqq_confidence.analyze_bias_competitors --reference-config hqq_confidence/analysis/dataset_relationship_20261010/raw_logit_run_config.json --output-dir hqq_confidence/analysis/bias_competitors_20261011',
        '```', '', 'Use a new output directory when reproducing; existing output directories are never overwritten.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    pd.DataFrame(summaries).to_csv(output/'summary.csv', index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    names = sorted(bias.dataset_name.unique())
    for i, condition in enumerate(CONDITIONS):
        b = bias[bias.condition == condition].set_index('dataset_name').loc[names]
        axes[0].plot(b.gain_pp, np.arange(len(names))+(i-1.5)*.14, 'o', label=condition)
        p = pred[(pred.scope == 'lodo') & (pred.condition == condition) & (pred.outcome == 'damage')].set_index('dataset_name').loc[names]
        axes[1].plot(p.delta_brier, np.arange(len(names))+(i-1.5)*.14, 'o', label=condition)
    for ax in axes:
        ax.axvline(0, color='gray', linewidth=1)
        ax.set_yticks(np.arange(len(names)), names)
        ax.grid(axis='x', alpha=.2)
    axes[0].set_xlabel('Cross-fit bias correction accuracy gain (pp)')
    axes[1].set_xlabel('Heldout damage Brier change (competitors - margin)')
    axes[0].set_title('Test 1: class-specific bias')
    axes[1].set_title('Test 2: additional competitor information')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'test_overview.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--chunk-size', type=int, default=1024)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error('chunk-size must be positive')
    reference = json.loads(args.reference_config.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen = protocol(args.reference_config, Path(reference['manifest']), args.output_dir)
    (args.output_dir/'protocol.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    data = analyze_pairs(reference, args.output_dir, args.chunk_size)
    predictive_tests(data, args.output_dir)
    write_report(args.output_dir)
    completion = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                      script_sha256=sha256_file(__file__), protocol_sha256=sha256_file(args.output_dir/'protocol.json'),
                      output_hashes={p.name: sha256_file(p) for p in args.output_dir.iterdir() if p.is_file()})
    (args.output_dir/'completion.json').write_text(json.dumps(completion, indent=2), encoding='utf-8')
    print(f'Complete: {args.output_dir / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
