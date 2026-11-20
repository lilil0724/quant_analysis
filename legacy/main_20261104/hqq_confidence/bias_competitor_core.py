"""Portable fixed diagnostic metrics; no TGDA, torch, HQQ or W&B imports.

Semantics match the fixed serial-100004 research snapshot. That historical
entrypoint remains unchanged so its executed-code checksums remain verifiable.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

SEED = 100004
FOLDS = 5
SHUFFLES = 5
THRESHOLDS = np.array([.25, .5, 1.])
KNOTS = np.array([.05, .25, .5, 1., 2., 4., 8.])
SOY = {'soyageing', 'soygene', 'soyglobal', 'soylocal'}
CONDITION = 'w3_g128'


class IncompletePrediction(ValueError):
    """A prespecified model cannot be evaluated; retain an explicit status."""


def assign_folds(target, n_folds=FOLDS, seed=SEED):
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
    original_winner = np.asarray(original_winner, dtype=int)
    winner = logits.argmax(axis=1)
    keep = logits[np.arange(len(logits)), original_winner] == logits.max(axis=1)
    return np.where(keep, original_winner, winner)


def competitor_features(logits, top1, top2):
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


def predict_dataset_split(data, outcome, train_names, test_names, expanded):
    if set(train_names) & set(test_names) or not train_names or not test_names:
        raise ValueError('Training and test datasets must be nonempty and disjoint')
    if outcome not in ('damage', 'swap'):
        raise ValueError('Unknown outcome')

    def select(name):
        item = data[name]
        mask = item['fp_correct'] if outcome == 'damage' else np.ones(len(item['fp_correct']), dtype=bool)
        x = item['features'][mask]
        if not expanded:
            x = x[:, :2+len(KNOTS)]
        if outcome == 'swap':
            x = np.column_stack([x, item['fp_correct'][mask]])
        return x, item['labels'][outcome][mask]

    blocks, labels, weights = [], [], []
    for name in train_names:
        x, y = select(name)
        if not len(x):
            raise IncompletePrediction(f'empty_training_population:{name}')
        blocks.append(x)
        labels.append(y)
        weights.append(np.full(len(x), 1/len(x)))
    train_x, train_y, sample_weights = np.concatenate(blocks), np.concatenate(labels), np.concatenate(weights)
    if np.unique(train_y).size != 2:
        raise IncompletePrediction('training_outcome_single_class')
    selected = [select(name) for name in test_names]
    if any(not len(x) for x, _ in selected):
        raise IncompletePrediction('empty_test_population')
    test_x = np.concatenate([x for x, _ in selected])
    train_x, test_x, mean, scale = train_standardize(train_x, test_x, sample_weights)
    model = LogisticRegression(C=100., max_iter=1000, tol=1e-7, solver='lbfgs')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train_x, train_y, sample_weight=sample_weights)
    except ConvergenceWarning as exc:
        raise IncompletePrediction('optimizer_did_not_converge') from exc
    p = model.predict_proba(test_x)[:, 1]
    fit = dict(intercept=float(model.intercept_[0]), coef=model.coef_[0].tolist(),
               standardizer_mean=mean.tolist(), standardizer_scale=scale.tolist(),
               iterations=int(model.n_iter_[0]), train_datasets=list(train_names), test_datasets=list(test_names))
    rows, offset = [], 0
    for name, (x, y) in zip(test_names, selected):
        probability = p[offset:offset+len(x)]
        offset += len(x)
        rows.append(dict(dataset_name=name, outcome=outcome,
                         model='margin_plus_competitors' if expanded else 'margin_only',
                         n_test=len(y), event_rate=float(y.mean()),
                         brier=brier_score_loss(y, probability), log_loss=log_loss(y, probability, labels=[False, True]),
                         auroc=roc_auc_score(y, probability) if np.unique(y).size == 2 else np.nan,
                         auroc_status='ok' if np.unique(y).size == 2 else 'undefined_single_class',
                         n_train=len(train_y), n_train_datasets=len(train_names), iterations=int(model.n_iter_[0])))
    return rows, fit
