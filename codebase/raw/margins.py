"""Within-dataset baseline-margin ranking helpers."""

import numpy as np
import pandas as pd


def binary_auroc(scores, outcomes):
    """Return rank-based AUROC and an explicit identifiability status."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=bool)
    if scores.ndim != 1 or outcomes.shape != scores.shape or not len(scores):
        raise ValueError('scores and outcomes must be non-empty aligned vectors')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite')
    positive = int(outcomes.sum())
    negative = int(len(outcomes) - positive)
    if not positive or not negative:
        return np.nan, 'undefined_single_class'
    ranks = pd.Series(scores).rank(method='average').to_numpy(dtype=float)
    rank_sum = float(ranks[outcomes].sum())
    auc = (
        rank_sum - positive * (positive + 1) / 2
    ) / (positive * negative)
    return float(auc), 'ok'


def assign_margin_deciles(normalized_margins):
    """Assign within-dataset deciles without splitting tied values."""
    values = np.asarray(normalized_margins, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError('normalized margins must be a non-empty vector')
    if not np.isfinite(values).all():
        raise ValueError('normalized margins must be finite')
    ranks = pd.Series(values).rank(method='average').to_numpy(dtype=float)
    deciles = np.floor((ranks - 1) * 10 / len(values)).astype(int) + 1
    return np.clip(deciles, 1, 10)
