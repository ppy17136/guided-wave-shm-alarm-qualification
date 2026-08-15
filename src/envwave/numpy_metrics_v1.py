"""NumPy-only binary metrics used to avoid SciPy ABI coupling on clusters."""
from __future__ import annotations

import numpy as np


def _binary(labels, scores):
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != s.size:
        raise ValueError("labels and scores must have equal length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    if not np.isfinite(s).all():
        raise ValueError("scores contain non-finite values")
    return y, s


def roc_auc(labels, scores) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""
    y, s = _binary(labels, scores)
    positive = y == 1
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if not n_positive or not n_negative:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    start = 0
    while start < s.size:
        stop = start + 1
        while stop < s.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    rank_sum = float(ranks[positive].sum())
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def average_precision(labels, scores) -> float:
    y, s = _binary(labels, scores)
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    positives = int(ranked.sum())
    if not positives:
        raise ValueError("Average precision requires positive samples")
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked == 1].sum() / positives)


def brier_score(labels, probability) -> float:
    y, p = _binary(labels, probability)
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return float(np.mean(np.square(p - y)))


def log_loss_binary(labels, probability, eps: float = 1e-6) -> float:
    y, p = _binary(labels, probability)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
