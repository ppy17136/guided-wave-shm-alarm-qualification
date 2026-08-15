"""Health-only calibration and support-aware alarm gates.

This module deliberately does not learn an anomaly score.  It audits a frozen
score threshold, models the environmental/operational support of healthy
reference data, and emits explicit abstentions before applying a sequential
alarm rule.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import ceil, sqrt
from typing import Any, Iterable, Sequence

import numpy as np


MAD_NORMAL_SCALE = 1.4826


@dataclass(frozen=True)
class CalibrationAudit:
    reliable: bool
    reason: str
    threshold: float | None
    median: float | None
    robust_scale: float | None
    samples: int
    blocks: int
    loo_threshold_min: float | None
    loo_threshold_max: float | None
    loo_relative_range: float | None
    max_loo_relative_range: float
    min_samples: int
    min_blocks: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupportModel:
    center: np.ndarray
    scale: np.ndarray
    constant_mask: np.ndarray
    constant_tolerance: np.ndarray
    reference_standardized: np.ndarray
    reference_categories: tuple[tuple[str, ...], ...]
    reference_blocks: np.ndarray
    k: int
    quantile: float
    distance_threshold: float
    reference_internal_distances: np.ndarray


@dataclass(frozen=True)
class SupportAssessment:
    supported: np.ndarray
    distance: np.ndarray
    reason: tuple[str, ...]


@dataclass(frozen=True)
class AlarmAssessment:
    point_alarm: np.ndarray
    sequential_alarm: np.ndarray
    final_status: tuple[str, ...]


def _as_1d(values: Iterable[Any], name: str) -> np.ndarray:
    array = np.asarray(list(values))
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _finite_scores(scores: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(scores), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    return array[np.isfinite(array)]


def robust_location_scale(values: Iterable[float]) -> tuple[float, float]:
    array = _finite_scores(values)
    if array.size == 0:
        raise ValueError("no finite values")
    median = float(np.median(array))
    scale = float(MAD_NORMAL_SCALE * np.median(np.abs(array - median)))
    return median, scale


def hampel_threshold(values: Iterable[float], multiplier: float = 6.0) -> tuple[float, float, float]:
    median, scale = robust_location_scale(values)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("robust scale is zero or non-finite")
    return median + multiplier * scale, median, scale


def audit_calibration(
    scores: Iterable[float],
    block_ids: Iterable[Any],
    *,
    multiplier: float = 6.0,
    max_loo_relative_range: float = 0.25,
    min_samples: int = 100,
    min_blocks: int = 10,
) -> CalibrationAudit:
    raw_scores = np.asarray(list(scores), dtype=np.float64)
    blocks = _as_1d(block_ids, "block_ids")
    if raw_scores.ndim != 1 or raw_scores.size != blocks.size:
        raise ValueError("scores and block_ids must be one-dimensional and equal length")
    valid = np.isfinite(raw_scores)
    scores_v = raw_scores[valid]
    blocks_v = blocks[valid]
    unique_blocks = np.unique(blocks_v)

    common = dict(
        samples=int(scores_v.size),
        blocks=int(unique_blocks.size),
        max_loo_relative_range=float(max_loo_relative_range),
        min_samples=int(min_samples),
        min_blocks=int(min_blocks),
    )
    if scores_v.size < min_samples:
        return CalibrationAudit(False, "insufficient_samples", None, None, None,
                                loo_threshold_min=None, loo_threshold_max=None,
                                loo_relative_range=None, **common)
    if unique_blocks.size < min_blocks:
        return CalibrationAudit(False, "insufficient_blocks", None, None, None,
                                loo_threshold_min=None, loo_threshold_max=None,
                                loo_relative_range=None, **common)
    try:
        threshold, median, scale = hampel_threshold(scores_v, multiplier)
    except ValueError as exc:
        return CalibrationAudit(False, str(exc).replace(" ", "_"), None, None, None,
                                loo_threshold_min=None, loo_threshold_max=None,
                                loo_relative_range=None, **common)

    loo: list[float] = []
    for block in unique_blocks:
        retained = scores_v[blocks_v != block]
        try:
            threshold_b, _, _ = hampel_threshold(retained, multiplier)
        except ValueError:
            return CalibrationAudit(False, "unresolvable_leave_one_block_out", threshold,
                                    median, scale, loo_threshold_min=None,
                                    loo_threshold_max=None, loo_relative_range=None, **common)
        loo.append(threshold_b)

    loo_array = np.asarray(loo, dtype=np.float64)
    denominator = max(abs(threshold), scale, np.finfo(np.float64).eps)
    relative_range = float((loo_array.max() - loo_array.min()) / denominator)
    reliable = bool(relative_range <= max_loo_relative_range)
    reason = "pass" if reliable else "loo_threshold_instability"
    return CalibrationAudit(
        reliable=reliable,
        reason=reason,
        threshold=float(threshold),
        median=float(median),
        robust_scale=float(scale),
        loo_threshold_min=float(loo_array.min()),
        loo_threshold_max=float(loo_array.max()),
        loo_relative_range=relative_range,
        **common,
    )


def _category_rows(categories: Sequence[Sequence[Any]] | None, n: int) -> tuple[tuple[str, ...], ...]:
    if categories is None:
        return tuple(() for _ in range(n))
    array = np.asarray(categories, dtype=object)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] != n:
        raise ValueError("categories must have one row per sample")
    return tuple(tuple(str(value) for value in row) for row in array.tolist())


def _kth_distance(point: np.ndarray, candidates: np.ndarray, k: int) -> float:
    if candidates.shape[0] < k:
        return float("inf")
    distances = np.sqrt(np.sum((candidates - point) ** 2, axis=1))
    return float(np.partition(distances, k - 1)[k - 1])


def fit_support_model(
    reference_numeric: Sequence[Sequence[float]],
    reference_blocks: Iterable[Any],
    reference_categories: Sequence[Sequence[Any]] | None = None,
    *,
    k: int | None = None,
    quantile: float = 0.99,
    constant_tolerance: float | Sequence[float] = 1e-8,
) -> SupportModel:
    numeric = np.asarray(reference_numeric, dtype=np.float64)
    blocks = _as_1d(reference_blocks, "reference_blocks")
    if numeric.ndim != 2 or numeric.shape[0] != blocks.size or numeric.shape[0] < 2:
        raise ValueError("reference_numeric must be a nonempty 2D array matching reference_blocks")
    if not np.all(np.isfinite(numeric)):
        raise ValueError("reference_numeric contains non-finite values")
    n, d = numeric.shape
    categories = _category_rows(reference_categories, n)
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be between 0.5 and 1")

    center = np.median(numeric, axis=0)
    mad_scale = MAD_NORMAL_SCALE * np.median(np.abs(numeric - center), axis=0)
    q25, q75 = np.quantile(numeric, [0.25, 0.75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    scale = np.where(mad_scale > 0.0, mad_scale, iqr_scale)
    constant_mask = ~(np.isfinite(scale) & (scale > 0.0))
    scale = np.where(constant_mask, 1.0, scale)
    standardized = (numeric - center) / scale

    tolerance = np.asarray(constant_tolerance, dtype=np.float64)
    if tolerance.ndim == 0:
        tolerance = np.full(d, float(tolerance))
    if tolerance.shape != (d,) or np.any(tolerance < 0) or not np.all(np.isfinite(tolerance)):
        raise ValueError("constant_tolerance must be finite, nonnegative, and match feature count")

    k_eff = int(k if k is not None else np.clip(ceil(sqrt(n)), 5, 30))
    if k_eff < 1:
        raise ValueError("k must be positive")

    internal = np.empty(n, dtype=np.float64)
    for index in range(n):
        allowed = np.array([
            blocks[j] != blocks[index] and categories[j] == categories[index]
            for j in range(n)
        ], dtype=bool)
        internal[index] = _kth_distance(standardized[index], standardized[allowed], k_eff)
    if not np.all(np.isfinite(internal)):
        raise ValueError("at least one category/block has fewer than k external-block neighbors")
    threshold = float(np.quantile(internal, quantile, method="linear"))
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("support distance threshold is invalid")

    return SupportModel(
        center=center,
        scale=scale,
        constant_mask=constant_mask,
        constant_tolerance=tolerance,
        reference_standardized=standardized,
        reference_categories=categories,
        reference_blocks=blocks,
        k=k_eff,
        quantile=float(quantile),
        distance_threshold=threshold,
        reference_internal_distances=internal,
    )


def assess_support(
    model: SupportModel,
    target_numeric: Sequence[Sequence[float]],
    target_categories: Sequence[Sequence[Any]] | None = None,
) -> SupportAssessment:
    numeric = np.asarray(target_numeric, dtype=np.float64)
    if numeric.ndim != 2 or numeric.shape[1] != model.center.size:
        raise ValueError("target_numeric feature count does not match support model")
    categories = _category_rows(target_categories, numeric.shape[0])
    supported = np.zeros(numeric.shape[0], dtype=bool)
    distance = np.full(numeric.shape[0], np.nan, dtype=np.float64)
    reasons: list[str] = []

    for index, row in enumerate(numeric):
        if not np.all(np.isfinite(row)):
            reasons.append("non_finite_numeric")
            continue
        if np.any(np.abs(row[model.constant_mask] - model.center[model.constant_mask])
                  > model.constant_tolerance[model.constant_mask]):
            reasons.append("constant_feature_mismatch")
            continue
        allowed = np.array([cat == categories[index] for cat in model.reference_categories], dtype=bool)
        if not np.any(allowed):
            reasons.append("unseen_category")
            continue
        standardized = (row - model.center) / model.scale
        kth = _kth_distance(standardized, model.reference_standardized[allowed], model.k)
        distance[index] = kth
        if not np.isfinite(kth):
            reasons.append("insufficient_category_neighbors")
        elif kth <= model.distance_threshold:
            supported[index] = True
            reasons.append("supported")
        else:
            reasons.append("outside_numeric_support")
    return SupportAssessment(supported=supported, distance=distance, reason=tuple(reasons))


def apply_alarm_rule(
    scores: Iterable[float],
    calibration: CalibrationAudit,
    support: Iterable[bool],
    *,
    window: int = 3,
    required: int = 2,
) -> AlarmAssessment:
    score_array = np.asarray(list(scores), dtype=np.float64)
    support_array = np.asarray(list(support), dtype=bool)
    if score_array.ndim != 1 or score_array.size != support_array.size:
        raise ValueError("scores and support must be equal-length one-dimensional arrays")
    if window < 1 or required < 1 or required > window:
        raise ValueError("sequential rule must satisfy 1 <= required <= window")

    point = np.zeros(score_array.size, dtype=bool)
    sequential = np.zeros(score_array.size, dtype=bool)
    status: list[str] = []
    history: deque[bool] = deque(maxlen=window)

    for index, score in enumerate(score_array):
        if not np.isfinite(score):
            history.clear()
            status.append("invalid")
            continue
        if not calibration.reliable or calibration.threshold is None:
            history.clear()
            status.append("abstain_calibration")
            continue
        if not support_array[index]:
            history.clear()
            status.append("abstain_support")
            continue
        point[index] = bool(score > calibration.threshold)
        history.append(bool(point[index]))
        sequential[index] = bool(len(history) == window and sum(history) >= required)
        status.append("alarm" if sequential[index] else "no_alarm")

    return AlarmAssessment(point_alarm=point, sequential_alarm=sequential,
                           final_status=tuple(status))
