"""Merge the three pre-specified P7 AUROC comparisons and apply Holm control."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs/p7_confirmatory_analysis_v1/01"


def exact_sign(differences):
    differences = np.asarray(differences, float)
    observed = float(differences.mean())
    null = np.array([
        np.mean(differences * signs)
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ])
    return observed, float(np.mean(null >= observed))


def holm(p_values):
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    metrics = pd.read_csv(ANALYSIS / "p7_event_metrics.csv")
    primary = metrics[(metrics.strategy == "static_empirical_alpha001") & (metrics.score == "path_iqr")].sort_values("event_index")
    max_median = metrics[(metrics.strategy == "static_empirical_alpha001") & (metrics.score == "max_minus_median")].sort_values("event_index")

    classical = pd.read_csv(ROOT / "runs/p7_classical_baselines_v1/event_metrics.csv")
    classical = classical[(classical.method == "environment_NN_OBS_mean") & np.isclose(classical.calibration_quantile, .999)].sort_values("event_index")
    cross = pd.read_csv(ROOT / "runs/p7_block_crossconformal_v1/crossconformal_metrics.csv")
    cross = cross[(cross.method == "environment_NN_OBS_mean") & np.isclose(cross.alpha, .01)].sort_values("event_index")
    if not (len(primary) == len(max_median) == len(classical) == len(cross) == 7):
        raise SystemExit("Expected exactly seven rows for each locked comparator")
    if not (
        np.array_equal(primary.event_index, max_median.event_index)
        and np.array_equal(primary.event_index, classical.event_index)
        and np.array_equal(primary.event_index, cross.event_index)
    ):
        raise SystemExit("Comparator event order mismatch")

    comparisons = {
        "max_minus_median": max_median.roc_auc.to_numpy(float),
        "classical_environment_NN_OBS_mean_q999": classical.diagnostic_auroc.to_numpy(float),
        "block_crossconformal_environment_NN_OBS_mean_alpha01": cross.diagnostic_auroc_cross_score.to_numpy(float),
    }
    rows = []
    primary_auc = primary.roc_auc.to_numpy(float)
    for name, values in comparisons.items():
        difference, p_value = exact_sign(primary_auc - values)
        rows.append({
            "comparison": name,
            "primary_macro_auc": float(primary_auc.mean()),
            "comparator_macro_auc": float(values.mean()),
            "paired_macro_auc_difference": difference,
            "one_sided_exact_sign_p": p_value,
        })
    adjusted = holm([row["one_sided_exact_sign_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(value)
        row["holm_reject_005"] = bool(value <= .05)
    frame = pd.DataFrame(rows)
    frame.to_csv(ANALYSIS / "p7_locked_comparator_tests.csv", index=False, encoding="utf-8-sig")
    payload = {
        "schema_version": "p7-locked-comparator-tests-v1",
        "familywise_alpha": .05,
        "alternative": "path_iqr event-level AUROC is greater",
        "exact_sign_permutations": 128,
        "comparisons": rows,
    }
    (ANALYSIS / "p7_locked_comparator_tests.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

