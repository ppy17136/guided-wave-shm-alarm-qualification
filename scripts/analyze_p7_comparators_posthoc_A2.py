"""Post-result A2 diagnostic comparator analysis.

This script is intentionally NOT part of the P7 confirmatory freeze.  The
frozen merger required seven rows for every comparator, whereas the frozen
block-crossconformal pipeline excluded D9->D10 under its older completeness
rule.  Here the seven-event classical comparison and six shared-event
crossconformal comparison are reported as exploratory diagnostics only.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs/p7_confirmatory_analysis_v1/01"


def exact_sign(differences: np.ndarray) -> tuple[float, float, int]:
    differences = np.asarray(differences, float)
    observed = float(differences.mean())
    null = np.array([
        np.mean(differences * signs)
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ])
    return observed, float(np.mean(null >= observed)), int(len(null))


def holm(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def compare(primary: pd.DataFrame, comparator: pd.DataFrame, name: str, column: str) -> dict:
    shared = primary.merge(
        comparator[["event_index", "transition", column]],
        on=["event_index", "transition"],
        how="inner",
        validate="one_to_one",
    ).sort_values("event_index")
    difference, p_value, permutations = exact_sign(
        shared["roc_auc"].to_numpy(float) - shared[column].to_numpy(float)
    )
    return {
        "comparison": name,
        "events": int(len(shared)),
        "transitions": shared["transition"].tolist(),
        "primary_macro_auc": float(shared["roc_auc"].mean()),
        "comparator_macro_auc": float(shared[column].mean()),
        "paired_macro_auc_difference": difference,
        "one_sided_exact_sign_p": p_value,
        "exact_sign_permutations": permutations,
    }


def main() -> None:
    metrics = pd.read_csv(ANALYSIS / "p7_event_metrics.csv")
    primary = metrics[
        (metrics.strategy == "static_empirical_alpha001")
        & (metrics.score == "path_iqr")
    ].sort_values("event_index")

    classical = pd.read_csv(ROOT / "runs/p7_classical_baselines_v1/event_metrics.csv")
    classical = classical[
        (classical.method == "environment_NN_OBS_mean")
        & np.isclose(classical.calibration_quantile, .999)
    ].sort_values("event_index")

    cross = pd.read_csv(ROOT / "runs/p7_block_crossconformal_v1/crossconformal_metrics.csv")
    cross = cross[
        (cross.method == "environment_NN_OBS_mean")
        & np.isclose(cross.alpha, .01)
    ].sort_values("event_index")

    rows = [
        compare(
            primary,
            classical,
            "classical_environment_NN_OBS_mean_q999_seven_events",
            "diagnostic_auroc",
        ),
        compare(
            primary,
            cross,
            "block_crossconformal_environment_NN_OBS_mean_alpha01_six_shared_events",
            "diagnostic_auroc_cross_score",
        ),
    ]
    adjusted = holm([row["one_sided_exact_sign_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_exploratory_family"] = float(value)
        row["holm_reject_005"] = bool(value <= .05)

    payload = {
        "schema_version": "p7-comparator-tests-posthoc-A2-v1",
        "analysis_status": "post_result_exploratory_diagnostic_not_confirmatory",
        "reason_frozen_merger_did_not_run": (
            "The frozen block-crossconformal comparator excluded D9->D10 under "
            "its older strict window-completeness rule and therefore supplied six rows."
        ),
        "primary_confirmatory_interpretation_unchanged": "both_confirmatory_endpoints_failed",
        "alternative": "path_iqr event-level AUROC is greater",
        "comparisons": rows,
    }
    frame = pd.DataFrame(rows)
    frame.to_csv(ANALYSIS / "p7_comparator_tests_posthoc_A2.csv", index=False, encoding="utf-8-sig")
    (ANALYSIS / "p7_comparator_tests_posthoc_A2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

