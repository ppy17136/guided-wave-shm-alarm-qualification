#!/usr/bin/env python3
"""Cross-fit the frozen P9D temperature-conditioned healthy threshold."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPSILON = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--p9b-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    p9b = json.loads(args.p9b_summary.read_text(encoding="utf-8"))
    p9b_recall = {item["task"]: item["recall"] for item in p9b["task_results"]}
    rows = []
    for task, task_frame in scores.groupby("task", sort=True):
        temperatures = sorted(task_frame.temperature_C.unique())
        healthy_quantiles = {}
        for temperature in temperatures:
            healthy = task_frame.loc[
                (task_frame.temperature_C == temperature) & (task_frame.label == 0),
                "score",
            ].to_numpy()
            healthy_quantiles[int(temperature)] = float(
                np.quantile(healthy, 0.95, method="higher")
            )
        for test_temperature in temperatures:
            train_temperatures = np.array(
                [value for value in temperatures if value != test_temperature], dtype=float
            )
            train_log_quantiles = np.log(
                np.array(
                    [healthy_quantiles[int(value)] for value in train_temperatures],
                    dtype=float,
                )
                + EPSILON
            )
            coefficients = np.polyfit(train_temperatures, train_log_quantiles, deg=2)
            threshold = float(np.exp(np.polyval(coefficients, test_temperature)))
            held = task_frame.loc[task_frame.temperature_C == test_temperature]
            healthy = held.loc[held.label == 0, "score"].to_numpy()
            diagnostic = held.loc[held.label == 1, "score"].to_numpy()
            rows.append(
                {
                    "task": task,
                    "temperature_C": int(test_temperature),
                    "threshold": threshold,
                    "observed_healthy_q95": healthy_quantiles[int(test_temperature)],
                    "fpr": float(np.mean(healthy > threshold)),
                    "false_positive_paths": int(np.sum(healthy > threshold)),
                    "recall": float(np.mean(diagnostic > threshold)),
                    "detected_paths": int(np.sum(diagnostic > threshold)),
                    "healthy_paths": int(healthy.size),
                    "diagnostic_paths": int(diagnostic.size),
                    "poly_c2": float(coefficients[0]),
                    "poly_c1": float(coefficients[1]),
                    "poly_c0": float(coefficients[2]),
                }
            )

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(args.output_dir / "p9d_crossfit_thresholds.csv", index=False)
    task_results = []
    for task, frame in result_frame.groupby("task", sort=True):
        mean_recall = float(frame.recall.mean())
        task_results.append(
            {
                "task": task,
                "mean_fpr": float(frame.fpr.mean()),
                "worst_temperature_fpr": float(frame.fpr.max()),
                "mean_recall": mean_recall,
                "minimum_temperature_recall": float(frame.recall.min()),
                "p9b_global_threshold_recall": float(p9b_recall[task]),
                "recall_change": float(mean_recall - p9b_recall[task]),
                "task_recall_gate": mean_recall >= p9b_recall[task] - 0.15,
            }
        )

    average_recall = float(result_frame.recall.mean())
    sensitivity = result_frame.loc[
        ~(
            result_frame.task.str.endswith("I2")
            & (result_frame.temperature_C == 45)
        )
    ]
    summary = {
        "schema_version": "p9d-temperature-conditioned-threshold-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "research_protocols/P9D_temperature_conditioned_threshold_protocol_v1.md",
        "score": "phase_aligned_quadratic normalized residual energy",
        "crossfit": "leave-one-temperature-out log-q95 quadratic regression",
        "task_results": task_results,
        "overall": {
            "mean_fpr": float(result_frame.fpr.mean()),
            "worst_temperature_fpr": float(result_frame.fpr.max()),
            "average_recall": average_recall,
            "sensitivity_average_recall_excluding_I2_45C": float(
                sensitivity.recall.mean()
            ),
            "minimum_threshold": float(result_frame.threshold.min()),
            "maximum_threshold": float(result_frame.threshold.max()),
        },
        "gates": {
            "all_temperature_fpr_at_most_0_10": bool(
                (result_frame.fpr <= 0.10).all()
            ),
            "average_recall_at_least_0_5896": average_recall >= 0.5896,
            "all_task_recall_gates": all(
                item["task_recall_gate"] for item in task_results
            ),
            "all_thresholds_finite_positive": bool(
                np.isfinite(result_frame.threshold).all()
                and (result_frame.threshold > 0).all()
            ),
        },
    }
    summary["passed_all_gates"] = all(summary["gates"].values())
    (args.output_dir / "p9d_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (task, frame) in zip(axes.ravel(), result_frame.groupby("task", sort=True)):
        ax.plot(frame.temperature_C, frame.observed_healthy_q95, "o-", label="Observed q95")
        ax.plot(frame.temperature_C, frame.threshold, "s--", label="Cross-fit threshold")
        ax.set_yscale("log")
        ax.set_title(task)
        ax.set_ylabel("P9B score threshold")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Temperature (C)")
    axes[-1, 1].set_xlabel("Temperature (C)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("P9D leave-one-temperature-out healthy threshold")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig01_crossfit_thresholds.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
