#!/usr/bin/env python3
"""Run the frozen P9B phase-aligned quadratic temperature interpolation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_p9a_lambnet_t_baselines import (
    ANCHORS,
    HELD_TEMPERATURES,
    TASKS,
    find_file,
    normalized_residual_energy,
    read_waveforms,
    summarize_rows,
)


STRETCH_GRID = np.linspace(0.99, 1.01, 81)
REFERENCE_TEMPERATURE = 30


def warp(signal: np.ndarray, factor: float) -> np.ndarray:
    index = np.arange(signal.size, dtype=float)
    return np.interp(index * factor, index, signal, left=0.0, right=0.0)


def best_factor(signal: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    index = np.arange(signal.size, dtype=float)
    reference_centered = reference - np.mean(reference)
    reference_norm = np.sqrt(np.sum(reference_centered**2))
    best_correlation = -np.inf
    selected_factor = 1.0
    for factor in STRETCH_GRID:
        candidate = np.interp(index * factor, index, signal, left=0.0, right=0.0)
        candidate -= np.mean(candidate)
        denominator = reference_norm * np.sqrt(np.sum(candidate**2))
        correlation = float(np.dot(reference_centered, candidate) / max(denominator, 1e-12))
        if correlation > best_correlation:
            best_correlation = correlation
            selected_factor = float(factor)
    return selected_factor, best_correlation


def quadratic_weights(temperature: float) -> np.ndarray:
    design = np.column_stack([np.ones_like(ANCHORS), ANCHORS, ANCHORS**2])
    target = np.array([1.0, temperature, temperature**2])
    return target @ np.linalg.pinv(design)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--p9a-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    factor_rows = []
    target_factor_rows = []
    reference_anchor_index = int(np.where(ANCHORS == REFERENCE_TEMPERATURE)[0][0])

    for task, baseline_name, diagnostic_name in TASKS:
        baseline, paths, temperatures = read_waveforms(find_file(args.raw_root, baseline_name))
        diagnostic, diagnostic_paths, diagnostic_temperatures = read_waveforms(
            find_file(args.raw_root, diagnostic_name)
        )
        if paths != diagnostic_paths or temperatures != diagnostic_temperatures:
            raise ValueError(f"Baseline/diagnostic structure mismatch for {task}")
        temp_index = {value: i for i, value in enumerate(temperatures)}
        anchor_wave = baseline[[temp_index[int(value)] for value in ANCHORS], :, :]
        factors = np.ones((len(ANCHORS), len(paths)), dtype=float)
        correlations = np.ones_like(factors)
        reference = anchor_wave[reference_anchor_index]

        for anchor_id, temperature in enumerate(ANCHORS.astype(int)):
            if anchor_id == reference_anchor_index:
                continue
            for path_id in range(len(paths)):
                factor, correlation = best_factor(
                    anchor_wave[anchor_id, path_id], reference[path_id]
                )
                factors[anchor_id, path_id] = factor
                correlations[anchor_id, path_id] = correlation

        aligned = np.empty_like(anchor_wave)
        for anchor_id in range(len(ANCHORS)):
            for path_id in range(len(paths)):
                aligned[anchor_id, path_id] = warp(
                    anchor_wave[anchor_id, path_id], factors[anchor_id, path_id]
                )
                factor_rows.append(
                    {
                        "task": task,
                        "path": paths[path_id],
                        "anchor_temperature_C": int(ANCHORS[anchor_id]),
                        "stretch_factor": float(factors[anchor_id, path_id]),
                        "alignment_correlation": float(correlations[anchor_id, path_id]),
                        "at_grid_boundary": bool(
                            factors[anchor_id, path_id] == STRETCH_GRID[0]
                            or factors[anchor_id, path_id] == STRETCH_GRID[-1]
                        ),
                    }
                )

        for temperature in HELD_TEMPERATURES:
            weights = quadratic_weights(float(temperature))
            aligned_prediction = np.einsum("a,apt->pt", weights, aligned)
            target_factors = np.einsum("a,ap->p", weights, factors)
            predicted = np.empty_like(aligned_prediction)
            for path_id in range(len(paths)):
                predicted[path_id] = warp(
                    aligned_prediction[path_id], 1.0 / target_factors[path_id]
                )
                target_factor_rows.append(
                    {
                        "task": task,
                        "path": paths[path_id],
                        "temperature_C": int(temperature),
                        "interpolated_stretch_factor": float(target_factors[path_id]),
                    }
                )
            idx = temp_index[int(temperature)]
            baseline_score = normalized_residual_energy(baseline[idx], predicted)
            diagnostic_score = normalized_residual_energy(diagnostic[idx], predicted)
            for path_id, path in enumerate(paths):
                rows.append(
                    {
                        "task": task,
                        "temperature_C": int(temperature),
                        "path": path,
                        "method": "phase_aligned_quadratic",
                        "label": 0,
                        "score": float(baseline_score[path_id]),
                    }
                )
                rows.append(
                    {
                        "task": task,
                        "temperature_C": int(temperature),
                        "path": path,
                        "method": "phase_aligned_quadratic",
                        "label": 1,
                        "score": float(diagnostic_score[path_id]),
                    }
                )

    event_summaries, macro = summarize_rows(rows)
    result = macro[0]
    p9a = json.loads(args.p9a_summary.read_text(encoding="utf-8"))
    p9a_quadratic = next(
        item for item in p9a["macro"] if item["method"] == "quadratic_interpolation"
    )
    factors_frame = pd.DataFrame(factor_rows)
    target_factors_frame = pd.DataFrame(target_factor_rows)
    pd.DataFrame(rows).to_csv(args.output_dir / "p9b_scores_long.csv.gz", index=False)
    pd.DataFrame(event_summaries).to_csv(
        args.output_dir / "p9b_task_summary.csv", index=False
    )
    factors_frame.to_csv(args.output_dir / "p9b_anchor_stretch_factors.csv", index=False)
    target_factors_frame.to_csv(
        args.output_dir / "p9b_target_stretch_factors.csv.gz", index=False
    )

    non_reference = factors_frame.anchor_temperature_C != REFERENCE_TEMPERATURE
    boundary_fraction = float(factors_frame.loc[non_reference, "at_grid_boundary"].mean())
    summary = {
        "schema_version": "p9b-phase-aligned-temperature-interpolation-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "research_protocols/P9B_phase_aligned_temperature_interpolation_protocol_v1.md",
        "reference_temperature_C": REFERENCE_TEMPERATURE,
        "stretch_grid": {
            "minimum": float(STRETCH_GRID[0]),
            "maximum": float(STRETCH_GRID[-1]),
            "points": int(STRETCH_GRID.size),
        },
        "result": result,
        "task_results": event_summaries,
        "factor_diagnostics": {
            "minimum_anchor_factor": float(factors_frame.stretch_factor.min()),
            "maximum_anchor_factor": float(factors_frame.stretch_factor.max()),
            "non_reference_boundary_fraction": boundary_fraction,
            "minimum_interpolated_target_factor": float(
                target_factors_frame.interpolated_stretch_factor.min()
            ),
            "maximum_interpolated_target_factor": float(
                target_factors_frame.interpolated_stretch_factor.max()
            ),
        },
        "comparisons": {
            "healthy_error_reduction_vs_p9a_quadratic": float(
                1.0 - result["mean_healthy_score"] / p9a_quadratic["mean_healthy_score"]
            ),
            "macro_auc_change_vs_p9a_quadratic": float(
                result["macro_auc"] - p9a_quadratic["macro_auc"]
            ),
        },
        "gates": {
            "healthy_error_lower_than_p9a_quadratic": result["mean_healthy_score"]
            < p9a_quadratic["mean_healthy_score"],
            "macro_auc_at_least_0_70": result["macro_auc"] >= 0.70,
            "worst_task_auc_at_least_0_60": result["worst_task_auc"] >= 0.60,
            "positive_direction_all_four_tasks": result["positive_direction_tasks"] == 4,
            "worst_temperature_fpr_at_most_0_10": result["worst_temperature_fpr"]
            <= 0.10,
            "less_than_10pct_nonreference_factors_at_boundary": boundary_fraction < 0.10,
        },
    }
    (args.output_dir / "p9b_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for temperature, group in factors_frame.groupby("anchor_temperature_C"):
        x = np.full(group.shape[0], temperature, dtype=float)
        jitter = np.linspace(-0.7, 0.7, group.shape[0])
        ax.scatter(x + jitter, group.stretch_factor, s=7, alpha=0.35)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Anchor temperature (C)")
    ax.set_ylabel("Stretch factor to 30 C phase coordinate")
    ax.set_title("Frozen path-wise stretch estimates")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig01_anchor_stretch_factors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    labels = ["P9A quadratic", "P9B phase-aligned"]
    values = [p9a_quadratic["macro_auc"], result["macro_auc"]]
    ax.bar(labels, values, color=["#2878B5", "#F28522"])
    ax.axhline(0.7, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Four-task macro AUROC")
    ax.set_title("Predeclared P9B comparison")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig02_p9a_p9b_macro_auc.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
