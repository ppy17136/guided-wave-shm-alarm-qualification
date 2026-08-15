#!/usr/bin/env python3
"""Run frozen P9E independent healthy-temperature transfer validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_p9e_composite_june2024 import read_sg2
from run_p9b_phase_aligned_interpolation import STRETCH_GRID, best_factor, warp


NOMINAL_ANCHORS = np.array([20.0, 35.0, 50.0, 65.0])
REFERENCE_NOMINAL_TEMPERATURE = 50.0
DIRECT_THRESHOLD = 0.02609810031847493
BASELINE_SAMPLES = 85


def quadratic_weights(temperature: float, anchor_temperatures: np.ndarray) -> np.ndarray:
    design = np.column_stack(
        [np.ones_like(anchor_temperatures), anchor_temperatures, anchor_temperatures**2]
    )
    target = np.array([1.0, temperature, temperature**2])
    return target @ np.linalg.pinv(design)


def normalized_residual_energy(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.mean((query - reference) ** 2, axis=-1)
    denominator = np.mean(reference**2, axis=-1)
    return numerator / np.maximum(denominator, 1e-18)


def temperature_bin(temperature: float) -> int:
    return int(5 * np.floor((temperature + 2.5) / 5.0))


def select_anchors(index: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for nominal in NOMINAL_ANCHORS:
        group = index.loc[
            np.isclose(index.temperature_program_C.to_numpy(dtype=float), nominal), :
        ].sort_values("sequence")
        if group.empty:
            raise ValueError(f"No plateau records for nominal anchor {nominal:g} C")
        row = group.iloc[len(group) // 2].copy()
        row["nominal_anchor_C"] = nominal
        selected.append(row)
    result = pd.DataFrame(selected).reset_index(drop=True)
    if result.sequence.duplicated().any():
        raise ValueError("Anchor selection produced duplicate acquisitions")
    return result


def load_signal(raw_root: Path, filename: str) -> np.ndarray:
    matches = list(raw_root.rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {filename}, found {len(matches)}")
    signal = read_sg2(matches[0], load_data=True)["data"].astype(np.float64)
    signal -= np.median(signal[:, :BASELINE_SAMPLES], axis=1, keepdims=True)
    return signal


def summarize_bins(scores: pd.DataFrame, alarm_column: str) -> pd.DataFrame:
    records = []
    for bin_c, group in scores.groupby("temperature_bin_C", sort=True):
        records.append(
            {
                "temperature_bin_C": int(bin_c),
                "acquisitions": int(group.sequence.nunique()),
                "paths": int(len(group)),
                "temperature_min_C": float(group.temperature_C.min()),
                "temperature_max_C": float(group.temperature_C.max()),
                "fpr": float(group[alarm_column].mean()),
                "acquisition_any_receiver_alarm_rate": float(
                    group.groupby("sequence")[alarm_column].any().mean()
                ),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--audit-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.audit_index)
    index = index.loc[~index.climate_metadata_missing.astype(bool)].copy()
    index["timestamp"] = pd.to_datetime(index.timestamp)
    index = index.sort_values("sequence").reset_index(drop=True)
    if len(index) != 219:
        raise ValueError(f"Expected 219 mapped acquisitions, found {len(index)}")

    anchors = select_anchors(index)
    anchor_sequences = set(anchors.sequence.astype(int))
    anchor_temperatures = anchors.temperature_C.to_numpy(dtype=float)
    reference_index = int(
        np.where(anchors.nominal_anchor_C.to_numpy() == REFERENCE_NOMINAL_TEMPERATURE)[0][0]
    )
    anchor_wave = np.stack(
        [load_signal(args.raw_root, name) for name in anchors.filename], axis=0
    )
    paths = anchor_wave.shape[1]

    factors = np.ones((len(anchors), paths), dtype=float)
    correlations = np.ones_like(factors)
    for anchor_id in range(len(anchors)):
        if anchor_id == reference_index:
            continue
        for path_id in range(paths):
            factor, correlation = best_factor(
                anchor_wave[anchor_id, path_id], anchor_wave[reference_index, path_id]
            )
            factors[anchor_id, path_id] = factor
            correlations[anchor_id, path_id] = correlation

    aligned_anchors = np.empty_like(anchor_wave)
    factor_rows = []
    for anchor_id in range(len(anchors)):
        for path_id in range(paths):
            aligned_anchors[anchor_id, path_id] = warp(
                anchor_wave[anchor_id, path_id], factors[anchor_id, path_id]
            )
            factor_rows.append(
                {
                    "nominal_anchor_C": float(anchors.iloc[anchor_id].nominal_anchor_C),
                    "actual_anchor_C": float(anchor_temperatures[anchor_id]),
                    "sequence": int(anchors.iloc[anchor_id].sequence),
                    "receiver": path_id + 1,
                    "stretch_factor": float(factors[anchor_id, path_id]),
                    "alignment_correlation": float(correlations[anchor_id, path_id]),
                    "at_grid_boundary": bool(
                        factors[anchor_id, path_id] == STRETCH_GRID[0]
                        or factors[anchor_id, path_id] == STRETCH_GRID[-1]
                    ),
                }
            )

    support_min = float(anchor_temperatures.min())
    support_max = float(anchor_temperatures.max())
    score_rows = []
    target_factor_rows = []
    for row in index.itertuples(index=False):
        if int(row.sequence) in anchor_sequences:
            continue
        temperature = float(row.temperature_C)
        support = support_min <= temperature <= support_max
        query = load_signal(args.raw_root, row.filename)
        weights = quadratic_weights(temperature, anchor_temperatures)
        aligned_prediction = np.einsum("a,apt->pt", weights, aligned_anchors)
        target_factors = np.einsum("a,ap->p", weights, factors)
        prediction = np.empty_like(aligned_prediction)
        for path_id in range(paths):
            prediction[path_id] = warp(
                aligned_prediction[path_id], 1.0 / target_factors[path_id]
            )
            target_factor_rows.append(
                {
                    "sequence": int(row.sequence),
                    "temperature_C": temperature,
                    "receiver": path_id + 1,
                    "interpolated_stretch_factor": float(target_factors[path_id]),
                    "in_support": support,
                }
            )
        scores = normalized_residual_energy(query, prediction)
        for path_id, score in enumerate(scores):
            score_rows.append(
                {
                    "sequence": int(row.sequence),
                    "filename": row.filename,
                    "timestamp": row.timestamp.isoformat(),
                    "temperature_C": temperature,
                    "temperature_program_C": float(row.temperature_program_C),
                    "temperature_bin_C": temperature_bin(temperature),
                    "receiver": path_id + 1,
                    "score": float(score),
                    "in_support": support,
                    "status": "healthy" if support else "unknown_extrapolation",
                    "direct_threshold": DIRECT_THRESHOLD,
                    "direct_alarm": bool(support and score > DIRECT_THRESHOLD),
                }
            )

    scores = pd.DataFrame(score_rows)
    support_scores = scores.loc[scores.in_support].copy()
    outside_scores = scores.loc[~scores.in_support].copy()
    if support_scores.empty:
        raise ValueError("No non-anchor acquisitions inside interpolation support")

    direct_bins = summarize_bins(support_scores, "direct_alarm")
    eligible_direct_bins = direct_bins.loc[direct_bins.acquisitions >= 5]
    direct_fpr = float(support_scores.direct_alarm.mean())
    direct_acquisition_alarm = float(
        support_scores.groupby("sequence").direct_alarm.any().mean()
    )

    acquisition_order = np.array(sorted(support_scores.sequence.unique()))
    calibration_count = max(1, int(np.floor(0.30 * acquisition_order.size)))
    calibration_sequences = set(acquisition_order[:calibration_count])
    calibration_mask = support_scores.sequence.isin(calibration_sequences)
    temporal_threshold = float(
        np.quantile(
            support_scores.loc[calibration_mask, "score"].to_numpy(),
            0.95,
            method="higher",
        )
    )
    support_scores["temporal_split"] = np.where(calibration_mask, "calibration", "test")
    support_scores["temporal_alarm"] = support_scores.score > temporal_threshold
    test_scores = support_scores.loc[support_scores.temporal_split == "test"].copy()
    temporal_bins = summarize_bins(test_scores, "temporal_alarm")
    eligible_temporal_bins = temporal_bins.loc[temporal_bins.acquisitions >= 5]

    leave_one_rows = []
    for receiver in range(1, paths + 1):
        subset = support_scores.loc[support_scores.receiver != receiver]
        leave_one_rows.append(
            {"excluded_receiver": receiver, "direct_fpr": float(subset.direct_alarm.mean())}
        )
    leave_one = pd.DataFrame(leave_one_rows)

    factors_frame = pd.DataFrame(factor_rows)
    target_factors_frame = pd.DataFrame(target_factor_rows)
    nonreference = factors_frame.nominal_anchor_C != REFERENCE_NOMINAL_TEMPERATURE
    boundary_fraction = float(factors_frame.loc[nonreference, "at_grid_boundary"].mean())
    gates = {
        "direct_overall_path_fpr_at_most_0_05": direct_fpr <= 0.05,
        "every_eligible_bin_fpr_at_most_0_10": bool(
            (eligible_direct_bins.fpr <= 0.10).all()
        ),
        "nonreference_factor_boundary_fraction_below_0_10": boundary_fraction < 0.10,
        "all_leave_one_receiver_fpr_at_most_0_05": bool(
            (leave_one.direct_fpr <= 0.05).all()
        ),
        "all_outside_support_marked_unknown": bool(
            outside_scores.empty or (outside_scores.status == "unknown_extrapolation").all()
        ),
    }
    summary = {
        "schema_version": "p9e-independent-healthy-temperature-transfer-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "research_protocols/P9E_independent_healthy_temperature_transfer_protocol_v1.md",
        "source": "https://zenodo.org/records/19209079",
        "anchors": [
            {
                "nominal_C": float(row.nominal_anchor_C),
                "actual_C": float(row.temperature_C),
                "sequence": int(row.sequence),
                "filename": row.filename,
            }
            for row in anchors.itertuples(index=False)
        ],
        "support_C": [support_min, support_max],
        "counts": {
            "mapped_acquisitions": int(len(index)),
            "anchor_acquisitions": int(len(anchors)),
            "support_nonanchor_acquisitions": int(support_scores.sequence.nunique()),
            "outside_support_acquisitions": int(outside_scores.sequence.nunique()),
            "receivers": paths,
        },
        "direct_transfer": {
            "threshold": DIRECT_THRESHOLD,
            "path_fpr": direct_fpr,
            "acquisition_any_receiver_alarm_rate": direct_acquisition_alarm,
            "worst_eligible_temperature_bin_fpr": float(eligible_direct_bins.fpr.max()),
            "eligible_temperature_bins": int(len(eligible_direct_bins)),
        },
        "temporal_split_diagnostic": {
            "calibration_acquisitions": calibration_count,
            "test_acquisitions": int(test_scores.sequence.nunique()),
            "threshold": temporal_threshold,
            "test_path_fpr": float(test_scores.temporal_alarm.mean()),
            "test_acquisition_any_receiver_alarm_rate": float(
                test_scores.groupby("sequence").temporal_alarm.any().mean()
            ),
            "worst_eligible_temperature_bin_fpr": float(eligible_temporal_bins.fpr.max()),
        },
        "factor_diagnostics": {
            "minimum_anchor_factor": float(factors_frame.stretch_factor.min()),
            "maximum_anchor_factor": float(factors_frame.stretch_factor.max()),
            "nonreference_boundary_fraction": boundary_fraction,
            "minimum_target_factor": float(target_factors_frame.interpolated_stretch_factor.min()),
            "maximum_target_factor": float(target_factors_frame.interpolated_stretch_factor.max()),
        },
        "leave_one_receiver": {
            "minimum_fpr": float(leave_one.direct_fpr.min()),
            "maximum_fpr": float(leave_one.direct_fpr.max()),
            "range": float(leave_one.direct_fpr.max() - leave_one.direct_fpr.min()),
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
        "interpretation_warning": (
            "Healthy-only independent structure: validates false alarms and interpolation "
            "support only; it cannot establish damage AUROC or recall."
        ),
    }

    scores.to_csv(args.output_dir / "p9e_scores_long.csv.gz", index=False)
    anchors.to_csv(args.output_dir / "p9e_selected_anchors.csv", index=False)
    factors_frame.to_csv(args.output_dir / "p9e_anchor_stretch_factors.csv", index=False)
    target_factors_frame.to_csv(
        args.output_dir / "p9e_target_stretch_factors.csv.gz", index=False
    )
    direct_bins.to_csv(args.output_dir / "p9e_direct_temperature_bins.csv", index=False)
    temporal_bins.to_csv(args.output_dir / "p9e_temporal_temperature_bins.csv", index=False)
    leave_one.to_csv(args.output_dir / "p9e_leave_one_receiver.csv", index=False)
    (args.output_dir / "p9e_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    acquisition_scores = support_scores.groupby(["sequence", "temperature_C"]).score.median()
    plot_frame = acquisition_scores.reset_index()
    ax.scatter(plot_frame.temperature_C, plot_frame.score, s=18, alpha=0.7, color="#2878B5")
    ax.axhline(DIRECT_THRESHOLD, color="#C82423", linestyle="--", label="P9B direct threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Measured temperature (C)")
    ax.set_ylabel("Median normalized residual energy across 44 receivers")
    ax.set_title("P9E independent healthy transfer scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig01_score_vs_temperature.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(direct_bins.temperature_bin_C.astype(str), direct_bins.fpr, color="#2878B5")
    ax.axhline(0.10, color="#C82423", linestyle="--", label="Per-bin gate")
    ax.set_ylim(0, max(0.12, float(direct_bins.fpr.max()) * 1.12))
    ax.set_xlabel("Measured-temperature bin center (C)")
    ax.set_ylabel("Healthy receiver-level FPR")
    ax.set_title("P9E direct-threshold false alarms by temperature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig02_direct_fpr_by_temperature.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
