#!/usr/bin/env python3
"""Run the frozen P10 structure-relative pipeline anomaly protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd


FILE_RE = re.compile(r"G4-288-(?P<number>\d+)-V-0-waveform\.xls$", re.I)
FREQUENCY_RE = re.compile(r"(?P<frequency>\d+(?:\.\d+)?)kHz", re.I)
FREQUENCIES = [14.0, 18.0, 24.0, 30.0, 37.0]
MODES = ["torsional", "flexural"]


def hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def load_archive(archive_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    records = []
    signals = []
    common_distance = None
    channel_names = [f"{int(freq)}kHz_{mode}" for freq in FREQUENCIES for mode in MODES]

    with zipfile.ZipFile(archive_path) as archive:
        items = []
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".xls"):
                continue
            match = FILE_RE.search(PurePosixPath(info.filename).name)
            if match is None:
                raise ValueError(f"Unexpected filename {info.filename}")
            items.append((int(match.group("number")), info))
        items.sort(key=lambda value: value[0])

        for number, info in items:
            payload = archive.read(info)
            workbook = xlrd.open_workbook(file_contents=payload, on_demand=True)
            by_frequency = {}
            for sheet_name in workbook.sheet_names():
                match = FREQUENCY_RE.search(sheet_name)
                if match is None:
                    raise ValueError(f"No frequency in {sheet_name}")
                frequency = float(match.group("frequency"))
                sheet = workbook.sheet_by_name(sheet_name)
                values = np.asarray(
                    [sheet.row_values(row, start_colx=0, end_colx=3) for row in range(2, sheet.nrows)],
                    dtype=np.float64,
                )
                by_frequency[frequency] = values
            workbook.release_resources()
            if set(by_frequency) != set(FREQUENCIES):
                raise ValueError(f"Frequency mismatch in {number}")
            distance = by_frequency[FREQUENCIES[0]][:, 0]
            if common_distance is None:
                common_distance = distance
            elif not np.array_equal(distance, common_distance):
                raise ValueError(f"Distance mismatch in {number}")
            sample = []
            signature_parts = []
            for frequency in FREQUENCIES:
                values = by_frequency[frequency]
                if not np.array_equal(values[:, 0], common_distance):
                    raise ValueError(f"Within-file distance mismatch in {number}")
                for column in (1, 2):
                    signal = values[:, column]
                    sample.append(signal)
                    signature_parts.append(hash_array(signal))
            signals.append(np.stack(sample, axis=0))
            records.append(
                {
                    "file_number": number,
                    "filename": info.filename,
                    "label": 0 if number <= 1901 else 1,
                    "workbook_signal_signature": hashlib.sha256(
                        "|".join(signature_parts).encode("ascii")
                    ).hexdigest(),
                }
            )

    frame = pd.DataFrame(records)
    waves = np.stack(signals, axis=0)
    return frame, waves, np.asarray(common_distance), channel_names


def deduplicate_and_split(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["status"] = "included"
    for _, group in result.groupby("workbook_signal_signature", sort=False):
        if len(group) > 1:
            later_indices = group.sort_values("file_number").index[1:]
            result.loc[later_indices, "status"] = "excluded_exact_duplicate_later_copy"
    result["split"] = "excluded_duplicate"
    healthy = result.loc[(result.label == 0) & (result.status == "included")].sort_values(
        "file_number"
    )
    damage = result.loc[(result.label == 1) & (result.status == "included")].sort_values(
        "file_number"
    )
    if len(healthy) != 199 or len(damage) != 28:
        raise ValueError(f"Expected 199 healthy and 28 damage after dedup, got {len(healthy)}, {len(damage)}")
    assignments = [
        (healthy.index[:109], "healthy_train"),
        (healthy.index[109:114], "embargo_A"),
        (healthy.index[114:154], "healthy_calibration"),
        (healthy.index[154:159], "embargo_B"),
        (healthy.index[159:199], "healthy_test"),
        (damage.index, "damage_test"),
    ]
    for indices, name in assignments:
        result.loc[indices, "split"] = name
    return result


def auc_high(negative: np.ndarray, positive: np.ndarray) -> float:
    delta = positive[:, None] - negative[None, :]
    return float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0))


def empirical_q95(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.95, method="higher"))


def spearman_file_order(numbers: np.ndarray, scores: np.ndarray) -> float:
    x = pd.Series(numbers).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def robust_scores(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    center = np.median(train, axis=0)
    raw_scale = 1.4826 * np.median(np.abs(train - center[None, :, :]), axis=0)
    scale = raw_scale.copy()
    floors = []
    for channel in range(train.shape[1]):
        positive = raw_scale[channel][raw_scale[channel] > 0]
        floor = 0.05 * float(np.median(positive))
        scale[channel] = np.maximum(scale[channel], floor)
        floors.append(floor)
    absolute_z = np.abs((query - center[None, :, :]) / scale[None, :, :])
    channel_scores = np.quantile(absolute_z, 0.99, axis=-1)
    top3 = np.partition(channel_scores, -3, axis=1)[:, -3:]
    scores = np.mean(top3, axis=1)
    return scores, channel_scores, {
        "scale_floor_by_channel": floors,
        "minimum_scale": float(scale.min()),
        "maximum_scale": float(scale.max()),
    }


def pca_scores(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, dict]:
    sampled_train = train[:, :, ::5].reshape(train.shape[0], -1)
    sampled_query = query[:, :, ::5].reshape(query.shape[0], -1)
    mean = sampled_train.mean(axis=0)
    std = sampled_train.std(axis=0, ddof=1)
    features_per_channel = train[:, 0, ::5].shape[1]
    floors = []
    for channel in range(train.shape[1]):
        start = channel * features_per_channel
        end = start + features_per_channel
        positive = std[start:end][std[start:end] > 0]
        floor = 0.05 * float(np.median(positive))
        std[start:end] = np.maximum(std[start:end], floor)
        floors.append(floor)
    x_train = (sampled_train - mean) / std
    x_query = (sampled_query - mean) / std
    _, singular, vt = np.linalg.svd(x_train, full_matrices=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / np.sum(variance)
    components_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    components = min(40, components_95)
    basis = vt[:components]
    reconstruction = (x_query @ basis.T) @ basis
    scores = np.mean((x_query - reconstruction) ** 2, axis=1)
    return scores, {
        "sample_step": 5,
        "features": int(x_train.shape[1]),
        "components_95pct_uncapped": components_95,
        "components_used": components,
        "variance_explained_used": float(cumulative[components - 1]),
        "std_floor_by_channel": floors,
    }


def method_summary(
    method: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    healthy_test = frame.split == "healthy_test"
    damage = frame.split == "damage_test"
    damage_frame = frame.loc[damage].sort_values("file_number")
    damage_scores = scores[damage_frame.index]
    early_scores = damage_scores[:5]
    negative = scores[healthy_test]
    return {
        "method": method,
        "threshold_calibration_q95_higher": threshold,
        "healthy_test_fpr": float(np.mean(negative > threshold)),
        "damage_recall": float(np.mean(damage_scores > threshold)),
        "earliest_five_damage_recall": float(np.mean(early_scores > threshold)),
        "auc_healthy_test_vs_damage": auc_high(negative, damage_scores),
        "damage_order_spearman": spearman_file_order(
            damage_frame.file_number.to_numpy(), damage_scores
        ),
        "healthy_test_score_median": float(np.median(negative)),
        "damage_score_median": float(np.median(damage_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, waves, distance, channel_names = load_archive(args.archive)
    frame = deduplicate_and_split(frame)
    duplicate_zero = np.flatnonzero(np.diff(distance) == 0)
    if duplicate_zero.size != 1 or distance[duplicate_zero[0]] != 0:
        raise ValueError("Unexpected duplicated-zero distance axis")
    keep_distance = np.ones(distance.size, dtype=bool)
    keep_distance[duplicate_zero[0] + 1] = False
    keep_distance &= np.abs(distance) >= 1.0
    waves = waves[:, :, keep_distance]
    retained_distance = distance[keep_distance]

    train_mask = frame.split == "healthy_train"
    calibration_mask = frame.split == "healthy_calibration"
    included_mask = frame.status == "included"
    robust, robust_channels, robust_diag = robust_scores(waves[train_mask], waves)
    pca, pca_diag = pca_scores(waves[train_mask], waves)
    energy = np.mean(np.sqrt(np.mean(waves**2, axis=-1)), axis=1)
    methods = {
        "robust_residual_top3": robust,
        "pca_reconstruction": pca,
        "raw_input_energy_negative_control": energy,
    }

    summaries = []
    score_rows = []
    for method, values in methods.items():
        threshold = empirical_q95(values[calibration_mask])
        summaries.append(method_summary(method, frame, values, threshold))
        for index, row in frame.iterrows():
            score_rows.append(
                {
                    "file_number": int(row.file_number),
                    "label": int(row.label),
                    "split": row.split,
                    "status": row.status,
                    "method": method,
                    "score": float(values[index]),
                    "threshold": threshold,
                    "alarm": bool(row.status == "included" and values[index] > threshold),
                }
            )

    robust_threshold = empirical_q95(robust[calibration_mask])
    channel_rows = []
    for channel, name in enumerate(channel_names):
        threshold = empirical_q95(robust_channels[calibration_mask, channel])
        negative = robust_channels[frame.split == "healthy_test", channel]
        positive = robust_channels[frame.split == "damage_test", channel]
        channel_rows.append(
            {
                "channel": name,
                "threshold": threshold,
                "healthy_test_fpr": float(np.mean(negative > threshold)),
                "damage_recall": float(np.mean(positive > threshold)),
                "auc": auc_high(negative, positive),
            }
        )

    summary_by_method = {row["method"]: row for row in summaries}
    primary = summary_by_method["robust_residual_top3"]
    negative_control = summary_by_method["raw_input_energy_negative_control"]
    gates = {
        "healthy_test_fpr_at_most_0_10": primary["healthy_test_fpr"] <= 0.10,
        "damage_recall_at_least_0_80": primary["damage_recall"] >= 0.80,
        "earliest_five_damage_recall_at_least_0_40": primary[
            "earliest_five_damage_recall"
        ]
        >= 0.40,
        "auc_at_least_0_80": primary["auc_healthy_test_vs_damage"] >= 0.80,
        "damage_order_spearman_at_least_0_30": primary["damage_order_spearman"]
        >= 0.30,
        "auc_above_input_energy_negative_control": primary[
            "auc_healthy_test_vs_damage"
        ]
        > negative_control["auc_healthy_test_vs_damage"],
    }
    output = {
        "schema_version": "p10-pipeline-structure-relative-anomaly-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "research_protocols/P10_pipeline_structure_relative_anomaly_protocol_v1.md",
        "data": {
            "archive": str(args.archive),
            "distance_retained_min_m": float(retained_distance.min()),
            "distance_retained_max_m": float(retained_distance.max()),
            "distance_points_retained": int(retained_distance.size),
            "channels": channel_names,
        },
        "split_counts": frame.split.value_counts().to_dict(),
        "method_results": summaries,
        "robust_diagnostics": robust_diag,
        "pca_diagnostics": pca_diag,
        "primary_gates": gates,
        "primary_passed": bool(all(gates.values())),
        "warnings": [
            "No per-file temperature labels: this is temporal environmental robustness, not supervised temperature compensation.",
            "No explicit six-stage damage labels: file order is only a severity-order surrogate.",
            "Five-repeat grouping lacks timestamps; five-file embargoes reduce but cannot prove independence.",
        ],
    }

    frame.to_csv(args.output_dir / "p10_sample_assignments.csv", index=False)
    pd.DataFrame(score_rows).to_csv(args.output_dir / "p10_scores_long.csv.gz", index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / "p10_method_summary.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(args.output_dir / "p10_robust_channel_diagnostics.csv", index=False)
    (args.output_dir / "p10_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(10.2, 5.5))
    included = frame.status == "included"
    colors = np.where(frame.label.to_numpy() == 0, "#2878B5", "#C82423")
    ax.scatter(frame.loc[included, "file_number"], robust[included], c=colors[included], s=22, alpha=0.8)
    ax.axhline(robust_threshold, color="black", linestyle="--", label="Healthy calibration q95")
    ax.axvline(1901.5, color="#666666", linestyle=":", label="Official healthy/damage boundary")
    ax.set_yscale("log")
    ax.set_xlabel("Acquisition file number")
    ax.set_ylabel("Frozen robust residual score")
    ax.set_title("P10 chronological structure-relative anomaly scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig01_robust_score_chronology.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    table = pd.DataFrame(summaries)
    x = np.arange(len(table))
    width = 0.25
    ax.bar(x - width, table.auc_healthy_test_vs_damage, width, label="AUROC")
    ax.bar(x, table.damage_recall, width, label="Damage recall")
    ax.bar(x + width, 1 - table.healthy_test_fpr, width, label="Healthy specificity")
    ax.set_xticks(x, table.method, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("P10 predeclared method comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig02_method_metrics.png", dpi=180)
    plt.close(fig)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
