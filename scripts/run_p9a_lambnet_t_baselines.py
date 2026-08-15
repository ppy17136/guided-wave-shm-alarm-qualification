#!/usr/bin/env python3
"""Run the first frozen LambNet-T waveform baselines from P9A protocol v1."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLUMN_RE = re.compile(r"^P(?P<a>\d{2})-(?P<b>\d{2})_T(?P<t>-?\d+)C$")
ANCHORS = np.array([-10, 10, 30, 50], dtype=float)
HELD_TEMPERATURES = np.array([-5, 0, 5, 15, 20, 25, 35, 40, 45], dtype=int)
TASKS = [
    ("40kHz_I1", "01_Baseline_40kHz_I1.csv", "01_Diagnostic_40kHz_I1.csv"),
    ("50kHz_I1", "02_Baseline_50kHz_I1.csv", "02_Diagnostic_50kHz_I1.csv"),
    ("40kHz_I2", "03_Baseline_40kHz_I2.csv", "03_Diagnostic_40kHz_I2.csv"),
    ("50kHz_I2", "04_Baseline_50kHz_I2.csv", "04_Diagnostic_50kHz_I2.csv"),
]


def find_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def read_waveforms(path: Path) -> tuple[np.ndarray, list[str], list[int]]:
    frame = pd.read_csv(path)
    parsed = []
    for column in frame.columns:
        match = COLUMN_RE.match(column)
        if match is None:
            raise ValueError(f"Malformed column {column!r} in {path.name}")
        parsed.append(
            (f"{int(match.group('a')):02d}-{int(match.group('b')):02d}", int(match.group("t")))
        )
    paths = sorted({item[0] for item in parsed})
    temperatures = sorted({item[1] for item in parsed})
    path_index = {name: i for i, name in enumerate(paths)}
    temp_index = {value: i for i, value in enumerate(temperatures)}
    waves = np.empty((len(temperatures), len(paths), frame.shape[0]), dtype=np.float64)
    for column, (path_name, temperature) in zip(frame.columns, parsed):
        waves[temp_index[temperature], path_index[path_name], :] = frame[column].to_numpy(
            dtype=np.float64
        )
    waves -= np.median(waves[..., :256], axis=-1, keepdims=True)
    return waves, paths, temperatures


def normalized_residual_energy(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.mean((query - reference) ** 2, axis=-1)
    denominator = np.mean(reference**2, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def obs_reference(query: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # query: path x time; anchors: anchor x path x time
    q = query - np.mean(query, axis=-1, keepdims=True)
    a = anchors - np.mean(anchors, axis=-1, keepdims=True)
    numerator = np.einsum("pt,apt->ap", q, a)
    denominator = np.sqrt(
        np.sum(q * q, axis=-1, keepdims=False)[None, :]
        * np.sum(a * a, axis=-1, keepdims=False)
    )
    correlation = numerator / np.maximum(denominator, 1e-12)
    selected = np.argmax(correlation, axis=0)
    path_ids = np.arange(query.shape[0])
    return anchors[selected, path_ids, :], selected


def quadratic_reference(target_temperature: float, anchors: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(ANCHORS), ANCHORS, ANCHORS**2])
    target = np.array([1.0, target_temperature, target_temperature**2])
    weights = target @ np.linalg.pinv(design)
    return np.einsum("a,apt->pt", weights, anchors)


def auc_high(negative: np.ndarray, positive: np.ndarray) -> float:
    negative = np.asarray(negative, dtype=float)
    positive = np.asarray(positive, dtype=float)
    # Pairwise form handles ties exactly and is small here (40 x 40 or 360 x 360).
    delta = positive[:, None] - negative[None, :]
    return float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0))


def summarize_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    methods = sorted({row["method"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    event_summaries = []
    for method in methods:
        for task in tasks:
            subset = [row for row in rows if row["method"] == method and row["task"] == task]
            negative = np.array([row["score"] for row in subset if row["label"] == 0])
            positive = np.array([row["score"] for row in subset if row["label"] == 1])
            threshold = float(np.quantile(negative, 0.95, method="higher"))
            by_temp_fpr = []
            by_temp_auc = []
            for temperature in HELD_TEMPERATURES:
                neg_t = np.array(
                    [
                        row["score"]
                        for row in subset
                        if row["label"] == 0 and row["temperature_C"] == int(temperature)
                    ]
                )
                pos_t = np.array(
                    [
                        row["score"]
                        for row in subset
                        if row["label"] == 1 and row["temperature_C"] == int(temperature)
                    ]
                )
                by_temp_fpr.append(float(np.mean(neg_t > threshold)))
                by_temp_auc.append(auc_high(neg_t, pos_t))
            event_summaries.append(
                {
                    "method": method,
                    "task": task,
                    "auc": auc_high(negative, positive),
                    "temperature_macro_auc": float(np.mean(by_temp_auc)),
                    "worst_temperature_auc": float(np.min(by_temp_auc)),
                    "healthy_score_mean": float(np.mean(negative)),
                    "diagnostic_score_mean": float(np.mean(positive)),
                    "effect_direction_positive": bool(np.mean(positive) > np.mean(negative)),
                    "threshold_healthy_q95": threshold,
                    "pooled_fpr": float(np.mean(negative > threshold)),
                    "worst_temperature_fpr": float(np.max(by_temp_fpr)),
                    "recall": float(np.mean(positive > threshold)),
                }
            )

    macro = []
    for method in methods:
        subset = [row for row in event_summaries if row["method"] == method]
        macro.append(
            {
                "method": method,
                "macro_auc": float(np.mean([row["auc"] for row in subset])),
                "worst_task_auc": float(np.min([row["auc"] for row in subset])),
                "macro_temperature_auc": float(
                    np.mean([row["temperature_macro_auc"] for row in subset])
                ),
                "worst_temperature_auc": float(
                    np.min([row["worst_temperature_auc"] for row in subset])
                ),
                "mean_healthy_score": float(
                    np.mean([row["healthy_score_mean"] for row in subset])
                ),
                "worst_temperature_fpr": float(
                    np.max([row["worst_temperature_fpr"] for row in subset])
                ),
                "mean_recall": float(np.mean([row["recall"] for row in subset])),
                "positive_direction_tasks": int(
                    sum(row["effect_direction_positive"] for row in subset)
                ),
            }
        )
    return event_summaries, macro


def save_figures(event_summaries: list[dict], macro: list[dict], output_dir: Path) -> None:
    methods = [row["method"] for row in macro]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    ax.bar(x, [row["macro_auc"] for row in macro], color="#2878B5")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.7, color="#C82423", linestyle=":", linewidth=1.2)
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.set_ylabel("Four-task macro AUROC")
    ax.set_ylim(0, 1)
    ax.set_title("P9A LambNet-T frozen waveform baselines")
    fig.tight_layout()
    fig.savefig(output_dir / "fig01_macro_auc.png", dpi=180)
    plt.close(fig)

    tasks = sorted({row["task"] for row in event_summaries})
    matrix = np.array(
        [
            [
                next(
                    row["auc"]
                    for row in event_summaries
                    if row["method"] == method and row["task"] == task
                )
                for task in tasks
            ]
            for method in methods
        ]
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(tasks)), tasks)
    ax.set_yticks(np.arange(len(methods)), methods)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="AUROC")
    ax.set_title("Task-wise damage separability")
    fig.tight_layout()
    fig.savefig(output_dir / "fig02_task_auc_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    selection_rows = []
    for task, baseline_name, diagnostic_name in TASKS:
        baseline, paths, temperatures = read_waveforms(find_file(args.raw_root, baseline_name))
        diagnostic, diagnostic_paths, diagnostic_temperatures = read_waveforms(
            find_file(args.raw_root, diagnostic_name)
        )
        if paths != diagnostic_paths or temperatures != diagnostic_temperatures:
            raise ValueError(f"Baseline/diagnostic structure mismatch for {task}")
        temp_index = {value: i for i, value in enumerate(temperatures)}
        anchor_wave = baseline[[temp_index[int(value)] for value in ANCHORS], :, :]

        for temperature in HELD_TEMPERATURES:
            idx = temp_index[int(temperature)]
            baseline_query = baseline[idx]
            diagnostic_query = diagnostic[idx]
            references = {
                "nearest_anchor": anchor_wave[
                    int(np.argmin(np.abs(ANCHORS - float(temperature))))
                ],
                "quadratic_interpolation": quadratic_reference(float(temperature), anchor_wave),
                "matched_temperature_oracle": baseline_query,
            }
            obs_baseline, obs_baseline_selected = obs_reference(baseline_query, anchor_wave)
            obs_diagnostic, obs_diagnostic_selected = obs_reference(diagnostic_query, anchor_wave)
            references["obs_query_selected_baseline"] = obs_baseline

            for method, reference in references.items():
                base_score = normalized_residual_energy(baseline_query, reference)
                diagnostic_reference = reference
                if method == "obs_query_selected_baseline":
                    diagnostic_reference = obs_diagnostic
                diagnostic_score = normalized_residual_energy(
                    diagnostic_query, diagnostic_reference
                )
                for path_id, path in enumerate(paths):
                    rows.append(
                        {
                            "task": task,
                            "temperature_C": int(temperature),
                            "path": path,
                            "method": method,
                            "label": 0,
                            "score": float(base_score[path_id]),
                        }
                    )
                    rows.append(
                        {
                            "task": task,
                            "temperature_C": int(temperature),
                            "path": path,
                            "method": method,
                            "label": 1,
                            "score": float(diagnostic_score[path_id]),
                        }
                    )

            baseline_rms = np.sqrt(np.mean(baseline_query**2, axis=-1))
            diagnostic_rms = np.sqrt(np.mean(diagnostic_query**2, axis=-1))
            for path_id, path in enumerate(paths):
                for label, value in ((0, baseline_rms[path_id]), (1, diagnostic_rms[path_id])):
                    rows.append(
                        {
                            "task": task,
                            "temperature_C": int(temperature),
                            "path": path,
                            "method": "raw_input_rms_high_negative_control",
                            "label": label,
                            "score": float(value),
                        }
                    )
                selection_rows.append(
                    {
                        "task": task,
                        "temperature_C": int(temperature),
                        "path": path,
                        "baseline_selected_anchor_C": int(
                            ANCHORS[obs_baseline_selected[path_id]]
                        ),
                        "diagnostic_selected_anchor_C": int(
                            ANCHORS[obs_diagnostic_selected[path_id]]
                        ),
                    }
                )

    event_summaries, macro = summarize_rows(rows)
    rows_frame = pd.DataFrame(rows)
    event_frame = pd.DataFrame(event_summaries)
    macro_frame = pd.DataFrame(macro).sort_values("macro_auc", ascending=False)
    selection_frame = pd.DataFrame(selection_rows)
    rows_frame.to_csv(args.output_dir / "p9a_scores_long.csv.gz", index=False)
    event_frame.to_csv(args.output_dir / "p9a_task_summary.csv", index=False)
    macro_frame.to_csv(args.output_dir / "p9a_macro_summary.csv", index=False)
    selection_frame.to_csv(args.output_dir / "p9a_obs_selection.csv.gz", index=False)
    save_figures(event_summaries, macro, args.output_dir)

    quadratic = next(row for row in macro if row["method"] == "quadratic_interpolation")
    nearest = next(row for row in macro if row["method"] == "nearest_anchor")
    negative = next(
        row for row in macro if row["method"] == "raw_input_rms_high_negative_control"
    )
    oracle = next(row for row in macro if row["method"] == "matched_temperature_oracle")
    summary = {
        "schema_version": "p9a-lambnet-t-frozen-baselines-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "research_protocols/P9A_lambnet_t_physics_baseline_protocol_v1.md",
        "anchors_C": ANCHORS.astype(int).tolist(),
        "held_temperatures_C": HELD_TEMPERATURES.tolist(),
        "tasks": [item[0] for item in TASKS],
        "methods": [row["method"] for row in macro],
        "macro": macro,
        "first_stage_checks": {
            "quadratic_healthy_error_reduction_vs_nearest": (
                1.0 - quadratic["mean_healthy_score"] / nearest["mean_healthy_score"]
                if nearest["mean_healthy_score"] > 0
                else math.nan
            ),
            "quadratic_positive_direction_all_tasks": quadratic[
                "positive_direction_tasks"
            ]
            == 4,
            "quadratic_macro_auc_at_least_0_70": quadratic["macro_auc"] >= 0.70,
            "quadratic_worst_task_auc_at_least_0_60": quadratic["worst_task_auc"]
            >= 0.60,
            "quadratic_worst_temperature_fpr_at_most_0_10": quadratic[
                "worst_temperature_fpr"
            ]
            <= 0.10,
            "quadratic_beats_raw_rms_negative_control": quadratic["macro_auc"]
            > negative["macro_auc"],
            "oracle_macro_auc": oracle["macro_auc"],
        },
        "interpretation_warning": (
            "Development-set benchmark. OBS uses the query waveform for baseline selection; "
            "matched-temperature is an oracle ceiling; neither is a deployable confirmation."
        ),
    }
    (args.output_dir / "p9a_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
