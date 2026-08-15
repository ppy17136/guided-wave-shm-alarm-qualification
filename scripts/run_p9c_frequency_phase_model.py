#!/usr/bin/env python3
"""Run the frozen P9C frequency-dependent amplitude/phase temperature model."""

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


SAMPLING_HZ = 12_000_000.0
SAMPLES = 6000


def quadratic_weights(temperature: float) -> np.ndarray:
    design = np.column_stack([np.ones_like(ANCHORS), ANCHORS, ANCHORS**2])
    target = np.array([1.0, temperature, temperature**2])
    return target @ np.linalg.pinv(design)


def band_for_task(task: str) -> tuple[float, float]:
    if task.startswith("40kHz"):
        return 20_000.0, 60_000.0
    if task.startswith("50kHz"):
        return 25_000.0, 75_000.0
    raise ValueError(task)


def bandlimit(waves: np.ndarray, mask: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(waves, axis=-1)
    spectrum[..., ~mask] = 0.0
    return np.fft.irfft(spectrum, n=SAMPLES, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--p9b-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frequencies = np.fft.rfftfreq(SAMPLES, d=1.0 / SAMPLING_HZ)
    rows = []
    model_diagnostics = []
    for task, baseline_name, diagnostic_name in TASKS:
        baseline, paths, temperatures = read_waveforms(find_file(args.raw_root, baseline_name))
        diagnostic, diagnostic_paths, diagnostic_temperatures = read_waveforms(
            find_file(args.raw_root, diagnostic_name)
        )
        if paths != diagnostic_paths or temperatures != diagnostic_temperatures:
            raise ValueError(f"Baseline/diagnostic structure mismatch for {task}")
        low, high = band_for_task(task)
        mask = (frequencies >= low) & (frequencies <= high)
        temp_index = {value: i for i, value in enumerate(temperatures)}
        anchor_wave = baseline[[temp_index[int(value)] for value in ANCHORS], :, :]
        anchor_spectrum = np.fft.rfft(anchor_wave, axis=-1)[..., mask]
        log_magnitude = np.log(np.abs(anchor_spectrum) + 1e-12)
        phase = np.unwrap(np.angle(anchor_spectrum), axis=0)

        for temperature in HELD_TEMPERATURES:
            weights = quadratic_weights(float(temperature))
            predicted_log_magnitude = np.einsum("a,apf->pf", weights, log_magnitude)
            predicted_phase = np.einsum("a,apf->pf", weights, phase)
            predicted_spectrum = np.zeros(
                (len(paths), frequencies.size), dtype=np.complex128
            )
            predicted_spectrum[:, mask] = np.exp(predicted_log_magnitude) * np.exp(
                1j * predicted_phase
            )
            predicted = np.fft.irfft(predicted_spectrum, n=SAMPLES, axis=-1)
            idx = temp_index[int(temperature)]
            baseline_band = bandlimit(baseline[idx], mask)
            diagnostic_band = bandlimit(diagnostic[idx], mask)
            baseline_score = normalized_residual_energy(baseline_band, predicted)
            diagnostic_score = normalized_residual_energy(diagnostic_band, predicted)
            for path_id, path in enumerate(paths):
                rows.append(
                    {
                        "task": task,
                        "temperature_C": int(temperature),
                        "path": path,
                        "method": "frequency_phase_quadratic",
                        "label": 0,
                        "score": float(baseline_score[path_id]),
                    }
                )
                rows.append(
                    {
                        "task": task,
                        "temperature_C": int(temperature),
                        "path": path,
                        "method": "frequency_phase_quadratic",
                        "label": 1,
                        "score": float(diagnostic_score[path_id]),
                    }
                )
        model_diagnostics.append(
            {
                "task": task,
                "band_low_Hz": low,
                "band_high_Hz": high,
                "fft_resolution_Hz": float(frequencies[1] - frequencies[0]),
                "band_bins": int(mask.sum()),
            }
        )

    event_summaries, macro = summarize_rows(rows)
    result = macro[0]
    p9b = json.loads(args.p9b_summary.read_text(encoding="utf-8"))
    p9b_result = p9b["result"]
    pd.DataFrame(rows).to_csv(args.output_dir / "p9c_scores_long.csv.gz", index=False)
    pd.DataFrame(event_summaries).to_csv(
        args.output_dir / "p9c_task_summary.csv", index=False
    )
    summary = {
        "schema_version": "p9c-frequency-phase-temperature-model-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "research_protocols/P9C_frequency_phase_temperature_model_protocol_v1.md",
        "sampling_Hz": SAMPLING_HZ,
        "samples": SAMPLES,
        "model_diagnostics": model_diagnostics,
        "result": result,
        "task_results": event_summaries,
        "comparisons": {
            "macro_auc_change_vs_p9b": float(
                result["macro_auc"] - p9b_result["macro_auc"]
            ),
            "worst_temperature_fpr_change_vs_p9b": float(
                result["worst_temperature_fpr"]
                - p9b_result["worst_temperature_fpr"]
            ),
        },
        "gates": {
            "macro_auc_at_least_p9b": result["macro_auc"] >= p9b_result["macro_auc"],
            "worst_task_auc_at_least_0_80": result["worst_task_auc"] >= 0.80,
            "positive_direction_all_four_tasks": result["positive_direction_tasks"] == 4,
            "worst_temperature_fpr_at_most_0_10": result["worst_temperature_fpr"]
            <= 0.10,
            "simultaneously_improves_auc_and_fpr": result["macro_auc"]
            >= p9b_result["macro_auc"]
            and result["worst_temperature_fpr"]
            < p9b_result["worst_temperature_fpr"],
        },
    }
    (args.output_dir / "p9c_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    labels = ["P9B phase stretch", "P9C frequency phase"]
    values = [p9b_result["macro_auc"], result["macro_auc"]]
    ax.bar(labels, values, color=["#F28522", "#2A9D8F"])
    ax.axhline(0.7, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Four-task macro AUROC")
    ax.set_title("Frozen P9C comparison")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig01_p9b_p9c_macro_auc.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
