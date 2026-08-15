#!/usr/bin/env python3
"""Postprocess P9A scores with an explicitly flagged duplicate-record sensitivity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def auc_high(negative: np.ndarray, positive: np.ndarray) -> float:
    delta = positive[:, None] - negative[None, :]
    return float(np.mean(delta > 0) + 0.5 * np.mean(delta == 0))


def summarize(frame: pd.DataFrame, variant: str) -> tuple[list[dict], list[dict]]:
    events = []
    for (method, task), subset in frame.groupby(["method", "task"], sort=True):
        negative = subset.loc[subset.label == 0, "score"].to_numpy()
        positive = subset.loc[subset.label == 1, "score"].to_numpy()
        threshold = float(np.quantile(negative, 0.95, method="higher"))
        temperature_metrics = []
        for temperature, temp_frame in subset.groupby("temperature_C", sort=True):
            negative_t = temp_frame.loc[temp_frame.label == 0, "score"].to_numpy()
            positive_t = temp_frame.loc[temp_frame.label == 1, "score"].to_numpy()
            temperature_metrics.append(
                {
                    "temperature_C": int(temperature),
                    "auc": auc_high(negative_t, positive_t),
                    "fpr": float(np.mean(negative_t > threshold)),
                    "recall": float(np.mean(positive_t > threshold)),
                }
            )
        events.append(
            {
                "variant": variant,
                "method": method,
                "task": task,
                "auc": auc_high(negative, positive),
                "worst_temperature_auc": min(item["auc"] for item in temperature_metrics),
                "worst_temperature_fpr": max(item["fpr"] for item in temperature_metrics),
                "recall": float(np.mean(positive > threshold)),
                "temperatures": temperature_metrics,
            }
        )
    macros = []
    for method in sorted({item["method"] for item in events}):
        subset = [item for item in events if item["method"] == method]
        macros.append(
            {
                "variant": variant,
                "method": method,
                "macro_auc": float(np.mean([item["auc"] for item in subset])),
                "worst_task_auc": float(np.min([item["auc"] for item in subset])),
                "worst_temperature_auc": float(
                    np.min([item["worst_temperature_auc"] for item in subset])
                ),
                "worst_temperature_fpr": float(
                    np.max([item["worst_temperature_fpr"] for item in subset])
                ),
                "mean_recall": float(np.mean([item["recall"] for item in subset])),
            }
        )
    return events, macros


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.scores)

    full_events, full_macros = summarize(frame, "full_primary")
    flagged = frame.task.str.endswith("I2") & (frame.temperature_C == 45)
    sensitivity_frame = frame.loc[~flagged].copy()
    sensitivity_events, sensitivity_macros = summarize(
        sensitivity_frame, "exclude_exact_duplicate_I2_45C_sensitivity"
    )
    events = full_events + sensitivity_events
    macros = full_macros + sensitivity_macros

    pd.DataFrame(
        [{key: value for key, value in item.items() if key != "temperatures"} for item in events]
    ).to_csv(args.output_dir / "p9a_semantic_sensitivity_tasks.csv", index=False)
    pd.DataFrame(macros).to_csv(
        args.output_dir / "p9a_semantic_sensitivity_macro.csv", index=False
    )
    summary = {
        "schema_version": "p9a-semantic-sensitivity-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_variant": "full_primary",
        "sensitivity_variant": "exclude_exact_duplicate_I2_45C_sensitivity",
        "exclusion_reason": (
            "Both I2 frequencies contain exact baseline/diagnostic duplicates at 45 C and "
            "50 C. The held-temperature benchmark includes 45 C but not the 50 C anchor."
        ),
        "excluded_score_rows": int(flagged.sum()),
        "main_result_remains_full": True,
        "macros": macros,
        "events": events,
    }
    (args.output_dir / "p9a_semantic_sensitivity.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"excluded_score_rows": int(flagged.sum()), "macros": macros}, indent=2))


if __name__ == "__main__":
    main()
