from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs" / "p10_pipeline_anomaly_v1" / "p10_scores_long.csv.gz"
OUT = ROOT / "runs" / "p11a_health_side_calibration_reliability_v1"
PROTOCOL = "research_protocols/P11A_health_side_calibration_reliability_exploratory_protocol_v1.md"
METHODS = [
    "robust_residual_top3",
    "pca_reconstruction",
    "raw_input_energy_negative_control",
]


def quantile_higher(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="higher"))


def median_mad(values: np.ndarray) -> tuple[float, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, 1.4826 * mad


def confirmed_flags(alarms: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(alarms), dtype=bool)
    for idx in range(2, len(alarms)):
        if int(np.sum(alarms[idx - 2 : idx + 1])) >= 2:
            flags[idx - 2 : idx + 1] |= alarms[idx - 2 : idx + 1]
    return flags


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(INPUT)
    scores = scores[scores["status"].eq("included")].copy()

    summaries: list[dict] = []
    threshold_rows: list[dict] = []
    block_rows: list[dict] = []

    for method in METHODS:
        data = scores[scores["method"].eq(method)].sort_values("file_number")
        cal = data[data["split"].eq("healthy_calibration")].copy()
        if len(cal) != 40:
            raise RuntimeError(f"{method}: expected 40 calibration samples, got {len(cal)}")
        cal_values = cal["score"].to_numpy(float)
        median, mad, robust_scale = median_mad(cal_values)
        t0 = quantile_higher(cal_values, 0.95)
        t1 = median + 6.0 * robust_scale
        cap = median + 3.0 * robust_scale
        t2 = quantile_higher(np.minimum(cal_values, cap), 0.95)

        block_ids = np.repeat(np.arange(1, 9), 5)
        cal["block"] = block_ids
        block_stats = []
        for block, group in cal.groupby("block", sort=True):
            block_stats.append({
                "method": method,
                "block": int(block),
                "file_min": int(group["file_number"].min()),
                "file_max": int(group["file_number"].max()),
                "block_median": float(group["score"].median()),
                "block_max": float(group["score"].max()),
            })
        block_rows.extend(block_stats)
        block_maxima = np.array([row["block_max"] for row in block_stats])
        tail_block_heterogeneity = float(block_maxima.max() / max(np.median(block_maxima), 1e-15))

        loo_thresholds = []
        for block in range(1, 9):
            loo = cal.loc[cal["block"].ne(block), "score"].to_numpy(float)
            loo_thresholds.append(quantile_higher(loo, 0.95))
        loo_min = float(np.min(loo_thresholds))
        loo_max = float(np.max(loo_thresholds))
        loo_relative_range = float((loo_max - loo_min) / max(abs(t0), 1e-15))

        top_n = max(1, math.ceil(0.05 * len(cal)))
        tail_support_blocks = int(cal.nlargest(top_n, "score")["block"].nunique())
        t0_t1_ratio = float(t0 / max(abs(t1), 1e-15))
        checks = {
            "tail_block_heterogeneity_at_most_2": tail_block_heterogeneity <= 2.0,
            "loo_t0_relative_range_at_most_0_5": loo_relative_range <= 0.5,
            "t0_t1_ratio_between_0_5_and_2": 0.5 <= t0_t1_ratio <= 2.0,
            "calibration_n_at_least_40": len(cal) >= 40,
            "eight_complete_blocks": len(block_stats) == 8 and all(
                len(cal[cal["block"].eq(block)]) == 5 for block in range(1, 9)
            ),
        }
        reliable = all(checks.values())

        method_summary = {
            "method": method,
            "calibration_n": int(len(cal)),
            "calibration_median": median,
            "calibration_mad": mad,
            "calibration_robust_scale": robust_scale,
            "t0_q95_higher": t0,
            "t1_hampel6": t1,
            "t2_winsor_q95": t2,
            "tail_block_heterogeneity": tail_block_heterogeneity,
            "loo_t0_min": loo_min,
            "loo_t0_max": loo_max,
            "loo_t0_relative_range": loo_relative_range,
            "tail_support_blocks_top5pct": tail_support_blocks,
            "t0_t1_ratio": t0_t1_ratio,
            "reliability_checks": checks,
            "calibration_status": "RELIABLE" if reliable else "CALIBRATION_UNRELIABLE",
        }
        summaries.append(method_summary)

        for threshold_name, threshold in [
            ("T0_q95_higher", t0),
            ("T1_hampel6", t1),
            ("T2_winsor_q95", t2),
        ]:
            result: dict[str, object] = {
                "method": method,
                "threshold_name": threshold_name,
                "threshold": threshold,
                "calibration_status": method_summary["calibration_status"],
            }
            for split in ["healthy_test", "damage_test"]:
                split_data = data[data["split"].eq(split)].sort_values("file_number")
                alarms = split_data["score"].to_numpy(float) > threshold
                confirmed = confirmed_flags(alarms)
                prefix = "healthy" if split == "healthy_test" else "damage"
                result[f"{prefix}_n"] = int(len(split_data))
                result[f"{prefix}_alarm_fraction"] = float(np.mean(alarms))
                result[f"{prefix}_confirmed_2of3_fraction"] = float(np.mean(confirmed))
                if split == "damage_test":
                    result["earliest_five_damage_alarm_fraction"] = float(np.mean(alarms[:5]))
                    confirmed_files = split_data.loc[confirmed, "file_number"]
                    result["first_confirmed_damage_file"] = (
                        int(confirmed_files.iloc[0]) if len(confirmed_files) else None
                    )
            threshold_rows.append(result)

    pd.DataFrame(threshold_rows).to_csv(OUT / "p11a_threshold_evaluation.csv", index=False)
    pd.DataFrame(block_rows).to_csv(OUT / "p11a_calibration_block_diagnostics.csv", index=False)

    summary = {
        "schema_version": "p11a-health-side-calibration-reliability-exploratory-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "posthoc_mechanism_exploration_not_confirmatory",
        "protocol": PROTOCOL,
        "input": str(INPUT.relative_to(ROOT)),
        "damage_labels_used_for_threshold_or_reliability_rule": False,
        "p10_status_changed": False,
        "method_summaries": summaries,
        "warnings": [
            "P11-A was designed after P10 damage outcomes were known.",
            "Threshold performance on P10 is descriptive and cannot be treated as external confirmation.",
            "A reliable calibration flag does not imply damage detection validity.",
        ],
    }
    (OUT / "p11a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    block_df = pd.DataFrame(block_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        sub = block_df[block_df["method"].eq(method)]
        info = next(item for item in summaries if item["method"] == method)
        ax.plot(sub["block"], sub["block_max"], marker="o", label="block maximum")
        ax.plot(sub["block"], sub["block_median"], marker="s", label="block median")
        ax.axhline(info["t0_q95_higher"], color="tab:red", linestyle="--", label="T0 q95")
        ax.axhline(info["t1_hampel6"], color="tab:green", linestyle=":", label="T1 Hampel-6")
        ax.set_title(method.replace("_", "\n"))
        ax.set_xlabel("calibration block (5 files)")
        ax.set_ylabel("score")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("P11-A health-side calibration tail stability (exploratory)")
    fig.savefig(OUT / "fig01_calibration_block_tail_stability.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
