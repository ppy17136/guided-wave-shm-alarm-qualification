from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "p11b_wind_blade_confirmatory_v1"
OUT = RUN / "posthoc_diagnostics"


def hampel6(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + 6.0 * 1.4826 * mad


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RUN / "p11b_sample_scores.csv.gz")
    summary = json.loads((RUN / "p11b_summary.json").read_text(encoding="utf-8"))
    threshold = float(summary["main_threshold"])

    cal = frame[frame["stage"].eq("reference_calibration")].sort_values("cycle").copy()
    block_rows = []
    for block_index, start in enumerate(range(0, len(cal), 10), start=1):
        block = cal.iloc[start : start + 10]
        if len(block) < 8:
            continue
        keep = cal.drop(block.index)["main_score"].dropna().to_numpy(float)
        block_rows.append({
            "block": block_index,
            "cycle_min": int(block["cycle"].min()),
            "cycle_max": int(block["cycle"].max()),
            "temperature_min_c": float(block["temperature_c"].min()),
            "temperature_max_c": float(block["temperature_c"].max()),
            "score_median": float(block["main_score"].median()),
            "score_max": float(block["main_score"].max()),
            "leave_this_block_out_hampel6": hampel6(keep),
        })
    pd.DataFrame(block_rows).to_csv(OUT / "calibration_block_influence.csv", index=False)

    support_rows = []
    for stage in ["reference_test", "E1_crack_plus_3mm", "E2_crack_plus_10mm", "E3_overload_40kg", "E4_overload_60kg", "E5_overload_70kg_failure"]:
        stage_frame = frame[frame["stage"].eq(stage)]
        for support_label, support_value in [("in_support", False), ("out_of_support", True)]:
            part = stage_frame[stage_frame["temperature_out_of_support"].eq(support_value)]
            scored = part[part["main_score"].notna()]
            support_rows.append({
                "stage": stage,
                "temperature_support": support_label,
                "n_total": int(len(part)),
                "n_scored": int(len(scored)),
                "temperature_min_c": float(scored["temperature_c"].min()) if len(scored) else np.nan,
                "temperature_max_c": float(scored["temperature_c"].max()) if len(scored) else np.nan,
                "score_median": float(scored["main_score"].median()) if len(scored) else np.nan,
                "alarm_fraction": float((scored["main_score"] > threshold).mean()) if len(scored) else np.nan,
                "confirmed_fraction": float(scored["main_confirmed_alarm"].mean()) if len(scored) else np.nan,
            })
    pd.DataFrame(support_rows).to_csv(OUT / "temperature_support_stage_metrics.csv", index=False)

    reference = frame[frame["stage"].eq("reference_test") & frame["main_score"].notna()].copy()
    e1 = frame[frame["stage"].eq("E1_crack_plus_3mm") & frame["main_score"].notna()].copy()
    false_alarms = reference[reference["main_score"] > threshold]
    misses = e1[e1["main_score"] <= threshold]
    false_alarms.to_csv(OUT / "reference_false_alarms.csv", index=False)
    misses.to_csv(OUT / "E1_missed_samples.csv", index=False)

    correlations = {}
    for stage in ["reference_calibration", "reference_test", "E1_crack_plus_3mm"]:
        part = frame[frame["stage"].eq(stage)].dropna(subset=["temperature_c", "main_score"])
        correlations[stage] = {
            "n": int(len(part)),
            "temperature_score_spearman": float(spearmanr(part["temperature_c"], part["main_score"]).statistic),
        }

    report = {
        "schema_version": "p11b-posthoc-diagnostics-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "posthoc_descriptive_no_threshold_or_method_changes",
        "main_threshold_unchanged": threshold,
        "reference_false_alarm_count": int(len(false_alarms)),
        "reference_false_alarm_cycles": [int(value) for value in false_alarms["cycle"]],
        "E1_miss_count": int(len(misses)),
        "E1_miss_cycle_min": int(misses["cycle"].min()) if len(misses) else None,
        "E1_miss_cycle_max": int(misses["cycle"].max()) if len(misses) else None,
        "E1_misses_out_of_temperature_support": int(misses["temperature_out_of_support"].sum()),
        "temperature_score_correlations": correlations,
        "calibration_leave_one_block_out_threshold_min": float(min(row["leave_this_block_out_hampel6"] for row in block_rows)),
        "calibration_leave_one_block_out_threshold_max": float(max(row["leave_this_block_out_hampel6"] for row in block_rows)),
        "warnings": [
            "P11-B primary status remains FAIL because the frozen calibration reliability gate failed.",
            "These diagnostics must not be used to alter the frozen threshold or success criteria.",
        ],
    }
    (OUT / "posthoc_diagnostic_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
