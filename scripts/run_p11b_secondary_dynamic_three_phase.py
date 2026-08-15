from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import run_p11b_wind_blade_confirmatory as frozen


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_references" / "wind_turbine_blade_shm_dataset.zip"
OUT = ROOT / "runs" / "p11b_secondary_dynamic_three_phase_v1"
PROTOCOL = "research_protocols/P11B_secondary_dynamic_three_phase_analysis_A2.md"

PHASES = {
    "plus": "niscopeWaveformsPlus",
    "zero_crossing": "niscopeWaveformsZeroCrossing",
    "minus": "niscopeWaveformsMinus",
}
DYNAMIC_RANGES = [
    (421_000, 1_137_000),
    (1_139_000, 1_641_000),
    (1_642_000, 1_692_000),
    (1_693_000, 1_737_000),
    (1_739_000, 1_795_000),
    (1_796_000, 1_805_000),
    (1_806_000, 2_248_000),
]
DYNAMIC_PATTERN = re.compile(
    r"/raw/Loading_cycle_(\d+)/5_cycles_50kHz/niscope_waveforms\.mat$"
)


def expected_cycles() -> np.ndarray:
    return np.concatenate([
        np.arange(start, stop + 1, 1_000, dtype=np.int64)
        for start, stop in DYNAMIC_RANGES
    ])


def member_map(archive: zipfile.ZipFile, cycles: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for info in archive.infolist():
        match = DYNAMIC_PATTERN.search(info.filename)
        if not match:
            continue
        cycle = int(match.group(1))
        if cycle in mapping:
            raise RuntimeError(f"duplicate dynamic cycle {cycle}")
        mapping[cycle] = info.filename
    expected = set(int(value) for value in cycles)
    if set(mapping) != expected:
        raise RuntimeError({
            "missing": sorted(expected - set(mapping))[:20],
            "extra": sorted(set(mapping) - expected)[:20],
        })
    return mapping


def load_phase(
    archive: zipfile.ZipFile,
    mapping: dict[int, str],
    cycles: np.ndarray,
    variable: str,
) -> tuple[np.ndarray, dict]:
    data = np.empty((len(cycles), frozen.N_SAMPLES, frozen.N_CHANNELS), dtype=np.float32)
    sample_rates = set()
    center_freqs = set()
    for idx, cycle in enumerate(cycles):
        with archive.open(mapping[int(cycle)], "r") as stream:
            payload = stream.read()
        mat = loadmat(io.BytesIO(payload), variable_names=[variable, "sampleRate", "centerFreq"])
        wave = np.asarray(mat[variable], dtype=np.float32)
        if wave.shape != (frozen.N_SAMPLES, frozen.N_CHANNELS):
            raise RuntimeError(f"unexpected dynamic shape {wave.shape} at {cycle}")
        data[idx] = wave
        sample_rates.add(float(np.asarray(mat["sampleRate"]).squeeze()))
        center_freqs.add(float(np.asarray(mat["centerFreq"]).squeeze()))
    data -= np.median(data[:, :500, :], axis=1, keepdims=True)
    return data, {"sample_rates": sorted(sample_rates), "center_freq_values": sorted(center_freqs)}


def phase_metrics(frame: pd.DataFrame, threshold: float, energy_threshold: float) -> tuple[dict, pd.DataFrame]:
    work = frame.copy()
    work["alarm"] = work["score"] > threshold
    work["confirmed_alarm"] = False
    work["energy_alarm"] = work["energy_score"] > energy_threshold
    stage_rows = []
    for stage, group in work.groupby("stage", sort=False):
        ordered = group.sort_values("cycle")
        valid = ordered["score"].notna()
        flags = frozen.confirmed_flags(
            ordered.loc[valid, "cycle"].to_numpy(int),
            ordered.loc[valid, "alarm"].to_numpy(bool),
        )
        work.loc[ordered.loc[valid].index, "confirmed_alarm"] = flags
        usable = work.loc[ordered.loc[valid].index]
        stage_rows.append({
            "stage": stage,
            "n_total": int(len(ordered)),
            "n_scored": int(len(usable)),
            "score_median": float(usable["score"].median()) if len(usable) else math.nan,
            "alarm_fraction": float(usable["alarm"].mean()) if len(usable) else math.nan,
            "confirmed_fraction": float(usable["confirmed_alarm"].mean()) if len(usable) else math.nan,
            "energy_alarm_fraction": float(usable["energy_alarm"].mean()) if len(usable) else math.nan,
        })

    reference = work[work["stage"].eq("reference_test") & work["score"].notna()]
    e1 = work[work["stage"].eq("E1_crack_plus_3mm") & work["score"].notna()]
    combined = pd.concat([reference.assign(label=0), e1.assign(label=1)], ignore_index=True)
    auc = float(roc_auc_score(combined["label"], combined["score"]))
    ap = float(average_precision_score(combined["label"], combined["score"]))
    energy_auc = float(roc_auc_score(combined["label"], combined["energy_score"]))
    first = e1[e1["confirmed_alarm"]]["cycle"]
    first_cycle = int(first.min()) if len(first) else None
    stages = pd.DataFrame(stage_rows).set_index("stage")
    progression = [
        float(stages.loc[name, "score_median"])
        for name in ["E1_crack_plus_3mm", "E2_crack_plus_10mm", "E3_overload_40kg", "E5_overload_70kg_failure"]
    ]
    result = {
        "threshold": threshold,
        "energy_threshold": energy_threshold,
        "reference_fpr": float(reference["alarm"].mean()),
        "reference_confirmed_fraction": float(reference["confirmed_alarm"].mean()),
        "E1_recall": float(e1["alarm"].mean()),
        "E1_confirmed_recall": float(e1["confirmed_alarm"].mean()),
        "first_confirmed_E1_cycle": first_cycle,
        "reference_vs_E1_auc": auc,
        "reference_vs_E1_average_precision": ap,
        "energy_reference_vs_E1_auc": energy_auc,
        "auc_margin_over_energy": auc - energy_auc,
        "progression_medians": progression,
        "progression_spearman": float(spearmanr(np.arange(1, 5), progression).statistic),
    }
    return result, work


def self_test() -> None:
    cycles = expected_cycles()
    assert len(cycles) == 1826
    assert cycles[0] == 421_000 and cycles[-1] == 2_248_000
    assert frozen.stage_for_cycle(1_025_000) == "E1_crack_plus_3mm"
    print(json.dumps({"self_test": "pass", "cycles": len(cycles)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.execute:
        raise SystemExit("Use --self-test or --execute")
    if frozen.sha256_file(ARCHIVE) != frozen.EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("archive SHA-256 mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    cycles = expected_cycles()
    stages = np.array([frozen.stage_for_cycle(int(value)) for value in cycles], dtype=object)
    train_indices = np.flatnonzero(stages == "reference_train")
    all_frames = []
    phase_results = {}
    stage_tables = []

    with zipfile.ZipFile(ARCHIVE) as archive:
        mapping = member_map(archive, cycles)
        temperatures, temperature_metadata = frozen.load_temperature_mapping(archive, cycles)
        for phase, variable in PHASES.items():
            waves, acquisition = load_phase(archive, mapping, cycles, variable)
            sensor_scores = np.full((len(cycles), 4), np.nan)
            sensor_lags = np.full((len(cycles), 4), 999, dtype=np.int16)
            sensor_energy = np.full((len(cycles), 4), np.nan)
            for sensor in range(4):
                score, lag, _ = frozen.score_one_channel(waves[:, :, sensor], train_indices, temperatures)
                sensor_scores[:, sensor] = score
                sensor_lags[:, sensor] = lag
                sensor_energy[:, sensor] = np.sqrt(np.mean(waves[:, :, sensor].astype(np.float64) ** 2, axis=1))
            del waves

            scores = np.full(len(cycles), np.nan)
            energies = np.full(len(cycles), np.nan)
            for idx in range(len(cycles)):
                valid = sensor_scores[idx, np.isfinite(sensor_scores[idx])]
                if len(valid) >= 3:
                    scores[idx] = float(np.mean(np.sort(valid)[-3:]))
                energy = sensor_energy[idx, np.isfinite(sensor_energy[idx])]
                if len(energy) >= 3:
                    energies[idx] = float(np.median(energy))

            frame = pd.DataFrame({
                "phase": phase,
                "cycle": cycles,
                "stage": stages,
                "temperature_c": temperatures,
                "score": scores,
                "energy_score": energies,
                "median_abs_lag": np.nanmedian(np.where(sensor_lags == 999, np.nan, np.abs(sensor_lags)), axis=1),
            })
            cal = frame[frame["stage"].eq("reference_calibration") & frame["score"].notna()]
            reliability = frozen.calibration_reliability(
                cal["score"].to_numpy(float), cal["temperature_c"].notna().to_numpy(bool)
            )
            threshold = float(reliability.get("t1_hampel6", math.nan))
            energy_threshold = frozen.hampel6(cal["energy_score"].dropna().to_numpy(float))[0]
            metrics, scored_frame = phase_metrics(frame, threshold, energy_threshold)
            metrics.update({"calibration_reliability": reliability, "acquisition_metadata": acquisition})
            phase_results[phase] = metrics
            scored_frame["phase"] = phase
            all_frames.append(scored_frame)

            stage_table = []
            for stage, group in scored_frame.groupby("stage", sort=False):
                usable = group[group["score"].notna()]
                stage_table.append({
                    "phase": phase,
                    "stage": stage,
                    "n": int(len(usable)),
                    "score_median": float(usable["score"].median()) if len(usable) else math.nan,
                    "alarm_fraction": float(usable["alarm"].mean()) if len(usable) else math.nan,
                    "confirmed_fraction": float(usable["confirmed_alarm"].mean()) if len(usable) else math.nan,
                })
            stage_tables.extend(stage_table)

    combined_frame = pd.concat(all_frames, ignore_index=True)
    combined_frame.to_csv(OUT / "p11b_dynamic_sample_scores.csv.gz", index=False, compression="gzip")
    pd.DataFrame(stage_tables).to_csv(OUT / "p11b_dynamic_stage_summary.csv", index=False)

    summary = {
        "schema_version": "p11b-secondary-dynamic-three-phase-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "predeclared_secondary_implemented_after_primary_no_effect_on_primary_status",
        "protocol": PROTOCOL,
        "primary_p11b_status_unchanged": "FAIL_calibration_reliability_gate",
        "archive_sha256": frozen.EXPECTED_ARCHIVE_SHA256,
        "dynamic_cycles": int(len(cycles)),
        "temperature_metadata": temperature_metadata,
        "phase_results": phase_results,
    }
    (OUT / "p11b_dynamic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "P11B_DYNAMIC_COMPLETED.json").write_text(
        json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(), "primary_status_unchanged": True}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
