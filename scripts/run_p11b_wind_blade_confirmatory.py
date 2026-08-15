from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_references" / "wind_turbine_blade_shm_dataset.zip"
OUT = ROOT / "runs" / "p11b_wind_blade_confirmatory_v1"
EXPECTED_ARCHIVE_SHA256 = "02394f981f2d9dc757dfe194d321bc8383ccb8d806bde009e60e23547a7c5e5e"
EXPECTED_LOCK1_MANIFEST_SHA256 = "7bec40dec4d5b4ea8023b8c63c2df3fe680043e9a3aa476dfcc0ea63f2606d5b"
LOCK1_MANIFEST = ROOT / "research_protocols" / "P11B_LOCK1_freeze_manifest_v1.json"
PROTOCOL = "research_protocols/P11B_wind_blade_confirmatory_preregistration_v1.md"
AMENDMENT = "research_protocols/P11B_preregistration_amendment_A1_schema_mapping.md"

FREQUENCIES = [20, 40, 60, 80, 100, 120]
SENSORS = ["S1", "S2", "S3", "S4"]
N_SAMPLES = 10_000
N_CHANNELS = 4
ALIGN_WINDOW = (79, 610)
MAX_LAG = 25
TEMP_NEIGHBORS = 30
MIN_TEMP_NEIGHBORS = 20

STATIC_RANGES = [
    (420_000, 1_136_000),
    (1_138_000, 1_640_000),
    (1_641_000, 1_691_000),
    (1_692_000, 1_737_000),
    (1_738_000, 1_794_000),
    (1_795_000, 1_804_000),
    (1_805_000, 2_248_000),
]

STAGES = [
    ("excluded_early_adjustment", 420_000, 501_000),
    ("reference_train", 502_000, 800_000),
    ("reference_calibration", 801_000, 950_000),
    ("reference_test", 951_000, 1_020_000),
    ("E1_crack_plus_3mm", 1_025_000, 1_637_000),
    ("E2_crack_plus_10mm", 1_642_000, 1_734_000),
    ("E3_overload_40kg", 1_739_000, 1_791_000),
    ("E4_overload_60kg", 1_796_000, 1_801_000),
    ("E5_overload_70kg_failure", 1_806_000, 2_248_000),
]

STATIC_PATTERN = re.compile(
    r"/averaged/Loading_cycle_(\d+)/5_cycles_(20|40|60|80|100|120)kHz/niscope_avg_waveform\.mat$"
)
TEMP_MEMBER = "wind_turbine_blade_shm_dataset/strains_temperature_curated.mat"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_static_cycles() -> np.ndarray:
    parts = [np.arange(start, stop + 1, 1_000, dtype=np.int64) for start, stop in STATIC_RANGES]
    return np.concatenate(parts)


def stage_for_cycle(cycle: int) -> str:
    for name, start, stop in STAGES:
        if start <= cycle <= stop:
            return name
    return "embargo"


def quantile_higher(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="higher"))


def hampel6(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    return median + 6.0 * scale, median, mad, scale


def best_lag(x: np.ndarray, template: np.ndarray) -> int:
    lo, hi = ALIGN_WINDOW
    best = (float("-inf"), 0)
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        t_start = max(lo, -lag)
        t_stop = min(hi, len(template) - lag)
        if t_stop - t_start < 100:
            continue
        a = x[t_start + lag : t_stop + lag].astype(np.float64, copy=False)
        b = template[t_start:t_stop].astype(np.float64, copy=False)
        a = a - a.mean()
        b = b - b.mean()
        denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
        corr = float(np.dot(a, b) / denom) if denom > 0 else -1.0
        if corr > best[0]:
            best = (corr, lag)
    return best[1]


def aligned_channel_score(x: np.ndarray, template: np.ndarray, scale: np.ndarray) -> tuple[float, int]:
    lag = best_lag(x, template)
    t_start = max(0, -lag)
    t_stop = min(len(template), len(template) - lag)
    residual = np.abs(
        x[t_start + lag : t_stop + lag].astype(np.float64, copy=False)
        - template[t_start:t_stop]
    ) / scale[t_start:t_stop]
    return float(np.quantile(residual, 0.99)), lag


def confirmed_flags(cycles: np.ndarray, alarms: np.ndarray) -> np.ndarray:
    flags = np.zeros(len(alarms), dtype=bool)
    for idx in range(2, len(alarms)):
        window_cycles = cycles[idx - 2 : idx + 1]
        if np.any(np.diff(window_cycles) > 3_000):
            continue
        window_alarm = alarms[idx - 2 : idx + 1]
        if int(window_alarm.sum()) >= 2:
            flags[idx - 2 : idx + 1] |= window_alarm
    return flags


def count_segments(cycles: np.ndarray, flags: np.ndarray) -> int:
    count = 0
    previous_cycle: int | None = None
    for cycle, flag in zip(cycles, flags):
        if not flag:
            continue
        if previous_cycle is None or cycle - previous_cycle > 3_000:
            count += 1
        previous_cycle = int(cycle)
    return count


def self_test() -> None:
    rng = np.random.default_rng(20260811)
    t = np.linspace(0, 1, N_SAMPLES, endpoint=False)
    template = np.sin(2 * np.pi * 50 * t) * np.hanning(N_SAMPLES)
    shifted = np.zeros_like(template)
    shifted[7:] = template[:-7]
    assert best_lag(shifted, template) == 7
    scale = np.ones_like(template) * 0.1
    score, lag = aligned_channel_score(shifted, template, scale)
    assert lag == 7 and score < 1e-8
    cycles = np.array([1_000, 2_000, 3_000, 4_000, 8_000])
    flags = confirmed_flags(cycles, np.array([False, True, True, False, True]))
    assert flags.tolist() == [False, True, True, False, False]
    values = rng.normal(0, 1, 1000)
    threshold, median, mad, robust_scale = hampel6(values)
    assert threshold > median and mad > 0 and robust_scale > 0
    assert len(expected_static_cycles()) == 1828
    print(json.dumps({"self_test": "pass", "lag": lag, "score": score}))


def load_temperature_mapping(archive: zipfile.ZipFile, cycles: np.ndarray) -> tuple[np.ndarray, dict]:
    temp_dir = Path(tempfile.gettempdir()) / "p11b_numeric_runtime"
    temp_dir.mkdir(parents=True, exist_ok=True)
    local = temp_dir / "strains_temperature_curated.mat"
    expected_size = archive.getinfo(TEMP_MEMBER).file_size
    if not local.exists() or local.stat().st_size != expected_size:
        with archive.open(TEMP_MEMBER, "r") as source, local.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    with h5py.File(local, "r") as handle:
        cycle_vector = np.asarray(handle["cyclingVectInterp"][:, 0], dtype=np.float64)
        temperature_vector = np.asarray(handle["tc"][0, :], dtype=np.float64)
    if len(cycle_vector) != len(temperature_vector):
        raise RuntimeError("temperature cycle and value length mismatch")
    if np.any(np.diff(cycle_vector) < 0):
        raise RuntimeError("cyclingVectInterp is not monotonic nondecreasing")
    mapped = np.full(len(cycles), np.nan, dtype=np.float64)
    counts = np.zeros(len(cycles), dtype=np.int64)
    for idx, cycle in enumerate(cycles):
        left = int(np.searchsorted(cycle_vector, cycle - 500, side="left"))
        right = int(np.searchsorted(cycle_vector, cycle + 500, side="right"))
        values = temperature_vector[left:right]
        values = values[np.isfinite(values)]
        counts[idx] = len(values)
        if len(values):
            mapped[idx] = float(np.median(values))
    metadata = {
        "source_points": int(len(cycle_vector)),
        "mapped_cycles": int(np.isfinite(mapped).sum()),
        "missing_cycles": int((~np.isfinite(mapped)).sum()),
        "minimum_window_points": int(counts.min()),
        "median_window_points": float(np.median(counts)),
        "maximum_window_points": int(counts.max()),
    }
    return mapped, metadata


def static_member_map(archive: zipfile.ZipFile, cycles: np.ndarray) -> dict[tuple[int, int], str]:
    mapping: dict[tuple[int, int], str] = {}
    for info in archive.infolist():
        match = STATIC_PATTERN.search(info.filename)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in mapping:
            raise RuntimeError(f"duplicate static member key {key}")
        mapping[key] = info.filename
    expected = {(int(cycle), freq) for cycle in cycles for freq in FREQUENCIES}
    missing = sorted(expected - set(mapping))
    extra = sorted(set(mapping) - expected)
    if missing or extra:
        raise RuntimeError({"missing_static": missing[:20], "extra_static": extra[:20]})
    return mapping


def load_frequency_waveforms(
    archive: zipfile.ZipFile,
    member_map: dict[tuple[int, int], str],
    cycles: np.ndarray,
    frequency: int,
) -> tuple[np.ndarray, dict]:
    data = np.empty((len(cycles), N_SAMPLES, N_CHANNELS), dtype=np.float32)
    sample_rates = set()
    center_freqs = set()
    averages = set()
    for idx, cycle in enumerate(cycles):
        member = member_map[(int(cycle), frequency)]
        with archive.open(member, "r") as stream:
            payload = stream.read()
        mat = loadmat(
            io.BytesIO(payload),
            variable_names=["niscopeAvgWaveform", "sampleRate", "centerFreq", "nAverages"],
        )
        wave = np.asarray(mat["niscopeAvgWaveform"], dtype=np.float32)
        if wave.shape != (N_SAMPLES, N_CHANNELS):
            raise RuntimeError(f"unexpected waveform shape {wave.shape}: {member}")
        data[idx] = wave
        sample_rates.add(float(np.asarray(mat["sampleRate"]).squeeze()))
        center_freqs.add(float(np.asarray(mat["centerFreq"]).squeeze()))
        averages.add(float(np.asarray(mat["nAverages"]).squeeze()))
    data -= np.median(data[:, :500, :], axis=1, keepdims=True)
    return data, {
        "frequency_khz": frequency,
        "sample_rates": sorted(sample_rates),
        "center_freq_values": sorted(center_freqs),
        "n_averages_values": sorted(averages),
    }


def neighbor_groups(train_temperatures: np.ndarray, target_temperatures: np.ndarray) -> tuple[list[tuple[int, ...] | None], np.ndarray]:
    groups: list[tuple[int, ...] | None] = []
    sufficient = np.zeros(len(target_temperatures), dtype=bool)
    finite_train = np.flatnonzero(np.isfinite(train_temperatures))
    for idx, temp in enumerate(target_temperatures):
        if not np.isfinite(temp) or len(finite_train) < MIN_TEMP_NEIGHBORS:
            groups.append(None)
            continue
        order = finite_train[np.argsort(np.abs(train_temperatures[finite_train] - temp), kind="stable")]
        selected = order[: min(TEMP_NEIGHBORS, len(order))]
        if len(selected) < MIN_TEMP_NEIGHBORS:
            groups.append(None)
            continue
        groups.append(tuple(int(value) for value in selected))
        sufficient[idx] = True
    return groups, sufficient


def score_one_channel(
    waveforms: np.ndarray,
    train_indices: np.ndarray,
    temperatures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = waveforms[train_indices]
    groups, sufficient = neighbor_groups(temperatures[train_indices], temperatures)
    scores = np.full(len(waveforms), np.nan, dtype=np.float64)
    lags = np.full(len(waveforms), 999, dtype=np.int16)
    cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    for idx, group in enumerate(groups):
        if group is None:
            continue
        if group not in cache:
            reference = train[np.asarray(group, dtype=int)].astype(np.float64)
            template = np.median(reference, axis=0)
            scale = 1.4826 * np.median(np.abs(reference - template), axis=0)
            positive = scale[scale > 0]
            floor = float(np.quantile(positive, 0.05)) if len(positive) else np.finfo(float).eps
            scale = np.maximum(scale, max(floor, np.finfo(float).eps))
            cache[group] = (template, scale)
        template, scale = cache[group]
        scores[idx], lags[idx] = aligned_channel_score(waveforms[idx], template, scale)
    return scores, lags, sufficient


def calibration_reliability(cal_scores: np.ndarray, cal_temps_valid: np.ndarray) -> dict:
    if len(cal_scores) < 100:
        return {"status": "CALIBRATION_UNRELIABLE", "reason": "fewer_than_100_valid_scores"}
    t1, median, mad, robust_scale = hampel6(cal_scores)
    t0 = quantile_higher(cal_scores, 0.95)
    blocks = [cal_scores[start : start + 10] for start in range(0, len(cal_scores), 10)]
    blocks = [block for block in blocks if len(block) >= 8]
    block_maxima = np.asarray([float(np.max(block)) for block in blocks])
    heterogeneity = float(block_maxima.max() / max(float(np.median(block_maxima)), np.finfo(float).eps))
    loo_t1 = []
    for block_index in range(len(blocks)):
        keep = np.concatenate([block for idx, block in enumerate(blocks) if idx != block_index])
        loo_t1.append(hampel6(keep)[0])
    loo_range = float((max(loo_t1) - min(loo_t1)) / max(abs(t1), np.finfo(float).eps))
    ratio = float(t0 / max(abs(t1), np.finfo(float).eps))
    checks = {
        "tail_block_heterogeneity_at_most_2": heterogeneity <= 2.0,
        "loo_t1_relative_range_at_most_0_25": loo_range <= 0.25,
        "t0_t1_ratio_between_0_5_and_2": 0.5 <= ratio <= 2.0,
        "at_least_100_valid_calibration_scores": len(cal_scores) >= 100,
        "at_least_10_blocks": len(blocks) >= 10,
        "temperature_coverage_at_least_0_90": float(np.mean(cal_temps_valid)) >= 0.90,
    }
    return {
        "status": "RELIABLE" if all(checks.values()) else "CALIBRATION_UNRELIABLE",
        "t1_hampel6": t1,
        "t0_q95_higher": t0,
        "median": median,
        "mad": mad,
        "robust_scale": robust_scale,
        "blocks": len(blocks),
        "tail_block_heterogeneity": heterogeneity,
        "loo_t1_min": min(loo_t1),
        "loo_t1_max": max(loo_t1),
        "loo_t1_relative_range": loo_range,
        "t0_t1_ratio": ratio,
        "temperature_coverage": float(np.mean(cal_temps_valid)),
        "checks": checks,
    }


def evaluate_method(frame: pd.DataFrame, score_column: str, threshold: float) -> tuple[dict, pd.DataFrame]:
    result = frame.copy()
    result["alarm"] = result[score_column] > threshold
    result["confirmed_alarm"] = False
    stage_rows = []
    for stage, group in result.groupby("stage", sort=False):
        ordered = group.sort_values("cycle")
        valid = ordered[score_column].notna().to_numpy()
        confirmed = np.zeros(len(ordered), dtype=bool)
        if valid.any():
            valid_pos = np.flatnonzero(valid)
            confirmed_valid = confirmed_flags(
                ordered.loc[valid, "cycle"].to_numpy(int),
                ordered.loc[valid, "alarm"].to_numpy(bool),
            )
            confirmed[valid_pos] = confirmed_valid
        result.loc[ordered.index, "confirmed_alarm"] = confirmed
        usable = ordered[ordered[score_column].notna()]
        stage_rows.append({
            "stage": stage,
            "n_total": int(len(ordered)),
            "n_scored": int(len(usable)),
            "rejected_fraction": float(1 - len(usable) / len(ordered)) if len(ordered) else math.nan,
            "score_median": float(usable[score_column].median()) if len(usable) else math.nan,
            "alarm_fraction": float(usable["alarm"].mean()) if len(usable) else math.nan,
            "confirmed_alarm_fraction": float(result.loc[usable.index, "confirmed_alarm"].mean()) if len(usable) else math.nan,
            "confirmed_segments": count_segments(
                usable["cycle"].to_numpy(int), result.loc[usable.index, "confirmed_alarm"].to_numpy(bool)
            ) if len(usable) else 0,
        })
    return {"threshold": threshold}, pd.DataFrame(stage_rows)


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

    if sha256_file(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("archive SHA-256 mismatch")
    if sha256_file(LOCK1_MANIFEST) != EXPECTED_LOCK1_MANIFEST_SHA256:
        raise RuntimeError("LOCK1 manifest SHA-256 mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    cycles = expected_static_cycles()
    stages = np.array([stage_for_cycle(int(cycle)) for cycle in cycles], dtype=object)
    train_indices = np.flatnonzero(stages == "reference_train")

    channel_scores = np.full((len(cycles), len(FREQUENCIES) * len(SENSORS)), np.nan, dtype=np.float64)
    channel_energy = np.full_like(channel_scores, np.nan)
    channel_lags = np.full(channel_scores.shape, 999, dtype=np.int16)
    channel_sufficient = np.zeros(channel_scores.shape, dtype=bool)
    acquisition_metadata = []

    with zipfile.ZipFile(ARCHIVE) as archive:
        member_map = static_member_map(archive, cycles)
        temperatures, temperature_metadata = load_temperature_mapping(archive, cycles)
        for frequency_index, frequency in enumerate(FREQUENCIES):
            waveforms, metadata = load_frequency_waveforms(archive, member_map, cycles, frequency)
            acquisition_metadata.append(metadata)
            for sensor_index, sensor in enumerate(SENSORS):
                channel_index = frequency_index * len(SENSORS) + sensor_index
                waves = waveforms[:, :, sensor_index]
                scores, lags, sufficient = score_one_channel(waves, train_indices, temperatures)
                channel_scores[:, channel_index] = scores
                channel_lags[:, channel_index] = lags
                channel_sufficient[:, channel_index] = sufficient
                channel_energy[:, channel_index] = np.sqrt(np.mean(waves.astype(np.float64) ** 2, axis=1))
            del waveforms

    valid_channel_counts = np.sum(np.isfinite(channel_scores), axis=1)
    main_scores = np.full(len(cycles), np.nan)
    energy_scores = np.full(len(cycles), np.nan)
    for idx in range(len(cycles)):
        valid_scores = channel_scores[idx, np.isfinite(channel_scores[idx])]
        valid_energy = channel_energy[idx, np.isfinite(channel_energy[idx])]
        if len(valid_scores) >= 18:
            main_scores[idx] = float(np.mean(np.sort(valid_scores)[-3:]))
        if len(valid_energy) >= 18:
            energy_scores[idx] = float(np.median(valid_energy))

    train_temp = temperatures[train_indices]
    train_temp_min = float(np.nanmin(train_temp))
    train_temp_max = float(np.nanmax(train_temp))
    temp_out_of_support = (temperatures < train_temp_min) | (temperatures > train_temp_max)

    frame = pd.DataFrame({
        "cycle": cycles,
        "stage": stages,
        "temperature_c": temperatures,
        "temperature_out_of_support": temp_out_of_support,
        "valid_channel_count": valid_channel_counts,
        "main_score": main_scores,
        "energy_score": energy_scores,
    })
    cal = frame[frame["stage"].eq("reference_calibration")]
    valid_cal = cal[cal["main_score"].notna()]
    reliability = calibration_reliability(
        valid_cal["main_score"].to_numpy(float),
        valid_cal["temperature_c"].notna().to_numpy(bool),
    )
    main_threshold = float(reliability.get("t1_hampel6", math.nan))
    energy_threshold = hampel6(cal["energy_score"].dropna().to_numpy(float))[0]

    _, main_stage = evaluate_method(frame, "main_score", main_threshold)
    _, energy_stage = evaluate_method(frame, "energy_score", energy_threshold)
    main_stage["method"] = "temperature_conditioned_structural_residual"
    energy_stage["method"] = "input_rms_energy_negative_control"
    stage_summary = pd.concat([main_stage, energy_stage], ignore_index=True)

    frame["main_alarm"] = frame["main_score"] > main_threshold
    frame["main_confirmed_alarm"] = False
    for stage, group in frame.groupby("stage", sort=False):
        ordered = group.sort_values("cycle")
        valid = ordered["main_score"].notna()
        flags = confirmed_flags(
            ordered.loc[valid, "cycle"].to_numpy(int),
            ordered.loc[valid, "main_alarm"].to_numpy(bool),
        )
        frame.loc[ordered.loc[valid].index, "main_confirmed_alarm"] = flags

    reference = frame[frame["stage"].eq("reference_test") & frame["main_score"].notna()]
    e1 = frame[frame["stage"].eq("E1_crack_plus_3mm") & frame["main_score"].notna()]
    combined = pd.concat([reference.assign(label=0), e1.assign(label=1)], ignore_index=True)
    main_auc = float(roc_auc_score(combined["label"], combined["main_score"]))
    main_ap = float(average_precision_score(combined["label"], combined["main_score"]))
    energy_auc = float(roc_auc_score(combined["label"], combined["energy_score"]))

    stage_lookup = stage_summary[stage_summary["method"].eq("temperature_conditioned_structural_residual")].set_index("stage")
    progression_names = ["E1_crack_plus_3mm", "E2_crack_plus_10mm", "E3_overload_40kg", "E5_overload_70kg_failure"]
    progression_medians = [float(stage_lookup.loc[name, "score_median"]) for name in progression_names]
    progression_spearman = float(spearmanr(np.arange(1, 5), progression_medians).statistic)

    e1_confirmed = e1[e1["main_confirmed_alarm"]]
    first_confirmed_cycle = int(e1_confirmed["cycle"].min()) if len(e1_confirmed) else None
    reference_confirmed_segments = count_segments(
        reference["cycle"].to_numpy(int), reference["main_confirmed_alarm"].to_numpy(bool)
    )

    gates = {
        "calibration_reliable": reliability.get("status") == "RELIABLE",
        "reference_point_fpr_at_most_0_05": float(reference["main_alarm"].mean()) <= 0.05,
        "reference_confirmed_fraction_at_most_0_01": float(reference["main_confirmed_alarm"].mean()) <= 0.01,
        "reference_confirmed_segments_at_most_1": reference_confirmed_segments <= 1,
        "E1_point_recall_at_least_0_80": float(e1["main_alarm"].mean()) >= 0.80,
        "E1_confirmed_recall_at_least_0_70": float(e1["main_confirmed_alarm"].mean()) >= 0.70,
        "first_confirmed_by_E1_plus_20000": first_confirmed_cycle is not None and first_confirmed_cycle <= 1_045_000,
        "E2_point_recall_at_least_0_80": float(stage_lookup.loc["E2_crack_plus_10mm", "alarm_fraction"]) >= 0.80,
        "E3_point_recall_at_least_0_80": float(stage_lookup.loc["E3_overload_40kg", "alarm_fraction"]) >= 0.80,
        "E5_point_recall_at_least_0_80": float(stage_lookup.loc["E5_overload_70kg_failure", "alarm_fraction"]) >= 0.80,
        "progression_spearman_at_least_0_60": progression_spearman >= 0.60,
        "reference_vs_E1_auc_at_least_0_80": main_auc >= 0.80,
        "auc_margin_over_energy_at_least_0_10": main_auc - energy_auc >= 0.10,
    }

    channel_rows = []
    for frequency_index, frequency in enumerate(FREQUENCIES):
        for sensor_index, sensor in enumerate(SENSORS):
            channel_index = frequency_index * len(SENSORS) + sensor_index
            for stage in ["reference_test", *progression_names]:
                mask = stages == stage
                values = channel_scores[mask, channel_index]
                channel_rows.append({
                    "frequency_khz": frequency,
                    "sensor": sensor,
                    "stage": stage,
                    "n": int(np.isfinite(values).sum()),
                    "median_channel_score": float(np.nanmedian(values)),
                    "median_abs_lag_samples": float(np.nanmedian(np.abs(channel_lags[mask, channel_index][channel_lags[mask, channel_index] != 999]))),
                })
    pd.DataFrame(channel_rows).to_csv(OUT / "p11b_channel_stage_summary.csv", index=False)
    stage_summary.to_csv(OUT / "p11b_stage_summary.csv", index=False)
    frame.to_csv(OUT / "p11b_sample_scores.csv.gz", index=False, compression="gzip")

    summary = {
        "schema_version": "p11b-wind-blade-confirmatory-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "confirmatory_single_execution",
        "protocol": PROTOCOL,
        "amendment": AMENDMENT,
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "lock1_manifest_sha256": EXPECTED_LOCK1_MANIFEST_SHA256,
        "static_cycles": int(len(cycles)),
        "stage_counts": frame["stage"].value_counts().to_dict(),
        "temperature_metadata": temperature_metadata,
        "training_temperature_range_c": [train_temp_min, train_temp_max],
        "temperature_out_of_support_count": int(temp_out_of_support.sum()),
        "acquisition_metadata": acquisition_metadata,
        "calibration_reliability": reliability,
        "main_threshold": main_threshold,
        "energy_threshold": energy_threshold,
        "reference_point_fpr": float(reference["main_alarm"].mean()),
        "reference_confirmed_fraction": float(reference["main_confirmed_alarm"].mean()),
        "reference_confirmed_segments": reference_confirmed_segments,
        "E1_point_recall": float(e1["main_alarm"].mean()),
        "E1_confirmed_recall": float(e1["main_confirmed_alarm"].mean()),
        "first_confirmed_E1_cycle": first_confirmed_cycle,
        "reference_vs_E1_auc": main_auc,
        "reference_vs_E1_average_precision": main_ap,
        "energy_reference_vs_E1_auc": energy_auc,
        "auc_margin_over_energy": main_auc - energy_auc,
        "progression_stage_medians": dict(zip(progression_names, progression_medians)),
        "progression_spearman": progression_spearman,
        "gates": gates,
        "primary_passed": all(gates.values()),
        "metadata_discrepancy": "paper reports approximately 14 Hz shaker excitation; frequencies.txt reports 124 Hz; neither value is used by the algorithm",
    }
    (OUT / "p11b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot = frame[frame["main_score"].notna()]
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    ax.plot(plot["cycle"], plot["main_score"], linewidth=0.8, color="tab:blue")
    ax.axhline(main_threshold, color="tab:red", linestyle="--", label="frozen Hampel-6 threshold")
    for cycle, label in [(1_023_000, "+3 mm"), (1_640_000, "+10 mm"), (1_737_000, "40 kg"), (1_794_000, "60 kg"), (1_804_000, "70 kg")]:
        ax.axvline(cycle, color="0.45", linewidth=0.8, linestyle=":")
        ax.text(cycle, ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("fatigue cycle")
    ax.set_ylabel("temperature-conditioned structural anomaly score")
    ax.set_title("P11-B sealed external confirmation: static guided waves")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(OUT / "fig01_score_chronology.png", dpi=180)
    plt.close(fig)

    (OUT / "P11B_COMPLETED.json").write_text(
        json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(), "primary_passed": summary["primary_passed"]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
