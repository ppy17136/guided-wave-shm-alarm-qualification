from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.p12_copv_pipeline import (  # noqa: E402
    FREQUENCIES_HZ,
    build_frequency_template,
    confirm_two_of_three,
    frequency_index_map,
    fuse_five_frequencies,
    score_frequency_repetitions,
    valid_path_mask,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(json_clean(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["frequencies_hz"]) != FREQUENCIES_HZ:
        raise ValueError("configuration frequencies differ from frozen implementation")
    return config


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        item = dict(row)
        item["nominal_temperature_c"] = int(item["nominal_temperature_c"])
        item["nominal_pressure_bar"] = int(item["nominal_pressure_bar"])
        item["sequence_validated"] = str(item["sequence_validated"]).lower() == "true"
        result.append(item)
    return result


def member_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["archive"]),
        int(row["nominal_temperature_c"]),
        int(row["nominal_pressure_bar"]),
        str(row["ramp_inferred"]),
    )


def cache_name(row: dict[str, Any]) -> str:
    digest = hashlib.sha256(str(row["path"]).encode("utf-8")).hexdigest()[:16]
    return f"{row['archive']}_T{row['nominal_temperature_c']}_P{row['nominal_pressure_bar']}_{row['ramp_inferred']}_{digest}.json"


def extract_member(archive: zipfile.ZipFile, member: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (hashlib.sha256(member.encode("utf-8")).hexdigest()[:16] + ".h5")
    if target.exists():
        target.unlink()
    with archive.open(member, "r") as source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
    if target.stat().st_size <= 0:
        raise RuntimeError(f"empty extracted H5 for {member}")
    return target


class MissingSupportMetadata(ValueError):
    pass


def scalar(handle: h5py.File, key: str) -> float:
    if key not in handle:
        raise MissingSupportMetadata(f"missing required support metadata: {key}")
    value = float(np.asarray(handle[key][...], dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError(f"non-finite metadata {key}")
    return value


def support_features(handle: h5py.File) -> list[float]:
    pressure = scalar(handle, "MetaData/Pressure")
    if "MetaData/Temp_Surface" not in handle:
        raise MissingSupportMetadata("missing required support metadata: MetaData/Temp_Surface")
    temperatures = np.asarray(handle["MetaData/Temp_Surface"][...], dtype=np.float64).reshape(-1)
    if temperatures.size != 4 or not np.isfinite(temperatures).all():
        raise MissingSupportMetadata("invalid surface-temperature metadata")
    return [pressure, float(np.median(temperatures))]


def structural_metadata(handle: h5py.File) -> dict[str, Any]:
    raw = handle["Data/Raw_Data"]
    channels = np.asarray(handle["MetaData/Channels"][...], dtype=np.float64)
    mapping = np.asarray(handle["MetaData/Index_FrequencyvsRepetition"][...], dtype=np.float64)
    excitation_frequencies = np.asarray(
        handle["MetaData/Signal_Frequency_Burst"][...], dtype=np.float64
    ).reshape(-1)
    if excitation_frequencies.shape != (5,) or not np.allclose(
        excitation_frequencies, np.asarray(FREQUENCIES_HZ, dtype=np.float64), rtol=0.0, atol=0.0
    ):
        raise ValueError(
            f"burst excitation frequencies differ from frozen frequencies: {excitation_frequencies.tolist()}"
        )
    fs = scalar(handle, "MetaData/Sampling_Frequency")
    if raw.shape != (18, 600, 7552) or raw.dtype != np.dtype("float64"):
        raise ValueError(f"unexpected raw dataset schema: {raw.shape}, {raw.dtype}")
    mask = valid_path_mask(channels)
    indices = frequency_index_map(mapping)
    return {
        "raw": raw,
        "path_mask": mask,
        "frequency_indices": indices,
        "sampling_frequency": fs,
        "support_features": support_features(handle),
        "channels_sha256": hashlib.sha256(np.ascontiguousarray(channels).tobytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256(np.ascontiguousarray(mapping).tobytes()).hexdigest(),
    }


def build_templates(path: Path) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        metadata = structural_metadata(handle)
        templates = {
            frequency: build_frequency_template(
                metadata["raw"], metadata["frequency_indices"][frequency],
                metadata["path_mask"], metadata["sampling_frequency"], frequency,
            )
            for frequency in FREQUENCIES_HZ
        }
        audit = {key: value for key, value in metadata.items() if key not in {"raw", "path_mask", "frequency_indices"}}
        audit["valid_paths"] = int(metadata["path_mask"].sum())
        audit["frequency_indices"] = {str(k): v for k, v in metadata["frequency_indices"].items()}
    return templates, audit


def score_target(path: Path, templates: dict[int, np.ndarray], device: str) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        metadata = structural_metadata(handle)
        frequencies = {}
        for frequency in FREQUENCIES_HZ:
            scored = score_frequency_repetitions(
                metadata["raw"], metadata["frequency_indices"][frequency],
                metadata["path_mask"], metadata["sampling_frequency"], frequency,
                templates[frequency], device=device,
            )
            frequencies[str(frequency)] = [item.__dict__ for item in scored]
        return {
            "schema_version": "p12-copv-h5-features-v1",
            "support_features": metadata["support_features"],
            "sampling_frequency": metadata["sampling_frequency"],
            "channels_sha256": metadata["channels_sha256"],
            "mapping_sha256": metadata["mapping_sha256"],
            "valid_paths": int(metadata["path_mask"].sum()),
            "frequencies": frequencies,
        }


def robust_scale_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    fallback = (q75 - q25) / 1.349
    scale = np.where(mad > 0, 1.4826 * mad, fallback)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("support robust scale is non-positive")
    return center, scale


def kth_distances(reference: np.ndarray, target: np.ndarray, k: int, self_exclude: bool) -> np.ndarray:
    squared = np.sum((target[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    if self_exclude:
        if len(target) != len(reference) or not np.allclose(target, reference):
            raise ValueError("self exclusion requires identical reference and target")
        np.fill_diagonal(squared, np.inf)
    return np.sqrt(np.partition(squared, k - 1, axis=1)[:, k - 1])


def fit_support(reference_features: np.ndarray, reference_blocks: list[str], quantile: float) -> dict[str, Any]:
    if len(reference_features) != len(reference_blocks):
        raise ValueError("support feature/block length mismatch")
    center, scale = robust_scale_fit(reference_features)
    standardized = (reference_features - center) / scale
    k = int(np.clip(math.ceil(math.sqrt(len(standardized))), 5, 30))
    squared = np.sum((standardized[:, None, :] - standardized[None, :, :]) ** 2, axis=2)
    block_array = np.asarray(reference_blocks, dtype=object)
    squared[block_array[:, None] == block_array[None, :]] = np.inf
    available = np.sum(np.isfinite(squared), axis=1)
    if np.any(available < k):
        raise ValueError("too few other-block neighbors for frozen support k")
    loo = np.sqrt(np.partition(squared, k - 1, axis=1)[:, k - 1])
    threshold = float(np.quantile(loo, quantile))
    return {
        "center": center.tolist(), "scale": scale.tolist(), "k": k,
        "quantile": quantile, "threshold": threshold,
        "loo_distances": loo.tolist(), "reference_blocks": reference_blocks,
    }


def apply_support(model: dict[str, Any], features: Iterable[float]) -> tuple[bool, float]:
    center = np.asarray(model["center"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    reference = np.asarray(model["reference_standardized"], dtype=np.float64)
    target = (np.asarray(list(features), dtype=np.float64) - center) / scale
    distance = float(kth_distances(reference, target[None, :], int(model["k"]), False)[0])
    return distance <= float(model["threshold"]), distance


def hampel_threshold(values: np.ndarray, multiplier: float) -> tuple[float, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + multiplier * 1.4826 * mad, median, mad


def fit_frequency_threshold(rows: list[dict[str, Any]], multiplier: float, max_relative_range: float) -> dict[str, Any]:
    values = np.asarray([row["score"] for row in rows], dtype=np.float64)
    blocks = sorted(set(row["block_id"] for row in rows))
    threshold, median, mad = hampel_threshold(values, multiplier)
    leave_one_out = []
    for block in blocks:
        kept = np.asarray([row["score"] for row in rows if row["block_id"] != block], dtype=np.float64)
        loo, _, loo_mad = hampel_threshold(kept, multiplier)
        leave_one_out.append({"deleted_block": block, "threshold": loo, "mad": loo_mad})
    loo_values = np.asarray([item["threshold"] for item in leave_one_out], dtype=np.float64)
    relative_range = float(np.ptp(loo_values) / max(abs(threshold), np.finfo(float).eps))
    checks = {
        "samples_at_least_100": len(values) >= 100,
        "blocks_at_least_10": len(blocks) >= 10,
        "thresholds_finite": bool(np.isfinite(threshold) and np.isfinite(loo_values).all()),
        "mad_positive": mad > 0 and all(item["mad"] > 0 for item in leave_one_out),
        "loo_relative_range_le_0_25": relative_range <= max_relative_range,
    }
    return {
        "threshold": threshold, "median": median, "mad": mad,
        "samples": len(values), "blocks": len(blocks),
        "loo_threshold_relative_range": relative_range,
        "leave_one_block_out": leave_one_out, "checks": checks,
        "pass": all(checks.values()),
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum()); negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    sorted_labels = labels[order]
    positives = int(sorted_labels.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def bootstrap_condition_metrics(
    healthy: list[dict[str, Any]], damaged: list[dict[str, Any]],
    repetitions: int, seed: int,
) -> dict[str, Any]:
    if not healthy or not damaged:
        return {"repetitions": repetitions, "status": "insufficient_class"}
    rng = np.random.default_rng(seed)
    fpr_values = np.empty(repetitions, dtype=np.float64)
    recall_values = np.empty(repetitions, dtype=np.float64)
    macro_auc_values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled_h = [healthy[i] for i in rng.integers(0, len(healthy), len(healthy))]
        sampled_d = [damaged[i] for i in rng.integers(0, len(damaged), len(damaged))]
        fpr_values[index] = np.mean([item["fusion"] == "alarm" for item in sampled_h])
        recall_values[index] = np.mean([item["fusion"] == "alarm" for item in sampled_d])
        labels = np.asarray([0] * len(sampled_h) + [1] * len(sampled_d), dtype=np.int64)
        aucs = []
        for frequency in FREQUENCIES_HZ:
            scores = np.asarray([
                item["frequencies"][str(frequency)]["condition_score"]
                for item in sampled_h + sampled_d
            ], dtype=np.float64)
            aucs.append(roc_auc(labels, scores))
        macro_auc_values[index] = np.mean(aucs)
    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {"lower_2_5pct": float(low), "upper_97_5pct": float(high)}
    return {
        "repetitions": repetitions, "seed": seed, "unit": "H5_condition_block",
        "healthy_fpr_95pct": interval(fpr_values),
        "irreversible_recall_95pct": interval(recall_values),
        "macro_roc_auc_95pct": interval(macro_auc_values),
    }


def flatten_features(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        if record.get("status") != "complete":
            continue
        for frequency in FREQUENCIES_HZ:
            for item in record["features"]["frequencies"][str(frequency)]:
                result.append({
                    "archive": record["archive"], "path": record["path"],
                    "temperature_c": record["temperature_c"], "pressure_bar": record["pressure_bar"],
                    "ramp": record["ramp"], "frequency_hz": frequency,
                    "repetition_index": item["repetition_index"], "score": item["score"],
                    "energy_score": item["energy_score"], "block_id": record["path"],
                    "support_features": record["features"]["support_features"],
                })
    return result


def count_alarm_runs(conditions: list[dict[str, Any]]) -> int:
    runs = 0
    by_temperature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in conditions:
        by_temperature[int(item["temperature_c"])].append(item)
    for items in by_temperature.values():
        ordered = sorted(items, key=lambda item: item["path"])
        previous_alarm = False
        for item in ordered:
            alarm = item["fusion"] == "alarm"
            if alarm and not previous_alarm:
                runs += 1
            previous_alarm = alarm
    return runs


def analyse(
    records: list[dict[str, Any]], config: dict[str, Any],
    reference_features: np.ndarray, reference_blocks: list[str],
) -> dict[str, Any]:
    long_rows = flatten_features(records)
    calibration_pressures = set(config["calibration_pressures_bar"])
    heldout_pressures = set(config["heldout_healthy_pressures_bar"])
    thresholds = {}
    for frequency in FREQUENCIES_HZ:
        selected = [row for row in long_rows if row["archive"] == "baseline" and row["ramp"] == "random"
                    and row["pressure_bar"] in calibration_pressures and row["frequency_hz"] == frequency]
        thresholds[str(frequency)] = fit_frequency_threshold(
            selected, float(config["hampel_multiplier"]), float(config["loo_relative_range_max"])
        )

    support = fit_support(reference_features, reference_blocks, float(config["support_quantile"]))
    center = np.asarray(support["center"]); scale = np.asarray(support["scale"])
    support["reference_standardized"] = ((reference_features - center) / scale).tolist()

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        by_condition[row["path"]].append(row)
    conditions = []
    for path, rows in sorted(by_condition.items()):
        first = rows[0]
        supported, distance = apply_support(support, first["support_features"])
        frequency_results = {}
        for frequency in FREQUENCIES_HZ:
            frequency_rows = sorted((row for row in rows if row["frequency_hz"] == frequency), key=lambda x: x["repetition_index"])
            threshold = thresholds[str(frequency)]["threshold"]
            alarms = [row["score"] > threshold for row in frequency_rows]
            frequency_results[str(frequency)] = {
                "scores": [row["score"] for row in frequency_rows],
                "energy_scores": [row["energy_score"] for row in frequency_rows],
                "threshold": threshold, "repeat_alarms": alarms,
                "confirmed": confirm_two_of_three(alarms) if supported and len(alarms) == 3 else None,
                "condition_score": float(np.median([row["score"] for row in frequency_rows])),
                "condition_energy_score": float(np.median([row["energy_score"] for row in frequency_rows])),
            }
        fusion = fuse_five_frequencies([frequency_results[str(f)]["confirmed"] for f in FREQUENCIES_HZ])
        conditions.append({
            "archive": first["archive"], "path": path, "temperature_c": first["temperature_c"],
            "pressure_bar": first["pressure_bar"], "ramp": first["ramp"],
            "support_distance": distance, "supported": supported, "fusion": fusion,
            "frequencies": frequency_results, "status": "complete",
        })

    count_in_coverage = {"invalid_missing_support_metadata", "invalid_missing_frozen_reference"}
    for record in records:
        if record.get("status") in count_in_coverage:
            conditions.append({
                "archive": record["archive"], "path": record["path"],
                "temperature_c": record["temperature_c"], "pressure_bar": record["pressure_bar"],
                "ramp": record["ramp"], "support_distance": None, "supported": False,
                "fusion": "abstain_support", "frequencies": {}, "status": record["status"],
                "invalid_reason": record.get("invalid_reason") or record.get("reference_exclusion"),
            })

    healthy = [item for item in conditions if item["archive"] == "baseline" and item["ramp"] == "random"
               and item["pressure_bar"] in heldout_pressures]
    damaged = [item for item in conditions if item["archive"] == "irreversible"]
    supported_healthy = [item for item in healthy if item["supported"]]
    supported_damaged = [item for item in damaged if item["supported"]]
    fpr = sum(item["fusion"] == "alarm" for item in supported_healthy) / max(1, len(supported_healthy))
    recall = sum(item["fusion"] == "alarm" for item in supported_damaged) / max(1, len(supported_damaged))
    false_runs = count_alarm_runs(supported_healthy)

    frequency_metrics = {}
    aucs = []
    energy_aucs = []
    for frequency in FREQUENCIES_HZ:
        usable = supported_healthy + supported_damaged
        labels = np.asarray([0] * len(supported_healthy) + [1] * len(supported_damaged))
        scores = np.asarray([item["frequencies"][str(frequency)]["condition_score"] for item in usable])
        energy = np.asarray([item["frequencies"][str(frequency)]["condition_energy_score"] for item in usable])
        auc = roc_auc(labels, scores); e_auc = roc_auc(labels, energy)
        aucs.append(auc); energy_aucs.append(e_auc)
        frequency_metrics[str(frequency)] = {
            "roc_auc": auc, "average_precision": average_precision(labels, scores),
            "energy_roc_auc": e_auc, "n_healthy": len(supported_healthy), "n_damage": len(supported_damaged),
        }

    temp_recall = {}
    for temperature in config["temperatures_c"]:
        subset = [item for item in supported_damaged if item["temperature_c"] == temperature]
        temp_recall[str(temperature)] = sum(item["fusion"] == "alarm" for item in subset) / max(1, len(subset))
    pressure_bins = {"50_250": (50, 250), "300_500": (300, 500), "550_700": (550, 700)}
    bin_recall = {}
    for name, (low, high) in pressure_bins.items():
        subset = [item for item in supported_damaged if low <= item["pressure_bar"] <= high]
        bin_recall[name] = sum(item["fusion"] == "alarm" for item in subset) / max(1, len(subset))

    reference_coverage = len(supported_healthy) / max(1, len(healthy))
    damage_coverage = len(supported_damaged) / max(1, len(damaged))
    temp_coverage = {}
    for temperature in config["temperatures_c"]:
        target = [item for item in damaged if item["temperature_c"] == temperature]
        temp_coverage[str(temperature)] = sum(item["supported"] for item in target) / max(1, len(target))

    macro_auc = float(np.nanmean(aucs)); macro_energy_auc = float(np.nanmean(energy_aucs))
    bootstrap = bootstrap_condition_metrics(
        supported_healthy, supported_damaged,
        int(config["bootstrap_repetitions"]), int(config["bootstrap_seed"]),
    )
    gates = {
        "five_frequency_calibration_reliable": all(item["pass"] for item in thresholds.values()),
        "healthy_support_coverage_ge_0_90": reference_coverage >= 0.90,
        "damage_support_coverage_ge_0_80": damage_coverage >= 0.80,
        "each_temperature_support_coverage_ge_0_60": min(temp_coverage.values()) >= 0.60,
        "supported_healthy_fpr_le_0_05": fpr <= 0.05,
        "healthy_false_alarm_blocks_le_2": false_runs <= 2,
        "supported_damage_recall_ge_0_80": recall >= 0.80,
        "worst_temperature_recall_ge_0_60": min(temp_recall.values()) >= 0.60,
        "worst_pressure_bin_recall_ge_0_60": min(bin_recall.values()) >= 0.60,
        "macro_auc_ge_0_80": macro_auc >= 0.80,
        "worst_frequency_auc_ge_0_65": min(aucs) >= 0.65,
        "macro_auc_advantage_over_energy_ge_0_10": macro_auc - macro_energy_auc >= 0.10,
        "no_unexplained_acquisition_asymmetry": True,
    }
    return {
        "schema_version": "p12-copv-confirmatory-summary-v1",
        "generated_utc": utc_now(), "thresholds": thresholds,
        "support_model": support,
        "coverage": {"healthy": reference_coverage, "irreversible": damage_coverage, "by_temperature": temp_coverage},
        "alarm_metrics": {"healthy_fpr": fpr, "healthy_false_alarm_blocks": false_runs,
                          "irreversible_recall": recall, "recall_by_temperature": temp_recall,
                          "recall_by_pressure_bin": bin_recall},
        "discrimination": {"by_frequency": frequency_metrics, "macro_roc_auc": macro_auc,
                           "worst_frequency_roc_auc": float(np.nanmin(aucs)),
                           "energy_macro_roc_auc": macro_energy_auc,
                           "macro_advantage_over_energy": macro_auc - macro_energy_auc},
        "bootstrap": bootstrap,
        "gates": gates, "primary_status": "PASS" if all(gates.values()) else "FAIL",
        "condition_count": len(conditions), "conditions": conditions,
    }


def execute(config: dict[str, Any], device: str, max_conditions: int | None) -> dict[str, Any]:
    output = ROOT / config["output_dir"]
    temp_dir = ROOT / config["temporary_dir"]
    feature_dir = output / "features"
    output.mkdir(parents=True, exist_ok=True); feature_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(ROOT / config["manifest"])
    if not all(row["sequence_validated"] for row in manifest):
        raise RuntimeError("manifest contains unvalidated pressure sequences")
    index: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in manifest:
        key = member_key(row)
        if key in index:
            raise RuntimeError(f"duplicate manifest key {key}")
        index[key] = row
    excluded = set(config["official_excluded_baseline_files"])
    archives = {role: zipfile.ZipFile(ROOT / relative, "r") for role, relative in config["archives"].items()}
    reference_features = []
    reference_blocks: list[str] = []
    records = []
    processed_conditions = 0
    try:
        for temperature in config["temperatures_c"]:
            for pressure in config["main_pressures_bar"]:
                processed_conditions += 1
                if max_conditions is not None and processed_conditions > max_conditions:
                    break
                targets = []
                for role in ("baseline", "irreversible", "reversible"):
                    ramps = ("random",) if role == "baseline" else ("descending", "random")
                    for ramp in ramps:
                        targets.append(index[(role, temperature, pressure, ramp)])
                cached = []
                all_cached = True
                for target in targets:
                    path = feature_dir / cache_name(target)
                    if path.is_file():
                        cached.append(json.loads(path.read_text(encoding="utf-8")))
                    else:
                        all_cached = False
                reference = index[("baseline", temperature, pressure, "descending")]
                reference_excluded = Path(reference["path"]).name in excluded
                if all_cached:
                    records.extend(cached)
                    features = next(
                        (item.get("reference_support_features") for item in cached
                         if item.get("reference_support_features") is not None),
                        None,
                    )
                    if features is not None and not reference_excluded:
                        reference_features.extend([features] * 3)
                        reference_blocks.extend([reference["path"]] * 3)
                    continue
                if reference_excluded:
                    for target in targets:
                        record = {
                            "schema_version": "p12-copv-h5-record-v1", "status": "invalid_missing_frozen_reference",
                            "archive": target["archive"], "path": target["path"], "temperature_c": temperature,
                            "pressure_bar": pressure, "ramp": target["ramp_inferred"],
                            "reference_path": reference["path"], "reference_exclusion": "official_baseline_anomaly",
                        }
                        atomic_json(feature_dir / cache_name(target), record); records.append(record)
                    continue
                extracted_reference = extract_member(archives["baseline"], reference["path"], temp_dir)
                try:
                    templates, reference_audit = build_templates(extracted_reference)
                finally:
                    extracted_reference.unlink(missing_ok=True)
                reference_features.extend([reference_audit["support_features"]] * 3)
                reference_blocks.extend([reference["path"]] * 3)
                for target in targets:
                    cache_path = feature_dir / cache_name(target)
                    if cache_path.is_file():
                        record = json.loads(cache_path.read_text(encoding="utf-8")); records.append(record); continue
                    if target["archive"] == "baseline" and Path(target["path"]).name in excluded:
                        record = {
                            "schema_version": "p12-copv-h5-record-v1", "status": "excluded_official_baseline_anomaly",
                            "archive": target["archive"], "path": target["path"], "temperature_c": temperature,
                            "pressure_bar": pressure, "ramp": target["ramp_inferred"],
                        }
                    else:
                        extracted = extract_member(archives[target["archive"]], target["path"], temp_dir)
                        try:
                            try:
                                features = score_target(extracted, templates, device)
                            except MissingSupportMetadata as exc:
                                features = None
                                invalid_reason = str(exc)
                        finally:
                            extracted.unlink(missing_ok=True)
                        if features is None:
                            record = {
                                "schema_version": "p12-copv-h5-record-v1",
                                "status": "invalid_missing_support_metadata",
                                "archive": target["archive"], "path": target["path"],
                                "temperature_c": temperature, "pressure_bar": pressure,
                                "ramp": target["ramp_inferred"], "reference_path": reference["path"],
                                "invalid_reason": invalid_reason,
                                "policy": "preregistered_unsupported_invalid_no_imputation",
                                "completed_utc": utc_now(),
                            }
                            atomic_json(cache_path, record); records.append(record)
                            continue
                        if features["channels_sha256"] != reference_audit["channels_sha256"] or features["mapping_sha256"] != reference_audit["mapping_sha256"]:
                            raise RuntimeError(f"acquisition mapping differs for {target['path']}")
                        if not math.isclose(float(features["sampling_frequency"]), float(reference_audit["sampling_frequency"]), rel_tol=0.0, abs_tol=0.0):
                            raise RuntimeError(f"sampling frequency differs for {target['path']}")
                        record = {
                            "schema_version": "p12-copv-h5-record-v1", "status": "complete",
                            "archive": target["archive"], "path": target["path"], "temperature_c": temperature,
                            "pressure_bar": pressure, "ramp": target["ramp_inferred"],
                            "reference_path": reference["path"],
                            "reference_support_features": reference_audit["support_features"],
                            "features": features, "completed_utc": utc_now(),
                        }
                    atomic_json(cache_path, record); records.append(record)
            if max_conditions is not None and processed_conditions >= max_conditions:
                break
    finally:
        for archive in archives.values():
            archive.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    if max_conditions is not None:
        smoke = {"schema_version": "p12-copv-smoke-v1", "status": "complete",
                 "conditions_processed": min(processed_conditions, max_conditions), "records": len(records),
                 "device": device, "completed_utc": utc_now()}
        atomic_json(output / "SMOKE_COMPLETED.json", smoke)
        return smoke
    summary = analyse(
        records, config, np.asarray(reference_features, dtype=np.float64), reference_blocks
    )
    atomic_json(output / "p12_copv_confirmatory_summary.json", summary)
    atomic_json(output / "COMPLETED.json", {"status": "complete", "primary_status": summary["primary_status"],
                                             "completed_utc": utc_now()})
    return {key: summary[key] for key in ("schema_version", "primary_status", "coverage", "alarm_metrics", "discrimination", "gates")}


def dry_run(config: dict[str, Any]) -> dict[str, Any]:
    manifest = read_manifest(ROOT / config["manifest"])
    counts = defaultdict(int)
    for row in manifest:
        counts[row["archive"]] += 1
    checks = {
        "three_archives_present": all((ROOT / path).is_file() for path in config["archives"].values()),
        "manifest_has_540_h5": len(manifest) == 540,
        "each_archive_has_180_h5": all(counts[role] == 180 for role in config["archives"]),
        "cuda_visible_on_current_host_informational": torch.cuda.is_available(),
        "five_frequencies_frozen": tuple(config["frequencies_hz"]) == FREQUENCIES_HZ,
        "six_temperatures": len(config["temperatures_c"]) == 6,
        "fourteen_main_pressures": len(config["main_pressures_bar"]) == 14,
        "official_four_exclusions": len(config["official_excluded_baseline_files"]) == 4,
    }
    required = [value for key, value in checks.items() if not key.endswith("_informational")]
    return {"schema_version": "p12-copv-preflight-v1", "passed": all(required), "checks": checks,
            "torch": torch.__version__, "numpy": np.__version__, "h5py": h5py.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p12_copv_v1/P12-COPV-01.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-conditions", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    if args.dry_run:
        result = dry_run(config)
    elif args.execute or args.smoke_conditions:
        result = execute(config, args.device, args.smoke_conditions)
    else:
        raise SystemExit("select --dry-run, --execute, or --smoke-conditions N")
    print(json.dumps(json_clean(result), ensure_ascii=False, indent=2, allow_nan=False))
    if args.dry_run and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
