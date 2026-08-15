from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "run_p12_copv_confirmatory.py"
SPEC = importlib.util.spec_from_file_location("p12_confirmatory", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def feature_payload(base_score: float, energy_base: float) -> dict:
    frequencies = {}
    for f_index, frequency in enumerate(MODULE.FREQUENCIES_HZ):
        frequencies[str(frequency)] = []
        for repetition in range(3):
            frequencies[str(frequency)].append({
                "repetition_index": repetition,
                "score": base_score + 0.0002 * f_index + 0.0001 * (repetition - 1),
                "energy_score": energy_base + 0.0001 * ((f_index + repetition) % 3),
                "valid_paths": 552,
                "top_paths": 28,
                "median_best_lag": 0.0,
            })
    return {
        "support_features": [],
        "sampling_frequency": 2_000_000.0,
        "channels_sha256": "synthetic",
        "mapping_sha256": "synthetic",
        "valid_paths": 552,
        "frequencies": frequencies,
    }


def make_record(role: str, temperature: int, pressure: int, ramp: str, sequence: int, score: float) -> dict:
    path = f"T{temperature}/{sequence:02d}_{role}_T{temperature}_{pressure}bar.h5"
    energy = 0.05 + 0.002 * ((temperature + pressure // 50 + sequence) % 5)
    payload = feature_payload(score, energy)
    payload["support_features"] = [float(pressure), float(temperature)]
    return {
        "status": "complete", "archive": role, "path": path,
        "temperature_c": temperature, "pressure_bar": pressure, "ramp": ramp,
        "features": payload,
    }


def main() -> None:
    config = json.loads((ROOT / "configs" / "p12_copv_v1" / "P12-COPV-01.json").read_text(encoding="utf-8"))
    records = []
    reference_features = []
    reference_blocks = []
    sequence = 0
    for temperature in config["temperatures_c"]:
        for pressure in config["main_pressures_bar"]:
            block = f"reference_T{temperature}_P{pressure}"
            reference_features.extend([[pressure, temperature]] * 3)
            reference_blocks.extend([block] * 3)
            if pressure in config["calibration_pressures_bar"]:
                sequence += 1
                calibration_score = 0.050 + 0.003 * np.sin(sequence * 0.83)
                records.append(make_record("baseline", temperature, pressure, "random", sequence, calibration_score))
            if pressure in config["heldout_healthy_pressures_bar"]:
                sequence += 1
                records.append(make_record("baseline", temperature, pressure, "random", sequence, 0.055))
            sequence += 1
            records.append(make_record("irreversible", temperature, pressure, "descending", sequence, 0.50))
            sequence += 1
            records.append(make_record("irreversible", temperature, pressure, "random", sequence, 0.52))

    summary = MODULE.analyse(
        records, config, np.asarray(reference_features, dtype=np.float64), reference_blocks
    )
    checks = {
        "synthetic_primary_pass": summary["primary_status"] == "PASS",
        "all_calibration_gates_pass": summary["gates"]["five_frequency_calibration_reliable"],
        "healthy_fpr_zero": summary["alarm_metrics"]["healthy_fpr"] == 0.0,
        "damage_recall_one": summary["alarm_metrics"]["irreversible_recall"] == 1.0,
        "all_support_covered": summary["coverage"]["healthy"] == 1.0 and summary["coverage"]["irreversible"] == 1.0,
        "five_frequency_metrics": len(summary["discrimination"]["by_frequency"]) == 5,
        "structural_auc_exceeds_energy": summary["discrimination"]["macro_advantage_over_energy"] >= 0.10,
        "same_block_repeats_excluded": len(set(summary["support_model"]["reference_blocks"])) == 84,
    }
    report = {
        "schema_version": "p12-copv-confirmatory-analysis-selftest-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "primary_status": summary["primary_status"],
        "alarm_metrics": summary["alarm_metrics"],
        "discrimination": summary["discrimination"],
    }
    target = ROOT / "data" / "reports" / "p12_copv_confirmatory_analysis_selftest_v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
