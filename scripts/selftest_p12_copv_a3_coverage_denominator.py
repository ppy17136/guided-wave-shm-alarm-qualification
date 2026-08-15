from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


RUNNER = load("p12_runner_a3", ROOT / "tools/run_p12_copv_confirmatory.py")
SYNTHETIC = load("p12_synthetic_a3", ROOT / "tools/selftest_p12_copv_confirmatory_analysis.py")


def main() -> None:
    config = json.loads((ROOT / "configs/p12_copv_v1/P12-COPV-01.json").read_text(encoding="utf-8"))
    records = []
    reference_features = []
    reference_blocks = []
    sequence = 0
    invalid_expected = 0
    for temperature in config["temperatures_c"]:
        for pressure in config["main_pressures_bar"]:
            block = f"reference_T{temperature}_P{pressure}"
            reference_features.extend([[pressure, temperature]] * 3)
            reference_blocks.extend([block] * 3)
            if pressure in config["calibration_pressures_bar"]:
                sequence += 1
                records.append(SYNTHETIC.make_record(
                    "baseline", temperature, pressure, "random", sequence,
                    0.050 + 0.003 * np.sin(sequence * 0.83),
                ))
            if pressure in config["heldout_healthy_pressures_bar"]:
                sequence += 1
                records.append(SYNTHETIC.make_record(
                    "baseline", temperature, pressure, "random", sequence, 0.055
                ))
            for ramp, score in (("descending", 0.50), ("random", 0.52)):
                sequence += 1
                if temperature == 37:
                    invalid_expected += 1
                    records.append({
                        "status": "invalid_missing_support_metadata",
                        "archive": "irreversible",
                        "path": f"T37/{sequence:02d}_irreversible_T37_{pressure}bar.h5",
                        "temperature_c": 37, "pressure_bar": pressure, "ramp": ramp,
                        "invalid_reason": "missing required support metadata",
                    })
                else:
                    records.append(SYNTHETIC.make_record(
                        "irreversible", temperature, pressure, ramp, sequence, score
                    ))
    summary = RUNNER.analyse(
        records, config, np.asarray(reference_features, dtype=np.float64), reference_blocks
    )
    expected_coverage = (168 - invalid_expected) / 168
    checks = {
        "twenty_eight_main_t37_h5_invalid": invalid_expected == 28,
        "invalids_remain_in_coverage_denominator": abs(summary["coverage"]["irreversible"] - expected_coverage) < 1e-12,
        "t37_temperature_coverage_zero": summary["coverage"]["by_temperature"]["37"] == 0.0,
        "support_coverage_gate_fails": not summary["gates"]["each_temperature_support_coverage_ge_0_60"],
        "primary_status_fails": summary["primary_status"] == "FAIL",
        "supported_damage_scores_still_analyzable": summary["discrimination"]["macro_roc_auc"] == 1.0,
    }
    report = {"schema_version": "p12-copv-a3-coverage-denominator-selftest-v1",
              "status": "pass" if all(checks.values()) else "fail", "checks": checks,
              "expected_damage_coverage": expected_coverage,
              "reported_coverage": summary["coverage"], "primary_status": summary["primary_status"]}
    path = ROOT / "data/reports/p12_copv_a3_coverage_denominator_selftest_v1.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
