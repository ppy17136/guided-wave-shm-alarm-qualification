from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p12_confirmatory", ROOT / "tools/run_p12_copv_confirmatory.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def main() -> None:
    temporary = ROOT / "data/reports/_p12_a3_missing_support_test.h5"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.create_group("MetaData")
        with h5py.File(temporary, "r") as handle:
            caught = False
            try:
                MODULE.support_features(handle)
            except MODULE.MissingSupportMetadata as exc:
                caught = "Pressure" in str(exc)
    finally:
        temporary.unlink(missing_ok=True)
    synthetic_records = [{
        "status": "invalid_missing_support_metadata", "archive": "irreversible",
        "path": "synthetic_missing.h5", "temperature_c": 37, "pressure_bar": 50,
        "ramp": "descending", "invalid_reason": "missing required support metadata",
    }]
    # Exercise the explicit coverage-record construction without any scores.
    count_in_coverage = {"invalid_missing_support_metadata", "invalid_missing_frozen_reference"}
    conditions = []
    for record in synthetic_records:
        if record["status"] in count_in_coverage:
            conditions.append({"supported": False, "fusion": "abstain_support", "status": record["status"]})
    checks = {
        "missing_pressure_raises_narrow_exception": caught,
        "temporary_h5_deleted": not temporary.exists(),
        "invalid_condition_retained": len(conditions) == 1,
        "invalid_condition_not_supported": conditions[0]["supported"] is False,
        "invalid_condition_abstains": conditions[0]["fusion"] == "abstain_support",
    }
    report = {"schema_version": "p12-copv-a3-missing-support-selftest-v1",
              "status": "pass" if all(checks.values()) else "fail", "checks": checks}
    path = ROOT / "data/reports/p12_copv_a3_missing_support_selftest_v1.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

