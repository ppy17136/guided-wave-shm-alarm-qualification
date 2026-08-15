from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/reports/p12_copv_schema_audit_lock1/official_json_records.json"
OUTPUT = ROOT / "data/reports/p12_copv_a3_missing_support_official_json_audit_v1.json"


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))["records"]
    required = {"/MetaData/Pressure", "/MetaData/Temp_Surface"}
    missing = []
    for record in records:
        for item in record["content"]:
            paths = {entry["path"] for entry in item.get("datasets", []) if isinstance(entry, dict) and "path" in entry}
            absent = sorted(required - paths)
            if absent:
                missing.append({"archive": record["archive"], "json": record["path"],
                                "file": item.get("file"), "missing": absent})
    t37 = [item for item in missing if item["archive"] == "irreversible" and "T37_" in item["json"]]
    t55 = [item for item in missing if item["archive"] == "irreversible" and "T55_" in item["json"]]
    checks = {"missing_total_32": len(missing) == 32, "irreversible_t37_30": len(t37) == 30,
              "irreversible_t55_2": len(t55) == 2,
              "both_required_fields_missing": all(set(item["missing"]) == required for item in missing),
              "no_other_state_missing": all(item["archive"] == "irreversible" for item in missing),
              "waveform_values_read": False}
    report = {"schema_version": "p12-copv-a3-official-json-missing-support-audit-v1",
              "status": "pass" if all(value is True for key, value in checks.items() if key != "waveform_values_read")
                        and checks["waveform_values_read"] is False else "fail",
              "checks": checks, "missing_files": missing}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
