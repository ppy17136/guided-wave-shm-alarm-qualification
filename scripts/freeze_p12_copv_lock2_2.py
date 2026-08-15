from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research_protocols/P12_COPV_LOCK2_2_execution_freeze_v1.json"
SHA_FILE = ROOT / "research_protocols/P12_COPV_LOCK2_2_SHA256_v1.txt"
FROZEN_FILES = [
    "configs/p12_copv_v1/P12-COPV-01.json",
    "src/p12_copv_pipeline.py",
    "tools/run_p12_copv_confirmatory.py",
    "tools/selftest_p12_copv_pipeline.py",
    "tools/selftest_p12_copv_confirmatory_analysis.py",
    "tools/selftest_p12_copv_a2_frequency_mapping.py",
    "tools/audit_p12_copv_a2_frequency_metadata.py",
    "tools/selftest_p12_copv_a3_missing_support.py",
    "tools/selftest_p12_copv_a3_coverage_denominator.py",
    "tools/audit_p12_copv_a3_missing_support_from_official_json.py",
    "research_protocols/P12_COPV_confirmatory_preregistration_v1.md",
    "research_protocols/P12_COPV_preregistration_amendment_A1_schema_mapping.md",
    "research_protocols/P12_COPV_preregistration_amendment_A2_frequency_index_mapping.md",
    "research_protocols/P12_COPV_preregistration_amendment_A3_missing_support_metadata_execution.md",
    "research_protocols/P12_COPV_LOCK1_freeze_manifest_v1.json",
    "research_protocols/P12_COPV_LOCK1_SHA256_v1.txt",
    "research_protocols/P12_COPV_LOCK2_execution_freeze_v1.json",
    "research_protocols/P12_COPV_LOCK2_SHA256_v1.txt",
    "research_protocols/P12_COPV_LOCK2_1_execution_freeze_v1.json",
    "research_protocols/P12_COPV_LOCK2_1_SHA256_v1.txt",
    "data/reports/p12_copv_download_verification_lock1.json",
    "data/reports/p12_copv_schema_audit_lock1/schema_audit_summary.json",
    "data/reports/p12_copv_schema_audit_lock1/h5_file_manifest.csv",
    "data/reports/p12_copv_pipeline_selftest_v1.json",
    "data/reports/p12_copv_confirmatory_analysis_selftest_v1.json",
    "data/reports/p12_copv_a2_frequency_mapping_selftest_v1.json",
    "data/reports/p12_copv_a2_frequency_metadata_audit_v1.json",
    "data/reports/p12_copv_a3_missing_support_selftest_v1.json",
    "data/reports/p12_copv_a3_coverage_denominator_selftest_v1.json",
    "data/reports/p12_copv_a3_missing_support_official_json_audit_v1.json",
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    verification = json.loads((ROOT / "data/reports/p12_copv_download_verification_lock1.json").read_text(encoding="utf-8"))
    reports = {}
    for name in (
        "p12_copv_pipeline_selftest_v1.json",
        "p12_copv_confirmatory_analysis_selftest_v1.json",
        "p12_copv_a2_frequency_mapping_selftest_v1.json",
        "p12_copv_a2_frequency_metadata_audit_v1.json",
        "p12_copv_a3_missing_support_selftest_v1.json",
        "p12_copv_a3_coverage_denominator_selftest_v1.json",
        "p12_copv_a3_missing_support_official_json_audit_v1.json",
    ):
        reports[name] = json.loads((ROOT / "data/reports" / name).read_text(encoding="utf-8"))
    if not verification["overall_pass"] or any(report["status"] != "pass" for report in reports.values()):
        raise RuntimeError("LOCK2.2 prerequisites did not pass")
    files = []
    for relative in FROZEN_FILES:
        path = ROOT / relative
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    archives = [{"name": item["name"], "bytes": item["actual_bytes"], "md5": item["actual_md5"],
                 "sha256": item["sha256"], "verified": item["verified"]} for item in verification["files"]]
    payload = {
        "schema_version": "p12-copv-lock2-2-execution-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "LOCK2_2_FROZEN_AFTER_PREREGISTERED_INVALID_METADATA_IMPLEMENTATION",
        "parent_lock2_1_sha256": (ROOT / "research_protocols/P12_COPV_LOCK2_1_SHA256_v1.txt").read_text().split()[0],
        "failed_job_before_lock2_2": {"job": "20087.master", "feature_cache_records_completed": 141,
            "failure_point": "missing MetaData/Pressure before waveform read for failing target H5"},
        "a3_scope": "record missing/nonfinite Pressure or Temp_Surface as preregistered unsupported/invalid and continue",
        "coverage_denominator_retains_invalid_conditions": True,
        "existing_141_cache_records_reused_without_score_inspection_or_selection": True,
        "scientific_parameters_or_gates_changed": False,
        "input_archives": archives, "frozen_files": files, "tests": reports,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "h5py": h5py.__version__, "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksum = sha256(OUTPUT)
    SHA_FILE.write_text(f"{checksum}  {OUTPUT.name}\n", encoding="ascii")
    print(json.dumps({"lock2_2": str(OUTPUT), "sha256": checksum, "files": len(files),
                      "state": payload["state"]}, indent=2))


if __name__ == "__main__":
    main()

