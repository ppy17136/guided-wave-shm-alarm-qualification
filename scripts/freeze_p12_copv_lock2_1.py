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
OUTPUT = ROOT / "research_protocols/P12_COPV_LOCK2_1_execution_freeze_v1.json"
SHA_FILE = ROOT / "research_protocols/P12_COPV_LOCK2_1_SHA256_v1.txt"
FROZEN_FILES = [
    "configs/p12_copv_v1/P12-COPV-01.json",
    "src/p12_copv_pipeline.py",
    "tools/run_p12_copv_confirmatory.py",
    "tools/selftest_p12_copv_pipeline.py",
    "tools/selftest_p12_copv_confirmatory_analysis.py",
    "tools/selftest_p12_copv_a2_frequency_mapping.py",
    "tools/audit_p12_copv_a2_frequency_metadata.py",
    "research_protocols/P12_COPV_confirmatory_preregistration_v1.md",
    "research_protocols/P12_COPV_preregistration_amendment_A1_schema_mapping.md",
    "research_protocols/P12_COPV_preregistration_amendment_A2_frequency_index_mapping.md",
    "research_protocols/P12_COPV_LOCK1_freeze_manifest_v1.json",
    "research_protocols/P12_COPV_LOCK1_SHA256_v1.txt",
    "research_protocols/P12_COPV_LOCK2_execution_freeze_v1.json",
    "research_protocols/P12_COPV_LOCK2_SHA256_v1.txt",
    "data/reports/p12_copv_download_verification_lock1.json",
    "data/reports/p12_copv_schema_audit_lock1/schema_audit_summary.json",
    "data/reports/p12_copv_schema_audit_lock1/h5_file_manifest.csv",
    "data/reports/p12_copv_pipeline_selftest_v1.json",
    "data/reports/p12_copv_confirmatory_analysis_selftest_v1.json",
    "data/reports/p12_copv_a2_frequency_mapping_selftest_v1.json",
    "data/reports/p12_copv_a2_frequency_metadata_audit_v1.json",
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    verification = json.loads((ROOT / "data/reports/p12_copv_download_verification_lock1.json").read_text(encoding="utf-8"))
    a2_selftest = json.loads((ROOT / "data/reports/p12_copv_a2_frequency_mapping_selftest_v1.json").read_text(encoding="utf-8"))
    a2_audit = json.loads((ROOT / "data/reports/p12_copv_a2_frequency_metadata_audit_v1.json").read_text(encoding="utf-8"))
    original_signal = json.loads((ROOT / "data/reports/p12_copv_pipeline_selftest_v1.json").read_text(encoding="utf-8"))
    original_analysis = json.loads((ROOT / "data/reports/p12_copv_confirmatory_analysis_selftest_v1.json").read_text(encoding="utf-8"))
    if not verification["overall_pass"] or any(x["status"] != "pass" for x in (a2_selftest, a2_audit, original_signal, original_analysis)):
        raise RuntimeError("LOCK2.1 prerequisites did not pass")
    files = []
    for relative in FROZEN_FILES:
        path = ROOT / relative
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    archives = [{"name": item["name"], "bytes": item["actual_bytes"], "md5": item["actual_md5"],
                 "sha256": item["sha256"], "verified": item["verified"]} for item in verification["files"]]
    payload = {
        "schema_version": "p12-copv-lock2-1-execution-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "LOCK2_1_FROZEN_AFTER_METADATA_ONLY_A2_BEFORE_ANY_RAW_WAVEFORM_VALUE_READ",
        "parent_lock2_sha256": (ROOT / "research_protocols/P12_COPV_LOCK2_SHA256_v1.txt").read_text().split()[0],
        "failed_jobs_before_lock2_1": {"formal": "20081.master", "diagnostic": "20082.master",
            "failure_point": "frequency index mapping before Data/Raw_Data value access"},
        "raw_numeric_values_read_before_freeze": False,
        "a2_scope": "exact 240 kHz fourth-group index label mapped to the declared 260 kHz fourth burst",
        "scientific_parameters_or_gates_changed": False,
        "input_archives": archives, "frozen_files": files,
        "tests": {"a2_mapping": a2_selftest, "three_state_metadata_audit": a2_audit,
                  "signal_processing": original_signal, "confirmatory_analysis": original_analysis},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "h5py": h5py.__version__, "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksum = sha256(OUTPUT)
    SHA_FILE.write_text(f"{checksum}  {OUTPUT.name}\n", encoding="ascii")
    print(json.dumps({"lock2_1": str(OUTPUT), "sha256": checksum, "files": len(files),
                      "state": payload["state"]}, indent=2))


if __name__ == "__main__":
    main()

