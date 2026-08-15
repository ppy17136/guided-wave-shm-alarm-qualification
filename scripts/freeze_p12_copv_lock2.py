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
OUTPUT = ROOT / "research_protocols" / "P12_COPV_LOCK2_execution_freeze_v1.json"
SHA_FILE = ROOT / "research_protocols" / "P12_COPV_LOCK2_SHA256_v1.txt"

FROZEN_FILES = [
    "configs/p12_copv_v1/P12-COPV-01.json",
    "src/p12_copv_pipeline.py",
    "tools/run_p12_copv_confirmatory.py",
    "tools/selftest_p12_copv_pipeline.py",
    "tools/selftest_p12_copv_confirmatory_analysis.py",
    "research_protocols/P12_COPV_confirmatory_preregistration_v1.md",
    "research_protocols/P12_COPV_preregistration_amendment_A1_schema_mapping.md",
    "research_protocols/P12_COPV_LOCK1_freeze_manifest_v1.json",
    "research_protocols/P12_COPV_LOCK1_SHA256_v1.txt",
    "data/reports/p12_copv_download_verification_lock1.json",
    "data/reports/p12_copv_schema_audit_lock1/schema_audit_summary.json",
    "data/reports/p12_copv_schema_audit_lock1/h5_file_manifest.csv",
    "data/reports/p12_copv_pipeline_selftest_v1.json",
    "data/reports/p12_copv_confirmatory_analysis_selftest_v1.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    verification = json.loads((ROOT / "data/reports/p12_copv_download_verification_lock1.json").read_text(encoding="utf-8"))
    signal_selftest = json.loads((ROOT / "data/reports/p12_copv_pipeline_selftest_v1.json").read_text(encoding="utf-8"))
    analysis_selftest = json.loads((ROOT / "data/reports/p12_copv_confirmatory_analysis_selftest_v1.json").read_text(encoding="utf-8"))
    if not verification.get("overall_pass"):
        raise RuntimeError("sealed archive verification did not pass")
    if signal_selftest.get("status") != "pass" or analysis_selftest.get("status") != "pass":
        raise RuntimeError("synthetic selftests did not pass")
    files = []
    for relative in FROZEN_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    archives = [{
        "name": item["name"], "bytes": item["actual_bytes"], "md5": item["actual_md5"],
        "sha256": item["sha256"], "verified": item["verified"],
    } for item in verification["files"]]
    payload = {
        "schema_version": "p12-copv-lock2-execution-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "LOCK2_FROZEN_BEFORE_ANY_RAW_WAVEFORM_VALUE_READ",
        "raw_numeric_values_read_before_freeze": False,
        "permission_after_freeze": "one-time smoke followed by unchanged formal execution",
        "scientific_rules": {
            "lock1_unchanged": True,
            "a1_schema_only_amendment_applied": True,
            "result_based_method_or_threshold_changes_prohibited": True,
            "new_methods_after_unseal_are_p13_exploratory_only": True,
        },
        "input_archives": archives,
        "frozen_files": files,
        "selftests": {
            "signal_processing": signal_selftest,
            "confirmatory_analysis": analysis_selftest,
        },
        "environment_used_for_freeze": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__,
            "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
            "cuda_visible": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lock_sha = sha256(OUTPUT)
    SHA_FILE.write_text(f"{lock_sha}  {OUTPUT.name}\n", encoding="ascii")
    print(json.dumps({"lock2": str(OUTPUT), "sha256": lock_sha, "files": len(files),
                      "archives": len(archives), "state": payload["state"]}, indent=2))


if __name__ == "__main__":
    main()
