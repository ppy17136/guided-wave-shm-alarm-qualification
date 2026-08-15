from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages/GW_P12_COPV_A3_LOCK2_2_Hotfix_v1_20260813.zip"
FILES = [
    "src/p12_copv_pipeline.py",
    "tools/run_p12_copv_confirmatory.py",
    "tools/selftest_p12_copv_a3_missing_support.py",
    "tools/selftest_p12_copv_a3_coverage_denominator.py",
    "tools/audit_p12_copv_a3_missing_support_from_official_json.py",
    "tools/verify_p12_copv_lock2_2.py",
    "pbs/34_p12_copv_confirmatory_serial_v1_2.pbs",
    "research_protocols/P12_COPV_preregistration_amendment_A3_missing_support_metadata_execution.md",
    "research_protocols/P12_COPV_LOCK2_2_execution_freeze_v1.json",
    "research_protocols/P12_COPV_LOCK2_2_SHA256_v1.txt",
    "data/reports/p12_copv_a3_missing_support_selftest_v1.json",
    "data/reports/p12_copv_a3_coverage_denominator_selftest_v1.json",
    "data/reports/p12_copv_a3_missing_support_official_json_audit_v1.json",
    "data/reports/p12_copv_pipeline_selftest_v1.json",
    "data/reports/p12_copv_confirmatory_analysis_selftest_v1.json",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "p12-copv-a3-lock2-2-hotfix-package-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "implement preregistered unsupported/invalid for missing Pressure or Temp_Surface",
        "scientific_parameters_or_gates_changed": False,
        "existing_cluster_feature_cache_included_or_overwritten": False,
        "lock2_2_sha256": (ROOT / "research_protocols/P12_COPV_LOCK2_2_SHA256_v1.txt").read_text().split()[0],
        "files": [{"path": item, "bytes": (ROOT / item).stat().st_size,
                   "sha256": digest(ROOT / item)} for item in FILES],
    }
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr("P12_COPV_A3_HOTFIX_MANIFEST.json",
                         json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    checksum = digest(PACKAGE)
    PACKAGE.with_suffix(PACKAGE.suffix + ".sha256").write_text(
        f"{checksum}  {PACKAGE.name}\n", encoding="ascii"
    )
    print(json.dumps({"package": str(PACKAGE), "bytes": PACKAGE.stat().st_size,
                      "sha256": checksum, "entries": len(FILES) + 1}, indent=2))


if __name__ == "__main__":
    main()
