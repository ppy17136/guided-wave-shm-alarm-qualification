from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "GW_P12_COPV_Confirmatory_Execution_Package_v1_20260812.zip"
FILES = [
    "configs/p12_copv_v1/P12-COPV-01.json",
    "src/p12_copv_pipeline.py",
    "tools/run_p12_copv_confirmatory.py",
    "tools/selftest_p12_copv_pipeline.py",
    "tools/selftest_p12_copv_confirmatory_analysis.py",
    "tools/install_p12_copv_offline_vendor.sh",
    "tools/verify_p12_copv_lock2.py",
    "pbs/32_p12_copv_confirmatory_serial.pbs",
    "research_protocols/P12_COPV_confirmatory_preregistration_v1.md",
    "research_protocols/P12_COPV_preregistration_amendment_A1_schema_mapping.md",
    "research_protocols/P12_COPV_LOCK1_freeze_manifest_v1.json",
    "research_protocols/P12_COPV_LOCK1_SHA256_v1.txt",
    "research_protocols/P12_COPV_LOCK2_execution_freeze_v1.json",
    "research_protocols/P12_COPV_LOCK2_SHA256_v1.txt",
    "data/reports/p12_copv_download_verification_lock1.json",
    "data/reports/p12_copv_schema_audit_lock1/schema_audit_summary.json",
    "data/reports/p12_copv_schema_audit_lock1/h5_file_manifest.csv",
    "data/reports/p12_copv_pipeline_selftest_v1.json",
    "data/reports/p12_copv_confirmatory_analysis_selftest_v1.json",
    "offline_wheels_p12_copv/scipy-1.12.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "offline_wheels_p12_copv/h5py-3.12.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    for relative in FILES:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    manifest = {
        "schema_version": "p12-copv-cluster-package-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lock2_sha256": (ROOT / "research_protocols/P12_COPV_LOCK2_SHA256_v1.txt").read_text().split()[0],
        "raw_archives_included": False,
        "raw_archive_destination": "_references/P12_COPV_SEALED",
        "files": [{"path": item, "bytes": (ROOT / item).stat().st_size,
                   "sha256": sha256(ROOT / item)} for item in FILES],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr("P12_COPV_CLUSTER_PACKAGE_MANIFEST.json", manifest_bytes)
    checksum = sha256(PACKAGE)
    checksum_path = PACKAGE.with_suffix(PACKAGE.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {PACKAGE.name}\n", encoding="ascii")
    print(json.dumps({"package": str(PACKAGE), "bytes": PACKAGE.stat().st_size,
                      "sha256": checksum, "files": len(FILES)}, indent=2))


if __name__ == "__main__":
    main()
