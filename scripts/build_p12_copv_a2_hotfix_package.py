from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages/GW_P12_COPV_A2_LOCK2_1_Hotfix_v1_20260812.zip"
FILES = [
    "src/p12_copv_pipeline.py",
    "tools/run_p12_copv_confirmatory.py",
    "tools/selftest_p12_copv_a2_frequency_mapping.py",
    "tools/audit_p12_copv_a2_frequency_metadata.py",
    "tools/verify_p12_copv_lock2_1.py",
    "pbs/33_p12_copv_confirmatory_serial_v1_1.pbs",
    "research_protocols/P12_COPV_preregistration_amendment_A2_frequency_index_mapping.md",
    "research_protocols/P12_COPV_LOCK2_1_execution_freeze_v1.json",
    "research_protocols/P12_COPV_LOCK2_1_SHA256_v1.txt",
    "data/reports/p12_copv_a2_frequency_mapping_selftest_v1.json",
    "data/reports/p12_copv_a2_frequency_metadata_audit_v1.json",
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
        "schema_version": "p12-copv-a2-lock2-1-hotfix-package-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "metadata-only 240-to-260 kHz fourth burst index mapping",
        "scientific_parameters_or_gates_changed": False,
        "lock2_1_sha256": (ROOT / "research_protocols/P12_COPV_LOCK2_1_SHA256_v1.txt").read_text().split()[0],
        "files": [{"path": item, "bytes": (ROOT / item).stat().st_size,
                   "sha256": digest(ROOT / item)} for item in FILES],
    }
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
        archive.writestr("P12_COPV_A2_HOTFIX_MANIFEST.json",
                         json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    checksum = digest(PACKAGE)
    PACKAGE.with_suffix(PACKAGE.suffix + ".sha256").write_text(
        f"{checksum}  {PACKAGE.name}\n", encoding="ascii"
    )
    print(json.dumps({"package": str(PACKAGE), "bytes": PACKAGE.stat().st_size,
                      "sha256": checksum, "entries": len(FILES) + 1}, indent=2))


if __name__ == "__main__":
    main()
