from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "GW_P12_COPV_LOCK1_v1_20260811.zip"
FILES = [
    "research_protocols/P12_COPV_candidate_source_audit_20260811.md",
    "research_protocols/P12_COPV_confirmatory_preregistration_v1.md",
    "research_protocols/P12_COPV_LOCK1_freeze_manifest_v1.json",
    "research_protocols/P12_COPV_LOCK1_SHA256_v1.txt",
    "research_protocols/P12_dual_gate_support_aware_alarm_protocol_v1.md",
    "research_protocols/P12_DUAL_GATE_PROTOTYPE_FREEZE_v1.json",
    "research_protocols/P12_DUAL_GATE_PROTOTYPE_FREEZE_SHA256_v1.txt",
    "src/dual_gate_alarm.py",
    "tools/selftest_p12_dual_gate.py",
    "tools/verify_p12_copv_downloads_lock1.py",
    "data/reports/p12_dual_gate_selftest_v1.json",
    "_references/P12_COPV_SEALED/README_DOWNLOAD_ONLY.md",
    "_references/COPV_Repository_Documentation_v1.pdf",
]
FORBIDDEN = {".mat", ".h5", ".hdf5", ".pt", ".zarr", ".rar", ".7z"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = [(rel, ROOT / rel) for rel in FILES]
    for rel, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() in FORBIDDEN:
            raise RuntimeError(f"forbidden data entry: {rel}")

    entries = [
        {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for rel, path in paths
    ]
    manifest = {
        "schema_version": "gw-p12-copv-lock1-evidence-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "LOCK1_before_large_archives_arrive",
        "numeric_data_included": False,
        "confirmatory_result_included": False,
        "entries": entries,
    }
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for rel, path in paths:
            archive.write(path, rel)

    with zipfile.ZipFile(PACKAGE) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        forbidden = [name for name in names if Path(name).suffix.lower() in FORBIDDEN]
    if bad or forbidden:
        raise RuntimeError({"crc_failure": bad, "forbidden": forbidden})

    digest = sha256(PACKAGE)
    PACKAGE.with_suffix(PACKAGE.suffix + ".sha256").write_text(
        f"{digest}  {PACKAGE.name}\n", encoding="ascii"
    )
    print(json.dumps({
        "package": str(PACKAGE),
        "bytes": PACKAGE.stat().st_size,
        "sha256": digest,
        "entries": len(names),
        "crc": "pass",
        "forbidden_entries": len(forbidden),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
