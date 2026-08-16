from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "GW_P11B_Wind_Blade_Confirmatory_Evidence_v2_20260811.zip"

EXACT_FILES = [
    "research_protocols/P11B_wind_blade_source_paper_summary_20260811.md",
    "research_protocols/P11B_wind_blade_confirmatory_preregistration_v1.md",
    "research_protocols/P11B_preregistration_amendment_A1_schema_mapping.md",
    "research_protocols/P11B_LOCK1_freeze_manifest_v1.json",
    "research_protocols/P11B_LOCK1_SHA256_v1.txt",
    "research_protocols/P11B_LOCK2_execution_freeze_manifest_v1.json",
    "research_protocols/P11B_LOCK2_SHA256_v1.txt",
    "research_protocols/P11B_UNSEAL_AND_EXECUTION_LOG_20260811.md",
    "research_protocols/P11B_wind_blade_confirmatory_results_report_20260811.md",
    "research_protocols/P11B_secondary_dynamic_three_phase_analysis_A2.md",
    "research_protocols/P11B_DYNAMIC_SECONDARY_LOCK_v1.json",
    "research_protocols/P11B_DYNAMIC_SECONDARY_LOCK_SHA256_v1.txt",
    "research_protocols/P11B_secondary_dynamic_three_phase_results_report_20260811.md",
    "tools/audit_p11b_archive_schema_lock1.py",
    "tools/audit_p11b_mat_schema_lock1.py",
    "tools/run_p11b_wind_blade_confirmatory.py",
    "tools/analyze_p11b_posthoc_diagnostics.py",
    "tools/run_p11b_secondary_dynamic_three_phase.py",
]
DIRECTORIES = [
    "data/reports/p11b_schema_audit_lock1",
    "runs/p11b_wind_blade_confirmatory_v1",
    "runs/p11b_secondary_dynamic_three_phase_v1",
]
FORBIDDEN_SUFFIXES = {".zip", ".mat", ".pt", ".zarr", ".xls", ".xlsx", ".sg2", ".rar", ".7z"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add(files: dict[str, Path], path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    rel = path.relative_to(ROOT).as_posix()
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"forbidden raw file: {rel}")
    files[rel] = path


def main() -> None:
    files: dict[str, Path] = {}
    for rel in EXACT_FILES:
        add(files, ROOT / rel)
    for rel in DIRECTORIES:
        for path in (ROOT / rel).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                add(files, path)

    entries = [
        {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for rel, path in sorted(files.items())
    ]
    manifest = {
        "schema_version": "gw-p11b-wind-blade-confirmatory-evidence-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_status": "FAIL_calibration_reliability_gate",
        "raw_data_included": False,
        "source_paper_pdf_included": False,
        "entry_count": len(entries),
        "entries": entries,
    }

    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for rel, path in sorted(files.items()):
            archive.write(path, rel)

    with zipfile.ZipFile(PACKAGE) as archive:
        bad = archive.testzip()
        forbidden = [name for name in archive.namelist() if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES]
        count = len(archive.namelist())
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
        "entries": count,
        "crc": "pass",
        "forbidden_entries": len(forbidden),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
