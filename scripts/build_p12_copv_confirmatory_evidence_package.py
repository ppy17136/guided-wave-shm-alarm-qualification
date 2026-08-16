from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "GW_P12_COPV_Confirmatory_Evidence_v1_20260814.zip"

EXACT_FILES = [
    "cluster_results/GW_P12_COPV_RESULTS_21006.master_20260813T161036Z.tar.gz",
    "cluster_results/GW_P12_COPV_RESULTS_21006.master_20260813T161036Z.tar.gz.sha256",
    "research_protocols/P12_COPV_candidate_source_audit_20260811.md",
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
    "research_protocols/P12_COPV_LOCK2_2_execution_freeze_v1.json",
    "research_protocols/P12_COPV_LOCK2_2_SHA256_v1.txt",
    "research_protocols/P12_COPV_confirmatory_results_report_20260814.md",
    "tools/build_p12_copv_results_evidence.py",
    "tools/build_p12_copv_confirmatory_evidence_package.py",
    "tools/run_p12_copv_confirmatory.py",
    "tools/verify_p12_copv_lock2_2.py",
    "tools/audit_p12_copv_a2_frequency_metadata.py",
    "tools/audit_p12_copv_a3_missing_support_from_official_json.py",
    "tools/selftest_p12_copv_pipeline.py",
    "tools/selftest_p12_copv_confirmatory_analysis.py",
    "tools/selftest_p12_copv_a2_frequency_mapping.py",
    "tools/selftest_p12_copv_a3_missing_support.py",
    "tools/selftest_p12_copv_a3_coverage_denominator.py",
    "src/p12_copv_pipeline.py",
    "configs/p12_copv_v1/P12-COPV-01.json",
    "data/reports/p12_copv_download_verification_lock1.json",
    "data/reports/p12_copv_schema_audit_lock1/schema_audit_summary.json",
    "data/reports/p12_copv_schema_audit_lock1/h5_file_manifest.csv",
    "data/reports/p12_copv_a2_frequency_metadata_audit_v1.json",
    "data/reports/p12_copv_a3_missing_support_official_json_audit_v1.json",
]

DIRECTORIES = [
    "runs/p12_copv_reporting_v1",
    "cluster_results/GW_P12_COPV_RESULTS_21006_extracted",
]

# Raw 19–20 GB source archives, H5 waveforms, and external papers are never
# copied into this compact evidence package.  The only nested archive allowed is
# the named 371 KB cluster result archive whose digest is already frozen.
FORBIDDEN_SUFFIXES = {".h5", ".mat", ".pt", ".zarr", ".rar", ".7z"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(files: dict[str, Path], path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    rel = path.relative_to(ROOT).as_posix()
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"forbidden raw-data entry: {rel}")
    files[rel] = path


def main() -> None:
    files: dict[str, Path] = {}
    for rel in EXACT_FILES:
        add(files, ROOT / rel)
    for rel in DIRECTORIES:
        for path in (ROOT / rel).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                add(files, path)

    source_archive = ROOT / EXACT_FILES[0]
    expected_source_sha = "7735d52e812c8fe930cf19e783c43574777145e12bedfa95982644dc3bbd4de5"
    if sha256(source_archive) != expected_source_sha:
        raise RuntimeError("source cluster archive hash mismatch")

    audit = json.loads((ROOT / "runs/p12_copv_reporting_v1/p12_reporting_audit.json").read_text(encoding="utf-8"))
    if audit["frozen_primary_status"] != "FAIL" or not all(audit["checks"].values()):
        raise RuntimeError("reporting audit does not preserve the frozen FAIL result")

    entries = [
        {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for rel, path in sorted(files.items())
    ]
    manifest = {
        "schema_version": "gw-p12-copv-confirmatory-evidence-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_status": "FAIL",
        "pbs_job": "21006.master",
        "pbs_exit_status": 0,
        "lock2_2_sha256": "63012f173bb1dcbbc287ae785d24c7a351a00a77cc619e0070af17c263cae64b",
        "cluster_archive_sha256": expected_source_sha,
        "raw_h5_included": False,
        "external_papers_included": False,
        "cluster_result_archive_included": True,
        "entry_count_excluding_manifest": len(entries),
        "entries": entries,
    }

    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for rel, path in sorted(files.items()):
            archive.write(path, rel)

    with zipfile.ZipFile(PACKAGE) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        duplicate_count = len(names) - len(set(names))
        forbidden = [name for name in names if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES]
        embedded_primary = archive.read(
            "cluster_results/GW_P12_COPV_RESULTS_21006_extracted/runs/p12_copv_v1/01/COMPLETED.json"
        ).decode("utf-8")
    if bad or duplicate_count or forbidden or '"primary_status": "FAIL"' not in embedded_primary:
        raise RuntimeError({
            "crc_failure": bad,
            "duplicates": duplicate_count,
            "forbidden": forbidden,
            "embedded_primary": embedded_primary,
        })

    digest = sha256(PACKAGE)
    PACKAGE.with_suffix(PACKAGE.suffix + ".sha256").write_text(
        f"{digest}  {PACKAGE.name}\n", encoding="ascii"
    )
    print(json.dumps({
        "package": str(PACKAGE),
        "bytes": PACKAGE.stat().st_size,
        "sha256": digest,
        "entries_including_manifest": len(names),
        "crc": "pass",
        "duplicate_entries": duplicate_count,
        "forbidden_raw_entries": len(forbidden),
        "confirmatory_status": "FAIL",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
