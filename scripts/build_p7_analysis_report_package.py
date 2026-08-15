"""Build a compact, auditable P7 analysis/report package without checkpoints."""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "packages/GW_P7_Confirmatory_Analysis_Report_v1_20260811.zip"
FILES = [
    "research_protocols/P7_confirmatory_results_report_20260811.md",
    "research_protocols/P7_predeep_baseline_audit_20260811.md",
    "research_protocols/P7_D7_D13_confirmatory_preregistration_v1.md",
    "research_protocols/P7_D7_D13_preregistration_amendment_A1.md",
    "research_protocols/P7_D7_D13_preregistration_amendment_A2.md",
    "research_protocols/P7_execution_code_freeze_v1.1.json",
    "research_protocols/P7_EXECUTION_CODE_SHA256_v1.1.txt",
    "tools/analyze_p7_confirmatory.py",
    "tools/analyze_p7_comparators_posthoc_A2.py",
    "cluster_results/GW_P7_CONFIRMATORY_20032.master_20260811T024511Z.tar.gz.sha256",
]
DIRECTORIES = [
    "runs/p7_confirmatory_analysis_v1/01",
    "runs/p7_classical_baselines_v1",
    "runs/p7_block_crossconformal_v1",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    paths = [ROOT / item for item in FILES]
    for directory in DIRECTORIES:
        paths.extend(path for path in sorted((ROOT / directory).rglob("*")) if path.is_file())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing analysis package files: {missing}")

    unique = sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())
    file_hashes = {}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in unique:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            file_hashes[relative] = sha256_bytes(data)
            archive.writestr(relative, data)
        manifest = {
            "schema_version": "p7-confirmatory-analysis-report-package-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "confirmatory_interpretation": "both_confirmatory_endpoints_failed",
            "posthoc_outputs_explicitly_labeled": True,
            "contains_checkpoints_or_raw_scores": False,
            "files": file_hashes,
        }
        archive.writestr(
            "ANALYSIS_PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{digest}  {OUTPUT.name}\n", encoding="ascii"
    )
    print(json.dumps({
        "archive": str(OUTPUT),
        "sha256": digest,
        "bytes": OUTPUT.stat().st_size,
        "files": len(unique) + 1,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

