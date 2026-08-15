"""Build the code-only P7 single-PBS execution package."""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packages" / "GW_P7_Confirmatory_Execution_Package_v1.1_20260811.zip"
FILES = [
    "README_P7_CONFIRMATORY.md",
    "research_protocols/P7_D7_D13_confirmatory_preregistration_v1.md",
    "research_protocols/P7_D7_D13_preregistration_amendment_A1.md",
    "research_protocols/P7_D7_D13_preregistration_amendment_A2.md",
    "research_protocols/P7_D7_D13_freeze_manifest_v1.json",
    "research_protocols/P7_D7_D13_freeze_manifest_v1.1.json",
    "research_protocols/P7_FREEZE_SHA256_v1.txt",
    "research_protocols/P7_FREEZE_SHA256_v1.1.txt",
    "research_protocols/P7_execution_code_freeze_v1.json",
    "research_protocols/P7_EXECUTION_CODE_SHA256_v1.txt",
    "research_protocols/P7_execution_code_freeze_v1.1.json",
    "research_protocols/P7_EXECUTION_CODE_SHA256_v1.1.txt",
    "configs/p7_confirmatory_v1/P7-CONFIRM-D7-D13-v1.1.yaml",
    "configs/p7_confirmatory_v1/P7-TRAIN-01.yaml",
    "configs/p7_confirmatory_v1/P7-RESCORE-01.yaml",
    "configs/p2_event_ssl_v1/P2-EVENT-SSL-01.yaml",
    "src/run_p2_event_ssl.py",
    "src/run_p7_confirmatory_train.py",
    "src/export_p3_checkpoint_scores.py",
    "src/export_p7_confirmatory_scores.py",
    "src/envwave/__init__.py",
    "src/envwave/data.py",
    "src/envwave/event_dual_branch.py",
    "src/envwave/model.py",
    "src/envwave/numpy_metrics_v1.py",
    "tools/build_damage_event_manifest.py",
    "tools/build_p7_confirmation_manifest.py",
    "tools/preflight_p7_confirmatory.py",
    "tools/preflight_p3_checkpoint_rescore.py",
    "tools/verify_p1_environment.py",
    "tools/collect_p2_event_ssl.py",
    "tools/collect_p3_checkpoint_rescore.py",
    "tools/analyze_p7_confirmatory.py",
    "tools/run_p7_classical_baselines.py",
    "tools/run_p7_block_crossconformal.py",
    "tools/merge_p7_confirmatory_comparators.py",
    "tools/run_event_classical_baselines.py",
    "tools/run_event_block_crossconformal.py",
    "pbs/32_p7_confirmatory_train_rescore_serial.pbs",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [item for item in FILES if not (ROOT / item).is_file()]
    if missing:
        raise SystemExit(f"Missing package files: {missing}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, relative)
    digest = sha256(OUT)
    sha_path = OUT.with_suffix(OUT.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {OUT.name}\n", encoding="ascii")
    report = {
        "schema_version": "p7-execution-package-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(OUT),
        "sha256": digest,
        "bytes": OUT.stat().st_size,
        "files": len(FILES),
        "contains_data": False,
    }
    (OUT.parent / "GW_P7_Confirmatory_Execution_Package_v1.1_20260811.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

