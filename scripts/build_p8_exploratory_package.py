"""Package P8 exploratory diagnostics, protocols, figures, and tables."""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "packages/GW_P8_Direction_Instability_Exploratory_v1_20260811.zip"
FILES = [
    "research_protocols/P7_confirmatory_results_report_20260811.md",
    "research_protocols/P8_direction_instability_exploratory_report_20260811.md",
    "research_protocols/P8_dynamic_path_graph_protocol_v1.md",
    "runs/p7_confirmatory_analysis_v1/01/p7_confirmatory_summary.json",
    "tools/analyze_p8_candidate_direction_stability.py",
    "tools/analyze_p8_two_sided_sequential.py",
    "tools/analyze_p8_common_local_modes.py",
    "tools/analyze_p8_environment_regime_library.py",
]
DIRECTORIES = [
    "runs/p8_exploratory_candidate_audit_v1",
    "runs/p8_two_sided_sequential_v1.1",
    "runs/p8_common_local_modes_v1",
    "runs/p8_environment_regime_library_v1",
]


def main() -> None:
    paths = [ROOT / item for item in FILES]
    for directory in DIRECTORIES:
        paths.extend(path for path in sorted((ROOT / directory).rglob("*")) if path.is_file())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing P8 package files: {missing}")
    unique = sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())
    hashes = {}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in unique:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr(relative, data)
        manifest = {
            "schema_version": "p8-direction-instability-exploratory-package-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_status": "post_P7_exploratory_not_confirmatory",
            "P7_interpretation_unchanged": "both_confirmatory_endpoints_failed",
            "environment_regime_branch_stopped": True,
            "next_bounded_protocol": "P8_dynamic_path_graph_protocol_v1.md",
            "files": hashes,
        }
        archive.writestr("P8_PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
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

