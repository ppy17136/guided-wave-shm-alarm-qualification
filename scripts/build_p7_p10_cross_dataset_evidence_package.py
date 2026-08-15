from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "GW_P7_P11_Cross_Dataset_Evidence_v1_20260811.zip"

EXACT_FILES = [
    "external_datasets/P9_DATA_ACQUISITION_MANIFEST_20260811.md",
    "external_datasets/MANUAL_DOWNLOAD_REQUESTS_20260811.md",
    "external_datasets/wind_turbine_blade_confirmatory_SEALED/SEAL_MANIFEST_20260811.md",
    "external_datasets/wind_turbine_blade_confirmatory_SEALED/wind_turbine_blade_shm_dataset.SEALED.zip.md5",
    "external_datasets/wind_turbine_blade_confirmatory_SEALED/wind_turbine_blade_shm_dataset.SEALED.zip.sha256",
]

RESEARCH_FILES = [
    "P7_confirmatory_results_report_20260811.md",
    "P7_D7_D13_confirmatory_preregistration_v1.md",
    "P7_D7_D13_preregistration_amendment_A1.md",
    "P7_D7_D13_preregistration_amendment_A2.md",
    "P7_D7_D13_freeze_manifest_v1.1.json",
    "P7_execution_code_freeze_v1.1.json",
    "P7_EXECUTION_CODE_SHA256_v1.1.txt",
    "P7_FREEZE_SHA256_v1.1.txt",
    "P8_direction_instability_exploratory_report_20260811.md",
    "P8_dynamic_path_graph_protocol_v1.md",
    "P9_external_validation_data_plan_v1.md",
    "P9_lambnet_t_development_results_report_20260811.md",
    "P9A_lambnet_t_physics_baseline_protocol_v1.md",
    "P9B_phase_aligned_temperature_interpolation_protocol_v1.md",
    "P9C_frequency_phase_temperature_model_protocol_v1.md",
    "P9D_temperature_conditioned_threshold_protocol_v1.md",
    "P9E_independent_healthy_temperature_transfer_protocol_v1.md",
    "P9E_independent_healthy_transfer_results_report_20260811.md",
    "P10_pipeline_dataset_audit_plan_v0_20260811.md",
    "P10_pipeline_structure_relative_anomaly_protocol_v1.md",
    "P10_pipeline_structure_relative_results_report_20260811.md",
    "P7_P10_cross_dataset_synthesis_and_paper_strategy_20260811.md",
    "P11A_health_side_calibration_reliability_exploratory_protocol_v1.md",
    "P11A_health_side_calibration_reliability_results_report_20260811.md",
]

TOOL_FILES = [
    "analyze_p7_confirmatory.py",
    "analyze_p7_comparators_posthoc_A2.py",
    "merge_p7_confirmatory_comparators.py",
    "run_p7_block_crossconformal.py",
    "run_p7_classical_baselines.py",
    "analyze_p8_candidate_direction_stability.py",
    "analyze_p8_two_sided_sequential.py",
    "analyze_p8_common_local_modes.py",
    "analyze_p8_environment_regime_library.py",
    "analyze_p8_dynamic_path_graph.py",
    "audit_p9_lambnet_t.py",
    "audit_p9_lambnet_t_semantic.py",
    "analyze_p9a_semantic_sensitivity.py",
    "run_p9a_lambnet_t_baselines.py",
    "run_p9b_phase_aligned_interpolation.py",
    "run_p9c_frequency_phase_model.py",
    "run_p9d_temperature_conditioned_threshold.py",
    "audit_p9e_composite_june2024.py",
    "run_p9e_independent_healthy_transfer.py",
    "audit_p10_pipeline_dataset.py",
    "run_p10_pipeline_anomaly.py",
    "run_p11a_health_side_calibration_reliability.py",
]

RESULT_DIRS = [
    "runs/p7_block_crossconformal_v1",
    "runs/p7_classical_baselines_v1",
    "runs/p7_confirmatory_analysis_v1",
    "runs/p8_common_local_modes_v1",
    "runs/p8_dynamic_path_graph_v1",
    "runs/p8_environment_regime_library_v1",
    "runs/p8_exploratory_candidate_audit_v1",
    "runs/p8_two_sided_sequential_v1",
    "runs/p8_two_sided_sequential_v1.1",
    "runs/p9a_lambnet_t_baselines_v1",
    "runs/p9b_phase_aligned_interpolation_v1",
    "runs/p9c_frequency_phase_model_v1",
    "runs/p9d_temperature_conditioned_threshold_v1",
    "runs/p9e_independent_healthy_transfer_v1",
    "runs/p10_pipeline_anomaly_v1",
    "runs/p11a_health_side_calibration_reliability_v1",
    "data/reports/p9_lambnet_t_audit_v1",
    "data/reports/p9e_composite_june2024_audit_v1",
    "data/reports/p10_pipeline_audit_v1",
]

FORBIDDEN_SUFFIXES = {
    ".zip", ".rar", ".7z", ".tar", ".sg2", ".xls", ".xlsx", ".mat", ".zarr", ".pt"
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_file(files: dict[str, Path], path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    rel = path.relative_to(ROOT).as_posix()
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"Forbidden raw/archive suffix in package: {rel}")
    files[rel] = path


def main() -> None:
    files: dict[str, Path] = {}

    for rel in EXACT_FILES:
        add_file(files, ROOT / rel)
    for name in RESEARCH_FILES:
        add_file(files, ROOT / "research_protocols" / name)
    for name in TOOL_FILES:
        add_file(files, ROOT / "tools" / name)
    for rel in RESULT_DIRS:
        directory = ROOT / rel
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                add_file(files, path)

    entries = []
    for rel, path in sorted(files.items()):
        entries.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})

    manifest = {
        "schema_version": "gw-p7-p10-cross-dataset-evidence-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reproducible protocols, code, compact results, negative controls and failure-preserving reports for P7-P10.",
        "raw_or_sealed_data_included": False,
        "sealed_wind_dataset_opened": False,
        "entry_count": len(entries),
        "entries": entries,
    }

    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for rel, path in sorted(files.items()):
            archive.write(path, rel)

    package_sha = sha256(PACKAGE)
    checksum_path = PACKAGE.with_suffix(PACKAGE.suffix + ".sha256")
    checksum_path.write_text(f"{package_sha}  {PACKAGE.name}\n", encoding="ascii")

    with zipfile.ZipFile(PACKAGE) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    forbidden = [name for name in names if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES]
    if bad is not None or forbidden:
        raise RuntimeError({"crc_failure": bad, "forbidden_entries": forbidden})

    print(json.dumps({
        "package": str(PACKAGE),
        "bytes": PACKAGE.stat().st_size,
        "sha256": package_sha,
        "entries": len(names),
        "crc": "pass",
        "forbidden_entries": len(forbidden),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
