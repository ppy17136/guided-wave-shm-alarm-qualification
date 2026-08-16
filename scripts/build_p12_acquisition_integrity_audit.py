"""Build and verify the machine-readable evidence for frozen P12 gate 13."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
DEFAULT_FREQUENCY_AUDIT = REPORTS / "p12_copv_a2_frequency_metadata_audit_v1.json"
DEFAULT_SCHEMA_SUMMARY = REPORTS / "p12_copv_schema_audit_public_summary_v1.json"
DEFAULT_CONDITION_RESULTS = ROOT / "results" / "derived_tables" / "p12_condition_level_results.csv"
DEFAULT_OUTPUT = ROOT / "results" / "derived_tables" / "p12_acquisition_integrity_audit.json"


def digest(path: Path) -> str:
    """Hash canonical JSON content so provenance is newline-platform invariant."""
    payload = json.dumps(
        json.loads(path.read_text(encoding="utf-8-sig")),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_digest(path: Path) -> str:
    """Hash UTF-8 tabular content after platform-invariant newline normalization."""
    payload = path.read_text(encoding="utf-8-sig")
    canonical = payload.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build(
    frequency_path: Path,
    schema_path: Path,
    condition_results_path: Path,
) -> dict[str, object]:
    frequency = load(frequency_path)
    schema = load(schema_path)
    with condition_results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        condition_rows = list(csv.DictReader(handle))
    records = frequency["records"]
    roles = [record["role"] for record in records]
    sampling = {record["role"]: record["sampling_frequency_hz"] for record in records}
    burst = {record["role"]: record["signal_frequency_burst_hz"] for record in records}
    raw_shape = {record["role"]: record["raw_dataset_shape_only"] for record in records}
    common_sampling = len({tuple(value) for value in sampling.values()}) == 1
    common_burst = len({tuple(value) for value in burst.values()}) == 1
    common_raw_shape = len({tuple(value) for value in raw_shape.values()}) == 1
    state_summary = schema["state_file_schema_summary"]
    expected_roles = {"baseline", "irreversible", "reversible"}
    all_540 = sum(item["files"] for item in state_summary.values()) == 540
    core_equal = all(item["core_acquisition_schema_equal"] for item in state_summary.values())
    known = schema["known_state_specific_differences"]
    known_routed = all(item["explained"] and item["disposition_fully_documented"] for item in known)
    raw_by_temperature = {
        "37": sum(item.get("distribution", {}).get("T37_all_two_pressure_ramps", 0) for item in known),
        "55": sum(item.get("distribution", {}).get("T55_selected_random_ramp", 0) for item in known),
    }
    invalid_rows = [
        row for row in condition_rows
        if row.get("status") == "invalid_missing_support_metadata"
    ]
    in_grid_by_temperature = {
        temperature: sum(row.get("temperature_c") == temperature for row in invalid_rows)
        for temperature in raw_by_temperature
    }
    outside_grid_by_temperature = {
        temperature: raw_by_temperature[temperature] - in_grid_by_temperature[temperature]
        for temperature in raw_by_temperature
    }
    raw_gap_files = sum(raw_by_temperature.values())
    in_grid_abstentions = len(invalid_rows)
    outside_grid_files = sum(outside_grid_by_temperature.values())
    scope_accounted = (
        len(condition_rows) == 419
        and all(value >= 0 for value in outside_grid_by_temperature.values())
        and raw_gap_files == in_grid_abstentions + outside_grid_files
    )
    scope_clarification = {
        "raw_archive_metadata_gap_files": raw_gap_files,
        "raw_archive_by_temperature_c": raw_by_temperature,
        "frozen_feature_records_expected": 420,
        "formal_condition_records_after_official_exclusion": len(condition_rows),
        "official_exclusion_records": 1,
        "in_frozen_evaluation_grid_and_counted_as_support_abstentions": in_grid_abstentions,
        "in_grid_by_temperature_c": in_grid_by_temperature,
        "outside_frozen_evaluation_grid_and_excluded_from_denominators": outside_grid_files,
        "outside_grid_by_temperature_c": outside_grid_by_temperature,
        "scope_accounted": scope_accounted,
        "scientific_effect": "none_reporting_scope_clarification_only",
    }
    unexplained = schema["unexplained_state_specific_differences"]
    checks = {
        "three_states_present": set(roles) == expected_roles,
        "sampling_frequency_equal_across_states": common_sampling,
        "burst_frequency_program_equal_across_states": common_burst,
        "raw_acquisition_shape_equal_across_states": common_raw_shape,
        "official_schema_inventory_covers_540_files": all_540,
        "core_channel_and_sampling_schema_equal_in_all_states": core_equal,
        "known_metadata_gaps_explained_and_disposition_documented": known_routed,
        "formal_condition_table_has_419_records": len(condition_rows) == 419,
        "metadata_gap_reporting_scope_fully_accounted": scope_clarification["scope_accounted"],
        "zero_unexplained_state_specific_differences": len(unexplained) == 0,
    }
    passed = all(checks.values())
    return {
        "schema_version": "p12-acquisition-integrity-gate-evidence-v1",
        "analysis_role": "post-outcome_machine_readable_binding_of_prespecified_qualitative_gate",
        "frozen_gate_key": "no_unexplained_acquisition_asymmetry",
        "manuscript_gate_label": "Data-integrity audit: no unexplained state-specific channel or sampling-regime difference",
        "gate_pass": passed,
        "checks": checks,
        "observed_sampling_frequency_hz_by_state": sampling,
        "observed_burst_frequency_program_hz_by_state": burst,
        "observed_raw_shape_by_state": raw_shape,
        "state_file_schema_summary": state_summary,
        "known_state_specific_differences": known,
        "reporting_scope_clarification": scope_clarification,
        "unexplained_state_specific_differences": unexplained,
        "unexplained_difference_count": len(unexplained),
        "source_evidence": [
            {"path": frequency_path.relative_to(ROOT).as_posix(), "sha256": digest(frequency_path)},
            {"path": schema_path.relative_to(ROOT).as_posix(), "sha256": digest(schema_path)},
            {"path": condition_results_path.relative_to(ROOT).as_posix(), "sha256": file_digest(condition_results_path)},
        ],
        "provenance_note": (
            "The frozen execution serializer stored this pre-specified qualitative gate as a Boolean. "
            "This post-outcome file binds that Boolean to auditable pre-existing metadata records; it "
            "does not modify a score, threshold, denominator, gate criterion, or the strict P12 FAIL outcome."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency-audit", type=Path, default=DEFAULT_FREQUENCY_AUDIT)
    parser.add_argument("--schema-summary", type=Path, default=DEFAULT_SCHEMA_SUMMARY)
    parser.add_argument("--condition-results", type=Path, default=DEFAULT_CONDITION_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(args.frequency_audit, args.schema_summary, args.condition_results)
    if args.check:
        committed = load(args.output)
        if committed != result:
            raise RuntimeError("Committed acquisition-integrity audit differs from recomputed evidence")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Write deterministic UTF-8/LF bytes on every platform so the
        # repository checksum is identical on Windows and Linux checkouts.
        args.output.write_bytes(json.dumps(result, indent=2).encode("utf-8"))
    print(json.dumps(result, indent=2))
    if not result["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
