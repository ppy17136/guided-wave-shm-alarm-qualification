"""Build and verify the machine-readable evidence for frozen P12 gate 13."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
DEFAULT_FREQUENCY_AUDIT = REPORTS / "p12_copv_a2_frequency_metadata_audit_v1.json"
DEFAULT_SCHEMA_SUMMARY = REPORTS / "p12_copv_schema_audit_public_summary_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "derived_tables" / "p12_acquisition_integrity_audit.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build(frequency_path: Path, schema_path: Path) -> dict[str, object]:
    frequency = load(frequency_path)
    schema = load(schema_path)
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
    known_routed = all(item["explained"] and item["routed_to_abstention"] for item in known)
    unexplained = schema["unexplained_state_specific_differences"]
    checks = {
        "three_states_present": set(roles) == expected_roles,
        "sampling_frequency_equal_across_states": common_sampling,
        "burst_frequency_program_equal_across_states": common_burst,
        "raw_acquisition_shape_equal_across_states": common_raw_shape,
        "official_schema_inventory_covers_540_files": all_540,
        "core_channel_and_sampling_schema_equal_in_all_states": core_equal,
        "known_metadata_gaps_explained_and_abstained": known_routed,
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
        "unexplained_state_specific_differences": unexplained,
        "unexplained_difference_count": len(unexplained),
        "source_evidence": [
            {"path": frequency_path.relative_to(ROOT).as_posix(), "sha256": digest(frequency_path)},
            {"path": schema_path.relative_to(ROOT).as_posix(), "sha256": digest(schema_path)},
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(args.frequency_audit, args.schema_summary)
    if args.check:
        committed = load(args.output)
        if committed != result:
            raise RuntimeError("Committed acquisition-integrity audit differs from recomputed evidence")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
