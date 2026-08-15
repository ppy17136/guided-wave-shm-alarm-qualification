"""Create the seven-event P7 manifest from metadata only.

This script never opens the guided-wave array. It selects and renumbers the
pre-registered D6->D7 ... D12->D13 transitions from the full event manifest.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    (6, 7, date(2022, 6, 11)),
    (7, 8, date(2022, 7, 31)),
    (8, 9, date(2022, 8, 7)),
    (9, 10, date(2022, 9, 11)),
    (10, 11, date(2022, 9, 18)),
    (11, 12, date(2022, 9, 25)),
    (12, 13, date(2022, 10, 2)),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/reports/damage_event_protocol_v1/damage_event_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/p7_confirmatory_protocol_v1/p7_event_manifest.json"),
    )
    args = parser.parse_args()
    source = json.loads((ROOT / args.source).read_text(encoding="utf-8"))
    by_pair = {(int(item["old_tag"]), int(item["new_tag"])): item for item in source["events"]}
    selected = []
    for new_index, (old_tag, new_tag, official_date) in enumerate(EXPECTED, start=1):
        pair = (old_tag, new_tag)
        if pair not in by_pair:
            raise SystemExit(f"Missing pre-registered transition D{old_tag}->D{new_tag}")
        event = deepcopy(by_pair[pair])
        observed = datetime.fromisoformat(event["event_time_utc"].replace("Z", "+00:00")).date()
        if observed != official_date:
            raise SystemExit(f"Date mismatch for D{old_tag}->D{new_tag}: {observed} != {official_date}")
        complete = all(bool(event.get(key)) for key in ("baseline_complete", "pre_event_complete", "post_event_complete"))
        approved_gap = (
            pair == (9, 10)
            and not bool(event.get("baseline_complete"))
            and bool(event.get("pre_event_complete"))
            and bool(event.get("post_event_complete"))
            and int(event.get("n_baseline", 0)) >= 4000
        )
        if not complete and not approved_gap:
            raise SystemExit(f"Incomplete causal windows for D{old_tag}->D{new_tag}")
        event["source_event_index"] = event["event_index"]
        event["event_index"] = new_index
        event["confirmatory_status"] = "untouched_at_preregistration"
        event["confirmatory_window_status"] = (
            "approved_A2_D9_D10_leading_baseline_gap" if approved_gap else "complete"
        )
        selected.append(event)

    payload = {
        "schema_version": "p7-confirmatory-event-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.source).replace("\\", "/"),
        "selection_uses_wave_amplitudes": False,
        "selection_rule": "exact ordered metadata transitions D6->D7 through D12->D13",
        "events": selected,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "events": len(selected),
        "transitions": [item["transition"] for item in selected],
        "all_windows_complete_or_A2_exception": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

