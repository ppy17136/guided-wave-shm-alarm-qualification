from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p12_copv_pipeline import FREQUENCIES_HZ, frequency_index_map


def mapping_row(fourth: int) -> np.ndarray:
    frequencies = [60_000] * 3 + [120_000] * 3 + [180_000] * 3 + [fourth] * 3 + [300_000] * 3
    return np.asarray([list(range(1, 16)), frequencies], dtype=np.float64)


def main() -> None:
    actual = frequency_index_map(mapping_row(240_000))
    canonical = frequency_index_map(mapping_row(260_000))
    malformed = mapping_row(240_000)
    malformed[0, 9] = 9
    rejected = False
    try:
        frequency_index_map(malformed)
    except ValueError:
        rejected = True
    checks = {
        "official_240_label_maps_to_260": actual[260_000] == [9, 10, 11],
        "canonical_260_still_supported": canonical[260_000] == [9, 10, 11],
        "all_five_frequencies_three_repeats": all(len(actual[f]) == 3 for f in FREQUENCIES_HZ),
        "malformed_indices_rejected": rejected,
    }
    report = {"schema_version": "p12-copv-a2-frequency-mapping-selftest-v1",
              "status": "pass" if all(checks.values()) else "fail", "checks": checks,
              "mapped_indices": {str(k): v for k, v in actual.items()}}
    path = ROOT / "data/reports/p12_copv_a2_frequency_mapping_selftest_v1.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
