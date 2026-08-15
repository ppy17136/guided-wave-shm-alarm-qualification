"""Run the frozen five-fold block cross-conformal baseline on P7."""
from __future__ import annotations

from pathlib import Path

import run_event_block_crossconformal as base


ROOT = Path(__file__).resolve().parents[1]
base.EVENTS_JSON = ROOT / "data/reports/p7_confirmatory_protocol_v1/p7_event_manifest.json"
base.OUT = ROOT / "runs/p7_block_crossconformal_v1"


if __name__ == "__main__":
    base.main()

