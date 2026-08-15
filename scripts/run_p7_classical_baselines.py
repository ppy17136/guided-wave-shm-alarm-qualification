"""Run the frozen classical q=.999 baseline on the P7 metadata manifest."""
from __future__ import annotations

from pathlib import Path

import run_event_classical_baselines as base


ROOT = Path(__file__).resolve().parents[1]
base.EVENTS_JSON = ROOT / "data/reports/p7_confirmatory_protocol_v1/p7_event_manifest.json"
base.OUT = ROOT / "runs/p7_classical_baselines_v1"


if __name__ == "__main__":
    base.main()

