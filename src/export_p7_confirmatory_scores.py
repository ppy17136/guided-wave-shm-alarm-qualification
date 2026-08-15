"""P7 thin wrapper: export frozen P3 component scores from 2022 stores."""
from __future__ import annotations

from src import export_p3_checkpoint_scores as base
from src.run_p7_confirmatory_train import build_records_2022


base.build_records = build_records_2022


if __name__ == "__main__":
    base.main()

