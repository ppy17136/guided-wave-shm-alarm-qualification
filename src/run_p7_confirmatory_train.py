"""P7 thin wrapper: reuse the frozen P2 trainer on 2022 monthly stores."""
from __future__ import annotations

import numpy as np
import zarr

from src import run_p2_event_ssl as base


def build_records_2022(start_ns: int, stop_ns: int):
    rows = []
    for store in sorted((base.ROOT / "data/zarr").glob("measurements_2022_*.zarr")):
        group = zarr.open_group(str(store), mode="r")
        times = np.asarray(group["datetime_ns"][:], np.int64)
        for idx in np.flatnonzero((times >= start_ns) & (times < stop_ns)):
            rows.append((int(times[idx]), str(store), int(idx)))
    rows.sort()
    return rows


base.build_records = build_records_2022


if __name__ == "__main__":
    base.main()

