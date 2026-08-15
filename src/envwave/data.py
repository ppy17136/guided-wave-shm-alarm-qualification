from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

@dataclass(frozen=True)
class Record:
    store: Path
    index: int
    timestamp_ns: int
    damage_tag: int

def to_ns(date_text: str, end_of_day=False):
    suffix = "T23:59:59.999999999" if end_of_day else "T00:00:00"
    return int(np.datetime64(date_text.replace("_", "-") + suffix, "ns").astype(np.int64))

def build_records(root: Path, start: str, end: str, allowed_damage=None, stride=1):
    lo, hi = to_ns(start), to_ns(end, True)
    records = []
    for store in sorted(root.glob("*.zarr")):
        group = zarr.open_group(str(store), mode="r")
        ts = np.asarray(group["datetime_ns"][:], np.int64)
        damage = np.asarray(group["damage_tag"][:], np.int16)
        keep = (ts >= lo) & (ts <= hi)
        if allowed_damage is not None:
            keep &= np.isin(damage, list(allowed_damage))
        for idx in np.flatnonzero(keep)[::stride]:
            records.append(Record(store, int(idx), int(ts[idx]), int(damage[idx])))
    return records

class GuidedWaveDataset(Dataset):
    def __init__(self, records: Iterable[Record], stats=None):
        self.records, self.stats = list(records), stats
        self._store_path, self._group = None, None
    def __len__(self): return len(self.records)
    def _open(self, path):
        if self._store_path != path:
            self._group, self._store_path = zarr.open_group(str(path), mode="r"), path
        return self._group
    def __getitem__(self, item):
        rec = self.records[item]
        g = self._open(rec.store)
        wave = np.asarray(g["guided_wave"][rec.index], np.float32)
        env = np.asarray([g["temperature"][rec.index], g["humidity"][rec.index], np.log1p(max(float(g["brightness"][rec.index]), 0.0)), g["pressure"][rec.index], g["weather_tag"][rec.index], (rec.timestamp_ns / 1e9) / 86400.0], np.float32)
        if self.stats:
            wave = (wave - np.asarray(self.stats["wave_mean"], np.float32)[:, None]) / (np.asarray(self.stats["wave_std"], np.float32)[:, None] + 1e-6)
            env = (env - np.asarray(self.stats["env_mean"], np.float32)) / (np.asarray(self.stats["env_std"], np.float32) + 1e-6)
        return {"wave": torch.from_numpy(wave), "environment": torch.from_numpy(env), "damage_tag": torch.tensor(rec.damage_tag), "timestamp_ns": torch.tensor(rec.timestamp_ns)}

