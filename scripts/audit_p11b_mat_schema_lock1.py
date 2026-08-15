from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
from scipy.io import whosmat


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_references" / "wind_turbine_blade_shm_dataset.zip"
OUT = ROOT / "data" / "reports" / "p11b_schema_audit_lock1"
TMP = Path(os.environ.get("TEMP", ".")) / "p11b_schema_mat"

TARGETS = {
    "static_representative": "wind_turbine_blade_shm_dataset/shaker_gw_sync_3/averaged/Loading_cycle_600000/5_cycles_20kHz/niscope_avg_waveform.mat",
    "dynamic_representative": "wind_turbine_blade_shm_dataset/shaker_gw_sync_3/raw/Loading_cycle_600000/5_cycles_50kHz/niscope_waveforms.mat",
    "strain_temperature": "wind_turbine_blade_shm_dataset/strains_temperature_curated.mat",
}


def matlab_header(path: Path) -> str:
    with path.open("rb") as stream:
        return stream.read(128).decode("latin-1", errors="replace").rstrip("\x00")


def inspect_schema(path: Path) -> dict:
    header = matlab_header(path)
    result = {"header": header, "bytes": path.stat().st_size}
    if h5py.is_hdf5(path):
        datasets: list[dict] = []
        groups: list[str] = []
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj) -> None:
                if isinstance(obj, h5py.Dataset):
                    datasets.append({
                        "name": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "chunks": list(obj.chunks) if obj.chunks else None,
                    })
                elif isinstance(obj, h5py.Group):
                    groups.append(name)
            handle.visititems(visitor)
        result.update({"format": "matlab_v7_3_hdf5", "datasets": datasets, "groups": groups})
    else:
        variables = [
            {"name": name, "shape": list(shape), "class": cls}
            for name, shape, cls in whosmat(path)
        ]
        result.update({"format": "matlab_pre_v7_3", "variables": variables})
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "p11b-mat-schema-audit-lock1-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "mat_headers_names_shapes_and_dtypes_only",
        "numeric_values_loaded": False,
        "numeric_summaries_computed": False,
        "targets": {},
    }

    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        for label, member in TARGETS.items():
            if member not in names:
                raise FileNotFoundError(member)
            local = TMP / f"{label}.mat"
            with archive.open(member, "r") as source, local.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
            info = inspect_schema(local)
            info["archive_path"] = member
            result["targets"][label] = info

    (OUT / "mat_schema_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
