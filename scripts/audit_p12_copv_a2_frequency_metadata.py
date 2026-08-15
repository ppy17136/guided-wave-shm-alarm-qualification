from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "baseline": ROOT / "_references/P12_COPV_SEALED/Baseline.zip",
    "irreversible": ROOT / "_references/P12_COPV_SEALED/Irreversible_Damage.zip",
    "reversible": ROOT / "_references/P12_COPV_SEALED/Reversible_Damage.zip",
}
OUTPUT = ROOT / "data/reports/p12_copv_a2_frequency_metadata_audit_v1.json"


def main() -> None:
    records = []
    with tempfile.TemporaryDirectory(prefix="p12_a2_metadata_") as temporary:
        directory = Path(temporary)
        for role, archive_path in ARCHIVES.items():
            with zipfile.ZipFile(archive_path, "r") as archive:
                member = next(name for name in archive.namelist() if name.lower().endswith(".h5"))
                target = directory / f"{role}.h5"
                with archive.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 16 * 1024 * 1024)
            with h5py.File(target, "r") as handle:
                index = np.asarray(handle["MetaData/Index_FrequencyvsRepetition"][...], dtype=np.float64)
                excitation = np.asarray(handle["MetaData/Signal_Frequency_Burst"][...], dtype=np.float64).reshape(-1)
                chirp = np.asarray(handle["MetaData/Index_ChirpvsRepetition"][...], dtype=np.float64)
                sampling = np.asarray(handle["MetaData/Sampling_Frequency"][...], dtype=np.float64).reshape(-1)
                records.append({
                    "role": role, "member": member,
                    "index_frequency_vs_repetition": index.tolist(),
                    "signal_frequency_burst_hz": excitation.tolist(),
                    "index_chirp_vs_repetition": chirp.tolist(),
                    "sampling_frequency_hz": sampling.tolist(),
                    "raw_dataset_shape_only": list(handle["Data/Raw_Data"].shape),
                    "raw_numeric_values_read": False,
                })
            target.unlink()
    expected_index_frequencies = [60_000] * 3 + [120_000] * 3 + [180_000] * 3 + [240_000] * 3 + [300_000] * 3
    expected_excitation = [60_000, 120_000, 180_000, 260_000, 300_000]
    checks = {
        "three_archives_audited": len(records) == 3,
        "all_index_rows_have_240_fourth_group": all(
            np.rint(item["index_frequency_vs_repetition"][1]).astype(int).tolist() == expected_index_frequencies
            for item in records
        ),
        "all_excitation_lists_have_260_fourth_frequency": all(
            np.rint(item["signal_frequency_burst_hz"]).astype(int).tolist() == expected_excitation
            for item in records
        ),
        "all_burst_indices_are_1_to_15": all(
            np.rint(item["index_frequency_vs_repetition"][0]).astype(int).tolist() == list(range(1, 16))
            for item in records
        ),
        "all_chirp_indices_are_16_to_18": all(
            np.rint(item["index_chirp_vs_repetition"][0]).astype(int).tolist() == [16, 17, 18]
            for item in records
        ),
        "raw_numeric_values_never_read": all(not item["raw_numeric_values_read"] for item in records),
    }
    report = {"schema_version": "p12-copv-a2-frequency-metadata-audit-v1",
              "status": "pass" if all(checks.values()) else "fail", "checks": checks, "records": records}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
