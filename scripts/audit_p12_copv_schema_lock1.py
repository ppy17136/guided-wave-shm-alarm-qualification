from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "_references" / "P12_COPV_SEALED"
OUT = ROOT / "data" / "reports" / "p12_copv_schema_audit_lock1"
ARCHIVES = [
    ("baseline", SEALED / "Baseline.zip"),
    ("irreversible", SEALED / "Irreversible_Damage.zip"),
    ("reversible", SEALED / "Reversible_Damage.zip"),
]
H5_PATTERN = re.compile(
    r"(?P<date>\d{2}-\d{2}-\d{2})[ _](?P<time>\d{2}-\d{2}-\d{2}).*?T(?P<temp>\d+)[ _](?P<pressure>\d+)bar\.h5$",
    re.IGNORECASE,
)


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"python_type": type(value).__name__}


def inspect_h5_schema(path: Path) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            item: dict[str, Any] = {
                "name": name,
                "object_type": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                "attribute_names": sorted(str(key) for key in obj.attrs.keys()),
                "attribute_types_only": {
                    str(key): json_safe(obj.attrs.get_id(key).dtype)
                    for key in obj.attrs.keys()
                },
            }
            if isinstance(obj, h5py.Dataset):
                item.update({
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                    "compression_options": json_safe(obj.compression_opts),
                    "fillvalue_type_only": type(obj.fillvalue).__name__,
                })
            objects.append(item)

        handle.visititems(visitor)
        root_attributes = {
            "names": sorted(str(key) for key in handle.attrs.keys()),
            "types_only": {
                str(key): json_safe(handle.attrs.get_id(key).dtype)
                for key in handle.attrs.keys()
            },
        }
    return {
        "raw_numeric_values_read": False,
        "root_attributes": root_attributes,
        "objects": objects,
    }


def parse_h5_member(archive_role: str, name: str, size: int, compressed: int, crc: int) -> dict[str, Any]:
    base = Path(name).name
    match = H5_PATTERN.search(base)
    row: dict[str, Any] = {
        "archive": archive_role,
        "path": name,
        "file_size": size,
        "compressed_size": compressed,
        "crc32": f"{crc:08x}",
        "timestamp": None,
        "nominal_temperature_c": None,
        "nominal_pressure_bar": None,
        "ramp_inferred": "unknown",
        "sequence_validated": False,
    }
    if match:
        row["timestamp"] = f"20{match.group('date')}T{match.group('time')}".replace("-", "", 2)
        row["nominal_temperature_c"] = int(match.group("temp"))
        row["nominal_pressure_bar"] = int(match.group("pressure"))
    return row


def infer_ramps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_descending = list(range(700, 49, -50)) + [20]
    # A1 schema mapping: the central directory shows that the randomized
    # sequence contains the additional 20 bar condition as well.
    expected_random = set(range(50, 701, 50)) | {20}
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["nominal_temperature_c"] is not None and row["timestamp"]:
            groups[int(row["nominal_temperature_c"])].append(row)
    validation = []
    for temp, group in sorted(groups.items()):
        group.sort(key=lambda item: str(item["timestamp"]))
        first = group[: len(expected_descending)]
        rest = group[len(expected_descending):]
        first_pressures = [item["nominal_pressure_bar"] for item in first]
        rest_pressures = [item["nominal_pressure_bar"] for item in rest]
        descending_ok = first_pressures == expected_descending
        random_ok = len(rest_pressures) == len(expected_random) and set(rest_pressures) == expected_random
        for item in first:
            item["ramp_inferred"] = "descending" if descending_ok else "unknown"
            item["sequence_validated"] = descending_ok
        for item in rest:
            item["ramp_inferred"] = "random" if random_ok else "unknown"
            item["sequence_validated"] = random_ok
        validation.append({
            "nominal_temperature_c": temp,
            "files": len(group),
            "descending_pressures": first_pressures,
            "random_pressures": rest_pressures,
            "descending_valid": descending_ok,
            "random_valid": random_ok,
        })
    return {
        "expected_descending_pressures": expected_descending,
        "expected_random_pressure_set": sorted(expected_random),
        "temperature_groups": validation,
        "all_sequences_valid": bool(validation) and all(
            item["descending_valid"] and item["random_valid"] for item in validation
        ),
    }


def choose_representative_h5(infos: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    h5 = [info for info in infos if not info.is_dir() and info.filename.lower().endswith(".h5")]
    preferred = [info for info in h5 if re.search(r"T25[ _]700bar\.h5$", info.filename, re.I)]
    if preferred:
        return sorted(preferred, key=lambda item: item.filename)[0]
    if not h5:
        raise RuntimeError("archive contains no H5 files")
    return sorted(h5, key=lambda item: item.filename)[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    temp_dir = OUT / "_temporary_representative_h5"
    temp_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    h5_rows: list[dict[str, Any]] = []
    archive_summaries = []
    schemas = []
    official_json_records = []

    try:
        for role, archive_path in ARCHIVES:
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            with zipfile.ZipFile(archive_path, "r") as archive:
                infos = archive.infolist()
                suffix_counts = Counter(
                    Path(info.filename).suffix.lower() or "<none>"
                    for info in infos if not info.is_dir()
                )
                for info in infos:
                    inventory_rows.append({
                        "archive": role,
                        "path": info.filename,
                        "is_directory": info.is_dir(),
                        "suffix": Path(info.filename).suffix.lower(),
                        "file_size": info.file_size,
                        "compressed_size": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                        "compression_type": info.compress_type,
                    })
                    if not info.is_dir() and info.filename.lower().endswith(".h5"):
                        h5_rows.append(parse_h5_member(
                            role, info.filename, info.file_size, info.compress_size, info.CRC
                        ))

                json_infos = [
                    info for info in infos
                    if not info.is_dir() and info.filename.lower().endswith(".json")
                ]
                for info in json_infos:
                    if info.file_size > 20 * 1024 * 1024:
                        raise RuntimeError(f"unexpectedly large official JSON: {info.filename}")
                    payload = archive.read(info)
                    record: dict[str, Any] = {
                        "archive": role,
                        "path": info.filename,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                    try:
                        parsed = json.loads(payload.decode("utf-8-sig"))
                        record["parsed"] = True
                        record["top_level_type"] = type(parsed).__name__
                        record["top_level_keys"] = sorted(parsed.keys()) if isinstance(parsed, dict) else None
                        record["content"] = parsed
                    except Exception as exc:
                        record.update({"parsed": False, "error": repr(exc)})
                    official_json_records.append(record)

                representative = choose_representative_h5(infos)
                extracted = temp_dir / f"{role}_representative.h5"
                with archive.open(representative, "r") as source, extracted.open("wb") as target:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                with extracted.open("rb") as stream:
                    extracted_sha256 = sha256_stream(stream)
                schema = inspect_h5_schema(extracted)
                schema.update({
                    "archive": role,
                    "member": representative.filename,
                    "member_file_size": representative.file_size,
                    "member_compressed_size": representative.compress_size,
                    "member_crc32": f"{representative.CRC:08x}",
                    "extracted_sha256": extracted_sha256,
                    "representative_file_deleted_after_schema_audit": True,
                })
                schemas.append(schema)
                extracted.unlink()

                archive_summaries.append({
                    "archive": role,
                    "path": str(archive_path),
                    "bytes": archive_path.stat().st_size,
                    "members": len(infos),
                    "files": sum(not info.is_dir() for info in infos),
                    "directories": sum(info.is_dir() for info in infos),
                    "total_uncompressed_bytes": sum(info.file_size for info in infos),
                    "total_compressed_bytes": sum(info.compress_size for info in infos),
                    "suffix_counts": dict(sorted(suffix_counts.items())),
                    "h5_files": sum(info.filename.lower().endswith(".h5") for info in infos),
                    "json_files": len(json_infos),
                    "full_crc_scan_performed": False,
                    "central_directory_read_only": True,
                })
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    ramp_validation = {}
    for role, _ in ARCHIVES:
        role_rows = [row for row in h5_rows if row["archive"] == role]
        ramp_validation[role] = infer_ramps(role_rows)

    with (OUT / "archive_inventory.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(inventory_rows)
    with (OUT / "h5_file_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(h5_rows[0].keys()))
        writer.writeheader()
        writer.writerows(h5_rows)

    (OUT / "representative_h5_schema.json").write_text(
        json.dumps({
            "schema_version": "p12-copv-representative-h5-schema-v1",
            "raw_numeric_values_read": False,
            "schemas": schemas,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT / "official_json_records.json").write_text(
        json.dumps({
            "schema_version": "p12-copv-official-json-records-v1",
            "records": official_json_records,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "p12-copv-schema-audit-lock1-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "zip_archives_opened_for_central_directory": True,
        "official_json_read": True,
        "representative_h5_schema_only": True,
        "raw_numeric_values_read": False,
        "waveform_statistics_computed": False,
        "full_zip_crc_scan_performed": False,
        "archives": archive_summaries,
        "ramp_validation": ramp_validation,
        "representative_h5": [
            {
                "archive": item["archive"],
                "member": item["member"],
                "raw_numeric_values_read": item["raw_numeric_values_read"],
                "object_count": len(item["objects"]),
            }
            for item in schemas
        ],
        "official_json_files": len(official_json_records),
    }
    (OUT / "schema_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
