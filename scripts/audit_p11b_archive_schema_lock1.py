from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_references" / "wind_turbine_blade_shm_dataset.zip"
OUT = ROOT / "data" / "reports" / "p11b_schema_audit_lock1"
DOCS = OUT / "official_docs_only"
ALLOWED_TEXT_SUFFIXES = {".txt", ".md", ".m"}
MAX_TEXT_BYTES = 2_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    extracted_docs: list[dict] = []
    ext_counts: Counter[str] = Counter()
    ext_bytes: Counter[str] = Counter()
    crc_groups: defaultdict[tuple[int, int], list[str]] = defaultdict(list)

    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC failure: {bad}")
        infos = archive.infolist()
        for index, info in enumerate(infos):
            suffix = Path(info.filename).suffix.lower() if not info.is_dir() else "<dir>"
            ext_counts[suffix] += 1
            ext_bytes[suffix] += info.file_size
            if not info.is_dir():
                crc_groups[(info.CRC, info.file_size)].append(info.filename)
            rows.append({
                "index": index,
                "path": info.filename,
                "is_directory": info.is_dir(),
                "extension": suffix,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
            })

            if (
                not info.is_dir()
                and suffix in ALLOWED_TEXT_SUFFIXES
                and info.file_size <= MAX_TEXT_BYTES
            ):
                safe_name = f"{index:06d}_{Path(info.filename).name}"
                target = DOCS / safe_name
                with archive.open(info, "r") as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                extracted_docs.append({
                    "archive_path": info.filename,
                    "local_path": str(target.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                })

    with (OUT / "archive_inventory.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    possible_duplicate_groups = [
        {"crc32": f"{crc:08x}", "bytes": size, "paths": paths}
        for (crc, size), paths in crc_groups.items()
        if len(paths) > 1
    ]
    summary = {
        "schema_version": "p11b-archive-schema-audit-lock1-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "central_directory_and_official_text_scripts_only_no_mat_numeric_content",
        "archive": str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256_file(ARCHIVE),
        "zip_crc_test": "pass",
        "entries": len(rows),
        "files": sum(not row["is_directory"] for row in rows),
        "directories": sum(row["is_directory"] for row in rows),
        "total_uncompressed_bytes": sum(row["uncompressed_bytes"] for row in rows),
        "extension_counts": dict(sorted(ext_counts.items())),
        "extension_uncompressed_bytes": dict(sorted(ext_bytes.items())),
        "official_text_or_script_files_extracted": extracted_docs,
        "possible_duplicate_groups_by_crc_and_size": possible_duplicate_groups,
        "numeric_mat_values_read": False,
        "numeric_signal_summaries_computed": False,
    }
    (OUT / "archive_schema_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
