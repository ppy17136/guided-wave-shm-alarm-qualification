"""Verify the P7 preregistration seal and, when present, official data integrity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research_protocols" / "P7_D7_D13_freeze_manifest_v1.json"
SEALED_MANIFEST_SHA256 = "4425939b94057d76cd2364be5048f265425e465fd625228343b1c328d1afd6df"


def digest(path: Path, algorithm: str, block: int = 16 * 1024 * 1024) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-future-data-absent",
        action="store_true",
        help="use only for reproducing the pre-download freeze-state audit",
    )
    args = parser.parse_args()

    checks: dict[str, object] = {}
    actual_manifest_hash = digest(MANIFEST, "sha256")
    checks["manifest_seal"] = {
        "expected": SEALED_MANIFEST_SHA256,
        "actual": actual_manifest_hash,
        "pass": actual_manifest_hash == SEALED_MANIFEST_SHA256,
    }
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    tracked = []
    for relative, expected in data["sha256"].items():
        path = ROOT / relative
        actual = digest(path, "sha256") if path.is_file() else None
        tracked.append({
            "path": relative,
            "expected": expected,
            "actual": actual,
            "pass": actual == expected,
        })
    checks["tracked_files"] = tracked

    official = []
    for entry in data["official_dataset"]["files"]:
        target = ROOT / "data" / "raw" / entry["name"].replace(" ", "_")
        if not target.is_file():
            official.append({"path": str(target.relative_to(ROOT)), "status": "absent"})
            continue
        size_ok = target.stat().st_size == int(entry["size"])
        actual_md5 = digest(target, "md5") if size_ok else None
        official.append({
            "path": str(target.relative_to(ROOT)),
            "status": "present",
            "size_pass": size_ok,
            "md5_expected": entry["md5"],
            "md5_actual": actual_md5,
            "md5_pass": actual_md5 == entry["md5"],
        })
    checks["official_future_files"] = official

    tracked_pass = all(item["pass"] for item in tracked)
    if args.require_future_data_absent:
        absent_pass = all(item["status"] == "absent" for item in official)
    else:
        absent_pass = True
    data_pass = all(
        item["status"] == "absent" or (item["size_pass"] and item["md5_pass"])
        for item in official
    )
    checks["require_future_data_absent"] = args.require_future_data_absent
    checks["future_data_absence_pass"] = absent_pass
    checks["official_data_integrity_pass"] = data_pass
    checks["passed"] = bool(checks["manifest_seal"]["pass"] and tracked_pass and absent_pass and data_pass)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

