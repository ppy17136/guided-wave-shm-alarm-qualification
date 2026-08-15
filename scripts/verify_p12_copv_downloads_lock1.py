from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "_references" / "P12_COPV_SEALED"
EXPECTED = {
    "Baseline.zip": {
        "bytes": 20_109_358_425,
        "md5": "c57efcd712d82c9f735dce969176699e",
        "role": "required_primary",
    },
    "Irreversible_Damage.zip": {
        "bytes": 20_276_822_123,
        "md5": "5bb46c4e92d5e873f2936e706001c366",
        "role": "required_primary",
    },
    "Reversible_Damage.zip": {
        "bytes": 20_410_262_593,
        "md5": "928e331d7b8fd4b3b2ca2189d57c5f79",
        "role": "optional_secondary",
    },
}


def hash_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify P12 COPV sealed downloads without opening ZIP archives."
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--require-secondary", action="store_true")
    args = parser.parse_args()

    directory = args.directory.resolve()
    results = []
    for name, expected in EXPECTED.items():
        path = directory / name
        item = {
            "name": name,
            "role": expected["role"],
            "path": str(path),
            "present": path.is_file(),
            "expected_bytes": expected["bytes"],
            "expected_md5": expected["md5"],
        }
        if path.is_file():
            size = path.stat().st_size
            item["actual_bytes"] = size
            item["bytes_match"] = size == expected["bytes"]
            if item["bytes_match"]:
                md5, sha256 = hash_file(path)
                item.update({
                    "actual_md5": md5,
                    "md5_match": md5 == expected["md5"],
                    "sha256": sha256,
                    "verified": md5 == expected["md5"],
                })
            else:
                item.update({"md5_match": False, "verified": False})
        else:
            item["verified"] = False
        results.append(item)

    required = [r for r in results if r["role"] == "required_primary"]
    secondary = next(r for r in results if r["role"] == "optional_secondary")
    passed = all(r["verified"] for r in required)
    if args.require_secondary:
        passed = passed and secondary["verified"]

    report = {
        "schema_version": "p12-copv-sealed-download-verification-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "directory": str(directory),
        "zip_archives_opened": False,
        "numeric_data_read": False,
        "required_primary_pass": all(r["verified"] for r in required),
        "secondary_present_and_verified": bool(secondary["verified"]),
        "overall_pass": passed,
        "files": results,
    }

    if args.write_report:
        output = ROOT / "data" / "reports" / "p12_copv_download_verification_lock1.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report["report_path"] = str(output)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
