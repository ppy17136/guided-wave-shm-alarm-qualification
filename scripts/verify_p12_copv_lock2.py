from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research_protocols" / "P12_COPV_LOCK2_execution_freeze_v1.json"
LOCK_SHA = ROOT / "research_protocols" / "P12_COPV_LOCK2_SHA256_v1.txt"
READY = ROOT / "P12_COPV_LOCK2_READY.ok"


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-archives", action="store_true")
    parser.add_argument("--require-ready-marker", action="store_true")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected_lock_sha = LOCK_SHA.read_text(encoding="ascii").split()[0]
    checks = {"lock2_sha256": digest(LOCK) == expected_lock_sha}
    for item in lock["frozen_files"]:
        path = ROOT / item["path"]
        checks[f"file:{item['path']}"] = (
            path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
        )
    archive_results = []
    for item in lock["input_archives"]:
        path = ROOT / "_references" / "P12_COPV_SEALED" / item["name"]
        result = {"name": item["name"], "present": path.is_file(),
                  "bytes_match": path.is_file() and path.stat().st_size == item["bytes"]}
        if args.full_archives and result["bytes_match"]:
            result["sha256"] = digest(path)
            result["sha256_match"] = result["sha256"] == item["sha256"]
        elif args.full_archives:
            result["sha256_match"] = False
        archive_results.append(result)
        checks[f"archive:{item['name']}:bytes"] = result["bytes_match"]
        if args.full_archives:
            checks[f"archive:{item['name']}:sha256"] = result["sha256_match"]
    if args.require_ready_marker:
        checks["ready_marker"] = READY.is_file()
    passed = all(checks.values())
    report = {"schema_version": "p12-copv-lock2-verification-v1", "passed": passed,
              "full_archive_hashing": args.full_archives, "checks": checks,
              "archives": archive_results, "lock2_sha256": expected_lock_sha}
    report_path = ROOT / "data" / "reports" / "p12_copv_lock2_cluster_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if passed and args.full_archives:
        READY.write_text(expected_lock_sha + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

