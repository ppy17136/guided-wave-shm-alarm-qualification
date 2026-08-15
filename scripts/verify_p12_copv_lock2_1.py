from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research_protocols/P12_COPV_LOCK2_1_execution_freeze_v1.json"
LOCK_SHA = ROOT / "research_protocols/P12_COPV_LOCK2_1_SHA256_v1.txt"
READY = ROOT / "P12_COPV_LOCK2_1_READY.ok"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-ready-marker", action="store_true")
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    expected = LOCK_SHA.read_text(encoding="ascii").split()[0]
    checks = {"lock2_1_sha256": digest(LOCK) == expected}
    for item in lock["frozen_files"]:
        path = ROOT / item["path"]
        checks[f"file:{item['path']}"] = path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    for item in lock["input_archives"]:
        path = ROOT / "_references/P12_COPV_SEALED" / item["name"]
        checks[f"archive:{item['name']}:bytes"] = path.is_file() and path.stat().st_size == item["bytes"]
    if args.require_ready_marker:
        checks["ready_marker"] = READY.is_file() and READY.read_text().strip() == expected
    passed = all(checks.values())
    report = {"schema_version": "p12-copv-lock2-1-verification-v1", "passed": passed,
              "lock2_1_sha256": expected, "checks": checks}
    path = ROOT / "data/reports/p12_copv_lock2_1_cluster_verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if passed and not args.require_ready_marker:
        READY.write_text(expected + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
