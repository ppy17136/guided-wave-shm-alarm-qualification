from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".csv", ".cff", ".txt", ""}
RAW_SUFFIXES = {".h5", ".hdf5", ".mat", ".zarr", ".pt", ".pth", ".ckpt", ".zip", ".tar", ".gz", ".7z"}
PRIVATE_PATTERNS = ("/" + "public/home/", "Users" + "\\BLL", "Users" + "/BLL", "lbl" + "20020033")
SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "generic_api_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if path.is_file())
    errors: list[dict[str, str]] = []
    counts = {"python": 0, "json": 0, "yaml": 0, "csv": 0, "pdf": 0, "png": 0}
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        suffix = path.suffix.lower()
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append({"file": rel, "error": "file exceeds 5 MiB"})
        if suffix in RAW_SUFFIXES:
            errors.append({"file": rel, "error": "raw-data, model, or archive suffix is excluded"})
        try:
            if suffix == ".py":
                counts["python"] += 1
                ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            elif suffix == ".json":
                counts["json"] += 1
                json.loads(path.read_text(encoding="utf-8-sig"))
            elif suffix in {".yaml", ".yml", ".cff"}:
                counts["yaml"] += 1
                yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            elif suffix == ".csv":
                counts["csv"] += 1
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    if not list(csv.reader(handle)):
                        raise ValueError("empty CSV")
            elif suffix == ".pdf":
                counts["pdf"] += 1
                if path.read_bytes()[:5] != b"%PDF-":
                    raise ValueError("bad PDF signature")
            elif suffix == ".png":
                counts["png"] += 1
                if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("bad PNG signature")
        except Exception as exc:
            errors.append({"file": rel, "error": repr(exc)})

        if suffix in TEXT_SUFFIXES and path.name != "SHA256SUMS.txt":
            content = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in PRIVATE_PATTERNS:
                if pattern in content:
                    errors.append({"file": rel, "error": f"private path marker: {pattern}"})
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(content):
                    errors.append({"file": rel, "error": f"secret marker: {label}"})

    checksum_errors = []
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, rel = line.split("  ", 1)
        target = ROOT / rel
        if not target.is_file() or sha256(target) != expected:
            checksum_errors.append(rel)
    errors.extend({"file": rel, "error": "SHA-256 mismatch"} for rel in checksum_errors)
    report = {
        "schema_version": "shm-public-static-validation-v1",
        "status": "pass" if not errors else "fail",
        "python": sys.version.split()[0],
        "file_count": len(files),
        "counts": counts,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

