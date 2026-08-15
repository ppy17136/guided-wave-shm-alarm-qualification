"""Resume the final P7 month using a browser-issued Figshare WAF token."""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/raw/measurements_2022_10.pickle"
PARTIAL = TARGET.with_suffix(TARGET.suffix + ".part")
EXPECTED_SIZE = 1_005_592_104
EXPECTED_MD5 = "a02f3ad17413fcad66b3e40cef63008b"
ORIGIN = "https://springernature.figshare.com/ndownloader/files/51426458"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    token = os.environ.get("FIGSHARE_WAF_TOKEN")
    if not token:
        raise SystemExit("FIGSHARE_WAF_TOKEN is required")
    if TARGET.is_file():
        if TARGET.stat().st_size == EXPECTED_SIZE and md5(TARGET) == EXPECTED_MD5:
            print("verified existing: measurements_2022_10.pickle", flush=True)
            return
        raise SystemExit("Existing October final file failed integrity check")
    start = PARTIAL.stat().st_size if PARTIAL.exists() else 0
    if start > EXPECTED_SIZE:
        raise SystemExit("October partial exceeds official size")
    request = urllib.request.Request(ORIGIN, headers={
        "User-Agent": "Mozilla/5.0",
        "Cookie": f"aws-waf-token={token}",
        "Range": f"bytes={start}-",
    })
    print(f"resume={start}/{EXPECTED_SIZE}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        if start and response.status != 206:
            raise RuntimeError(f"Expected HTTP 206 for resume, received {response.status}")
        with PARTIAL.open("ab" if start else "wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    actual_size = PARTIAL.stat().st_size
    print(f"downloaded_bytes={actual_size}", flush=True)
    if actual_size != EXPECTED_SIZE:
        raise RuntimeError(f"Size mismatch: {actual_size} != {EXPECTED_SIZE}")
    actual_md5 = md5(PARTIAL)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: {actual_md5} != {EXPECTED_MD5}")
    os.replace(PARTIAL, TARGET)
    print("P7_OCTOBER_DOWNLOAD_COMPLETE_AND_MD5_VERIFIED", flush=True)


if __name__ == "__main__":
    main()

