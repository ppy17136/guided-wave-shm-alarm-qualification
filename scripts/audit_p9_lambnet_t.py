#!/usr/bin/env python3
"""Audit the public LambNet-T CSV release without loading it all into memory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


HEADER_RE = re.compile(r"^P(?P<a>\d{2})-(?P<b>\d{2})_T(?P<t>-?\d+)C$")
FILE_RE = re.compile(
    r"^(?P<order>\d{2})_(?P<state>Baseline|Diagnostic)_"
    r"(?P<frequency>\d+)kHz_(?P<impact>I\d+)\.csv$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_csv(path: Path) -> dict:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected CSV filename: {path.name}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        parsed = []
        malformed_headers = []
        for name in header:
            item = HEADER_RE.match(name.strip())
            if item is None:
                malformed_headers.append(name)
            else:
                parsed.append(
                    (int(item.group("a")), int(item.group("b")), int(item.group("t")))
                )

        rows = 0
        short_rows = 0
        nonfinite = 0
        invalid_numeric = 0
        minimum = math.inf
        maximum = -math.inf
        sum_squares = 0.0
        values = 0

        for row in reader:
            rows += 1
            if len(row) != len(header):
                short_rows += 1
                continue
            for cell in row:
                try:
                    value = float(cell)
                except ValueError:
                    invalid_numeric += 1
                    continue
                if not math.isfinite(value):
                    nonfinite += 1
                    continue
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                sum_squares += value * value
                values += 1

    temperatures = sorted({item[2] for item in parsed})
    paths = sorted({(item[0], item[1]) for item in parsed})
    header_counts = Counter((item[0], item[1]) for item in parsed)
    result = {
        **match.groupdict(),
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": rows,
        "columns": len(header),
        "samples_expected": 6000,
        "paths": len(paths),
        "temperatures": temperatures,
        "temperature_count": len(temperatures),
        "malformed_header_count": len(malformed_headers),
        "malformed_headers": malformed_headers,
        "path_temperature_counts_unique": sorted(set(header_counts.values())),
        "row_width_mismatch_count": short_rows,
        "invalid_numeric_count": invalid_numeric,
        "nonfinite_count": nonfinite,
        "finite_value_count": values,
        "minimum_mV": minimum if values else None,
        "maximum_mV": maximum if values else None,
        "global_rms_mV": math.sqrt(sum_squares / values) if values else None,
        "header": header,
        "path_list": [f"{a:02d}-{b:02d}" for a, b in paths],
    }
    result["structural_checks"] = {
        "rows_6000": rows == 6000,
        "columns_520": len(header) == 520,
        "forty_paths": len(paths) == 40,
        "thirteen_temperatures": temperatures == list(range(-10, 51, 5)),
        "each_path_has_13_temperatures": set(header_counts.values()) == {13},
        "headers_parse": not malformed_headers,
        "row_widths_match": short_rows == 0,
        "all_values_finite_numeric": invalid_numeric == 0 and nonfinite == 0,
    }
    result["passed"] = all(result["structural_checks"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    csv_paths = sorted(args.raw_root.rglob("*.csv"))
    if len(csv_paths) != 8:
        raise SystemExit(f"Expected 8 CSV files, found {len(csv_paths)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audits = [audit_csv(path.resolve()) for path in csv_paths]

    pairs = {}
    for item in audits:
        key = f"{item['frequency']}kHz_{item['impact']}"
        pairs.setdefault(key, {})[item["state"]] = item

    pair_checks = {}
    for key, pair in sorted(pairs.items()):
        baseline = pair.get("Baseline")
        diagnostic = pair.get("Diagnostic")
        pair_checks[key] = {
            "has_baseline_and_diagnostic": baseline is not None and diagnostic is not None,
            "headers_identical": bool(
                baseline and diagnostic and baseline["header"] == diagnostic["header"]
            ),
            "paths_identical": bool(
                baseline
                and diagnostic
                and baseline["path_list"] == diagnostic["path_list"]
            ),
        }

    summary = {
        "schema_version": "p9-lambnet-t-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://data.mendeley.com/datasets/d9pzsm33gy/1",
        "doi": "10.17632/d9pzsm33gy.1",
        "license": "CC BY 4.0",
        "raw_root": str(args.raw_root.resolve()),
        "files": len(audits),
        "total_bytes": sum(item["bytes"] for item in audits),
        "all_files_pass": all(item["passed"] for item in audits),
        "all_pair_checks_pass": all(
            all(checks.values()) for checks in pair_checks.values()
        ),
        "pair_checks": pair_checks,
        "audits": audits,
    }

    json_path = args.output_dir / "lambnet_t_audit.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "lambnet_t_file_summary.csv"
    columns = [
        "order",
        "state",
        "frequency",
        "impact",
        "bytes",
        "rows",
        "columns",
        "paths",
        "temperature_count",
        "minimum_mV",
        "maximum_mV",
        "global_rms_mV",
        "sha256",
        "passed",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audits)

    markdown = [
        "# LambNet-T 数据完整性审计",
        "",
        f"- 数据文件：{len(audits)} 个 CSV",
        f"- CSV 总字节数：{summary['total_bytes']}",
        f"- 逐文件结构检查：{'通过' if summary['all_files_pass'] else '失败'}",
        f"- 基线/诊断配对检查：{'通过' if summary['all_pair_checks_pass'] else '失败'}",
        "",
        "| 文件 | 状态 | 频率/kHz | 冲击 | 形状 | 路径 | 温度 | RMS/mV | 通过 |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in audits:
        markdown.append(
            f"| {Path(item['file']).name} | {item['state']} | {item['frequency']} | "
            f"{item['impact']} | {item['rows']}×{item['columns']} | {item['paths']} | "
            f"{item['temperature_count']} | {item['global_rms_mV']:.6g} | "
            f"{'是' if item['passed'] else '否'} |"
        )
    markdown.extend(
        [
            "",
            "配对规则要求每个频率/冲击组合同时存在Baseline与Diagnostic，且列顺序完全相同。",
        ]
    )
    (args.output_dir / "lambnet_t_audit_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "json": str(json_path.resolve()),
                "csv": str(csv_path.resolve()),
                "files": len(audits),
                "all_files_pass": summary["all_files_pass"],
                "all_pair_checks_pass": summary["all_pair_checks_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
