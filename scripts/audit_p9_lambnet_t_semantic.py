#!/usr/bin/env python3
"""Semantic pair audit for exact duplicate LambNet-T measurements."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_RE = re.compile(r"^P(?P<a>\d{2})-(?P<b>\d{2})_T(?P<t>-?\d+)C$")
PAIRS = [
    ("40kHz_I1", "01_Baseline_40kHz_I1.csv", "01_Diagnostic_40kHz_I1.csv"),
    ("50kHz_I1", "02_Baseline_50kHz_I1.csv", "02_Diagnostic_50kHz_I1.csv"),
    ("40kHz_I2", "03_Baseline_40kHz_I2.csv", "03_Diagnostic_40kHz_I2.csv"),
    ("50kHz_I2", "04_Baseline_50kHz_I2.csv", "04_Diagnostic_50kHz_I2.csv"),
]


def find(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_results = []
    column_rows = []
    for task, baseline_name, diagnostic_name in PAIRS:
        baseline = pd.read_csv(find(args.raw_root, baseline_name))
        diagnostic = pd.read_csv(find(args.raw_root, diagnostic_name))
        headers_identical = baseline.columns.tolist() == diagnostic.columns.tolist()
        shapes_identical = baseline.shape == diagnostic.shape
        if not headers_identical or not shapes_identical:
            raise ValueError(f"Cannot compare malformed pair {task}")
        base_values = baseline.to_numpy(dtype=np.float64, copy=False)
        diagnostic_values = diagnostic.to_numpy(dtype=np.float64, copy=False)
        exact_by_column = np.all(base_values == diagnostic_values, axis=0)
        max_abs_by_column = np.max(np.abs(base_values - diagnostic_values), axis=0)
        fully_identical_temperatures = []
        for temperature in range(-10, 51, 5):
            indices = []
            for index, column in enumerate(baseline.columns):
                match = COLUMN_RE.match(column)
                if match and int(match.group("t")) == temperature:
                    indices.append(index)
            if len(indices) != 40:
                raise ValueError(f"Expected 40 paths at {temperature} C in {task}")
            if bool(np.all(exact_by_column[indices])):
                fully_identical_temperatures.append(temperature)
        for index, column in enumerate(baseline.columns):
            match = COLUMN_RE.match(column)
            column_rows.append(
                {
                    "task": task,
                    "column": column,
                    "temperature_C": int(match.group("t")) if match else None,
                    "exactly_identical": bool(exact_by_column[index]),
                    "max_abs_difference_mV": float(max_abs_by_column[index]),
                }
            )
        pair_results.append(
            {
                "task": task,
                "shape": list(baseline.shape),
                "headers_identical": headers_identical,
                "exactly_identical_columns": int(np.sum(exact_by_column)),
                "fully_identical_temperatures_C": fully_identical_temperatures,
                "semantic_duplicate_warning": bool(fully_identical_temperatures),
            }
        )

    summary = {
        "schema_version": "p9-lambnet-t-semantic-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": pair_results,
        "semantic_duplicate_warning": any(
            item["semantic_duplicate_warning"] for item in pair_results
        ),
        "interpretation": (
            "I2 at 45 C and 50 C is an exact baseline/diagnostic duplicate at both frequencies. "
            "Retain and flag in the full analysis; treat exclusion only as a separately "
            "reported missing-measurement sensitivity analysis."
        ),
    }
    (args.output_dir / "lambnet_t_semantic_pair_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(column_rows).to_csv(
        args.output_dir / "lambnet_t_pair_column_differences.csv", index=False
    )
    report = [
        "# LambNet-T 基线/诊断语义重复审计",
        "",
        "| 任务 | 完全相同列数 | 完全相同的整个温度 | 警告 |",
        "|---|---:|---|---|",
    ]
    for item in pair_results:
        temperatures = ", ".join(map(str, item["fully_identical_temperatures_C"])) or "无"
        report.append(
            f"| {item['task']} | {item['exactly_identical_columns']} | {temperatures} | "
            f"{'是' if item['semantic_duplicate_warning'] else '否'} |"
        )
    report.extend(
        [
            "",
            "I2在45 ℃和50 ℃时，40 kHz和50 kHz的40条路径Baseline与Diagnostic均逐点完全相同。",
            "全量分析必须保留并标记该温度；排除它只能作为‘疑似缺失/占位记录’敏感性分析，不能替代主结果。",
        ]
    )
    (args.output_dir / "lambnet_t_semantic_pair_audit.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
