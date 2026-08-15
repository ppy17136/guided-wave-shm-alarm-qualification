#!/usr/bin/env python3
"""Audit the Mendeley pipeline guided-wave XLS archive without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import PurePosixPath, Path

import numpy as np
import pandas as pd
import xlrd


FILE_RE = re.compile(r"G4-288-(?P<number>\d+)-V-0-waveform\.xls$", re.I)
FREQUENCY_RE = re.compile(r"(?P<frequency>\d+(?:\.\d+)?)kHz", re.I)
WINDOW_RE = re.compile(r"Window-(?P<window>\d+)\.sC\.raw", re.I)
EXPECTED_FREQUENCIES = {14.0, 18.0, 24.0, 30.0, 37.0}
EXPECTED_HEADERS = ["distance [m]", "torsional [V]", "flexural [V]", "pure DAC [V]"]


def hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def contiguous_runs(numbers: list[int], labels: dict[int, str]) -> list[dict]:
    runs = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number != previous + 1 or labels[number] != labels[previous]:
            runs.append(
                {
                    "label": labels[previous],
                    "start_file_number": start,
                    "end_file_number": previous,
                    "files": previous - start + 1,
                }
            )
            start = number
        previous = number
    runs.append(
        {
            "label": labels[previous],
            "start_file_number": start,
            "end_file_number": previous,
            "files": previous - start + 1,
        }
    )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    signal_hash_locations: dict[str, list[str]] = defaultdict(list)
    distance_hashes = Counter()
    sheet_shapes = Counter()
    sheet_orders = Counter()
    text_hits = []
    zip_total_uncompressed = 0

    with zipfile.ZipFile(args.archive) as archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        xls_infos = [item for item in infos if item.filename.lower().endswith(".xls")]
        if len(xls_infos) != 236:
            raise ValueError(f"Expected 236 XLS files, found {len(xls_infos)}")

        parsed_infos = []
        for info in xls_infos:
            match = FILE_RE.search(PurePosixPath(info.filename).name)
            if match is None:
                raise ValueError(f"Unexpected XLS filename: {info.filename}")
            parsed_infos.append((int(match.group("number")), info))
        parsed_infos.sort(key=lambda item: item[0])

        for sequence, (number, info) in enumerate(parsed_infos, start=1):
            payload = archive.read(info)
            zip_total_uncompressed += len(payload)
            workbook = xlrd.open_workbook(file_contents=payload, on_demand=True)
            frequencies = []
            window_ids = []
            file_finite = True
            file_header_ok = True
            file_distance_axis_expected = True
            file_signal_hashes = []
            torsional_rms = []
            flexural_rms = []
            dac_nonzero_fraction = []

            for sheet_name in workbook.sheet_names():
                sheet = workbook.sheet_by_name(sheet_name)
                sheet_shapes[(sheet.nrows, sheet.ncols)] += 1
                frequency_match = FREQUENCY_RE.search(sheet_name)
                if frequency_match is None:
                    raise ValueError(f"Missing frequency in {number}: {sheet_name}")
                frequency = float(frequency_match.group("frequency"))
                frequencies.append(frequency)

                header = [str(sheet.cell_value(1, column)).strip() for column in range(sheet.ncols)]
                file_header_ok &= header == EXPECTED_HEADERS
                window_text = str(sheet.cell_value(0, 0)).strip()
                window_match = WINDOW_RE.fullmatch(window_text)
                window_ids.append(int(window_match.group("window")) if window_match else -1)

                values = np.asarray(
                    [sheet.row_values(row, start_colx=0, end_colx=4) for row in range(2, sheet.nrows)],
                    dtype=np.float64,
                )
                finite = bool(np.isfinite(values).all())
                file_finite &= finite
                distance = values[:, 0]
                differences = np.diff(distance)
                duplicated_zero = np.flatnonzero(differences == 0)
                file_distance_axis_expected &= bool(
                    np.all(differences >= 0)
                    and duplicated_zero.size == 1
                    and distance[duplicated_zero[0]] == 0
                    and distance[duplicated_zero[0] + 1] == 0
                )
                distance_hash = hash_array(distance)
                distance_hashes[distance_hash] += 1
                signal_hash = hash_array(values[:, 1:3])
                signal_hash_locations[signal_hash].append(f"{number}:{frequency:g}kHz")
                file_signal_hashes.append(signal_hash)
                torsional_rms.append(float(np.sqrt(np.mean(values[:, 1] ** 2))))
                flexural_rms.append(float(np.sqrt(np.mean(values[:, 2] ** 2))))
                dac_nonzero_fraction.append(float(np.mean(values[:, 3] != 0)))

                for row_id in range(min(2, sheet.nrows)):
                    for column_id in range(sheet.ncols):
                        value = sheet.cell_value(row_id, column_id)
                        if isinstance(value, str) and re.search(
                            r"temp|temperature|degree|celsius|damage|defect|corrosion|stage",
                            value,
                            flags=re.I,
                        ):
                            text_hits.append(
                                {
                                    "file_number": number,
                                    "sheet": sheet_name,
                                    "row": row_id,
                                    "column": column_id,
                                    "text": value,
                                }
                            )

            sheet_orders[tuple(frequencies)] += 1
            label = "healthy" if number <= 1901 else "damage"
            rows.append(
                {
                    "sequence": sequence,
                    "file_number": number,
                    "label_from_official_boundary": label,
                    "filename": info.filename,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "xls_sha256": hashlib.sha256(payload).hexdigest(),
                    "sheets": len(workbook.sheet_names()),
                    "frequencies_kHz": ",".join(f"{value:g}" for value in sorted(frequencies)),
                    "frequency_set_expected": set(frequencies) == EXPECTED_FREQUENCIES,
                    "sheet_order": ",".join(f"{value:g}" for value in frequencies),
                    "window_ids": ",".join(map(str, window_ids)),
                    "one_window_id_across_sheets": len(set(window_ids)) == 1,
                    "header_expected": file_header_ok,
                    "all_finite": file_finite,
                    "distance_axis_expected": file_distance_axis_expected,
                    "one_signal_hash_per_frequency": len(set(file_signal_hashes)) == len(file_signal_hashes),
                    "workbook_signal_signature": hashlib.sha256(
                        "|".join(sorted(file_signal_hashes)).encode("ascii")
                    ).hexdigest(),
                    "torsional_rms_min": min(torsional_rms),
                    "torsional_rms_max": max(torsional_rms),
                    "flexural_rms_min": min(flexural_rms),
                    "flexural_rms_max": max(flexural_rms),
                    "dac_nonzero_fraction_min": min(dac_nonzero_fraction),
                    "dac_nonzero_fraction_max": max(dac_nonzero_fraction),
                }
            )
            workbook.release_resources()

    frame = pd.DataFrame(rows)
    labels = dict(zip(frame.file_number.astype(int), frame.label_from_official_boundary))
    numbers = frame.file_number.astype(int).tolist()
    runs = pd.DataFrame(contiguous_runs(numbers, labels))
    present = set(numbers)
    missing_numbers = sorted(set(range(min(numbers), max(numbers) + 1)) - present)
    duplicate_signals = {
        digest: locations
        for digest, locations in signal_hash_locations.items()
        if len(locations) > 1
    }

    duplicate_workbooks = []
    for signature, group in frame.groupby("workbook_signal_signature", sort=False):
        if len(group) > 1:
            duplicate_workbooks.append(
                {
                    "workbook_signal_signature": signature,
                    "copies": int(len(group)),
                    "file_numbers": ";".join(map(str, group.file_number.astype(int))),
                    "labels": ";".join(group.label_from_official_boundary),
                }
            )

    frame.to_csv(args.output_dir / "pipeline_xls_index.csv", index=False)
    runs.to_csv(args.output_dir / "pipeline_contiguous_runs.csv", index=False)
    pd.DataFrame(text_hits).to_csv(args.output_dir / "pipeline_metadata_text_hits.csv", index=False)
    duplicate_rows = [
        {"signal_sha256": digest, "copies": len(locations), "locations": ";".join(locations)}
        for digest, locations in duplicate_signals.items()
    ]
    pd.DataFrame(duplicate_rows).to_csv(
        args.output_dir / "pipeline_exact_duplicate_signals.csv", index=False
    )
    pd.DataFrame(duplicate_workbooks).to_csv(
        args.output_dir / "pipeline_exact_duplicate_workbooks.csv", index=False
    )

    checks = {
        "exactly_236_xls_files": len(frame) == 236,
        "official_207_healthy_29_damage_boundary": (
            int((frame.label_from_official_boundary == "healthy").sum()) == 207
            and int((frame.label_from_official_boundary == "damage").sum()) == 29
        ),
        "five_expected_frequencies_every_file": bool(frame.frequency_set_expected.all()),
        "shape_2509_by_4_every_sheet": set(sheet_shapes) == {(2509, 4)},
        "expected_headers_every_file": bool(frame.header_expected.all()),
        "finite_every_file": bool(frame.all_finite.all()),
        "distance_nondecreasing_with_one_duplicated_zero_every_file": bool(frame.distance_axis_expected.all()),
        "one_common_distance_axis": len(distance_hashes) == 1,
        "one_window_id_across_frequencies": bool(frame.one_window_id_across_sheets.all()),
        "no_exact_duplicate_frequency_signals": len(duplicate_signals) == 0,
        "no_explicit_temperature_or_damage_metadata_in_headers": len(text_hits) == 0,
        "dac_column_all_zero_every_file": bool((frame.dac_nonzero_fraction_max == 0).all()),
    }
    summary = {
        "schema_version": "p10-pipeline-dataset-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://data.mendeley.com/datasets/ttb63krg6d/1",
        "doi": "10.17632/ttb63krg6d.1",
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "zip_uncompressed_bytes": zip_total_uncompressed,
        "xls_files": len(frame),
        "file_number_range": [min(numbers), max(numbers)],
        "missing_file_numbers_count": len(missing_numbers),
        "missing_file_numbers": missing_numbers,
        "labels": frame.label_from_official_boundary.value_counts().to_dict(),
        "contiguous_runs": int(len(runs)),
        "sheet_shapes": {f"{key[0]}x{key[1]}": value for key, value in sheet_shapes.items()},
        "sheet_order_variants": {"|".join(map(str, key)): value for key, value in sheet_orders.items()},
        "distance_axis_unique_hashes": len(distance_hashes),
        "distance_axis_occurrences": int(sum(distance_hashes.values())),
        "exact_duplicate_signal_groups": len(duplicate_signals),
        "exact_duplicate_workbook_groups": len(duplicate_workbooks),
        "paper_vs_archive_shape_discrepancy": {
            "paper_text": "2057 x 4 matrix",
            "archive_workbook_shape": "2509 x 4 including two header rows",
            "archive_numeric_shape": "2507 x 4",
        },
        "explicit_temperature_or_damage_text_hits": len(text_hits),
        "window_id_unique": sorted(
            {
                int(value)
                for values in frame.window_ids
                for value in str(values).split(",")
                if int(value) >= 0
            }
        ),
        "checks": checks,
        "passed_structural_audit": bool(all(checks.values())),
        "semantic_limitations": [
            "The paper reports 2057 x 4, whereas all archive sheets are 2509 x 4 including two header rows (2507 x 4 numeric).",
            "No per-file temperature value was found in filenames, sheet names, or header text.",
            "The six damage stages are not explicitly encoded in filenames or workbook headers.",
            "Consecutive files may be technical repeats; independent grouping must be frozen before modelling.",
        ],
    }
    (args.output_dir / "pipeline_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = f"""# P10钢管导波数据结构与语义审计

- ZIP：{summary['archive_bytes']:,}字节；CRC已在下载阶段通过。
- XLS：{summary['xls_files']}个；健康207、损伤29（按官方1901/1902边界）。
- 文件号：{min(numbers)}–{max(numbers)}，缺号{len(missing_numbers)}个，形成{len(runs)}个连续批次。
- 每文件：5个频率，全部为14/18/24/30/37 kHz；每表2509×4，含2507个数值行。
- 论文正文写2057×4，与全部原始工作表的2509×4不一致；这是数据描述中的形状笔误，不据此裁剪样本。
- 距离轴：{len(distance_hashes)}种精确哈希，共检查{sum(distance_hashes.values())}个工作表。
- 精确重复频率信号组：{len(duplicate_signals)}；对应完整工作簿重复组：{len(duplicate_workbooks)}。
- 温度/损伤阶段显式文本命中：{len(text_hits)}。
- 结构审计通过：{'是' if summary['passed_structural_audit'] else '否'}。

## 关键语义限制

数据论文说明环境温度约19–26 ℃、末日分6步制造损伤，但ZIP中的文件名、工作表名和表头没有逐文件温度值，也没有六级损伤阶段标签。现阶段只能可靠使用官方健康/损伤总边界和文件时序；不得自行按每5个文件强行推断损伤等级，也不得声称进行有监督温度回归。

建模前必须把连续采集文件视作潜在技术重复，采用成块时间切分；DAC列只作为显示校正曲线，不作为独立传感通道。
"""
    (args.output_dir / "pipeline_audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
