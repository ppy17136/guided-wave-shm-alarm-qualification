from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "cluster_results"
    / "GW_P12_COPV_RESULTS_21006_extracted"
    / "runs"
    / "p12_copv_v1"
    / "01"
    / "p12_copv_confirmatory_summary.json"
)
FEATURE_DIR = SOURCE.parent / "features"
OUT = ROOT / "runs" / "p12_copv_reporting_v1"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
FREQUENCIES = ["60000", "120000", "180000", "260000", "300000"]
TEMPERATURES = [25, 31, 37, 43, 49, 55]
PRESSURES = list(range(50, 701, 50))
HELDOUT_HEALTHY_PRESSURES = {50, 150, 250, 350, 450, 550, 650}


GATE_LABELS = {
    "five_frequency_calibration_reliable": "Calibration reliable (5/5 frequencies)",
    "healthy_support_coverage_ge_0_90": "Healthy support coverage >= 0.90",
    "damage_support_coverage_ge_0_80": "Damage support coverage >= 0.80",
    "each_temperature_support_coverage_ge_0_60": "Each-temperature support >= 0.60",
    "supported_healthy_fpr_le_0_05": "Supported healthy FPR <= 0.05",
    "healthy_false_alarm_blocks_le_2": "Healthy false-alarm blocks <= 2",
    "supported_damage_recall_ge_0_80": "Supported damage recall >= 0.80",
    "worst_temperature_recall_ge_0_60": "Worst-temperature recall >= 0.60",
    "worst_pressure_bin_recall_ge_0_60": "Worst pressure-bin recall >= 0.60",
    "macro_auc_ge_0_80": "Macro AUROC >= 0.80",
    "worst_frequency_auc_ge_0_65": "Worst-frequency AUROC >= 0.65",
    "macro_auc_advantage_over_energy_ge_0_10": "AUROC advantage over energy >= 0.10",
    "no_unexplained_acquisition_asymmetry": "No unexplained acquisition asymmetry",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    conditions = data["conditions"]

    # Frozen primary gate table.
    gate_rows = []
    for name, passed in data["gates"].items():
        gate_rows.append(
            {
                "gate": name,
                "display_label": GATE_LABELS[name],
                "passed": bool(passed),
                "status": "PASS" if passed else "FAIL",
            }
        )
    write_csv(TABLES / "p12_primary_gate_table.csv", list(gate_rows[0]), gate_rows)

    # Calibration and discrimination tables.
    threshold_rows = []
    frequency_rows = []
    for frequency in FREQUENCIES:
        threshold = data["thresholds"][frequency]
        loo_values = [item["threshold"] for item in threshold["leave_one_block_out"]]
        metric = data["discrimination"]["by_frequency"][frequency]
        threshold_rows.append(
            {
                "frequency_hz": int(frequency),
                "threshold": threshold["threshold"],
                "median": threshold["median"],
                "mad": threshold["mad"],
                "samples": threshold["samples"],
                "blocks": threshold["blocks"],
                "loo_threshold_min": min(loo_values),
                "loo_threshold_max": max(loo_values),
                "loo_threshold_relative_range": threshold["loo_threshold_relative_range"],
                "frozen_reliability_limit": 0.25,
                "reliable": threshold["loo_threshold_relative_range"] <= 0.25,
            }
        )
        frequency_rows.append(
            {
                "frequency_hz": int(frequency),
                "roc_auc": metric["roc_auc"],
                "average_precision": metric["average_precision"],
                "energy_roc_auc": metric["energy_roc_auc"],
                "auc_advantage_over_energy": metric["roc_auc"] - metric["energy_roc_auc"],
                "n_healthy": metric["n_healthy"],
                "n_damage": metric["n_damage"],
            }
        )
    write_csv(TABLES / "p12_calibration_thresholds.csv", list(threshold_rows[0]), threshold_rows)
    write_csv(TABLES / "p12_frequency_discrimination.csv", list(frequency_rows[0]), frequency_rows)

    # Condition-level table and independently reproduced counts.
    condition_rows = []
    for item in conditions:
        row = {
            "archive": item["archive"],
            "path": item["path"],
            "temperature_c": item.get("temperature_c"),
            "pressure_bar": item.get("pressure_bar"),
            "ramp": item.get("ramp"),
            "status": item["status"],
            "invalid_reason": item.get("invalid_reason"),
            "support_distance": item.get("support_distance"),
            "supported": item.get("supported"),
            "fusion": item.get("fusion"),
        }
        for frequency in FREQUENCIES:
            fdata = item.get("frequencies", {}).get(frequency, {})
            row[f"score_{frequency}"] = fdata.get("condition_score")
            row[f"energy_score_{frequency}"] = fdata.get("condition_energy_score")
            row[f"confirmed_{frequency}"] = fdata.get("confirmed")
        condition_rows.append(row)
    condition_fields = list(condition_rows[0])
    write_csv(TABLES / "p12_condition_level_results.csv", condition_fields, condition_rows)

    status_counts = Counter((x["archive"], x["status"], x.get("fusion")) for x in conditions)
    status_rows = [
        {"archive": key[0], "status": key[1], "fusion": key[2], "count": value}
        for key, value in sorted(status_counts.items())
    ]
    write_csv(TABLES / "p12_record_status_counts.csv", list(status_rows[0]), status_rows)

    # Coverage and observed decisions by temperature.  T37 damage recall is not
    # estimable because zero irreversible records are supported; frozen gate output
    # is retained separately rather than presented as an observed 0/28 recall.
    temperature_rows = []
    for temperature in TEMPERATURES:
        healthy_all = [
            x for x in conditions
            if x["archive"] == "baseline"
            and x["temperature_c"] == temperature
            and x["pressure_bar"] in HELDOUT_HEALTHY_PRESSURES
        ]
        damage_all = [x for x in conditions if x["archive"] == "irreversible" and x["temperature_c"] == temperature]
        healthy_supported = [x for x in healthy_all if x.get("supported")]
        damage_supported = [x for x in damage_all if x.get("supported")]
        damage_alarms = [x for x in damage_supported if x.get("fusion") == "alarm"]
        observed_recall = len(damage_alarms) / len(damage_supported) if damage_supported else None
        temperature_rows.append(
            {
                "temperature_c": temperature,
                "healthy_total": len(healthy_all),
                "healthy_supported": len(healthy_supported),
                "healthy_coverage": len(healthy_supported) / len(healthy_all) if healthy_all else None,
                "irreversible_total": len(damage_all),
                "irreversible_supported": len(damage_supported),
                "irreversible_support_coverage": len(damage_supported) / len(damage_all) if damage_all else None,
                "irreversible_alarms": len(damage_alarms),
                "observed_supported_recall": observed_recall,
                "frozen_gate_recall_value": data["alarm_metrics"]["recall_by_temperature"][str(temperature)],
                "interpretation": "not_estimable_no_supported_damage" if observed_recall is None else "observed_supported_subset",
            }
        )
    write_csv(TABLES / "p12_temperature_coverage_and_recall.csv", list(temperature_rows[0]), temperature_rows)

    pressure_bins = [(50, 250), (300, 500), (550, 700)]
    pressure_rows = []
    for lower, upper in pressure_bins:
        subset = [
            x
            for x in conditions
            if x["archive"] == "irreversible"
            and x.get("supported")
            and lower <= x["pressure_bar"] <= upper
        ]
        alarms = [x for x in subset if x.get("fusion") == "alarm"]
        key = f"{lower}_{upper}"
        pressure_rows.append(
            {
                "pressure_bin_bar": f"{lower}-{upper}",
                "supported_irreversible": len(subset),
                "alarms": len(alarms),
                "observed_recall": len(alarms) / len(subset),
                "frozen_summary_recall": data["alarm_metrics"]["recall_by_pressure_bin"][key],
            }
        )
    write_csv(TABLES / "p12_pressure_bin_recall.csv", list(pressure_rows[0]), pressure_rows)

    # Reversible damage is a frozen secondary outcome; report detection fraction,
    # never substitute it for the irreversible primary endpoint.
    reversible_rows = []
    for temperature in TEMPERATURES:
        subset = [x for x in conditions if x["archive"] == "reversible" and x["temperature_c"] == temperature]
        supported = [x for x in subset if x.get("supported")]
        alarms = [x for x in supported if x.get("fusion") == "alarm"]
        reversible_rows.append(
            {
                "temperature_c": temperature,
                "total": len(subset),
                "supported": len(supported),
                "support_coverage": len(supported) / len(subset),
                "alarms": len(alarms),
                "supported_detection_fraction": len(alarms) / len(supported) if supported else None,
            }
        )
    write_csv(TABLES / "p12_reversible_secondary_by_temperature.csv", list(reversible_rows[0]), reversible_rows)

    # Figure 1: all pre-registered gates.
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    y = np.arange(len(gate_rows))
    values = [1 if row["passed"] else 0 for row in gate_rows]
    colors = ["#138A72" if value else "#C73E1D" for value in values]
    ax.barh(y, [1] * len(y), color=colors, height=0.66)
    ax.set_yticks(y, [row["display_label"] for row in gate_rows], fontsize=9)
    ax.set_xlim(0, 1.13)
    ax.set_xticks([])
    ax.invert_yaxis()
    for yi, value in zip(y, values):
        ax.text(0.5, yi, "PASS" if value else "FAIL", ha="center", va="center", color="white", weight="bold")
    ax.set_title("P12-COPV pre-registered primary gates: overall FAIL", loc="left", weight="bold")
    ax.text(0, 1.02, f"{sum(values)} passed / {len(values)} total; six failed gates determine the frozen primary outcome",
            transform=ax.transAxes, fontsize=10, color="#444444")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, "fig01_primary_gate_dashboard")

    # Figure 2: discrimination versus the pre-registered energy control.
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    x = np.arange(len(FREQUENCIES))
    width = 0.36
    structural = [row["roc_auc"] for row in frequency_rows]
    energy = [row["energy_roc_auc"] for row in frequency_rows]
    ax.bar(x - width / 2, structural, width, label="Structural residual score", color="#2C7FB8")
    ax.bar(x + width / 2, energy, width, label="Energy control", color="#F28E2B")
    ax.set_xticks(x, [f"{int(f) // 1000} kHz" for f in FREQUENCIES])
    ax.set_ylim(0.90, 1.008)
    ax.set_ylabel("AUROC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.set_title("High ranking performance does not imply structural-score superiority", loc="left", weight="bold")
    ax.text(0, 1.02,
            f"Macro AUROC: structural {data['discrimination']['macro_roc_auc']:.3f}; energy {data['discrimination']['energy_macro_roc_auc']:.3f}; difference {data['discrimination']['macro_advantage_over_energy']:+.3f}",
            transform=ax.transAxes, fontsize=9.5, color="#444444")
    save_figure(fig, "fig02_frequency_auc_vs_energy_control")

    # Figure 3: support coverage and conditional recall by temperature.
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.05]})
    x = np.arange(len(TEMPERATURES))
    healthy_cov = [row["healthy_coverage"] for row in temperature_rows]
    damage_cov = [row["irreversible_support_coverage"] for row in temperature_rows]
    width = 0.36
    axes[0].bar(x - width / 2, healthy_cov, width, label="Healthy", color="#4E79A7")
    axes[0].bar(x + width / 2, damage_cov, width, label="Irreversible damage", color="#E15759")
    axes[0].axhline(0.60, color="#555555", linestyle="--", linewidth=1, label="Per-temperature gate (0.60)")
    axes[0].set_ylabel("Support coverage")
    axes[0].set_ylim(0, 1.08)
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)
    recalls = [row["observed_supported_recall"] for row in temperature_rows]
    plotted = [0 if value is None else value for value in recalls]
    bars = axes[1].bar(x, plotted, width=0.58, color=["#BDBDBD" if value is None else "#59A14F" for value in recalls])
    axes[1].axhline(0.60, color="#555555", linestyle="--", linewidth=1, label="Worst-temperature recall gate (0.60)")
    for bar, value, row in zip(bars, recalls, temperature_rows):
        label = "N/E\n0 supported" if value is None else f"{100*value:.1f}%\n({row['irreversible_alarms']}/{row['irreversible_supported']})"
        axes[1].text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0.01) + 0.035, label,
                     ha="center", va="bottom", fontsize=8.5)
    axes[1].set_ylabel("Observed recall\n(supported subset)")
    axes[1].set_ylim(0, 1.13)
    axes[1].set_xticks(x, [f"{t} °C" for t in TEMPERATURES])
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Support failure and missed alarms are distinct failure modes", x=0.12, ha="left", weight="bold")
    fig.text(0.12, 0.94, "T37 is not an observed 0% recall: no irreversible record had valid support metadata; T43 is a true 0/28 supported miss.", fontsize=9.2, color="#444444")
    fig.subplots_adjust(top=0.87, hspace=0.18)
    save_figure(fig, "fig03_temperature_support_and_recall")

    # Figure 4: pressure-bin recall.
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    bars = ax.bar([row["pressure_bin_bar"] for row in pressure_rows], [row["observed_recall"] for row in pressure_rows], color="#76B7B2", width=0.62)
    ax.axhline(0.60, color="#C73E1D", linestyle="--", linewidth=1.4, label="Frozen minimum (0.60)")
    for bar, row in zip(bars, pressure_rows):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                f"{100*row['observed_recall']:.1f}%\n({row['alarms']}/{row['supported_irreversible']})", ha="center", fontsize=9)
    ax.set_ylim(0, 0.74)
    ax.set_ylabel("Irreversible recall (supported subset)")
    ax.set_xlabel("Pressure bin (bar)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title("Detection weakens at the highest pressure range", loc="left", weight="bold")
    save_figure(fig, "fig04_pressure_bin_recall")

    # Figure 5: threshold-normalized score distributions.
    fig, axes = plt.subplots(1, len(FREQUENCIES), figsize=(14.5, 4.8), sharey=True)
    archive_order = ["baseline", "reversible", "irreversible"]
    archive_labels = ["Healthy", "Reversible", "Irreversible"]
    archive_colors = ["#4E79A7", "#F28E2B", "#E15759"]
    for ax, frequency in zip(axes, FREQUENCIES):
        values = []
        for archive in archive_order:
            values.append([
                x["frequencies"][frequency]["condition_score"] / data["thresholds"][frequency]["threshold"]
                for x in conditions
                if x["archive"] == archive and x.get("supported") and frequency in x.get("frequencies", {})
            ])
        box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.62, medianprops={"color": "black"})
        for patch, color in zip(box["boxes"], archive_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.axhline(1, color="#222222", linestyle="--", linewidth=1)
        ax.set_title(f"{int(frequency)//1000} kHz")
        ax.set_xticks([1, 2, 3], ["H", "R", "I"])
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Condition score / frozen threshold")
    fig.suptitle("Ranking separation coexists with conservative alarm thresholds", x=0.08, ha="left", weight="bold")
    fig.text(0.08, 0.92, "H = healthy, R = reversible damage, I = irreversible damage; dashed line is the per-frequency alarm threshold.", fontsize=9.2, color="#444444")
    fig.subplots_adjust(top=0.82, wspace=0.12)
    save_figure(fig, "fig05_threshold_normalized_score_distributions")

    # Figure 6: record disposition, preserving abstentions and exclusions.
    categories = [
        ("Complete", 374, "#59A14F"),
        ("Missing support metadata", 30, "#EDC948"),
        ("Missing frozen reference", 15, "#B07AA1"),
        ("Official baseline exclusion", 1, "#9C755F"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    bars = ax.barh([item[0] for item in categories], [item[1] for item in categories], color=[item[2] for item in categories])
    for bar, item in zip(bars, categories):
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2, f"{item[1]} ({100*item[1]/420:.1f}%)", va="center", fontsize=9.5)
    ax.set_xlim(0, 430)
    ax.set_xlabel("H5 condition records")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("All 420 records remain auditable in the analysis flow", loc="left", weight="bold")
    save_figure(fig, "fig06_record_disposition")

    # Figure 7: irreversible alarm fractions across the operating grid.
    matrix = np.full((len(TEMPERATURES), len(PRESSURES)), np.nan)
    supported_n = np.zeros_like(matrix, dtype=int)
    alarm_n = np.zeros_like(matrix, dtype=int)
    for i, temperature in enumerate(TEMPERATURES):
        for j, pressure in enumerate(PRESSURES):
            subset = [x for x in conditions if x["archive"] == "irreversible" and x["temperature_c"] == temperature and x["pressure_bar"] == pressure and x.get("supported")]
            alarms = [x for x in subset if x.get("fusion") == "alarm"]
            supported_n[i, j] = len(subset)
            alarm_n[i, j] = len(alarms)
            if subset:
                matrix[i, j] = len(alarms) / len(subset)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#D9D9D9")
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(PRESSURES)), PRESSURES)
    ax.set_yticks(np.arange(len(TEMPERATURES)), TEMPERATURES)
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("Temperature (°C)")
    for i in range(len(TEMPERATURES)):
        for j in range(len(PRESSURES)):
            text = "N/E" if supported_n[i, j] == 0 else f"{alarm_n[i,j]}/{supported_n[i,j]}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color="black")
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Alarm fraction among supported irreversible records")
    ax.set_title("Irreversible alarm map exposes operating-condition failure structure", loc="left", weight="bold")
    ax.text(0, 1.04, "Each valid cell contains alarms / supported records (two ramp orders expected); N/E = not evaluable under the frozen support rule.", transform=ax.transAxes, fontsize=9.2, color="#444444")
    save_figure(fig, "fig07_irreversible_alarm_operating_map")

    # Figure 8: calibration stability.
    fig, ax = plt.subplots(figsize=(8.7, 4.7))
    relative_ranges = [row["loo_threshold_relative_range"] for row in threshold_rows]
    bars = ax.bar([f"{int(f)//1000} kHz" for f in FREQUENCIES], relative_ranges, color="#4E79A7", width=0.62)
    ax.axhline(0.25, color="#C73E1D", linestyle="--", linewidth=1.4, label="Frozen reliability limit (0.25)")
    for bar, value in zip(bars, relative_ranges):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.008, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 0.29)
    ax.set_ylabel("Leave-one-block-out threshold relative range")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title("Healthy calibration reliability passed at all five frequencies", loc="left", weight="bold")
    save_figure(fig, "fig08_calibration_stability")

    # Machine-readable reporting audit, including internal consistency checks.
    supported_healthy = [
        x for x in conditions
        if x["archive"] == "baseline"
        and x.get("supported")
        and x["pressure_bar"] in HELDOUT_HEALTHY_PRESSURES
    ]
    supported_irreversible = [x for x in conditions if x["archive"] == "irreversible" and x.get("supported")]
    supported_reversible = [x for x in conditions if x["archive"] == "reversible" and x.get("supported")]
    audit = {
        "schema_version": "p12-copv-reporting-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(SOURCE),
        "frozen_primary_status": data["primary_status"],
        "checks": {
            "summary_condition_count_419": len(conditions) == data["condition_count"] == 419,
            "feature_json_count_420": len(list(FEATURE_DIR.glob("*.json"))) == 420,
            "supported_healthy_39": len(supported_healthy) == 39,
            "healthy_false_alarms_0": sum(x["fusion"] == "alarm" for x in supported_healthy) == 0,
            "supported_irreversible_132": len(supported_irreversible) == 132,
            "irreversible_alarms_67": sum(x["fusion"] == "alarm" for x in supported_irreversible) == 67,
            "supported_reversible_162": len(supported_reversible) == 162,
            "reversible_alarms_46": sum(x["fusion"] == "alarm" for x in supported_reversible) == 46,
            "primary_gates_7_pass_6_fail": sum(data["gates"].values()) == 7 and len(data["gates"]) == 13,
            "primary_status_fail": data["primary_status"] == "FAIL",
        },
        "derived_counts": {
            "supported_healthy": len(supported_healthy),
            "healthy_alarms": sum(x["fusion"] == "alarm" for x in supported_healthy),
            "supported_irreversible": len(supported_irreversible),
            "irreversible_alarms": sum(x["fusion"] == "alarm" for x in supported_irreversible),
            "supported_reversible": len(supported_reversible),
            "reversible_alarms": sum(x["fusion"] == "alarm" for x in supported_reversible),
        },
        "interpretive_guardrails": [
            "T37 irreversible recall is not estimable because no irreversible record is supported; the frozen gate implementation records zero.",
            "Reversible-damage detection fraction is secondary and does not replace the irreversible primary endpoint.",
            "No thresholds, support limits, scores, or gates were recomputed or tuned by this reporting script.",
        ],
    }
    if not all(audit["checks"].values()):
        raise RuntimeError(audit["checks"])
    (OUT / "p12_reporting_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    outputs = [path for path in OUT.rglob("*") if path.is_file()]
    manifest = {
        "schema_version": "p12-copv-reporting-manifest-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_status": "FAIL",
        "files": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(outputs)
            if path.name != "p12_reporting_manifest.json"
        ],
    }
    (OUT / "p12_reporting_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "tables": len(list(TABLES.glob("*.csv"))),
        "figures_png": len(list(FIGURES.glob("*.png"))),
        "figures_pdf": len(list(FIGURES.glob("*.pdf"))),
        "checks_passed": all(audit["checks"].values()),
        "primary_status": data["primary_status"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
