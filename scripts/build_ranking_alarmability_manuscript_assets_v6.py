"""Build publication figures and audit tables for the SHM v6 manuscript.

The script consumes only frozen, derived CSV files.  It does not inspect raw
waveforms, refit a model, tune a threshold, or alter a confirmatory outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "ranking_not_alarmability_v1"
PUBLIC_LAYOUT = (ROOT / "results" / "derived_tables").is_dir() and not MANUSCRIPT.is_dir()
if PUBLIC_LAYOUT:
    OUT = ROOT / "figures"
    TABLES = ROOT / "results" / "derived_tables"
    SOURCE_TABLES = ROOT / "results" / "derived_tables"
    EVIDENCE = SOURCE_TABLES / "table01_cross_dataset_evidence.csv"
    AUDIT_OUT = ROOT / "results" / "asset_build_audit.json"
else:
    OUT = MANUSCRIPT / "figures_v6"
    TABLES = MANUSCRIPT / "tables_v6"
    SOURCE_TABLES = ROOT / "runs" / "p12_copv_reporting_v1" / "tables"
    EVIDENCE = MANUSCRIPT / "tables" / "table01_cross_dataset_evidence.csv"
    AUDIT_OUT = MANUSCRIPT / "asset_build_audit_v6.json"

NAVY = "#17365D"
BLUE = "#2F75B5"
TEAL = "#1B998B"
GREEN = "#2E8B57"
AMBER = "#D98E04"
RED = "#C23B22"
GRAY = "#6B7280"
LIGHT = "#EEF3F8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def as_float(value: str) -> float:
    if value in {"", "NA", "N/A", "nan"}:
        return float("nan")
    return float(value)


def configure_output(output_root: Path | None) -> None:
    """Redirect generated assets without changing committed input tables."""
    global OUT, TABLES, AUDIT_OUT
    if output_root is None:
        return
    OUT = output_root / "figures"
    TABLES = output_root / "results" / "derived_tables"
    AUDIT_OUT = output_root / "results" / "asset_build_audit.json"


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def figure_1() -> None:
    fig = plt.figure(figsize=(13.4, 6.9), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.0], wspace=0.07)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0, 9.65, "High AUROC ≠ reliable alarms", fontsize=18,
            weight="bold", color=NAVY, va="top")
    ax.text(0, 9.05, "Qualification is a chain: failure at any required layer limits the claim.",
            fontsize=10.5, color=GRAY, va="top")

    layers = [
        ("1  Representation", "Score reproducible?"),
        ("2  Ranking", "Damage ranked higher?"),
        ("3  Healthy calibration", "Frozen threshold stable?"),
        ("4  Operating support", "Condition eligible?"),
        ("5  Alarm decision", "FPR / recall acceptable?"),
    ]
    colors = ["#DCE6F1", "#C6E0F5", "#CDE9E3", "#FCE4C2", "#F7D0CB"]
    y = 8.2
    for index, ((title, question), color) in enumerate(zip(layers, colors)):
        box = FancyBboxPatch((0.15, y - 0.85), 6.78, 1.03,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             linewidth=1.2, edgecolor=NAVY, facecolor=color)
        ax.add_patch(box)
        ax.text(0.45, y - 0.25, title, fontsize=11.6, weight="bold", color=NAVY)
        ax.text(4.25, y - 0.25, question, fontsize=8.3, color="#263238")
        if index < len(layers) - 1:
            ax.annotate("", xy=(3.6, y - 1.15), xytext=(3.6, y - 0.87),
                        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.3))
        y -= 1.35

    ctrl = FancyBboxPatch((7.35, 3.05), 2.50, 4.1,
                          boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5, edgecolor=AMBER, facecolor="#FFF5DE")
    ax.add_patch(ctrl)
    ax.text(8.60, 6.78, "Simple physical\nbaseline", ha="center", va="top",
            fontsize=10.9, weight="bold", color="#8A5700", linespacing=1.05)
    ax.text(8.60, 5.76, "e.g., energy change", ha="center", va="top",
            fontsize=9.2, color="#5F4B25")
    ax.text(8.60, 4.65, "Does the structural\nscore add\ninformation?", ha="center", va="top",
            fontsize=9.0, color="#5F4B25", linespacing=1.12)
    ax.annotate("", xy=(6.93, 5.05), xytext=(7.35, 5.05),
                arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.35, mutation_scale=7))
    ax.text(0.15, 0.55, "Output: score + calibration + support + alarm / abstain",
            fontsize=10.6, weight="bold", color=NAVY)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    panel = FancyBboxPatch((0.03, 0.06), 0.94, 0.88,
                           boxstyle="round,pad=0.025,rounding_size=0.035",
                           linewidth=1.4, edgecolor=NAVY, facecolor="#F8FAFC")
    ax2.add_patch(panel)
    ax2.text(0.08, 0.89, "External COPV confirmation", fontsize=15, weight="bold", color=NAVY)
    metrics = [
        ("Structural macro AUROC", "0.9890", GREEN),
        ("Observed healthy alarms", "0 / 39", GREEN),
        ("Supported damage recall", "67/132 = 0.5076", RED),
        ("Damage support", "132/168 = 0.7857", RED),
        ("Structural - energy AUROC", "-0.0065", RED),
        ("Mandatory gates", "7 / 13 passed", RED),
    ]
    y = 0.80
    for label, value, color in metrics:
        ax2.text(0.075, y, label, fontsize=9.6, color=GRAY, va="center")
        ax2.text(0.94, y, value, fontsize=11.2, weight="bold", color=color,
                 ha="right", va="center")
        ax2.plot([0.075, 0.94], [y - 0.055, y - 0.055], color="#D8DEE8", lw=0.8)
        y -= 0.115
    ax2.text(0.50, 0.105, "STRICT OUTCOME: FAIL", ha="center", va="center",
             fontsize=16, weight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.38", facecolor=RED, edgecolor=RED))
    save(fig, "fig01_qualification_chain_and_p12_headline")


def figure_2() -> None:
    rows = read_csv(EVIDENCE)
    names = [
        "Long-term\nbridge", "Composite temp.\nbenchmark", "Independent\nhealthy transfer",
        "Monitored\npipeline", "Wind-blade\nlocked test", "External COPV\nconfirmation",
    ]
    matrix = np.array([
        [as_float(row["ranking_auc"]), 1 - as_float(row["healthy_fpr"]), as_float(row["damage_recall"])]
        for row in rows
    ])
    masked = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(figsize=(10.9, 5.9))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#E5E7EB")
    im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(3), ["AUROC\n(higher is better)", "Healthy specificity\n1 - FPR", "Frozen-rule recall\n(higher is better)"])
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(axis="both", labelsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "N/A" if np.isnan(value) else f"{value:.3f}"
            color = "#4B5563" if np.isnan(value) else ("white" if value < 0.25 or value > 0.78 else "black")
            ax.text(j, i, label, ha="center", va="center", fontsize=10.5, weight="bold", color=color)
    ax.set_title("Cross-dataset evidence with all color directions aligned", fontsize=14, weight="bold", color=NAVY, pad=14)
    ax.text(0.5, 1.005, "Green always means a more favourable value; grey denotes an inapplicable or unavailable endpoint.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.3, color=GRAY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("More favourable  ->", rotation=90)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    save(fig, "fig02_cross_dataset_aligned_metrics")


def figure_3() -> None:
    rows = [row for row in read_csv(EVIDENCE) if row["ranking_auc"] != "NA" and row["damage_recall"] != "NA"]
    labels = ["Long-term bridge", "Composite temp. benchmark", "Monitored pipeline", "Wind-blade locked test", "External COPV"]
    markers = ["X", "s", "D", "^", "o"]
    colors = [GRAY, AMBER, BLUE, TEAL, RED]
    fig, ax = plt.subplots(figsize=(9.3, 6.5))
    for row, label, marker, color in zip(rows, labels, markers, colors):
        x = float(row["ranking_auc"])
        y = float(row["damage_recall"])
        size = 230 if label == "External COPV" else 150
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        dx, dy = {
            "Long-term bridge": (0.012, 0.035),
            "Composite temp. benchmark": (-0.205, 0.045),
            "Monitored pipeline": (-0.20, -0.07),
            "Wind-blade locked test": (-0.205, 0.02),
            "External COPV": (-0.19, -0.08),
        }[label]
        ax.annotate(label, (x, y), xytext=(x + dx, y + dy), fontsize=9.7, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    ax.set_xlim(0.47, 1.02)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Ranking AUROC", fontsize=11)
    ax.set_ylabel("Recall at the frozen healthy-only decision rule", fontsize=11)
    ax.set_title("High ranking can coexist with weak or disqualified alarms", fontsize=14, weight="bold", color=NAVY)
    ax.grid(alpha=0.22)
    ax.text(0.485, 0.98, "Marker shapes aid visual separation only;\nthey do not encode a common pass/fail boundary.",
            fontsize=9.2, color=GRAY, va="top")
    save(fig, "fig03_ranking_vs_frozen_alarm_recall")


def figure_4() -> None:
    temp = read_csv(SOURCE_TABLES / "p12_temperature_coverage_and_recall.csv")
    pressure = read_csv(SOURCE_TABLES / "p12_pressure_bin_recall.csv")
    t = np.array([float(row["temperature_c"]) for row in temp])
    support = np.array([float(row["irreversible_support_coverage"]) for row in temp])
    recall = np.array([float(row["frozen_gate_recall_value"]) for row in temp])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), gridspec_kw={"width_ratios": [1.5, 0.85]})
    ax = axes[0]
    width = 2.1
    ax.bar(t - width / 2, support, width=width, color="#A8C5E5", label="Damage support coverage")
    ax.bar(t + width / 2, recall, width=width, color=RED, alpha=0.82, label="Frozen-rule recall")
    ax.axhline(0.60, color=GRAY, ls="--", lw=1.1, label="Frozen stratum gate = 0.60")
    ax.set_xticks(t)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Temperature (degrees C)")
    ax.set_ylabel("Fraction")
    ax.set_title("Temperature operating envelope", weight="bold", color=NAVY)
    ax.legend(fontsize=8.5, loc="upper center", ncol=2)
    ax.annotate("No supported\ndamage cases", xy=(37, 0.02), xytext=(34.2, 0.28),
                ha="center", fontsize=8.5, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))
    ax.annotate("Supported, but\nno alarms", xy=(43.9, 0.02), xytext=(46.0, 0.20),
                ha="center", fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    bins = [row["pressure_bin_bar"] for row in pressure]
    values = [float(row["observed_recall"]) for row in pressure]
    bars = ax.bar(bins, values, color=[AMBER, AMBER, RED], width=0.62)
    ax.axhline(0.60, color=GRAY, ls="--", lw=1.1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Pressure bin (bar)")
    ax.set_ylabel("Supported damage recall")
    ax.set_title("Pressure-bin sensitivity", weight="bold", color=NAVY)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}", ha="center", fontsize=9.5, weight="bold")
    fig.suptitle("P12 failure is localized to support and conditional sensitivity", fontsize=14, weight="bold", color=NAVY, y=1.02)
    save(fig, "fig04_p12_operating_envelope")


def gate_rows() -> list[dict[str, object]]:
    gates = read_csv(SOURCE_TABLES / "p12_primary_gate_table.csv")
    display_label = {
        "supported_healthy_fpr_le_0_05": "Observed supported healthy FPR <= 0.05",
        "healthy_false_alarm_blocks_le_2": "Confirmed healthy false-alarm runs <= 2",
        "no_unexplained_acquisition_asymmetry": "Data-integrity audit: no unexplained state-specific channel or sampling-regime difference",
    }
    observed = {
        "five_frequency_calibration_reliable": "5/5 reliable",
        "healthy_support_coverage_ge_0_90": "0.9286",
        "damage_support_coverage_ge_0_80": "0.7857",
        "each_temperature_support_coverage_ge_0_60": "minimum 0.0000",
        "supported_healthy_fpr_le_0_05": "0/39 = 0.0000",
        "healthy_false_alarm_blocks_le_2": "0",
        "supported_damage_recall_ge_0_80": "67/132 = 0.5076",
        "worst_temperature_recall_ge_0_60": "minimum 0.0000",
        "worst_pressure_bin_recall_ge_0_60": "minimum 0.3056",
        "macro_auc_ge_0_80": "0.9890",
        "worst_frequency_auc_ge_0_65": "0.9711",
        "macro_auc_advantage_over_energy_ge_0_10": "-0.0065",
        "no_unexplained_acquisition_asymmetry": "0 unexplained differences; known metadata gaps abstained",
    }
    rationale = {
        "five_frequency_calibration_reliable": "Prevents an unstable healthy tail from defining the alarm.",
        "healthy_support_coverage_ge_0_90": "Requires broad evaluability of healthy operation.",
        "damage_support_coverage_ge_0_80": "Prevents abstention from hiding too much declared damage space.",
        "each_temperature_support_coverage_ge_0_60": "Rejects an operating envelope with an unsupported temperature stratum.",
        "supported_healthy_fpr_le_0_05": "Caps the empirical inspection burden within support.",
        "healthy_false_alarm_blocks_le_2": "Limits clustered false-alarm runs not visible in marginal FPR.",
        "supported_damage_recall_ge_0_80": "Requires useful sensitivity after support filtering.",
        "worst_temperature_recall_ge_0_60": "Prevents strong strata from masking a blind temperature regime.",
        "worst_pressure_bin_recall_ge_0_60": "Prevents strong pressure ranges from masking a weak range.",
        "macro_auc_ge_0_80": "Requires useful overall ordering information.",
        "worst_frequency_auc_ge_0_65": "Requires every retained frequency to carry information.",
        "macro_auc_advantage_over_energy_ge_0_10": "Requires structural information beyond a simple energy change.",
        "no_unexplained_acquisition_asymmetry": "Guards against undocumented state-specific acquisition or schema artefacts.",
    }
    sensitivity = {
        "five_frequency_calibration_reliable": "Pass; not threshold-sensitive.",
        "healthy_support_coverage_ge_0_90": "Passes by 0.0286.",
        "damage_support_coverage_ge_0_80": "Would pass at 0.75; a narrow coverage miss.",
        "each_temperature_support_coverage_ge_0_60": "Still fails at 0.50 because T37 has zero support.",
        "supported_healthy_fpr_le_0_05": "Observed pass; 95% two-sided upper bound is 0.0903.",
        "healthy_false_alarm_blocks_le_2": "Observed pass; two-run margin under the frozen rule.",
        "supported_damage_recall_ge_0_80": "Would pass only if relaxed to about 0.50.",
        "worst_temperature_recall_ge_0_60": "Fails for any positive threshold because T43 recall is zero.",
        "worst_pressure_bin_recall_ge_0_60": "Would pass only near 0.30.",
        "macro_auc_ge_0_80": "Passes by 0.1890.",
        "worst_frequency_auc_ge_0_65": "Passes by 0.3211.",
        "macro_auc_advantage_over_energy_ge_0_10": "Still fails at zero advantage.",
        "no_unexplained_acquisition_asymmetry": "Pass; machine-readable audit documents equal sampling/channel regimes and explained metadata gaps.",
    }
    return [
        {
            "gate": display_label.get(row["gate"], row["display_label"]),
            "observed": observed[row["gate"]],
            "outcome": row["status"],
            "engineering_rationale": rationale[row["gate"]],
            "one_at_a_time_sensitivity": sensitivity[row["gate"]],
        }
        for row in gates
    ]


def figure_5(rows: list[dict[str, object]]) -> None:
    status = np.array([[1 if row["outcome"] == "PASS" else 0, 2] for row in rows])
    fig, ax = plt.subplots(figsize=(12.0, 7.3))
    ax.imshow(status, cmap=ListedColormap(["#F4B6AD", "#B7DFC2", "#F7F9FC"]),
              vmin=0, vmax=2, aspect="auto")
    labels = [str(row["gate"]) for row in rows]
    ax.set_yticks(range(len(labels)), labels, fontsize=9.2)
    ax.set_xticks([0, 1], ["Frozen outcome", "Observed value"])
    for i, row in enumerate(rows):
        ax.text(0, i, str(row["outcome"]), ha="center", va="center", weight="bold",
                color=GREEN if row["outcome"] == "PASS" else RED)
        ax.text(1, i, str(row["observed"]), ha="center", va="center",
                fontsize=9.0, color="#263238")
    ax.set_title("P12 conjunctive qualification: 7 of 13 mandatory gates passed", fontsize=14,
                 weight="bold", color=NAVY, pad=14)
    ax.set_xlim(-0.55, 1.55)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.subplots_adjust(left=0.43, right=0.96, top=0.90, bottom=0.08)
    save(fig, "fig05_p12_gate_dashboard")

def interval_table() -> list[dict[str, object]]:
    x, n, alpha = 0, 39, 0.05
    lower = 0.0 if x == 0 else beta.ppf(alpha / 2, x, n - x + 1)
    upper_two = beta.ppf(1 - alpha / 2, x + 1, n - x)
    upper_one = beta.ppf(1 - alpha, x + 1, n - x)
    return [{
        "endpoint": "Supported healthy alarm probability",
        "events": x,
        "trials": n,
        "empirical_estimate": x / n,
        "clopper_pearson_95pct_two_sided_lower": lower,
        "clopper_pearson_95pct_two_sided_upper": upper_two,
        "clopper_pearson_95pct_one_sided_upper": upper_one,
        "interpretation": "Observed 0/39 is not proof that the prospective false-alarm probability is zero.",
    }]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional clean output root; inputs remain the committed derived tables.",
    )
    args = parser.parse_args()
    configure_output(args.output_root)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    rows = gate_rows()
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5(rows)
    write_csv(TABLES / "table04_p12_gate_rationale_and_sensitivity.csv", rows)
    write_csv(TABLES / "table05_zero_event_exact_interval.csv", interval_table())
    audit = {
        "schema_version": "shm-v6-assets-v1",
        "input_evidence": display_path(EVIDENCE),
        "input_p12_tables": display_path(SOURCE_TABLES),
        "figures": sorted(path.name for path in OUT.glob("*.png")),
        "tables": sorted(path.name for path in TABLES.glob("*.csv")),
        "raw_data_accessed": False,
        "models_refit": False,
        "thresholds_changed": False,
    }
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
