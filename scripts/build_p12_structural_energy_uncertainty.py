"""Reproduce the post-outcome paired condition-level P12 AUROC uncertainty audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "derived_tables" / "p12_condition_level_results.csv"
DEFAULT_OUTPUT = ROOT / "results" / "derived_tables" / "p12_structural_energy_paired_bootstrap.csv"
FREQUENCIES = (60000, 120000, 180000, 260000, 300000)
HELDOUT_HEALTHY_PRESSURES = {50, 150, 250, 350, 450, 550, 650}


def auc(healthy: np.ndarray, damage: np.ndarray) -> float:
    scores = np.concatenate([healthy, damage])
    ranks = rankdata(scores, method="average")
    n0, n1 = len(healthy), len(damage)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def eligible(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    healthy = [
        row for row in rows
        if row["archive"] == "baseline"
        and row["supported"] == "True"
        and int(float(row["pressure_bar"])) in HELDOUT_HEALTHY_PRESSURES
    ]
    damage = [
        row for row in rows
        if row["archive"] == "irreversible" and row["supported"] == "True"
    ]
    if (len(healthy), len(damage)) != (39, 132):
        raise RuntimeError(f"Unexpected P12 denominators: healthy={len(healthy)}, damage={len(damage)}")
    return healthy, damage


def matrix(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}_{frequency}"]) for frequency in FREQUENCIES] for row in rows],
        dtype=float,
    )


def macro_auc(healthy: np.ndarray, damage: np.ndarray) -> float:
    return float(np.mean([auc(healthy[:, j], damage[:, j]) for j in range(len(FREQUENCIES))]))


def calculate(path: Path, repetitions: int, seed: int) -> dict[str, object]:
    healthy_rows, damage_rows = eligible(read_rows(path))
    hs, ds = matrix(healthy_rows, "score"), matrix(damage_rows, "score")
    he, de = matrix(healthy_rows, "energy_score"), matrix(damage_rows, "energy_score")
    structural = macro_auc(hs, ds)
    energy = macro_auc(he, de)
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        hi = rng.integers(0, len(hs), len(hs))
        di = rng.integers(0, len(ds), len(ds))
        differences[i] = macro_auc(hs[hi], ds[di]) - macro_auc(he[hi], de[di])
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "analysis_role": "post-outcome_secondary_uncertainty_characterization",
        "bootstrap_unit": "paired_condition_within_class",
        "healthy_conditions": len(hs),
        "irreversible_damage_conditions": len(ds),
        "frequencies": ";".join(str(value) for value in FREQUENCIES),
        "structural_macro_auroc": structural,
        "energy_macro_auroc": energy,
        "structural_minus_energy_macro_auroc": structural - energy,
        "paired_bootstrap_95pct_lower": float(lower),
        "paired_bootstrap_95pct_upper": float(upper),
        "paired_bootstrap_probability_difference_gt_0": float(np.mean(differences > 0)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "interpretation": "Does not alter the frozen +0.10 physical-increment gate or the P12 FAIL outcome.",
    }


def write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def compare(path: Path, row: dict[str, object]) -> None:
    existing = read_rows(path)
    if len(existing) != 1:
        raise RuntimeError("Expected one committed uncertainty row")
    for key, value in row.items():
        observed = existing[0][key]
        if isinstance(value, float):
            if not np.isclose(float(observed), value, rtol=0, atol=1e-12):
                raise RuntimeError(f"Mismatch for {key}: committed={observed}, recomputed={value}")
        elif str(value) != observed:
            raise RuntimeError(f"Mismatch for {key}: committed={observed!r}, recomputed={value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    row = calculate(args.input, args.repetitions, args.seed)
    if args.check:
        compare(args.output, row)
    else:
        write_csv(args.output, row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
