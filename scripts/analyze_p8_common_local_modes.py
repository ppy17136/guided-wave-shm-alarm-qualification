"""Exploratory common-mode/local-path decomposition using frozen P7 exports.

Designed after P7.  All D7-D13 results are internal exploratory evidence.  The
script fits path normalization and a fixed linear environment model on each
event's training window only, then evaluates direction stability across events.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_p8_candidate_direction_stability import (
    ROOT,
    SEEDS,
    TRANSITIONS,
    average_precision,
    roc_auc,
)


SPLITS = ("train", "calibration", "pre_event", "post_event")
COMPONENTS = ("path_shape_error", "path_amplitude_error", "input_amplitude")
RIDGE_ALPHA = 1.0


def load_events(source: Path):
    events = []
    for event_index, expected in enumerate(TRANSITIONS, start=1):
        accum = {split: {name: [] for name in (*COMPONENTS, "environment")} for split in SPLITS}
        transition = None
        for seed in SEEDS:
            folder = source / f"event_{event_index:02d}" / f"seed_{seed}"
            metadata = json.loads((folder / "score_metadata.json").read_text(encoding="utf-8"))
            transition = transition or metadata["transition"]
            if metadata["transition"] != transition:
                raise SystemExit(f"Transition mismatch for event {event_index}")
            with np.load(folder / "component_scores.npz") as data:
                for split in SPLITS:
                    for name in accum[split]:
                        accum[split][name].append(data[f"{split}_{name}"].astype(np.float64))
        if transition != expected:
            raise SystemExit(f"Unexpected transition {transition}; expected {expected}")
        events.append({
            "event_index": event_index,
            "transition": transition,
            **{
                split: {name: np.mean(np.stack(values), axis=0) for name, values in content.items()}
                for split, content in accum.items()
            },
        })
    return events


def path_modes(train: np.ndarray, values: np.ndarray):
    train_log = np.log1p(np.maximum(np.asarray(train, float), 0))
    value_log = np.log1p(np.maximum(np.asarray(values, float), 0))
    median = np.median(train_log, axis=0)
    q25, q75 = np.quantile(train_log, (.25, .75), axis=0)
    z = (value_log - median) / np.maximum(q75 - q25, 1e-6)
    common = np.median(z, axis=1)
    local = z - common[:, None]
    return {
        "common_median": common,
        "local_l2": np.sqrt(np.mean(np.square(local), axis=1)),
        "local_iqr": np.quantile(local, .75, axis=1) - np.quantile(local, .25, axis=1),
        "local_max_abs": np.max(np.abs(local), axis=1),
        "local_tail_asymmetry": np.abs(np.quantile(local, .9, axis=1) + np.quantile(local, .1, axis=1)),
    }


def robust_variants(train: np.ndarray, values: np.ndarray):
    median = float(np.median(train))
    q25, q75 = np.quantile(train, (.25, .75))
    z = (np.asarray(values, float) - median) / max(float(q75 - q25), 1e-6)
    return np.maximum(z, 0), np.abs(z)


def ridge_residuals(train_env: np.ndarray, split_env: np.ndarray, train_y: np.ndarray, split_y: np.ndarray):
    mean = train_env.mean(axis=0)
    scale = np.maximum(train_env.std(axis=0), 1e-6)
    train_x = (train_env - mean) / scale
    split_x = (split_env - mean) / scale
    train_x = np.column_stack((np.ones(len(train_x)), train_x))
    split_x = np.column_stack((np.ones(len(split_x)), split_x))
    penalty = np.eye(train_x.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ train_y)
    return train_y - train_x @ coefficients, split_y - split_x @ coefficients


def derive_event_candidates(event):
    base_by_split = {split: {} for split in SPLITS}
    for component in COMPONENTS:
        train = event["train"][component]
        for split in SPLITS:
            modes = path_modes(train, event[split][component])
            for mode, values in modes.items():
                base_by_split[split][f"{component}__{mode}"] = values

    candidates = {split: {} for split in SPLITS}
    for base_name, train_values in base_by_split["train"].items():
        train_high, train_two = robust_variants(train_values, train_values)
        candidates["train"][f"{base_name}__raw_high"] = train_high
        candidates["train"][f"{base_name}__raw_two"] = train_two
        train_residual, _ = ridge_residuals(
            event["train"]["environment"], event["train"]["environment"], train_values, train_values
        )
        train_residual_high, train_residual_two = robust_variants(train_residual, train_residual)
        candidates["train"][f"{base_name}__env_residual_high"] = train_residual_high
        candidates["train"][f"{base_name}__env_residual_two"] = train_residual_two
        for split in ("calibration", "pre_event", "post_event"):
            values = base_by_split[split][base_name]
            high, two = robust_variants(train_values, values)
            candidates[split][f"{base_name}__raw_high"] = high
            candidates[split][f"{base_name}__raw_two"] = two
            _, residual = ridge_residuals(
                event["train"]["environment"], event[split]["environment"], train_values, values
            )
            residual_high, residual_two = robust_variants(train_residual, residual)
            candidates[split][f"{base_name}__env_residual_high"] = residual_high
            candidates[split][f"{base_name}__env_residual_two"] = residual_two
    names = tuple(sorted(candidates["train"]))
    return names, {split: np.column_stack([candidates[split][name] for name in names]) for split in SPLITS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("cluster_results/extracted_P7_20032_20260811/runs/p7_confirmatory_rescore_v1/01"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/p8_common_local_modes_v1"))
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    events = load_events(source)
    event_rows = []
    names = None
    auc_rows = []
    for event in events:
        current_names, matrices = derive_event_candidates(event)
        if names is None:
            names = current_names
        elif current_names != names:
            raise SystemExit("Derived-candidate mismatch")
        labels = np.r_[np.zeros(len(matrices["pre_event"]), np.int8), np.ones(len(matrices["post_event"]), np.int8)]
        scores = np.vstack((matrices["pre_event"], matrices["post_event"]))
        event_aucs = []
        for index, name in enumerate(names):
            auc = roc_auc(labels, scores[:, index])
            ap = average_precision(labels, scores[:, index])
            event_aucs.append(auc)
            event_rows.append({
                "event_index": event["event_index"],
                "transition": event["transition"],
                "candidate_index": index,
                "candidate": name,
                "roc_auc": auc,
                "average_precision": ap,
                "direction": "higher_after" if auc >= .5 else "lower_after",
                "direction_free_oracle_auc": max(auc, 1.0 - auc),
            })
        auc_rows.append(event_aucs)
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(output / "p8_common_local_event_metrics.csv", index=False, encoding="utf-8-sig")
    auc_matrix = np.asarray(auc_rows, float)

    summary_rows = []
    for index, name in enumerate(names):
        aucs = auc_matrix[:, index]
        signs = np.where(aucs >= .5, 1, -1)
        summary_rows.append({
            "candidate_index": index,
            "candidate": name,
            "component": name.split("__", 1)[0],
            "mode": name.split("__")[1],
            "variant": "__".join(name.split("__")[2:]),
            "macro_auc": float(aucs.mean()),
            "worst_auc": float(aucs.min()),
            "best_auc": float(aucs.max()),
            "events_auc_above_half": int((aucs > .5).sum()),
            "direction_sign_changes": int(np.count_nonzero(signs[1:] != signs[:-1])),
            "macro_direction_free_oracle_auc": float(np.maximum(aucs, 1.0 - aucs).mean()),
            "worst_direction_free_oracle_auc": float(np.maximum(aucs, 1.0 - aucs).min()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["macro_auc", "worst_auc", "candidate"], ascending=[False, False, True]
    )
    summary.to_csv(output / "p8_common_local_candidate_summary.csv", index=False, encoding="utf-8-sig")

    loeo_rows = []
    for held_index, event in enumerate(events):
        development_raw = np.delete(auc_matrix, held_index, axis=0).mean(axis=0)
        directions = np.where(development_raw >= .5, 1, -1)
        selected = int(np.argmax(np.maximum(development_raw, 1.0 - development_raw)))
        held_raw = float(auc_matrix[held_index, selected])
        held_auc = held_raw if directions[selected] > 0 else 1.0 - held_raw
        loeo_rows.append({
            "held_event_index": event["event_index"],
            "held_transition": event["transition"],
            "selected_candidate": names[selected],
            "selected_direction_from_other_six": "higher_after" if directions[selected] > 0 else "lower_after",
            "development_oriented_macro_auc": float(max(development_raw[selected], 1.0 - development_raw[selected])),
            "held_raw_auc": held_raw,
            "held_auc_using_development_direction": held_auc,
        })
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(output / "p8_common_local_loeo.csv", index=False, encoding="utf-8-sig")

    best_raw = summary.iloc[0]
    best_free = summary.sort_values(
        ["macro_direction_free_oracle_auc", "worst_direction_free_oracle_auc"], ascending=False
    ).iloc[0]
    payload = {
        "schema_version": "p8-common-local-modes-exploratory-v1",
        "analysis_status": "designed_after_P7_internal_exploratory_not_confirmatory",
        "ridge_alpha_fixed": RIDGE_ALPHA,
        "candidate_count": len(names),
        "best_posthoc_raw_candidate": best_raw.to_dict(),
        "best_posthoc_direction_free_candidate": best_free.to_dict(),
        "loeo": {
            "macro_held_auc": float(loeo.held_auc_using_development_direction.mean()),
            "worst_held_auc": float(loeo.held_auc_using_development_direction.min()),
            "events_above_half": int((loeo.held_auc_using_development_direction > .5).sum()),
            "distinct_selected_candidates": int(loeo.selected_candidate.nunique()),
        },
        "interpretation_guard": "All feature-family conclusions use D7-D13 labels and require external confirmation.",
    }
    (output / "p8_common_local_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top = summary.head(16).candidate.tolist()
    indices = [names.index(name) for name in top]
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(auc_matrix[:, indices], vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(range(len(events)), [event["transition"] for event in events])
    ax.set_xticks(range(len(top)), top, rotation=75, ha="right", fontsize=7)
    ax.set_title("P8 exploratory common/local-mode AUROC")
    fig.colorbar(image, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(output / "fig01_common_local_top_auc.png", dpi=220)
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

