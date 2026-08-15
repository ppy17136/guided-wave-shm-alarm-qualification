"""Exploratory nonlinear environment-regime reference library for P8.

The experiment is deliberately bounded: K in {4, 8, 16}, three frozen path
components, and three fixed local aggregations.  Environment k-means and all
path reference statistics are fitted on each event's training window only.
Designed after P7; results are not confirmatory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_p8_candidate_direction_stability import ROOT, average_precision, roc_auc
from analyze_p8_common_local_modes import COMPONENTS, load_events


K_VALUES = (4, 8, 16)
AGGREGATIONS = ("mean", "top2", "max")
SEED = 20260811


def kmeans(train: np.ndarray, k: int, seed: int, iterations: int = 60):
    rng = np.random.default_rng(seed)
    centers = [train[rng.integers(0, len(train))]]
    distance = np.sum(np.square(train - centers[0]), axis=1)
    while len(centers) < k:
        probability = distance / max(float(distance.sum()), 1e-12)
        centers.append(train[rng.choice(len(train), p=probability)])
        newest = np.sum(np.square(train - centers[-1]), axis=1)
        distance = np.minimum(distance, newest)
    centers = np.asarray(centers, float)
    labels = np.zeros(len(train), int)
    for _ in range(iterations):
        current = np.argmin(np.sum(np.square(train[:, None, :] - centers[None, :, :]), axis=2), axis=1)
        if np.array_equal(current, labels) and _ > 0:
            break
        labels = current
        for index in range(k):
            members = train[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return centers, labels


def assign(values: np.ndarray, centers: np.ndarray):
    distance = np.sum(np.square(values[:, None, :] - centers[None, :, :]), axis=2)
    return np.argmin(distance, axis=1), np.sqrt(np.min(distance, axis=1))


def robust_path_coordinates(train: np.ndarray, values: np.ndarray):
    train_log = np.log1p(np.maximum(train, 0))
    values_log = np.log1p(np.maximum(values, 0))
    median = np.median(train_log, axis=0)
    q25, q75 = np.quantile(train_log, (.25, .75), axis=0)
    z = (values_log - median) / np.maximum(q75 - q25, 1e-6)
    common = np.median(z, axis=1)
    return z - common[:, None]


def local_scores(train_local, train_labels, values_local, value_labels, k):
    global_center = np.median(train_local, axis=0)
    g25, g75 = np.quantile(train_local, (.25, .75), axis=0)
    global_scale = np.maximum(g75 - g25, 1e-6)
    centers, scales = [], []
    for state in range(k):
        members = train_local[train_labels == state]
        if len(members) < 32:
            centers.append(global_center)
            scales.append(global_scale)
        else:
            centers.append(np.median(members, axis=0))
            q25, q75 = np.quantile(members, (.25, .75), axis=0)
            scales.append(np.maximum(q75 - q25, global_scale * .10))
    centers, scales = np.asarray(centers), np.asarray(scales)
    deviation = np.abs((values_local - centers[value_labels]) / scales[value_labels])
    return {
        "mean": deviation.mean(axis=1),
        "top2": np.sort(deviation, axis=1)[:, -2:].mean(axis=1),
        "max": deviation.max(axis=1),
    }


def derive_event(event, event_seed):
    train_env = event["train"]["environment"]
    env_mean = train_env.mean(axis=0)
    env_scale = np.maximum(train_env.std(axis=0), 1e-6)
    env = {split: (event[split]["environment"] - env_mean) / env_scale for split in event if split in ("train", "calibration", "pre_event", "post_event")}
    candidates = {split: {} for split in env}
    for k in K_VALUES:
        centers, train_labels = kmeans(env["train"], k, event_seed + k)
        labels = {"train": train_labels}
        support = {}
        for split in ("calibration", "pre_event", "post_event"):
            labels[split], support[split] = assign(env[split], centers)
        for split in ("calibration", "pre_event", "post_event"):
            candidates[split][f"environment_support__k{k}"] = support[split]
        for component in COMPONENTS:
            train_local = robust_path_coordinates(event["train"][component], event["train"][component])
            for split in ("calibration", "pre_event", "post_event"):
                values_local = robust_path_coordinates(event["train"][component], event[split][component])
                scores = local_scores(train_local, train_labels, values_local, labels[split], k)
                for aggregation in AGGREGATIONS:
                    candidates[split][f"{component}__regime_k{k}__local_{aggregation}"] = scores[aggregation]
    names = tuple(sorted(candidates["pre_event"]))
    matrices = {
        split: np.column_stack([candidates[split][name] for name in names])
        for split in ("calibration", "pre_event", "post_event")
    }
    return names, matrices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path("cluster_results/extracted_P7_20032_20260811/runs/p7_confirmatory_rescore_v1/01"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/p8_environment_regime_library_v1"))
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    events = load_events(source)
    names = None
    auc_rows, rows = [], []
    for event in events:
        current_names, matrices = derive_event(event, SEED + event["event_index"] * 100)
        if names is None:
            names = current_names
        elif names != current_names:
            raise SystemExit("Candidate mismatch")
        labels = np.r_[np.zeros(len(matrices["pre_event"]), np.int8), np.ones(len(matrices["post_event"]), np.int8)]
        scores = np.vstack((matrices["pre_event"], matrices["post_event"]))
        event_aucs = []
        for index, name in enumerate(names):
            auc = roc_auc(labels, scores[:, index])
            event_aucs.append(auc)
            rows.append({
                "event_index": event["event_index"],
                "transition": event["transition"],
                "candidate_index": index,
                "candidate": name,
                "roc_auc": auc,
                "average_precision": average_precision(labels, scores[:, index]),
                "direction_free_oracle_auc": max(auc, 1.0 - auc),
            })
        auc_rows.append(event_aucs)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "p8_regime_event_metrics.csv", index=False, encoding="utf-8-sig")
    auc_matrix = np.asarray(auc_rows, float)

    summary_rows = []
    for index, name in enumerate(names):
        aucs = auc_matrix[:, index]
        signs = np.where(aucs >= .5, 1, -1)
        summary_rows.append({
            "candidate_index": index,
            "candidate": name,
            "macro_auc": float(aucs.mean()),
            "worst_auc": float(aucs.min()),
            "best_auc": float(aucs.max()),
            "events_auc_above_half": int((aucs > .5).sum()),
            "direction_sign_changes": int(np.count_nonzero(signs[1:] != signs[:-1])),
            "macro_direction_free_oracle_auc": float(np.maximum(aucs, 1.0 - aucs).mean()),
            "worst_direction_free_oracle_auc": float(np.maximum(aucs, 1.0 - aucs).min()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["macro_auc", "worst_auc"], ascending=False)
    summary.to_csv(output / "p8_regime_candidate_summary.csv", index=False, encoding="utf-8-sig")

    loeo_rows = []
    for held_index, event in enumerate(events):
        dev = np.delete(auc_matrix, held_index, axis=0).mean(axis=0)
        direction = np.where(dev >= .5, 1, -1)
        selected = int(np.argmax(np.maximum(dev, 1.0 - dev)))
        held_raw = float(auc_matrix[held_index, selected])
        held_auc = held_raw if direction[selected] > 0 else 1.0 - held_raw
        loeo_rows.append({
            "held_event_index": event["event_index"],
            "held_transition": event["transition"],
            "selected_candidate": names[selected],
            "selected_direction_from_other_six": "higher_after" if direction[selected] > 0 else "lower_after",
            "development_oriented_macro_auc": float(max(dev[selected], 1.0 - dev[selected])),
            "held_raw_auc": held_raw,
            "held_auc_using_development_direction": held_auc,
        })
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(output / "p8_regime_loeo.csv", index=False, encoding="utf-8-sig")

    best = summary.iloc[0]
    best_free = summary.sort_values(["macro_direction_free_oracle_auc", "worst_direction_free_oracle_auc"], ascending=False).iloc[0]
    payload = {
        "schema_version": "p8-environment-regime-library-exploratory-v1",
        "analysis_status": "bounded_mechanism_test_after_P7_not_confirmatory",
        "k_values": list(K_VALUES),
        "aggregations": list(AGGREGATIONS),
        "candidate_count": len(names),
        "best_posthoc_raw_candidate": best.to_dict(),
        "best_posthoc_direction_free_candidate": best_free.to_dict(),
        "loeo": {
            "macro_held_auc": float(loeo.held_auc_using_development_direction.mean()),
            "worst_held_auc": float(loeo.held_auc_using_development_direction.min()),
            "events_above_half": int((loeo.held_auc_using_development_direction > .5).sum()),
            "distinct_selected_candidates": int(loeo.selected_candidate.nunique()),
        },
        "stop_rule": "Stop this regime-library branch if leave-one-event-out transfer remains below 0.5 macro AUROC.",
        "branch_passed_stop_rule": bool(loeo.held_auc_using_development_direction.mean() >= .5),
    }
    (output / "p8_regime_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    top = summary.head(14).candidate.tolist()
    indices = [names.index(name) for name in top]
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(auc_matrix[:, indices], vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(range(len(events)), [event["transition"] for event in events])
    ax.set_xticks(range(len(top)), top, rotation=75, ha="right", fontsize=7)
    ax.set_title("P8 bounded environment-regime reference experiment")
    fig.colorbar(image, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(output / "fig01_regime_candidate_auc.png", dpi=220)
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

