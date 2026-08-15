"""Exploratory post-P7 audit of candidate direction stability.

This analysis was designed after the P7 confirmatory results were known.  It
uses D7-D13 labels and must never be represented as confirmatory or external
validation.  Its purpose is to decide whether P7 failed because all learned
representations were uninformative, or because score direction changed across
damage stages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260809, 20260810, 20260811, 20260812, 20260813)
TRANSITIONS = ("D6->D7", "D7->D8", "D8->D9", "D9->D10", "D10->D11", "D11->D12", "D12->D13")


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, np.int8)
    scores = np.asarray(scores, np.float64)
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg or not np.isfinite(scores).all():
        raise ValueError("invalid AUROC input")
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(len(scores), np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    ranked = np.asarray(labels, np.int8)[np.argsort(-np.asarray(scores, float), kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].mean())


def load_ensembles(source: Path):
    names = None
    events = []
    for event_index, expected_transition in enumerate(TRANSITIONS, start=1):
        seed_pre, seed_post = [], []
        pre_time = post_time = None
        transition = None
        for seed in SEEDS:
            folder = source / f"event_{event_index:02d}" / f"seed_{seed}"
            score_path = folder / "component_scores.npz"
            metadata_path = folder / "score_metadata.json"
            if not score_path.is_file() or not metadata_path.is_file():
                raise SystemExit(f"Missing score export: {folder}")
            with np.load(score_path) as data:
                current_names = tuple(data["candidate_names"].tolist())
                if names is None:
                    names = current_names
                elif current_names != names:
                    raise SystemExit("Candidate-name mismatch")
                current_pre_time = data["pre_event_time_ns"]
                current_post_time = data["post_event_time_ns"]
                if pre_time is None:
                    pre_time, post_time = current_pre_time, current_post_time
                elif not np.array_equal(pre_time, current_pre_time) or not np.array_equal(post_time, current_post_time):
                    raise SystemExit(f"Seed time mismatch in event {event_index}")
                seed_pre.append(data["pre_event_candidate_scores"].astype(np.float64))
                seed_post.append(data["post_event_candidate_scores"].astype(np.float64))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            transition = transition or metadata["transition"]
            if metadata["transition"] != transition:
                raise SystemExit(f"Seed transition mismatch in event {event_index}")
        if transition != expected_transition:
            raise SystemExit(f"Unexpected transition {transition}; expected {expected_transition}")
        events.append({
            "event_index": event_index,
            "transition": transition,
            "pre": np.mean(np.stack(seed_pre), axis=0),
            "post": np.mean(np.stack(seed_post), axis=0),
        })
    return names, events


def candidate_family(name: str) -> str:
    base = name.split("__", 1)[0]
    for prefix in (
        "shape_path", "amplitude_path", "input_amplitude", "environment_control",
        "shape_embedding", "amplitude_embedding", "fused", "fusion_equal",
    ):
        if base.startswith(prefix):
            return prefix
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("cluster_results/extracted_P7_20032_20260811/runs/p7_confirmatory_rescore_v1/01"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/p8_exploratory_candidate_audit_v1"))
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    names, events = load_ensembles(source)
    rows = []
    auc_matrix = np.empty((len(events), len(names)), float)
    ap_matrix = np.empty_like(auc_matrix)
    for event_offset, event in enumerate(events):
        labels = np.r_[np.zeros(len(event["pre"]), np.int8), np.ones(len(event["post"]), np.int8)]
        values = np.vstack((event["pre"], event["post"]))
        for candidate_index, name in enumerate(names):
            auc = roc_auc(labels, values[:, candidate_index])
            ap = average_precision(labels, values[:, candidate_index])
            auc_matrix[event_offset, candidate_index] = auc
            ap_matrix[event_offset, candidate_index] = ap
            rows.append({
                "event_index": event["event_index"],
                "transition": event["transition"],
                "candidate_index": candidate_index,
                "candidate": name,
                "family": candidate_family(name),
                "roc_auc": auc,
                "average_precision": ap,
                "direction": "higher_after" if auc >= .5 else "lower_after",
                "direction_free_oracle_auc": max(auc, 1.0 - auc),
            })
    event_frame = pd.DataFrame(rows)
    event_frame.to_csv(output / "p8_candidate_event_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for candidate_index, name in enumerate(names):
        aucs = auc_matrix[:, candidate_index]
        aps = ap_matrix[:, candidate_index]
        signs = np.where(aucs >= .5, 1, -1)
        summary_rows.append({
            "candidate_index": candidate_index,
            "candidate": name,
            "family": candidate_family(name),
            "macro_auc": float(aucs.mean()),
            "worst_auc": float(aucs.min()),
            "best_auc": float(aucs.max()),
            "macro_average_precision": float(aps.mean()),
            "events_auc_above_half": int((aucs > .5).sum()),
            "higher_after_events": int((signs > 0).sum()),
            "direction_sign_changes": int(np.count_nonzero(signs[1:] != signs[:-1])),
            "macro_direction_free_oracle_auc": float(np.maximum(aucs, 1.0 - aucs).mean()),
            "worst_direction_free_oracle_auc": float(np.maximum(aucs, 1.0 - aucs).min()),
        })
    candidate_frame = pd.DataFrame(summary_rows).sort_values(
        ["macro_auc", "worst_auc", "candidate"], ascending=[False, False, True]
    )
    candidate_frame.to_csv(output / "p8_candidate_summary.csv", index=False, encoding="utf-8-sig")

    loeo_rows = []
    for held_index, event in enumerate(events):
        development = np.delete(auc_matrix, held_index, axis=0)
        development_raw = development.mean(axis=0)
        direction = np.where(development_raw >= .5, 1, -1)
        development_oriented = np.maximum(development_raw, 1.0 - development_raw)
        selected = int(np.argmax(development_oriented))
        held_raw = float(auc_matrix[held_index, selected])
        held_auc = held_raw if direction[selected] > 0 else 1.0 - held_raw
        loeo_rows.append({
            "held_event_index": event["event_index"],
            "held_transition": event["transition"],
            "selected_candidate": names[selected],
            "selected_direction_from_other_six": "higher_after" if direction[selected] > 0 else "lower_after",
            "development_six_event_oriented_macro_auc": float(development_oriented[selected]),
            "held_raw_auc": held_raw,
            "held_auc_using_development_direction": held_auc,
        })
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(output / "p8_leave_one_event_out_selection.csv", index=False, encoding="utf-8-sig")

    raw_best_index = int(np.argmax(auc_matrix.mean(axis=0)))
    direction_free_best_index = int(np.argmax(np.maximum(auc_matrix, 1.0 - auc_matrix).mean(axis=0)))
    oracle_rows = []
    for event_index, event in enumerate(events):
        raw_index = int(np.argmax(auc_matrix[event_index]))
        free_index = int(np.argmax(np.maximum(auc_matrix[event_index], 1.0 - auc_matrix[event_index])))
        oracle_rows.append({
            "event_index": event["event_index"],
            "transition": event["transition"],
            "best_high_direction_candidate": names[raw_index],
            "best_high_direction_auc": float(auc_matrix[event_index, raw_index]),
            "best_direction_free_candidate": names[free_index],
            "best_direction_free_oracle_auc": float(max(auc_matrix[event_index, free_index], 1.0 - auc_matrix[event_index, free_index])),
            "direction_free_oracle_orientation": "higher_after" if auc_matrix[event_index, free_index] >= .5 else "lower_after",
        })
    oracle = pd.DataFrame(oracle_rows)
    oracle.to_csv(output / "p8_event_oracle_diagnostics.csv", index=False, encoding="utf-8-sig")

    locked = json.loads((ROOT / "runs/p7_confirmatory_analysis_v1/01/p7_confirmatory_summary.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "p8-candidate-direction-stability-exploratory-v1",
        "analysis_status": "designed_after_P7_results_exploratory_not_confirmatory",
        "uses_D7_D13_labels": True,
        "candidate_count": len(names),
        "events": list(TRANSITIONS),
        "locked_P7_reference_macro_auc": locked["primary_ranking"]["macro_mean_auc"],
        "best_posthoc_raw_macro_candidate": candidate_frame.iloc[0].to_dict(),
        "best_posthoc_direction_free_candidate": candidate_frame.sort_values(
            ["macro_direction_free_oracle_auc", "worst_direction_free_oracle_auc"], ascending=False
        ).iloc[0].to_dict(),
        "loeo_selection": {
            "macro_held_auc_using_development_direction": float(loeo.held_auc_using_development_direction.mean()),
            "worst_held_auc_using_development_direction": float(loeo.held_auc_using_development_direction.min()),
            "held_events_above_half": int((loeo.held_auc_using_development_direction > .5).sum()),
            "distinct_selected_candidates": int(loeo.selected_candidate.nunique()),
        },
        "eventwise_oracle": {
            "macro_best_high_direction_auc": float(oracle.best_high_direction_auc.mean()),
            "macro_best_direction_free_auc": float(oracle.best_direction_free_oracle_auc.mean()),
        },
        "interpretation_guard": (
            "Per-event oracle direction and posthoc candidate maxima quantify available signal only; "
            "they are not deployable estimates and cannot rescue the failed P7 confirmation."
        ),
    }
    (output / "p8_exploratory_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top = candidate_frame.head(16).candidate.tolist()
    top_indices = [names.index(name) for name in top]
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(auc_matrix[:, top_indices], vmin=0, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(range(len(events)), [event["transition"] for event in events])
    ax.set_xticks(range(len(top)), top, rotation=75, ha="right", fontsize=8)
    ax.set_title("P8 exploratory: event AUROC of top posthoc raw candidates")
    fig.colorbar(image, ax=ax, label="AUROC (higher score = post-event)")
    fig.tight_layout()
    fig.savefig(output / "fig01_top_candidate_event_auc_heatmap.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        candidate_frame.macro_auc,
        candidate_frame.macro_direction_free_oracle_auc,
        c=candidate_frame.direction_sign_changes,
        cmap="viridis",
        s=55,
        alpha=.85,
    )
    ax.axvline(.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Seven-event macro AUROC (fixed high direction)")
    ax.set_ylabel("Macro direction-free oracle AUROC")
    ax.set_title("Representation signal versus direction instability")
    fig.colorbar(scatter, ax=ax, label="Direction sign changes across events")
    fig.tight_layout()
    fig.savefig(output / "fig02_direction_instability_scatter.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(loeo.held_transition, loeo.held_auc_using_development_direction, color="#326b9b")
    ax.axhline(.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-event AUROC")
    ax.set_title("Exploratory leave-one-event-out candidate and direction selection")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output / "fig03_loeo_held_event_auc.png", dpi=220)
    plt.close(fig)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

