"""Exploratory bidirectional sequential detector after failed P7 confirmation.

All P5b operating constants are retained.  The sole conceptual change is to
use a two-tailed empirical p-value against the slow reference, so both score
increases and decreases can accumulate evidence.  Candidate screening and all
D7-D13 results are exploratory and label-informed.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_p8_candidate_direction_stability import (
    ROOT,
    SEEDS,
    TRANSITIONS,
    average_precision,
    candidate_family,
    roc_auc,
)


SLOW_WINDOW = 1024
SLOW_STRIDE = 32
KAPPA = 4.0
THRESHOLD = 8.0
DECAY = 0.95
GUARD_P = 0.05


def load_events(source: Path):
    names = None
    events = []
    for event_index, expected in enumerate(TRANSITIONS, start=1):
        split_by_seed = {split: [] for split in ("calibration", "pre_event", "post_event")}
        split_time = {}
        transition = None
        event_ns = None
        for seed in SEEDS:
            folder = source / f"event_{event_index:02d}" / f"seed_{seed}"
            metadata = json.loads((folder / "score_metadata.json").read_text(encoding="utf-8"))
            transition = transition or metadata["transition"]
            if metadata["transition"] != transition:
                raise SystemExit(f"Transition mismatch for event {event_index}")
            event_ns = int(np.datetime64(metadata["event_time_utc"].removesuffix("Z"), "ns").astype(np.int64))
            with np.load(folder / "component_scores.npz") as data:
                current_names = tuple(data["candidate_names"].tolist())
                if names is None:
                    names = current_names
                elif current_names != names:
                    raise SystemExit("Candidate-name mismatch")
                for split in split_by_seed:
                    current_time = data[f"{split}_time_ns"]
                    if split not in split_time:
                        split_time[split] = current_time
                    elif not np.array_equal(split_time[split], current_time):
                        raise SystemExit(f"Seed time mismatch for event {event_index} {split}")
                    split_by_seed[split].append(data[f"{split}_candidate_scores"].astype(np.float64))
        if transition != expected:
            raise SystemExit(f"Unexpected transition {transition}; expected {expected}")
        event = {"event_index": event_index, "transition": transition, "event_ns": event_ns}
        for split, values in split_by_seed.items():
            event[split] = np.mean(np.stack(values), axis=0)
            event[f"{split}_time"] = split_time[split]
        events.append(event)
    return names, events


def two_tailed_empirical_p(reference: np.ndarray, value: float) -> float:
    reference = np.asarray(reference, float)
    lower = (1.0 + np.count_nonzero(reference <= value)) / (len(reference) + 1.0)
    upper = (1.0 + np.count_nonzero(reference >= value)) / (len(reference) + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def bidirectional_cusum(initial: np.ndarray, stream: np.ndarray):
    slow = deque(np.asarray(initial, float)[-SLOW_WINDOW:].tolist(), maxlen=SLOW_WINDOW)
    p_values = np.empty(len(stream), float)
    cusum_values = np.empty(len(stream), float)
    alarms = np.empty(len(stream), bool)
    cusum = 0.0
    for index, raw in enumerate(np.asarray(stream, float)):
        p_value = two_tailed_empirical_p(np.fromiter(slow, float), float(raw))
        cusum = max(0.0, DECAY * cusum - np.log(max(p_value, 1e-12)) - KAPPA)
        alarm = cusum >= THRESHOLD
        p_values[index], cusum_values[index], alarms[index] = p_value, cusum, alarm
        if not alarm and p_value > GUARD_P and (index + 1) % SLOW_STRIDE == 0:
            slow.append(float(raw))
    return p_values, cusum_values, alarms


def run_count(mask: np.ndarray, minimum: int = 3) -> int:
    padded = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.sum((edges[1::2] - edges[::2]) >= minimum))


def detection_delay(mask: np.ndarray, times: np.ndarray, event_ns: int, minimum: int = 3):
    if len(mask) < minimum:
        return None
    hit = np.flatnonzero(np.convolve(np.asarray(mask, np.int8), np.ones(minimum, np.int8), mode="valid") == minimum)
    return None if not len(hit) else float((int(times[int(hit[0])]) - event_ns) / 3_600_000_000_000)


def candidate_rank(frame: pd.DataFrame, event_count: int):
    false_limit = int(np.floor(14 * event_count / 7))
    detected_limit = int(np.ceil(5 * event_count / 7))
    rows = []
    for candidate, group in frame.groupby("candidate", sort=False):
        worst_fpr = float(group.pre_fpr.max())
        false_runs = int(group.pre_false_runs_k3.sum())
        detected = int(group.detection_delay_hours_k3.notna().sum())
        violations = int(worst_fpr > .10) + int(false_runs > false_limit) + int(detected < detected_limit)
        rows.append({
            "candidate": candidate,
            "family": group.family.iloc[0],
            "events": event_count,
            "constraint_violations": violations,
            "worst_pre_fpr": worst_fpr,
            "median_pre_fpr": float(group.pre_fpr.median()),
            "total_pre_false_runs_k3": false_runs,
            "events_detected_k3": detected,
            "median_post_recall": float(group.post_recall.median()),
            "median_detection_delay_hours_k3": (
                None if not group.detection_delay_hours_k3.notna().any()
                else float(group.detection_delay_hours_k3.dropna().median())
            ),
            "macro_evidence_auc": float(group.evidence_auc.mean()),
            "false_run_limit_scaled": false_limit,
            "detected_event_limit_scaled": detected_limit,
        })
    return pd.DataFrame(rows).sort_values(
        ["constraint_violations", "events_detected_k3", "total_pre_false_runs_k3", "worst_pre_fpr", "median_post_recall", "candidate"],
        ascending=[True, False, True, True, False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("cluster_results/extracted_P7_20032_20260811/runs/p7_confirmatory_rescore_v1/01"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/p8_two_sided_sequential_v1"))
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    names, events = load_events(source)
    rows = []
    for event in events:
        for candidate_index, candidate in enumerate(names):
            calibration = event["calibration"][:, candidate_index]
            pre = event["pre_event"][:, candidate_index]
            post = event["post_event"][:, candidate_index]
            p_values, cusum, alarms = bidirectional_cusum(calibration, np.r_[pre, post])
            pre_alarm, post_alarm = alarms[:len(pre)], alarms[len(pre):]
            evidence = -np.log10(np.maximum(p_values, 1e-12))
            labels = np.r_[np.zeros(len(pre), np.int8), np.ones(len(post), np.int8)]
            rows.append({
                "event_index": event["event_index"],
                "transition": event["transition"],
                "candidate_index": candidate_index,
                "candidate": candidate,
                "family": candidate_family(candidate),
                "pre_fpr": float(pre_alarm.mean()),
                "pre_false_runs_k3": run_count(pre_alarm),
                "post_recall": float(post_alarm.mean()),
                "detection_delay_hours_k3": detection_delay(
                    post_alarm, event["post_event_time"], event["event_ns"]
                ),
                "evidence_auc": roc_auc(labels, evidence),
                "evidence_average_precision": average_precision(labels, evidence),
                "max_cusum": float(cusum.max()),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "p8_two_sided_event_metrics.csv", index=False, encoding="utf-8-sig")
    ranking = candidate_rank(metrics, 7)
    ranking.to_csv(output / "p8_two_sided_candidate_summary.csv", index=False, encoding="utf-8-sig")

    loeo_rows = []
    for held_index, held in enumerate(events, start=1):
        development = metrics[metrics.event_index != held_index]
        selected = candidate_rank(development, 6).iloc[0]
        held_row = metrics[(metrics.event_index == held_index) & (metrics.candidate == selected.candidate)].iloc[0]
        loeo_rows.append({
            "held_event_index": held_index,
            "held_transition": held["transition"],
            "selected_candidate_from_other_six": selected.candidate,
            "development_constraint_violations": int(selected.constraint_violations),
            "development_events_detected_k3": int(selected.events_detected_k3),
            "development_total_false_runs_k3": int(selected.total_pre_false_runs_k3),
            "held_pre_fpr": float(held_row.pre_fpr),
            "held_pre_false_runs_k3": int(held_row.pre_false_runs_k3),
            "held_post_recall": float(held_row.post_recall),
            "held_detection_delay_hours_k3": (
                None if pd.isna(held_row.detection_delay_hours_k3)
                else float(held_row.detection_delay_hours_k3)
            ),
            "held_evidence_auc": float(held_row.evidence_auc),
        })
    loeo = pd.DataFrame(loeo_rows)
    loeo.to_csv(output / "p8_two_sided_loeo_selection.csv", index=False, encoding="utf-8-sig")

    best = ranking.iloc[0]
    payload = {
        "schema_version": "p8-two-sided-sequential-exploratory-v1",
        "analysis_status": "designed_after_P7_results_candidate_screening_not_confirmatory",
        "fixed_from_P5b": {
            "slow_window": SLOW_WINDOW,
            "slow_stride": SLOW_STRIDE,
            "kappa": KAPPA,
            "threshold": THRESHOLD,
            "decay": DECAY,
            "guard_p": GUARD_P,
        },
        "conceptual_change": "two-tailed empirical p-value detects both score increases and decreases",
        "candidates_screened": len(names),
        "candidates_meeting_all_locked_style_endpoints_posthoc": int((ranking.constraint_violations == 0).sum()),
        "best_posthoc_candidate_by_declared_lexicographic_rule": best.to_dict(),
        "loeo": {
            "distinct_selected_candidates": int(loeo.selected_candidate_from_other_six.nunique()),
            "held_events_detected_k3": int(loeo.held_detection_delay_hours_k3.notna().sum()),
            "held_total_false_runs_k3": int(loeo.held_pre_false_runs_k3.sum()),
            "worst_held_pre_fpr": float(loeo.held_pre_fpr.max()),
            "median_held_post_recall": float(loeo.held_post_recall.median()),
            "macro_held_evidence_auc": float(loeo.held_evidence_auc.mean()),
        },
        "interpretation_guard": (
            "All candidates and D7-D13 labels were screened after P7. Even a successful candidate "
            "would require a new untouched external confirmation."
        ),
    }
    (output / "p8_two_sided_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top = ranking.head(12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(top.candidate[::-1], top.events_detected_k3[::-1], color="#2f8f5b")
    axes[0].set_xlabel("Events detected with K3")
    axes[0].set_xlim(0, 7.2)
    axes[1].barh(top.candidate[::-1], top.total_pre_false_runs_k3[::-1], color="#d48624")
    axes[1].axvline(14, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Total pre-event false runs (K3)")
    fig.suptitle("P8 exploratory bidirectional sequential candidates")
    fig.tight_layout()
    fig.savefig(output / "fig01_two_sided_candidate_operating.png", dpi=220)
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

