"""Pre-registered P7 confirmation analysis for untouched D7-D13 scores."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRANSITIONS = ("D6->D7", "D7->D8", "D8->D9", "D9->D10", "D10->D11", "D11->D12", "D12->D13")
SEEDS = (20260809, 20260810, 20260811, 20260812, 20260813)
ALPHA = 0.01
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_REPETITIONS = 5000
SENSITIVITY_REPETITIONS = 2000


def roc_auc(labels, scores) -> float:
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


def average_precision(labels, scores) -> float:
    labels = np.asarray(labels, np.int8)
    ranked = labels[np.argsort(-np.asarray(scores, float), kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].mean())


def empirical_p(reference, target):
    ordered = np.sort(np.asarray(reference, float))
    count_ge = len(ordered) - np.searchsorted(ordered, np.asarray(target, float), side="left")
    return (1.0 + count_ge) / (len(ordered) + 1.0)


def run_count(mask, minimum=3) -> int:
    padded = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.sum((edges[1::2] - edges[::2]) >= minimum))


def detection_delay(alarm, times, event_ns, minimum=3):
    alarm = np.asarray(alarm, np.int8)
    if len(alarm) < minimum:
        return None
    hit = np.flatnonzero(np.convolve(alarm, np.ones(minimum, np.int8), mode="valid") == minimum)
    return None if not len(hit) else float((int(times[int(hit[0])]) - event_ns) / 3_600_000_000_000)


def path_scores(train_path, values):
    train_log = np.log1p(np.maximum(np.asarray(train_path, float), 0))
    value_log = np.log1p(np.maximum(np.asarray(values, float), 0))
    median = np.median(train_log, axis=0)
    q25, q75 = np.quantile(train_log, (0.25, 0.75), axis=0)
    z = (value_log - median) / np.maximum(q75 - q25, 1e-6)
    path_iqr = np.quantile(z, 0.75, axis=1) - np.quantile(z, 0.25, axis=1)
    max_minus_median = np.max(z, axis=1) - np.median(z, axis=1)
    return np.column_stack((path_iqr, max_minus_median))


def load_events(source: Path):
    events = {}
    for event_index in range(1, 8):
        seed_items = []
        for seed in SEEDS:
            folder = source / f"event_{event_index:02d}" / f"seed_{seed}"
            npz_path = folder / "component_scores.npz"
            metadata_path = folder / "score_metadata.json"
            if not npz_path.is_file() or not metadata_path.is_file():
                raise SystemExit(f"Missing P7 score result: {folder}")
            z = np.load(npz_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            names = tuple(z["candidate_names"].tolist())
            fused_index = names.index("fused_diag__two_sided")
            item = {"seed": seed, "metadata": metadata}
            for split in ("calibration", "pre_event", "post_event"):
                path = path_scores(z["train_path_shape_error"], z[f"{split}_path_shape_error"])
                item[split] = {
                    "path_iqr": path[:, 0],
                    "max_minus_median": path[:, 1],
                    "fused_diag__two_sided": z[f"{split}_candidate_scores"][:, fused_index],
                    "time": z[f"{split}_time_ns"],
                }
            seed_items.append(item)
        transition = seed_items[0]["metadata"]["transition"]
        if transition != EXPECTED_TRANSITIONS[event_index - 1]:
            raise SystemExit(f"Unexpected event order: {transition}")
        for other in seed_items[1:]:
            if other["metadata"]["transition"] != transition:
                raise SystemExit("Seed transition mismatch")
            for split in ("calibration", "pre_event", "post_event"):
                if not np.array_equal(seed_items[0][split]["time"], other[split]["time"]):
                    raise SystemExit(f"Seed time mismatch for event {event_index} {split}")
        ensemble = {"transition": transition, "event_ns": int(np.datetime64(seed_items[0]["metadata"]["event_time_utc"].removesuffix("Z"), "ns").astype(np.int64))}
        for split in ("calibration", "pre_event", "post_event"):
            ensemble[f"{split}_time"] = seed_items[0][split]["time"]
            for score in ("path_iqr", "max_minus_median", "fused_diag__two_sided"):
                ensemble[f"{split}_{score}"] = np.mean(np.stack([item[split][score] for item in seed_items]), axis=0)
        events[event_index] = ensemble
    return events


def rolling_p(initial, stream, window=1024):
    reference = deque(np.asarray(initial, float)[-window:].tolist(), maxlen=window)
    output = np.empty(len(stream), float)
    for index, value in enumerate(np.asarray(stream, float)):
        array = np.fromiter(reference, float)
        output[index] = float(empirical_p(array, [value])[0])
        reference.append(float(value))
    return output


def leaky_cusum(initial, stream):
    fast_window, slow_window, slow_stride = 128, 1024, 32
    kappa, threshold, decay, guard_p = 4.0, 8.0, 0.95, 0.05
    fast = deque(np.asarray(initial, float)[-fast_window:].tolist(), maxlen=fast_window)
    slow = deque(np.asarray(initial, float)[-slow_window:].tolist(), maxlen=slow_window)
    p_values = np.empty(len(stream), float)
    cusum_values = np.empty(len(stream), float)
    alarms = np.empty(len(stream), bool)
    cusum = 0.0
    for index, raw in enumerate(np.asarray(stream, float)):
        fast_array = np.fromiter(fast, float)
        slow_array = np.fromiter(slow, float)
        adjusted = raw - np.median(fast_array) + np.median(slow_array)
        p_value = (1.0 + np.count_nonzero(slow_array >= adjusted)) / (len(slow_array) + 1.0)
        cusum = max(0.0, decay * cusum - np.log(max(p_value, 1e-12)) - kappa)
        alarm = cusum >= threshold
        p_values[index], cusum_values[index], alarms[index] = p_value, cusum, alarm
        fast.append(float(raw))
        if not alarm and p_value > guard_p and (index + 1) % slow_stride == 0:
            slow.append(float(raw))
    return p_values, cusum_values, alarms


def metric_row(event_index, item, score_name, strategy, p_pre, p_post):
    pre = item[f"pre_event_{score_name}"]
    post = item[f"post_event_{score_name}"]
    alarm_pre, alarm_post = np.asarray(p_pre) <= ALPHA, np.asarray(p_post) <= ALPHA
    labels = np.r_[np.zeros(len(pre), np.int8), np.ones(len(post), np.int8)]
    scores = np.r_[pre, post]
    return {
        "event_index": event_index,
        "transition": item["transition"],
        "strategy": strategy,
        "score": score_name,
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "pre_fpr": float(alarm_pre.mean()),
        "pre_false_runs_k3": run_count(alarm_pre, 3),
        "post_recall": float(alarm_post.mean()),
        "detection_delay_hours_k3": detection_delay(alarm_post, item["post_event_time"], item["event_ns"], 3),
    }


def evaluate(events):
    rows = []
    traces = []
    for event_index, item in events.items():
        for score in ("path_iqr", "max_minus_median"):
            calibration = item[f"calibration_{score}"]
            pre, post = item[f"pre_event_{score}"], item[f"post_event_{score}"]
            rows.append(metric_row(event_index, item, score, "static_empirical_alpha001", empirical_p(calibration, pre), empirical_p(calibration, post)))
            p = rolling_p(calibration, np.r_[pre, post], 1024)
            rows.append(metric_row(event_index, item, score, "rolling_w1024_guard0_alpha001", p[:len(pre)], p[len(pre):]))

        score = "fused_diag__two_sided"
        pre, post = item[f"pre_event_{score}"], item[f"post_event_{score}"]
        p, cusum, alarm = leaky_cusum(item[f"calibration_{score}"], np.r_[pre, post])
        pseudo_p = np.where(alarm, 0.0, 1.0)
        rows.append(metric_row(event_index, item, score, "p5b_leaky_f128_s32_k4_h8_r095_guard005", pseudo_p[:len(pre)], pseudo_p[len(pre):]))
        traces.append(pd.DataFrame({
            "event_index": event_index,
            "transition": item["transition"],
            "phase": np.r_[np.repeat("pre", len(pre)), np.repeat("post", len(post))],
            "time_ns": np.r_[item["pre_event_time"], item["post_event_time"]],
            "score": np.r_[pre, post],
            "p_value": p,
            "cusum": cusum,
            "alarm": alarm,
        }))
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True)


def natural_blocks(values, times, block_hours):
    values = np.asarray(values, float)
    times = np.asarray(times, np.int64)
    block_ns = int(block_hours * 3_600_000_000_000)
    ids = (times - times[0]) // block_ns
    return [values[ids == value] for value in np.unique(ids)]


def block_bootstrap(events, block_hours, repetitions, rng):
    prepared = []
    for item in events.values():
        prepared.append((
            natural_blocks(item["pre_event_path_iqr"], item["pre_event_time"], block_hours),
            natural_blocks(item["post_event_path_iqr"], item["post_event_time"], block_hours),
        ))
    macro = np.empty(repetitions, float)
    for replicate in range(repetitions):
        aucs = []
        for pre_blocks, post_blocks in prepared:
            pre = np.concatenate([pre_blocks[i] for i in rng.integers(0, len(pre_blocks), len(pre_blocks))])
            post = np.concatenate([post_blocks[i] for i in rng.integers(0, len(post_blocks), len(post_blocks))])
            labels = np.r_[np.zeros(len(pre), np.int8), np.ones(len(post), np.int8)]
            aucs.append(roc_auc(labels, np.r_[pre, post]))
        macro[replicate] = np.mean(aucs)
    return macro


def exact_sign_test(differences):
    differences = np.asarray(differences, float)
    observed = float(differences.mean())
    null = np.array([np.mean(differences * signs) for signs in itertools.product((-1.0, 1.0), repeat=len(differences))])
    return {"mean_difference": observed, "one_sided_exact_p": float(np.mean(null >= observed)), "permutations": int(len(null))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("runs/p7_confirmatory_rescore_v1/01"))
    parser.add_argument("--output", type=Path, default=Path("runs/p7_confirmatory_analysis_v1/01"))
    args = parser.parse_args()
    source, output = ROOT / args.source, ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    events = load_events(source)
    metrics, traces = evaluate(events)
    metrics.to_csv(output / "p7_event_metrics.csv", index=False, encoding="utf-8-sig")
    traces.to_csv(output / "p7_p5b_alarm_traces.csv.gz", index=False, compression="gzip")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    for hours, repetitions in ((24, BOOTSTRAP_REPETITIONS), (6, SENSITIVITY_REPETITIONS), (12, SENSITIVITY_REPETITIONS), (48, SENSITIVITY_REPETITIONS)):
        values = block_bootstrap(events, hours, repetitions, rng)
        np.save(output / f"path_iqr_macro_auc_bootstrap_{hours}h.npy", values)
        bootstrap_rows.append({
            "block_hours": hours,
            "repetitions": repetitions,
            "mean": float(values.mean()),
            "lower_one_sided_95": float(np.quantile(values, 0.05)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
        })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output / "p7_path_iqr_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    static = metrics[metrics.strategy == "static_empirical_alpha001"].set_index(["score", "event_index"])
    primary_event = static.loc["path_iqr"].sort_index()
    comparison_event = static.loc["max_minus_median"].sort_index()
    macro_auc = float(primary_event.roc_auc.mean())
    lower95 = float(bootstrap.loc[bootstrap.block_hours == 24, "lower_one_sided_95"].iloc[0])
    events_above_half = int((primary_event.roc_auc > 0.5).sum())
    ranking_success = bool(macro_auc >= 0.65 and lower95 > 0.50 and events_above_half >= 5)

    events_without_gap = {key: value for key, value in events.items() if value["transition"] != "D9->D10"}
    gap_sensitivity_values = block_bootstrap(
        events_without_gap, 24, BOOTSTRAP_REPETITIONS, np.random.default_rng(BOOTSTRAP_SEED + 1)
    )
    np.save(output / "path_iqr_macro_auc_bootstrap_24h_excluding_D9_D10.npy", gap_sensitivity_values)
    sensitivity_event = primary_event[primary_event.transition != "D9->D10"]

    alarm = metrics[metrics.strategy == "p5b_leaky_f128_s32_k4_h8_r095_guard005"].copy()
    alarm_success = bool(alarm.pre_fpr.max() <= 0.10 and alarm.pre_false_runs_k3.sum() <= 14 and alarm.detection_delay_hours_k3.notna().sum() >= 5)
    paired = exact_sign_test(primary_event.roc_auc.to_numpy() - comparison_event.roc_auc.to_numpy())
    summary = {
        "schema_version": "p7-confirmatory-analysis-v1",
        "confirmation_events": list(EXPECTED_TRANSITIONS),
        "primary_ranking": {
            "score": "path_iqr",
            "macro_mean_auc": macro_auc,
            "worst_auc": float(primary_event.roc_auc.min()),
            "events_auc_above_half": events_above_half,
            "bootstrap_24h_one_sided_lower95": lower95,
            "success": ranking_success,
        },
        "primary_alarm": {
            "strategy": "p5b_leaky_f128_s32_k4_h8_r095_guard005",
            "worst_pre_fpr": float(alarm.pre_fpr.max()),
            "median_pre_fpr": float(alarm.pre_fpr.median()),
            "total_pre_false_runs_k3": int(alarm.pre_false_runs_k3.sum()),
            "events_detected_k3": int(alarm.detection_delay_hours_k3.notna().sum()),
            "median_post_recall": float(alarm.post_recall.median()),
            "median_detection_delay_hours_k3": None if not alarm.detection_delay_hours_k3.notna().any() else float(alarm.detection_delay_hours_k3.dropna().median()),
            "success": alarm_success,
        },
        "A2_metadata_gap_sensitivity_not_part_of_primary_decision": {
            "excluded_transition": "D9->D10",
            "reason": "training baseline begins 27.076 h after planned start; calibration and both test windows are complete",
            "events": 6,
            "macro_mean_auc": float(sensitivity_event.roc_auc.mean()),
            "worst_auc": float(sensitivity_event.roc_auc.min()),
            "events_auc_above_half": int((sensitivity_event.roc_auc > .5).sum()),
            "bootstrap_24h_one_sided_lower95": float(np.quantile(gap_sensitivity_values, .05)),
        },
        "paired_path_iqr_vs_max_minus_median": paired,
        "overall_interpretation_code": (
            "ranking_and_alarm_confirmed" if ranking_success and alarm_success else
            "ranking_confirmed_alarm_failed" if ranking_success else
            "ranking_failed_alarm_confirmed" if alarm_success else
            "both_confirmatory_endpoints_failed"
        ),
        "no_hyperparameter_selection_on_confirmation_events": True,
    }
    (output / "p7_confirmatory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_data = primary_event.reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(figure_data.transition, figure_data.roc_auc, color="#2369a8")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.axhline(0.65, color="#c0392b", linestyle=":", linewidth=1.5)
    ax.set_ylim(0, 1); ax.set_ylabel("AUROC"); ax.set_title("P7 untouched confirmation: path-IQR by event")
    ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(output / "fig01_path_iqr_event_auc.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(alarm.transition, alarm.pre_fpr, color="#d48624"); axes[0].axhline(.10, color="black", linestyle="--")
    axes[0].set_ylabel("Pre-event FPR"); axes[0].set_ylim(0, max(.12, float(alarm.pre_fpr.max()) * 1.15)); axes[0].tick_params(axis="x", rotation=35)
    detected = alarm.detection_delay_hours_k3.notna().astype(int)
    axes[1].bar(alarm.transition, detected, color="#2f8f5b"); axes[1].set_ylim(0, 1.15); axes[1].set_ylabel("K3 detected within 5 days")
    axes[1].tick_params(axis="x", rotation=35)
    fig.suptitle("P7 locked P5b operating behavior"); fig.tight_layout(); fig.savefig(output / "fig02_p5b_operating.png", dpi=220); plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

