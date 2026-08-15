"""P8-B bounded internal experiment: dynamic environment-conditioned path graph.

This implementation follows research_protocols/P8_dynamic_path_graph_protocol_v1.md.
It was designed after P7 and is strictly exploratory/internal.  No branch or
operating constant is selected using D7-D13 labels; the primary branch is the
fixed equal fusion of shape and amplitude-error graph residuals.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_p8_candidate_direction_stability import ROOT, SEEDS, TRANSITIONS, average_precision, roc_auc


SPLITS = ("train", "calibration", "pre_event", "post_event")
COMPONENTS = ("path_shape_error", "path_amplitude_error", "input_amplitude")
BRANCHES = ("shape_graph", "amplitude_graph", "input_graph_negative_control", "shape_amplitude_equal_fusion")
PRIMARY_BRANCH = "shape_amplitude_equal_fusion"
RFF_DIMENSION = 256
RFF_SEED = 20260811
RIDGE_ALPHA = 1.0
FAST_WINDOW = 128
SLOW_WINDOW = 2048
UPDATE_STRIDE = 32
SHRINKAGE = 0.25
CALIBRATION_QUANTILE = 0.999
ENERGY_FAST = 32
ENERGY_SLOW_SAMPLE = 64
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260811
DAY_NS = 86_400_000_000_000


def load_events(source: Path):
    events = []
    for event_index, expected in enumerate(TRANSITIONS, start=1):
        arrays = {split: {name: [] for name in (*COMPONENTS, "environment")} for split in SPLITS}
        times = {}
        transition = None
        event_ns = None
        for seed in SEEDS:
            folder = source / f"event_{event_index:02d}" / f"seed_{seed}"
            metadata = json.loads((folder / "score_metadata.json").read_text(encoding="utf-8"))
            transition = transition or metadata["transition"]
            if metadata["transition"] != transition:
                raise SystemExit(f"Transition mismatch in event {event_index}")
            event_ns = int(np.datetime64(metadata["event_time_utc"].removesuffix("Z"), "ns").astype(np.int64))
            with np.load(folder / "component_scores.npz") as data:
                for split in SPLITS:
                    current_time = data[f"{split}_time_ns"]
                    if split not in times:
                        times[split] = current_time
                    elif not np.array_equal(times[split], current_time):
                        raise SystemExit(f"Time mismatch in event {event_index} {split}")
                    for name in arrays[split]:
                        arrays[split][name].append(data[f"{split}_{name}"].astype(np.float64))
        if transition != expected:
            raise SystemExit(f"Unexpected transition {transition}; expected {expected}")
        event = {"event_index": event_index, "transition": transition, "event_ns": event_ns}
        for split in SPLITS:
            event[split] = {name: np.mean(np.stack(values), axis=0) for name, values in arrays[split].items()}
            event[f"{split}_time"] = times[split]
        events.append(event)
    return events


def path_edges(train: np.ndarray, values: np.ndarray):
    train_log = np.log1p(np.maximum(np.asarray(train, float), 0))
    value_log = np.log1p(np.maximum(np.asarray(values, float), 0))
    median = np.median(train_log, axis=0)
    q25, q75 = np.quantile(train_log, (.25, .75), axis=0)
    z = (value_log - median) / np.maximum(q75 - q25, 1e-6)
    pairs = [(left, right) for left in range(z.shape[1]) for right in range(left + 1, z.shape[1])]
    return np.column_stack([z[:, left] - z[:, right] for left, right in pairs])


def median_environment_scale(train: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    count = min(len(train), 768)
    sample = train[rng.choice(len(train), count, replace=False)]
    left = sample[:count // 2]
    right = sample[count // 2:count // 2 * 2]
    distance = np.linalg.norm(left - right, axis=1)
    positive = distance[distance > 1e-12]
    return max(float(np.median(positive)) if len(positive) else 1.0, 1e-3)


def rff_features(train_env: np.ndarray, values_env: np.ndarray, seed: int):
    mean = train_env.mean(axis=0)
    scale = np.maximum(train_env.std(axis=0), 1e-6)
    train_z = (train_env - mean) / scale
    values_z = (values_env - mean) / scale
    length = median_environment_scale(train_z, seed)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0 / length, size=(train_z.shape[1], RFF_DIMENSION))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=RFF_DIMENSION)
    factor = np.sqrt(2.0 / RFF_DIMENSION)
    train_phi = factor * np.cos(train_z @ weights + phase)
    value_phi = factor * np.cos(values_z @ weights + phase)
    return np.column_stack((np.ones(len(train_phi)), train_phi)), np.column_stack((np.ones(len(value_phi)), value_phi)), train_z, values_z


def fit_event_residuals(event):
    train_env = event["train"]["environment"]
    phi_by_split = {}
    train_phi = None
    environment_z = {}
    for split in SPLITS:
        current_train_phi, current_phi, train_z, values_z = rff_features(
            train_env, event[split]["environment"], RFF_SEED + event["event_index"]
        )
        train_phi = current_train_phi
        phi_by_split[split] = current_phi
        environment_z[split] = values_z

    edges = {split: {} for split in SPLITS}
    for component in COMPONENTS:
        train_component = event["train"][component]
        for split in SPLITS:
            edges[split][component] = path_edges(train_component, event[split][component])

    train_targets = np.column_stack([edges["train"][component] for component in COMPONENTS])
    penalty = np.eye(train_phi.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(train_phi.T @ train_phi + penalty, train_phi.T @ train_targets)
    widths = [edges["train"][component].shape[1] for component in COMPONENTS]
    offsets = np.cumsum([0, *widths])
    residuals = {split: {} for split in SPLITS}
    for split in SPLITS:
        prediction = phi_by_split[split] @ coefficients
        for index, component in enumerate(COMPONENTS):
            residuals[split][component] = edges[split][component] - prediction[:, offsets[index]:offsets[index + 1]]

    for component in COMPONENTS:
        train_residual = residuals["train"][component]
        center = np.median(train_residual, axis=0)
        q25, q75 = np.quantile(train_residual, (.25, .75), axis=0)
        scale = np.maximum(q75 - q25, 1e-6)
        for split in SPLITS:
            residuals[split][component] = (residuals[split][component] - center) / scale

    branch = {split: {} for split in SPLITS}
    for split in SPLITS:
        branch[split]["shape_graph"] = residuals[split]["path_shape_error"]
        branch[split]["amplitude_graph"] = residuals[split]["path_amplitude_error"]
        branch[split]["input_graph_negative_control"] = residuals[split]["input_amplitude"]
        branch[split]["shape_amplitude_equal_fusion"] = np.column_stack((
            residuals[split]["path_shape_error"] / np.sqrt(2.0),
            residuals[split]["path_amplitude_error"] / np.sqrt(2.0),
        ))

    train_cov = np.cov(environment_z["train"], rowvar=False)
    level = max(float(np.trace(train_cov) / train_cov.shape[0]), 1e-6)
    covariance = (1.0 - SHRINKAGE) * train_cov + SHRINKAGE * np.eye(train_cov.shape[0]) * level
    inverse = np.linalg.pinv(covariance, rcond=1e-6)
    environment_distance = {
        split: np.einsum("ni,ij,nj->n", environment_z[split], inverse, environment_z[split])
        for split in SPLITS
    }
    support_threshold = float(np.quantile(environment_distance["calibration"], CALIBRATION_QUANTILE))
    support = {split: environment_distance[split] <= support_threshold for split in SPLITS}
    return branch, environment_distance, support, support_threshold


class DynamicGraphState:
    def __init__(self, train: np.ndarray):
        self.fast = deque(np.asarray(train[-FAST_WINDOW:], float).tolist(), maxlen=FAST_WINDOW)
        self.slow = deque(np.asarray(train[-SLOW_WINDOW:], float).tolist(), maxlen=SLOW_WINDOW)
        self.fast_sum = np.sum(np.asarray(self.fast), axis=0)
        slow_array = np.asarray(self.slow, float)
        self.slow_sum = np.sum(slow_array, axis=0)
        self.slow_outer = slow_array.T @ slow_array
        self.inverse = None
        self.refresh_inverse()

    def refresh_inverse(self):
        count = len(self.slow)
        mean = self.slow_sum / count
        covariance = (self.slow_outer - count * np.outer(mean, mean)) / max(count - 1, 1)
        if np.ndim(covariance) == 0:
            covariance = np.asarray([[float(covariance)]])
        dimension = covariance.shape[0]
        level = max(float(np.trace(covariance) / dimension), 1e-6)
        covariance = (1.0 - SHRINKAGE) * covariance + SHRINKAGE * np.eye(dimension) * level
        self.inverse = np.linalg.pinv(covariance, rcond=1e-5)

    def append_fast(self, value: np.ndarray):
        if len(self.fast) == self.fast.maxlen:
            self.fast_sum -= np.asarray(self.fast[0], float)
        self.fast.append(np.asarray(value, float))
        self.fast_sum += value

    def statistic(self):
        fast_mean = self.fast_sum / len(self.fast)
        slow_mean = self.slow_sum / len(self.slow)
        delta = fast_mean - slow_mean
        return float(delta @ self.inverse @ delta)

    def update_slow(self, value: np.ndarray):
        if len(self.slow) == self.slow.maxlen:
            removed = np.asarray(self.slow[0], float)
            self.slow_sum -= removed
            self.slow_outer -= np.outer(removed, removed)
        self.slow.append(np.asarray(value, float))
        self.slow_sum += value
        self.slow_outer += np.outer(value, value)
        self.refresh_inverse()


def run_dynamic(state: DynamicGraphState, values: np.ndarray, support: np.ndarray, threshold=None):
    statistics = np.empty(len(values), float)
    alarms = np.zeros(len(values), bool)
    for index, value in enumerate(np.asarray(values, float)):
        state.append_fast(value)
        statistic = state.statistic()
        alarm = bool(threshold is not None and support[index] and statistic > threshold)
        statistics[index], alarms[index] = statistic, alarm
        if support[index] and not alarm and (index + 1) % UPDATE_STRIDE == 0:
            state.update_slow(value)
    return statistics, alarms


def energy_distance(fast: np.ndarray, slow: np.ndarray):
    cross = np.linalg.norm(fast[:, None, :] - slow[None, :, :], axis=2).mean()
    within_fast = np.linalg.norm(fast[:, None, :] - fast[None, :, :], axis=2).mean()
    within_slow = np.linalg.norm(slow[:, None, :] - slow[None, :, :], axis=2).mean()
    return float(max(0.0, 2.0 * cross - within_fast - within_slow))


def rolling_energy(train: np.ndarray, values: np.ndarray, support: np.ndarray):
    fast = deque(np.asarray(train[-ENERGY_FAST:], float).tolist(), maxlen=ENERGY_FAST)
    slow = deque(np.asarray(train[-SLOW_WINDOW:], float).tolist(), maxlen=SLOW_WINDOW)
    output = np.empty(len(values), float)
    last = 0.0
    for index, value in enumerate(np.asarray(values, float)):
        fast.append(value)
        if index % UPDATE_STRIDE == 0:
            slow_array = np.asarray(slow, float)
            selected = np.linspace(0, len(slow_array) - 1, min(ENERGY_SLOW_SAMPLE, len(slow_array)), dtype=int)
            last = energy_distance(np.asarray(fast, float), slow_array[selected])
        output[index] = last
        if support[index] and (index + 1) % UPDATE_STRIDE == 0:
            slow.append(value)
    return output


def run_count(mask: np.ndarray, minimum: int = 3):
    padded = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.sum((edges[1::2] - edges[::2]) >= minimum))


def alarm_duration_hours(mask: np.ndarray, times: np.ndarray):
    mask = np.asarray(mask, bool)
    times = np.asarray(times, np.int64)
    if not mask.any():
        return 0.0, 0.0
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    durations = []
    typical_step = int(np.median(np.diff(times))) if len(times) > 1 else 0
    for start, stop in zip(edges[::2], edges[1::2]):
        end_time = int(times[stop - 1]) + typical_step
        durations.append((end_time - int(times[start])) / 3_600_000_000_000)
    return float(np.sum(durations)), float(np.max(durations))


def detection_delay(mask: np.ndarray, times: np.ndarray, event_ns: int, minimum: int = 3):
    hit = np.flatnonzero(np.convolve(np.asarray(mask, np.int8), np.ones(minimum, np.int8), mode="valid") == minimum)
    return None if not len(hit) else float((int(times[int(hit[0])]) - event_ns) / 3_600_000_000_000)


def natural_blocks(values: np.ndarray, times: np.ndarray, hours: int = 24):
    ids = (np.asarray(times, np.int64) - int(times[0])) // int(hours * 3_600_000_000_000)
    return [np.asarray(values, float)[ids == block] for block in np.unique(ids)]


def block_bootstrap(event_values, repetitions, seed):
    rng = np.random.default_rng(seed)
    output = np.empty(repetitions, float)
    prepared = [(natural_blocks(pre, pre_time), natural_blocks(post, post_time)) for pre, pre_time, post, post_time in event_values]
    for replicate in range(repetitions):
        aucs = []
        for pre_blocks, post_blocks in prepared:
            pre = np.concatenate([pre_blocks[i] for i in rng.integers(0, len(pre_blocks), len(pre_blocks))])
            post = np.concatenate([post_blocks[i] for i in rng.integers(0, len(post_blocks), len(post_blocks))])
            labels = np.r_[np.zeros(len(pre), np.int8), np.ones(len(post), np.int8)]
            aucs.append(roc_auc(labels, np.r_[pre, post]))
        output[replicate] = np.mean(aucs)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path("cluster_results/extracted_P7_20032_20260811/runs/p7_confirmatory_rescore_v1/01"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/p8_dynamic_path_graph_v1"))
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    events = load_events(source)
    rows, traces, primary_bootstrap_values = [], [], []
    for event in events:
        branch, environment_distance, support, support_threshold = fit_event_residuals(event)
        for branch_name in BRANCHES:
            state = DynamicGraphState(branch["train"][branch_name])
            calibration_stat, _ = run_dynamic(
                state, branch["calibration"][branch_name], support["calibration"], threshold=None
            )
            valid_calibration = calibration_stat[support["calibration"]]
            threshold = float(np.quantile(valid_calibration, CALIBRATION_QUANTILE))
            stream = np.vstack((branch["pre_event"][branch_name], branch["post_event"][branch_name]))
            stream_support = np.r_[support["pre_event"], support["post_event"]]
            stream_stat, stream_alarm = run_dynamic(state, stream, stream_support, threshold=threshold)
            pre_count = len(branch["pre_event"][branch_name])
            pre_stat, post_stat = stream_stat[:pre_count], stream_stat[pre_count:]
            pre_alarm, post_alarm = stream_alarm[:pre_count], stream_alarm[pre_count:]
            labels = np.r_[np.zeros(len(pre_stat), np.int8), np.ones(len(post_stat), np.int8)]
            total_duration, max_duration = alarm_duration_hours(pre_alarm, event["pre_event_time"])

            energy_stream = rolling_energy(branch["train"][branch_name], stream, stream_support)
            energy_pre, energy_post = energy_stream[:pre_count], energy_stream[pre_count:]
            rows.append({
                "event_index": event["event_index"],
                "transition": event["transition"],
                "branch": branch_name,
                "threshold": threshold,
                "roc_auc": roc_auc(labels, stream_stat),
                "average_precision": average_precision(labels, stream_stat),
                "energy_sensitivity_auc": roc_auc(labels, energy_stream),
                "pre_fpr": float(pre_alarm.mean()),
                "pre_false_runs_k3": run_count(pre_alarm),
                "pre_alarm_total_hours": total_duration,
                "pre_alarm_max_run_hours": max_duration,
                "post_recall": float(post_alarm.mean()),
                "detection_delay_hours_k3": detection_delay(post_alarm, event["post_event_time"], event["event_ns"]),
                "pre_unknown_rate": float((~support["pre_event"]).mean()),
                "post_unknown_rate": float((~support["post_event"]).mean()),
                "environment_support_threshold": support_threshold,
            })
            if branch_name == PRIMARY_BRANCH:
                primary_bootstrap_values.append((
                    pre_stat, event["pre_event_time"], post_stat, event["post_event_time"]
                ))
                traces.append(pd.DataFrame({
                    "event_index": event["event_index"],
                    "transition": event["transition"],
                    "phase": np.r_[np.repeat("pre", len(pre_stat)), np.repeat("post", len(post_stat))],
                    "time_ns": np.r_[event["pre_event_time"], event["post_event_time"]],
                    "statistic": stream_stat,
                    "energy_statistic": energy_stream,
                    "threshold": threshold,
                    "environment_supported": stream_support,
                    "alarm": stream_alarm,
                    "environment_distance": np.r_[environment_distance["pre_event"], environment_distance["post_event"]],
                }))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "p8_dynamic_graph_event_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(traces, ignore_index=True).to_csv(
        output / "p8_dynamic_graph_primary_traces.csv.gz", index=False, compression="gzip"
    )

    summaries = []
    for branch_name, group in metrics.groupby("branch", sort=False):
        summaries.append({
            "branch": branch_name,
            "macro_auc": float(group.roc_auc.mean()),
            "worst_auc": float(group.roc_auc.min()),
            "events_auc_above_half": int((group.roc_auc > .5).sum()),
            "macro_energy_sensitivity_auc": float(group.energy_sensitivity_auc.mean()),
            "worst_pre_fpr": float(group.pre_fpr.max()),
            "median_pre_fpr": float(group.pre_fpr.median()),
            "total_pre_false_runs_k3": int(group.pre_false_runs_k3.sum()),
            "total_pre_alarm_hours": float(group.pre_alarm_total_hours.sum()),
            "max_pre_alarm_run_hours": float(group.pre_alarm_max_run_hours.max()),
            "events_detected_k3": int(group.detection_delay_hours_k3.notna().sum()),
            "median_post_recall": float(group.post_recall.median()),
            "median_detection_delay_hours_k3": (
                None if not group.detection_delay_hours_k3.notna().any()
                else float(group.detection_delay_hours_k3.dropna().median())
            ),
            "worst_pre_unknown_rate": float(group.pre_unknown_rate.max()),
            "worst_post_unknown_rate": float(group.post_unknown_rate.max()),
        })
    branch_summary = pd.DataFrame(summaries)
    branch_summary.to_csv(output / "p8_dynamic_graph_branch_summary.csv", index=False, encoding="utf-8-sig")

    bootstrap = block_bootstrap(primary_bootstrap_values, BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED)
    np.save(output / "p8_dynamic_graph_primary_macro_auc_bootstrap_24h.npy", bootstrap)
    primary = branch_summary[branch_summary.branch == PRIMARY_BRANCH].iloc[0]
    go_no_go = {
        "macro_auc_at_least_055": bool(primary.macro_auc >= .55),
        "worst_auc_at_least_040": bool(primary.worst_auc >= .40),
        "at_least_four_events_above_half": bool(primary.events_auc_above_half >= 4),
        "worst_pre_fpr_at_most_015": bool(primary.worst_pre_fpr <= .15),
        "false_runs_at_most_14": bool(primary.total_pre_false_runs_k3 <= 14),
        "at_least_five_events_detected": bool(primary.events_detected_k3 >= 5),
    }
    payload = {
        "schema_version": "p8-dynamic-path-graph-internal-v1",
        "analysis_status": "bounded_internal_exploratory_after_P7_not_confirmatory",
        "protocol": "research_protocols/P8_dynamic_path_graph_protocol_v1.md",
        "primary_branch_fixed_before_execution": PRIMARY_BRANCH,
        "parameters": {
            "rff_dimension": RFF_DIMENSION,
            "rff_seed": RFF_SEED,
            "ridge_alpha": RIDGE_ALPHA,
            "fast_window": FAST_WINDOW,
            "slow_window": SLOW_WINDOW,
            "update_stride": UPDATE_STRIDE,
            "covariance_shrinkage": SHRINKAGE,
            "calibration_quantile": CALIBRATION_QUANTILE,
        },
        "primary_branch": primary.to_dict(),
        "primary_24h_bootstrap": {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "mean": float(bootstrap.mean()),
            "one_sided_lower95": float(np.quantile(bootstrap, .05)),
            "q025": float(np.quantile(bootstrap, .025)),
            "q975": float(np.quantile(bootstrap, .975)),
        },
        "go_no_go_checks": go_no_go,
        "branch_passed_all_internal_gates": bool(all(go_no_go.values())),
        "interpretation_guard": "D7-D13 were already inspected; passing would justify external validation only, not confirm efficacy.",
    }
    (output / "p8_dynamic_graph_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary_events = metrics[metrics.branch == PRIMARY_BRANCH]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(primary_events.transition, primary_events.roc_auc, color="#31688e")
    axes[0].axhline(.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("AUROC"); axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(primary_events.transition, primary_events.pre_fpr, color="#d48624")
    axes[1].axhline(.15, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Pre-event FPR"); axes[1].tick_params(axis="x", rotation=35)
    fig.suptitle("P8-B fixed shape-amplitude path-graph fusion")
    fig.tight_layout(); fig.savefig(output / "fig01_primary_event_performance.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(branch_summary))
    ax.bar(x - .18, branch_summary.macro_auc, width=.36, label="Mahalanobis fast-slow")
    ax.bar(x + .18, branch_summary.macro_energy_sensitivity_auc, width=.36, label="Energy sensitivity")
    ax.axhline(.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, branch_summary.branch, rotation=25, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Macro AUROC"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "fig02_branch_ablation.png", dpi=220); plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

