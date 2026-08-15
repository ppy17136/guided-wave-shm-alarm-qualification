"""Export rich causal split scores from the 35 completed P2 checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.envwave.event_dual_branch import EventDualBranchSSL
from src.envwave.model import masked_wave
from src.run_p2_event_ssl import DAY_NS, EventDataset, build_records, select

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "calibration", "pre_event", "post_event")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_loader(records, stats, batch_size, workers):
    return DataLoader(EventDataset(records, stats), batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True, persistent_workers=False, drop_last=False)


@torch.no_grad()
def extract_components(model, records, stats, device, model_cfg, inference_cfg, seed):
    data_loader = make_loader(records, stats, int(inference_cfg["batch_size"]), int(inference_cfg["workers"]))
    model.eval()
    seed_all(seed)
    result = {name: [] for name in ("time", "environment", "input_amplitude", "z_fused", "z_shape",
                                     "z_amplitude", "path_shape_error", "path_amplitude_error")}
    repeats = int(inference_cfg["mask_repeats"])
    ratio = float(model_cfg["mask_ratio"])
    block = int(model_cfg["mask_block_samples"])
    for batch in data_loader:
        shape = batch["shape"].to(device, non_blocking=True)
        amplitude = batch["amp"].to(device, non_blocking=True)
        environment = batch["env"].to(device, non_blocking=True)
        accum = None
        for _ in range(repeats):
            corrupted, mask = masked_wave(shape, ratio, block)
            output = model(corrupted, amplitude, environment)
            path_shape = (torch.square(output["wave_reconstruction"] - shape) * mask).sum(2) / mask.sum(2).clamp_min(1)
            path_amplitude = torch.square(output["amplitude_prediction"] - amplitude)
            current = {
                "z_fused": output["z_fused"], "z_shape": output["z_shape"], "z_amplitude": output["z_amplitude"],
                "path_shape_error": path_shape, "path_amplitude_error": path_amplitude,
            }
            if accum is None:
                accum = {key: value.clone() for key, value in current.items()}
            else:
                for key, value in current.items():
                    accum[key].add_(value)
        for key in accum:
            accum[key].div_(repeats)
        result["time"].append(batch["time"].numpy())
        result["environment"].append(batch["env"].numpy())
        result["input_amplitude"].append(batch["amp"].numpy())
        for key in ("z_fused", "z_shape", "z_amplitude", "path_shape_error", "path_amplitude_error"):
            result[key].append(accum[key].float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


def diagonal_distance(train, values):
    center = train.mean(0)
    scale = np.maximum(train.std(0), 1e-5)
    return np.mean(np.square((values - center) / scale), axis=1)


def shrinkage_distance(train, values):
    center = train.mean(0)
    centered = train - center
    covariance = np.cov(centered, rowvar=False)
    dimension = covariance.shape[0]
    diagonal_level = max(float(np.trace(covariance) / dimension), 1e-6)
    covariance = .75 * covariance + .25 * np.eye(dimension) * diagonal_level
    inverse = np.linalg.pinv(covariance, rcond=1e-5)
    delta = values - center
    return np.einsum("ni,ij,nj->n", delta, inverse, delta)


def cosine_distance(train, values):
    center = train.mean(0)
    center /= max(float(np.linalg.norm(center)), 1e-8)
    norm = np.maximum(np.linalg.norm(values, axis=1), 1e-8)
    return 1.0 - (values @ center) / norm


def path_aggregates(train, values, prefix, raw):
    median = np.median(train, axis=0)
    q25, q75 = np.quantile(train, [.25, .75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    z = (values - median) / scale
    high = np.maximum(z, 0)
    two = np.abs(z)
    raw[f"{prefix}_high_mean"] = high.mean(1)
    raw[f"{prefix}_high_top2"] = np.sort(high, axis=1)[:, -2:].mean(1)
    raw[f"{prefix}_high_max"] = high.max(1)
    raw[f"{prefix}_two_mean"] = two.mean(1)
    raw[f"{prefix}_two_top2"] = np.sort(two, axis=1)[:, -2:].mean(1)
    raw[f"{prefix}_two_max"] = two.max(1)


def robust_variants(train_raw, values_by_split):
    candidate_by_split = {split: {} for split in SPLITS}
    for name, train_values in train_raw.items():
        median = float(np.median(train_values))
        q25, q75 = np.quantile(train_values, [.25, .75])
        scale = max(float(q75 - q25), 1e-6)
        for split in SPLITS:
            z = (values_by_split[split][name] - median) / scale
            candidate_by_split[split][f"{name}__high"] = np.maximum(z, 0)
            candidate_by_split[split][f"{name}__two_sided"] = np.abs(z)
    for split in SPLITS:
        candidates = candidate_by_split[split]
        candidates["fusion_equal_high"] = np.mean(np.column_stack([
            candidates["fused_diag__high"], candidates["shape_path_high_top2__high"],
            candidates["amplitude_path_high_top2__high"]]), axis=1)
        candidates["fusion_equal_two_sided"] = np.mean(np.column_stack([
            candidates["fused_diag__two_sided"], candidates["shape_path_two_top2__high"],
            candidates["amplitude_path_two_top2__high"]]), axis=1)
    names = sorted(candidate_by_split["train"])
    matrices = {split: np.column_stack([candidate_by_split[split][name] for name in names]).astype(np.float32)
                for split in SPLITS}
    return names, matrices


def derive_scores(components):
    train = components["train"]
    raw_by_split = {split: {} for split in SPLITS}
    for split in SPLITS:
        data = components[split]
        for embedding_name, prefix in (("z_fused", "fused"), ("z_shape", "shape_embedding"),
                                       ("z_amplitude", "amplitude_embedding")):
            raw_by_split[split][f"{prefix}_diag"] = diagonal_distance(train[embedding_name], data[embedding_name])
            raw_by_split[split][f"{prefix}_cosine"] = cosine_distance(train[embedding_name], data[embedding_name])
        raw_by_split[split]["fused_shrinkage"] = shrinkage_distance(train["z_fused"], data["z_fused"])
        path_aggregates(train["path_shape_error"], data["path_shape_error"], "shape_path", raw_by_split[split])
        path_aggregates(train["path_amplitude_error"], data["path_amplitude_error"], "amplitude_path", raw_by_split[split])
        path_aggregates(train["input_amplitude"], data["input_amplitude"], "input_amplitude", raw_by_split[split])
        path_aggregates(train["environment"], data["environment"], "environment_control", raw_by_split[split])
    return robust_variants(raw_by_split["train"], raw_by_split)


def split_records(event_ns):
    records = build_records(event_ns - 15 * DAY_NS, event_ns + 5 * DAY_NS)
    return {
        "train": select(records, event_ns - 15 * DAY_NS, event_ns - 8 * DAY_NS),
        "calibration": select(records, event_ns - 8 * DAY_NS, event_ns - 5 * DAY_NS),
        "pre_event": select(records, event_ns - 5 * DAY_NS, event_ns),
        "post_event": select(records, event_ns, event_ns + 5 * DAY_NS),
    }


def export_one(config, event, seed, device, smoke=False, force=False):
    event_index = int(event["event_index"])
    source = ROOT / config["source"]["checkpoint_root"] / f"event_{event_index:02d}" / f"seed_{seed}" / "checkpoint.pt"
    base = config["runtime"]["smoke_output_dir"] if smoke else config["runtime"]["output_dir"]
    output = ROOT / base / f"event_{event_index:02d}" / f"seed_{seed}"
    completed = output / "SCORES_COMPLETED.json"
    if completed.exists() and not force:
        return {"event_index": event_index, "seed": seed, "status": "skipped"}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "SCORES_RUNNING.json", {"started_utc": utc_now(), "source": str(source), "smoke": smoke})
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint["config"]
        model_cfg = checkpoint_config["model"]
        model = EventDualBranchSSL(int(model_cfg["embedding_dim"]), 7, int(model_cfg["transformer_layers"]),
                                   int(model_cfg["attention_heads"]), float(model_cfg["dropout"]))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        stats = tuple(np.asarray(value, dtype=np.float64) for value in checkpoint["stats"])
        event_ns = int(np.datetime64(event["event_time_utc"].removesuffix("Z"), "ns").astype(np.int64))
        records = split_records(event_ns)
        if smoke:
            for split in records:
                step = max(1, len(records[split]) // 96)
                records[split] = records[split][::step][:96]
        components = {}
        for split_index, split in enumerate(SPLITS):
            components[split] = extract_components(model, records[split], stats, device, model_cfg,
                                                    config["inference"], seed + 10000 * split_index)
        candidate_names, matrices = derive_scores(components)
        arrays = {"candidate_names": np.asarray(candidate_names, dtype="U80")}
        for split in SPLITS:
            arrays[f"{split}_time_ns"] = components[split]["time"].astype(np.int64)
            arrays[f"{split}_environment"] = components[split]["environment"].astype(np.float32)
            arrays[f"{split}_path_shape_error"] = components[split]["path_shape_error"].astype(np.float32)
            arrays[f"{split}_path_amplitude_error"] = components[split]["path_amplitude_error"].astype(np.float32)
            arrays[f"{split}_input_amplitude"] = components[split]["input_amplitude"].astype(np.float32)
            arrays[f"{split}_candidate_scores"] = matrices[split]
        np.savez_compressed(output / "component_scores.npz", **arrays)
        metadata = {"schema_version": "p3-checkpoint-score-export-v1", "completed_utc": utc_now(),
                    "event_index": event_index, "transition": event["transition"], "event_time_utc": event["event_time_utc"],
                    "seed": seed, "smoke": smoke, "source_checkpoint": str(source.relative_to(ROOT)),
                    "candidate_count": len(candidate_names), "candidate_names": candidate_names,
                    "split_counts": {split: len(records[split]) for split in SPLITS},
                    "mask_repeats": int(config["inference"]["mask_repeats"]), "device": str(device),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "leakage_guard": "all score transformations are fitted on train only; no post labels are used"}
        atomic_json(output / "score_metadata.json", metadata)
        atomic_json(completed, {"completed_utc": utc_now(), "event_index": event_index, "seed": seed})
        (output / "SCORES_RUNNING.json").unlink(missing_ok=True)
        return {"event_index": event_index, "seed": seed, "status": "completed", "candidates": len(candidate_names)}
    except Exception as error:
        atomic_json(output / "SCORES_FAILED.json", {"failed_utc": utc_now(), "error": str(error),
                                                     "traceback": traceback.format_exc()})
        (output / "SCORES_RUNNING.json").unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["data"]["event_manifest"]).read_text(encoding="utf-8"))
    events = manifest["events"]
    seeds = [int(seed) for seed in config["source"]["seeds"]]
    checkpoints = [ROOT / config["source"]["checkpoint_root"] / f"event_{int(event['event_index']):02d}" /
                   f"seed_{seed}" / "checkpoint.pt" for event in events for seed in seeds]
    audit = {"schema_version": "p3-score-export-audit-v1", "events": len(events), "seeds": len(seeds),
             "checkpoints_expected": len(checkpoints), "checkpoints_present": sum(path.is_file() for path in checkpoints),
             "cuda": torch.cuda.is_available(), "config": str(args.config)}
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["checkpoints_present"] != 35:
        raise SystemExit("Expected 35 source checkpoints")
    if args.dry_run or not args.execute:
        if not args.execute:
            return
    if args.execute and not torch.cuda.is_available():
        raise SystemExit("Score export requires a visible CUDA GPU")
    device = torch.device("cuda")
    selected_events = events[:1] if args.smoke else events
    selected_seeds = seeds[:1] if args.smoke else seeds
    results = [export_one(config, event, seed, device, args.smoke, args.force)
               for event in selected_events for seed in selected_seeds]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
