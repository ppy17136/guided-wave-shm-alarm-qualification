"""Execute one submission-safe P1 job.

Only runners explicitly marked ready by ``build_p1_execution_package.py`` are
allowed to train.  The first implemented runner is a strict one-way masked-wave
self-supervised anomaly pilot: 2018 healthy data are split chronologically into
train/calibration, while 2021 remains completely unseen until locked evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from src.envwave.numpy_metrics_v1 import average_precision, brier_score, log_loss_binary, roc_auc
from torch.utils.data import DataLoader, Dataset

from src.envwave.model import EnvWaveSSL, masked_wave


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ZarrWaveDataset(Dataset):
    def __init__(self, store: Path, indices: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self._group = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _open(self):
        if self._group is None:
            self._group = zarr.open_group(str(self.store), mode="r")
        return self._group

    def __getitem__(self, item: int):
        group = self._open()
        index = int(self.indices[item])
        wave = np.asarray(group["guided_wave"][index], dtype=np.float32)
        wave = (wave - self.mean[:, None]) / (self.std[:, None] + 1e-6)
        return {
            "wave": torch.from_numpy(wave),
            "label": torch.tensor(int(group["damage_tag"][index] > 0), dtype=torch.int64),
            "timestamp_ns": torch.tensor(int(group["datetime_ns"][index]), dtype=torch.int64),
            "index": torch.tensor(index, dtype=torch.int64),
        }


def choose_evenly(indices: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or indices.size <= limit:
        return indices
    positions = np.linspace(0, indices.size - 1, num=limit, dtype=np.int64)
    return indices[positions]


def store_metadata(store: Path) -> dict[str, object]:
    group = zarr.open_group(str(store), mode="r")
    required = ["guided_wave", "datetime_ns", "damage_tag", "humidity"]
    missing = [name for name in required if name not in group]
    if missing:
        raise ValueError(f"{store} missing arrays: {missing}")
    shape = tuple(int(x) for x in group["guided_wave"].shape)
    if shape[1:] != (8, 2000):
        raise ValueError(f"Unexpected guided_wave shape {shape}; expected (N, 8, 2000)")
    return {
        "store": str(store),
        "shape": shape,
        "time_min": str(np.asarray(group["datetime_ns"][:], dtype="datetime64[ns]").min()),
        "time_max": str(np.asarray(group["datetime_ns"][:], dtype="datetime64[ns]").max()),
        "damage_counts": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(np.asarray(group["damage_tag"][:]), return_counts=True))
        },
    }


def build_split(config: dict[str, object], smoke: bool) -> dict[str, np.ndarray]:
    data = config["data"]
    train_store = ROOT / data["train_store"]
    test_store = ROOT / data["test_store"]
    train_group = zarr.open_group(str(train_store), mode="r")
    test_group = zarr.open_group(str(test_store), mode="r")

    train_ts = np.asarray(train_group["datetime_ns"][:], dtype=np.int64)
    train_damage = np.asarray(train_group["damage_tag"][:], dtype=np.int16)
    train_humidity = np.asarray(train_group["humidity"][:], dtype=np.float64)
    valid = (train_damage == 0) & np.isfinite(train_humidity) & (train_humidity >= 0) & (train_humidity <= 100)
    valid_indices = np.flatnonzero(valid)
    valid_indices = valid_indices[np.argsort(train_ts[valid_indices], kind="stable")]
    split = int(round(valid_indices.size * float(data["chronological_train_fraction"])))
    if split <= 0 or split >= valid_indices.size:
        raise ValueError("Chronological train/calibration split is empty")
    train_indices, calibration_indices = valid_indices[:split], valid_indices[split:]

    test_ts = np.asarray(test_group["datetime_ns"][:], dtype=np.int64)
    test_indices = np.argsort(test_ts, kind="stable").astype(np.int64)
    if smoke:
        train_indices = choose_evenly(train_indices, 64)
        calibration_indices = choose_evenly(calibration_indices, 32)
        labels = np.asarray(test_group["damage_tag"][:], dtype=np.int16) > 0
        healthy = choose_evenly(np.flatnonzero(~labels), 32)
        damaged = choose_evenly(np.flatnonzero(labels), 32)
        test_indices = np.sort(np.concatenate([healthy, damaged])).astype(np.int64)
    return {
        "train": train_indices,
        "calibration": calibration_indices,
        "test": test_indices,
    }


def wave_stats(store: Path, indices: np.ndarray, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    group = zarr.open_group(str(store), mode="r")
    array = group["guided_wave"]
    total = np.zeros(8, dtype=np.float64)
    total_sq = np.zeros(8, dtype=np.float64)
    count = 0
    for start in range(0, indices.size, batch):
        selected = indices[start:start + batch]
        values = np.asarray(array.oindex[selected, :, :], dtype=np.float64)
        total += values.sum(axis=(0, 2))
        total_sq += np.square(values).sum(axis=(0, 2))
        count += values.shape[0] * values.shape[2]
    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def make_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def reconstruction_loss(model: EnvWaveSSL, wave: torch.Tensor, ratio: float, block: int) -> torch.Tensor:
    corrupted, mask = masked_wave(wave, ratio=ratio, block=block)
    reconstruction = model(corrupted)["wave_reconstruction"]
    error = torch.square(reconstruction - wave)
    return error.masked_select(mask).mean()


@torch.no_grad()
def evaluate_scores(
    model: EnvWaveSSL, loader: DataLoader, device: torch.device,
    ratio: float, block: int, seed: int
) -> dict[str, np.ndarray]:
    model.eval()
    seed_all(seed)
    scores, labels, timestamps, indices = [], [], [], []
    for batch in loader:
        wave = batch["wave"].to(device, non_blocking=True)
        corrupted, mask = masked_wave(wave, ratio=ratio, block=block)
        reconstruction = model(corrupted)["wave_reconstruction"]
        sample_error = (torch.square(reconstruction - wave) * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)
        scores.append(sample_error.detach().cpu().numpy())
        labels.append(batch["label"].numpy())
        timestamps.append(batch["timestamp_ns"].numpy())
        indices.append(batch["index"].numpy())
    return {
        "score": np.concatenate(scores),
        "label": np.concatenate(labels),
        "timestamp_ns": np.concatenate(timestamps),
        "index": np.concatenate(indices),
    }


def empirical_percentile(calibration_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    ranks = np.searchsorted(reference, scores, side="right")
    return np.clip((ranks + 1) / (reference.size + 2), 1e-6, 1 - 1e-6)


def expected_calibration_error(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            result += mask.mean() * abs(float(probability[mask].mean()) - float(labels[mask].mean()))
    return float(result)


def metrics_from_scores(calibration: dict[str, np.ndarray], test: dict[str, np.ndarray], quantile: float) -> dict[str, object]:
    cal_score = calibration["score"]
    score = test["score"]
    labels = test["label"].astype(np.int8)
    if np.unique(labels).size < 2:
        raise ValueError("Locked test set must contain both healthy and damaged samples")
    threshold = float(np.quantile(cal_score, quantile, method="higher"))
    probability = empirical_percentile(cal_score, score)
    alarm = score > threshold
    healthy, damaged = labels == 0, labels == 1
    healthy_times = test["timestamp_ns"][healthy].astype("datetime64[ns]")
    duration_days = max(float((healthy_times.max() - healthy_times.min()) / np.timedelta64(1, "D")), 1e-6)
    return {
        "roc_auc": roc_auc(labels, score),
        "average_precision": average_precision(labels, score),
        "threshold_quantile": quantile,
        "threshold": threshold,
        "test_recall": float(alarm[damaged].mean()),
        "test_false_positive_rate": float(alarm[healthy].mean()),
        "test_false_alarms": int(alarm[healthy].sum()),
        "test_false_alarms_per_30_days": float(alarm[healthy].sum() / duration_days * 30),
        "pilot_brier_empirical_percentile": brier_score(labels, probability),
        "pilot_nll_empirical_percentile": log_loss_binary(labels, probability),
        "pilot_ece_empirical_percentile": expected_calibration_error(probability, labels),
        "calibration_samples": int(cal_score.size),
        "test_healthy_samples": int(healthy.sum()),
        "test_damage_samples": int(damaged.sum()),
        "probability_warning": "Calibration metrics use an empirical anomaly percentile, not a supervised damage probability.",
    }


def write_scores(path: Path, result: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["zarr_index", "timestamp_ns", "timestamp", "damage_label", "anomaly_score"])
        for index, timestamp, label, score in zip(result["index"], result["timestamp_ns"], result["label"], result["score"]):
            writer.writerow([int(index), int(timestamp), str(np.datetime64(int(timestamp), "ns")), int(label), float(score)])


def train_seed(config: dict[str, object], seed: int, smoke: bool, force: bool) -> dict[str, object]:
    job_id = config["job"]["id"]
    out = (ROOT / "runs/p1_matrix_v1_smoke" / job_id / f"seed_{seed}") if smoke else (ROOT / config["runtime"]["output_dir"] / f"seed_{seed}")
    metrics_path = out / "metrics.json"
    if metrics_path.exists() and not force:
        return {"seed": seed, "status": "skipped_existing", "metrics": str(metrics_path.relative_to(ROOT))}
    out.mkdir(parents=True, exist_ok=True)
    running = {
        "job_id": job_id, "seed": seed, "status": "running", "started_utc": utc_now(),
        "pbs_jobid": os.environ.get("PBS_JOBID"), "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
    }
    atomic_json(out / "RUNNING.json", running)
    seed_all(seed)
    try:
        split = build_split(config, smoke=smoke)
        train_store = ROOT / config["data"]["train_store"]
        test_store = ROOT / config["data"]["test_store"]
        mean, std = wave_stats(train_store, split["train"])
        training = config["training"]
        batch_size = 8 if smoke else int(training["batch_size"])
        workers = 0 if smoke else min(int(training["workers"]), int(config["runtime"]["cpu_workers_cap"]))
        train_loader = make_loader(ZarrWaveDataset(train_store, split["train"], mean, std), batch_size, workers, True)
        cal_loader = make_loader(ZarrWaveDataset(train_store, split["calibration"], mean, std), batch_size, workers, False)
        test_loader = make_loader(ZarrWaveDataset(test_store, split["test"], mean, std), batch_size, workers, False)

        model_cfg = config["model"]
        model = EnvWaveSSL(
            dim=int(model_cfg["embedding_dim"]), layers=int(model_cfg["transformer_layers"]),
            heads=int(model_cfg["attention_heads"]), dropout=float(model_cfg["dropout"]),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
        )
        use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        epochs = 1 if smoke else int(training["epochs"])
        patience = 1 if smoke else int(training["early_stopping_patience"])
        best_loss, best_state, stale = float("inf"), None, 0
        history = []
        ratio, block = float(model_cfg["mask_ratio"]), int(model_cfg["mask_block_samples"])
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                wave = batch["wave"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = reconstruction_loss(model, wave, ratio, block)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            calibration = evaluate_scores(model, cal_loader, device, ratio, block, seed + epoch)
            validation_loss = float(calibration["score"].mean())
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "calibration_loss": validation_loss}
            history.append(record)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Training produced no checkpoint")
        model.load_state_dict(best_state)
        model.to(device)
        calibration = evaluate_scores(model, cal_loader, device, ratio, block, seed + 10000)
        test = evaluate_scores(model, test_loader, device, ratio, block, seed + 20000)
        metrics = metrics_from_scores(calibration, test, float(config["evaluation"]["threshold_quantile"]))
        metrics.update({
            "job_id": job_id, "seed": seed, "runner": config["readiness"]["runner"],
            "smoke": smoke, "epochs_completed": len(history), "best_calibration_loss": best_loss,
        })
        torch.save({"model_state": best_state, "config": config, "wave_mean": mean, "wave_std": std}, out / "checkpoint.pt")
        with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
        write_scores(out / "calibration_scores.csv", calibration)
        write_scores(out / "test_scores.csv", test)
        provenance = {
            "job_id": job_id, "seed": seed, "started_utc": running["started_utc"], "completed_utc": utc_now(),
            "config": config, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "train_store": store_metadata(train_store), "test_store": store_metadata(test_store),
            "split_counts": {k: int(v.size) for k, v in split.items()},
            "split_hash_inputs": {k: [int(v[0]), int(v[-1]), int(v.size)] for k, v in split.items()},
            "leakage_policy": "No target-year samples used for training, normalization, validation, threshold selection, or early stopping.",
            "pilot_scope": "Unsupervised masked-reconstruction anomaly pilot; label_fraction_pct is not used.",
        }
        atomic_json(out / "provenance.json", provenance)
        atomic_json(metrics_path, metrics)
        atomic_json(out / "COMPLETED.json", {**running, "status": "completed", "completed_utc": utc_now()})
        (out / "RUNNING.json").unlink(missing_ok=True)
        return {"seed": seed, "status": "completed", "metrics": str(metrics_path.relative_to(ROOT))}
    except Exception as exc:
        failure = {
            **running, "status": "failed", "failed_utc": utc_now(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        }
        atomic_json(out / "FAILED.json", failure)
        (out / "RUNNING.json").unlink(missing_ok=True)
        raise


def dry_run(config_path: Path, config: dict[str, object]) -> dict[str, object]:
    data = config["data"]
    train_store, test_store = ROOT / data["train_store"], ROOT / data["test_store"]
    result = {
        "config": str(config_path), "job_id": config["job"]["id"],
        "readiness": config["readiness"], "train_store_exists": train_store.exists(),
        "test_store_exists": test_store.exists(), "seeds": config["training"]["seeds"],
        "pbs_wave": config["runtime"]["pbs_wave"], "output_dir": config["runtime"]["output_dir"],
    }
    if train_store.exists(): result["train"] = store_metadata(train_store)
    if test_store.exists(): result["test"] = store_metadata(test_store)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.dry_run or not args.execute:
        print(json.dumps(dry_run(config_path, config), ensure_ascii=False, indent=2))
        if not args.execute:
            return
    if config["readiness"]["status"] != "ready_pilot_ssl_anomaly":
        raise SystemExit(f"Blocked by readiness gate: {config['readiness']['status']} :: {config['readiness']['reason']}")
    if config["readiness"]["runner"] != "masked_reconstruction_anomaly":
        raise SystemExit(f"Unsupported runner: {config['readiness']['runner']}")
    seeds = list(config["training"]["seeds"])
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    results = [train_seed(config, int(seed), args.smoke, args.force) for seed in seeds]
    print(json.dumps({"job_id": config["job"]["id"], "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
