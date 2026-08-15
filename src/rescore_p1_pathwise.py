"""Rescore completed P1-015 checkpoints with path-resolved diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from src.envwave.model import EnvWaveSSL, masked_wave
from src.run_p1_job import ROOT, ZarrWaveDataset, build_split, make_loader, seed_all


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_existing_scores(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {int(row["zarr_index"]): float(row["anomaly_score"]) for row in csv.DictReader(stream)}


@torch.no_grad()
def pathwise_scores(
    model: EnvWaveSSL, loader, device: torch.device, ratio: float, block: int, seed: int
) -> dict[str, np.ndarray]:
    model.eval()
    seed_all(seed)
    output: dict[str, list[np.ndarray]] = {
        "aggregate_mse": [], "path_mse": [], "path_relative_mse": [],
        "path_input_energy": [], "label": [], "timestamp_ns": [], "index": [],
    }
    for batch in loader:
        wave = batch["wave"].to(device, non_blocking=True)
        corrupted, mask = masked_wave(wave, ratio=ratio, block=block)
        reconstruction = model(corrupted)["wave_reconstruction"]
        squared = torch.square(reconstruction - wave)
        mask_float = mask.to(squared.dtype)
        path_count = mask_float.sum(dim=2).clamp_min(1)
        path_error_sum = (squared * mask_float).sum(dim=2)
        path_mse = path_error_sum / path_count
        input_energy_sum = (torch.square(wave) * mask_float).sum(dim=2)
        path_input_energy = input_energy_sum / path_count
        path_relative = path_error_sum / input_energy_sum.clamp_min(1e-6)
        aggregate = path_error_sum.sum(dim=1) / path_count.sum(dim=1).clamp_min(1)
        output["aggregate_mse"].append(aggregate.cpu().numpy())
        output["path_mse"].append(path_mse.cpu().numpy())
        output["path_relative_mse"].append(path_relative.cpu().numpy())
        output["path_input_energy"].append(path_input_energy.cpu().numpy())
        output["label"].append(batch["label"].numpy())
        output["timestamp_ns"].append(batch["timestamp_ns"].numpy())
        output["index"].append(batch["index"].numpy())
    return {key: np.concatenate(values) for key, values in output.items()}


def write_pathwise_csv(path: Path, result: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["zarr_index", "timestamp_ns", "timestamp", "damage_label", "aggregate_mse"]
    fields += [f"path_mse_{i}" for i in range(8)]
    fields += [f"path_relative_mse_{i}" for i in range(8)]
    fields += [f"path_input_energy_{i}" for i in range(8)]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for n in range(result["index"].size):
            timestamp = int(result["timestamp_ns"][n])
            writer.writerow([
                int(result["index"][n]), timestamp, str(np.datetime64(timestamp, "ns")),
                int(result["label"][n]), float(result["aggregate_mse"][n]),
                *[float(value) for value in result["path_mse"][n]],
                *[float(value) for value in result["path_relative_mse"][n]],
                *[float(value) for value in result["path_input_energy"][n]],
            ])


def compare_existing(result: dict[str, np.ndarray], existing_path: Path) -> dict[str, float]:
    existing = read_existing_scores(existing_path)
    differences = [
        abs(float(score) - existing[int(index)])
        for index, score in zip(result["index"], result["aggregate_mse"])
        if int(index) in existing
    ]
    if not differences:
        raise RuntimeError(f"No overlapping indices with {existing_path}")
    return {
        "overlap": len(differences),
        "max_abs_difference": float(max(differences)),
        "mean_abs_difference": float(np.mean(differences)),
    }


def load_model(checkpoint_path: Path, config: dict, device: torch.device) -> tuple[EnvWaveSSL, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = config["model"]
    model = EnvWaveSSL(
        dim=int(model_config["embedding_dim"]),
        layers=int(model_config["transformer_layers"]),
        heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/p1_matrix_v1/P1-015.yaml"))
    parser.add_argument("--input-run-root", type=Path, default=Path("runs/p1_matrix_v1/015"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/p1_matrix_v1/015_pathwise_rescore"))
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--limit-calibration", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    input_root = args.input_run_root if args.input_run_root.is_absolute() else ROOT / args.input_run_root
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    if args.max_seeds is not None:
        seeds = seeds[:args.max_seeds]
    split = build_split(config, smoke=False)
    if args.limit_calibration is not None:
        split["calibration"] = split["calibration"][:args.limit_calibration]
    if args.limit_test is not None:
        split["test"] = split["test"][:args.limit_test]
    train_store = ROOT / config["data"]["train_store"]
    test_store = ROOT / config["data"]["test_store"]
    batch_size = int(config["training"]["batch_size"])
    workers = min(int(config["training"]["workers"]), int(config["runtime"]["cpu_workers_cap"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratio = float(config["model"]["mask_ratio"])
    block = int(config["model"]["mask_block_samples"])
    campaign = {
        "schema_version": "p1-pathwise-rescore-v1", "started_utc": utc_now(),
        "device": str(device), "seeds": seeds, "results": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        seed_output = output_root / f"seed_{seed}"
        completed_path = seed_output / "PATHWISE_COMPLETED.json"
        if completed_path.exists() and not args.force:
            campaign["results"].append({"seed": seed, "status": "skipped_existing"})
            continue
        seed_output.mkdir(parents=True, exist_ok=True)
        checkpoint_path = input_root / f"seed_{seed}/checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model, checkpoint = load_model(checkpoint_path, config, device)
        mean = np.asarray(checkpoint["wave_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["wave_std"], dtype=np.float32)
        cal_loader = make_loader(
            ZarrWaveDataset(train_store, split["calibration"], mean, std), batch_size, workers, False
        )
        test_loader = make_loader(
            ZarrWaveDataset(test_store, split["test"], mean, std), batch_size, workers, False
        )
        cal_result = pathwise_scores(model, cal_loader, device, ratio, block, seed + 10000)
        test_result = pathwise_scores(model, test_loader, device, ratio, block, seed + 20000)
        cal_check = compare_existing(cal_result, input_root / f"seed_{seed}/calibration_scores.csv")
        test_check = compare_existing(test_result, input_root / f"seed_{seed}/test_scores.csv")
        tolerance = 5e-4 if device.type == "cpu" else 5e-5
        if cal_check["max_abs_difference"] > tolerance or test_check["max_abs_difference"] > tolerance:
            raise RuntimeError(f"Aggregate-score reproduction failed for seed {seed}: {cal_check}, {test_check}")
        write_pathwise_csv(seed_output / "calibration_pathwise_scores.csv", cal_result)
        write_pathwise_csv(seed_output / "test_pathwise_scores.csv", test_result)
        record = {
            "seed": seed, "status": "completed", "completed_utc": utc_now(),
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "calibration_samples": int(cal_result["index"].size), "test_samples": int(test_result["index"].size),
            "calibration_aggregate_reproduction": cal_check, "test_aggregate_reproduction": test_check,
            "score_definitions": {
                "path_mse": "masked squared reconstruction error divided by masked count per path",
                "path_relative_mse": "masked squared reconstruction error divided by masked normalized-input energy per path",
                "path_input_energy": "masked normalized-input squared amplitude divided by masked count per path",
            },
        }
        completed_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        campaign["results"].append(record)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    campaign["completed_utc"] = utc_now()
    campaign["status"] = "complete"
    (output_root / "PATHWISE_CAMPAIGN.json").write_text(json.dumps(campaign, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(campaign, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
