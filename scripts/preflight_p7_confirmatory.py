"""Strict metadata, store, model, and freeze preflight for P7."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
import zarr

from src.envwave.event_dual_branch import EventDualBranchSSL


ROOT = Path(__file__).resolve().parents[1]
MONTHS = ("2022_05", "2022_06", "2022_07", "2022_08", "2022_09", "2022_10")
TRANSITIONS = ("D6->D7", "D7->D8", "D8->D9", "D9->D10", "D10->D11", "D11->D12", "D12->D13")
V11_SEAL = "570036f8201371edff7cba9f2b9e350f72726a1d8b9901f00d19908b9dbab4fa"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, default=Path("configs/p7_confirmatory_v1/P7-TRAIN-01.yaml"))
    parser.add_argument("--rescore-config", type=Path, default=Path("configs/p7_confirmatory_v1/P7-RESCORE-01.yaml"))
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    train = yaml.safe_load((ROOT / args.train_config).read_text(encoding="utf-8"))
    rescore = yaml.safe_load((ROOT / args.rescore_config).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / train["data"]["event_manifest"]).read_text(encoding="utf-8"))
    events = manifest["events"]

    checks: dict[str, bool] = {}
    checks["seven_exact_transitions"] = tuple(item["transition"] for item in events) == TRANSITIONS
    checks["all_windows_complete_or_A2_exception"] = all(
        (
            all(bool(item.get(key)) for key in ("baseline_complete", "pre_event_complete", "post_event_complete"))
            or item.get("confirmatory_window_status") == "approved_A2_D9_D10_leading_baseline_gap"
        )
        for item in events
    )
    checks["five_frozen_seeds"] = train["training"]["seeds"] == [20260809, 20260810, 20260811, 20260812, 20260813]
    checks["rescore_same_seeds"] = rescore["source"]["seeds"] == train["training"]["seeds"]
    checks["same_event_manifest"] = rescore["data"]["event_manifest"] == train["data"]["event_manifest"]
    seal = ROOT / "research_protocols/P7_D7_D13_freeze_manifest_v1.1.json"
    checks["v1_1_freeze_seal"] = sha256(seal) == V11_SEAL

    stores = []
    for month in MONTHS:
        path = ROOT / "data/zarr" / f"measurements_{month}.zarr"
        report = {"month": month, "exists": path.is_dir()}
        valid = path.is_dir()
        if valid:
            group = zarr.open_group(str(path), mode="r")
            shape = tuple(group["guided_wave"].shape)
            report.update({"measurements": int(shape[0]), "wave_shape": list(shape)})
            valid = shape[1:] == (8, 2000) and len(group["datetime_ns"]) == shape[0]
        checks[f"store_{month}"] = bool(valid)
        stores.append(report)

    cfg = train["model"]
    model = EventDualBranchSSL(cfg["embedding_dim"], 7, cfg["transformer_layers"], cfg["attention_heads"], cfg["dropout"])
    with torch.no_grad():
        output = model(torch.zeros(2, 8, 2000), torch.zeros(2, 8), torch.zeros(2, 7))
    checks["model_forward"] = output["z_fused"].shape == (2, cfg["embedding_dim"]) and output["wave_reconstruction"].shape == (2, 8, 2000)
    checks["cuda_visible"] = torch.cuda.is_available()
    if args.require_cuda:
        checks["required_cuda"] = checks["cuda_visible"]
    passed = all(value for key, value in checks.items() if key != "cuda_visible")
    report = {
        "schema_version": "p7-confirmatory-preflight-v1",
        "passed": passed,
        "checks": checks,
        "events": [item["transition"] for item in events],
        "stores": stores,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

