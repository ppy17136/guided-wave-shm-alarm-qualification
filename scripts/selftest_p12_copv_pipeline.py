from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.p12_copv_pipeline import (  # noqa: E402
    FREQUENCIES_HZ,
    aggregate_top_fraction,
    bandpass_waveforms,
    confirm_two_of_three,
    frequency_index_map,
    fuse_five_frequencies,
    lag_gain_path_scores,
    valid_path_mask,
)


def directed_channels() -> np.ndarray:
    return np.asarray(
        [(sender, receiver) for sender in range(1, 26) for receiver in range(1, 26)
         if sender != receiver],
        dtype=np.float64,
    )


def main() -> None:
    channels = directed_channels()
    mask = valid_path_mask(channels)
    assert channels.shape == (600, 2)
    assert int(mask.sum()) == 552

    indices = np.arange(1, 16, dtype=np.float64)
    frequencies = np.repeat(np.asarray(FREQUENCIES_HZ, dtype=np.float64), 3)
    mapping = frequency_index_map(np.vstack([indices, frequencies]))
    assert all(len(mapping[frequency]) == 3 for frequency in FREQUENCIES_HZ)
    assert mapping[60_000] == [0, 1, 2]

    rng = np.random.default_rng(20260812)
    paths, samples = 32, 512
    template = rng.normal(0.0, 1.0, size=(paths, samples)).astype(np.float32)
    target = np.zeros_like(template)
    target[:, 3:] = 1.3 * template[:, :-3]
    clean_di, clean_energy, clean_lag = lag_gain_path_scores(target, template, 8, device="cpu")
    assert float(np.max(clean_di)) < 2e-3, float(np.max(clean_di))
    assert np.all(clean_lag == 3), np.unique(clean_lag, return_counts=True)
    assert abs(float(np.median(clean_energy)) - 0.3) < 0.02

    damaged = target.copy()
    damaged[:8, 180:260] += rng.normal(0.0, 3.0, size=(8, 80)).astype(np.float32)
    damaged_di, _, _ = lag_gain_path_scores(damaged, template, 8, device="cpu")
    clean_top, clean_count = aggregate_top_fraction(clean_di, 0.05)
    damaged_top, damaged_count = aggregate_top_fraction(damaged_di, 0.05)
    assert clean_count == damaged_count == 2
    assert damaged_top > clean_top + 0.1

    fs, fc = 2_000_000.0, 120_000.0
    time = np.arange(1024) / fs
    burst = np.sin(2 * np.pi * fc * time) * np.hanning(time.size)
    filtered = bandpass_waveforms(np.stack([burst, 0.5 * burst]), fs, fc)
    assert filtered.shape == (2, 1024)
    assert np.isfinite(filtered).all()
    assert float(np.max(np.abs(filtered))) > 0.1

    assert confirm_two_of_three([True, True, False])
    assert not confirm_two_of_three([True, False, False])
    assert fuse_five_frequencies([True, True, True, False, False]) == "alarm"
    assert fuse_five_frequencies([True, False, False, False, None]) == "no_alarm"
    assert fuse_five_frequencies([True, True, True, None, None]) == "abstain"

    result = {
        "schema_version": "p12-copv-pipeline-selftest-v1",
        "status": "pass",
        "checks": {
            "directed_channel_map_600": True,
            "sensor20_mask_leaves_552": True,
            "five_frequencies_three_repeats": True,
            "lag_recovered_exactly": True,
            "gain_invariance": True,
            "damage_residual_increases": True,
            "butterworth_bandpass_finite": True,
            "two_of_three_confirmation": True,
            "three_of_five_fusion": True,
            "insufficient_frequency_support_abstains": True,
        },
        "numeric": {
            "maximum_clean_gain_matched_DI": float(np.max(clean_di)),
            "median_clean_energy_change": float(np.median(clean_energy)),
            "clean_top_score": clean_top,
            "damaged_top_score": damaged_top,
            "recovered_lag": int(np.median(clean_lag)),
        },
    }
    output = ROOT / "data" / "reports" / "p12_copv_pipeline_selftest_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**result, "report": str(output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
