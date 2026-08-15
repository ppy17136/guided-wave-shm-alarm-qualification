"""Frozen signal-processing primitives for the P12 COPV confirmation.

The module contains no dataset-specific result selection.  It implements the
pre-registered five-frequency filtering, gain-and-lag matched path residual,
sensor-20 mask, top-five-percent aggregation, and repeat/frequency voting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt


FREQUENCIES_HZ = (60_000, 120_000, 180_000, 260_000, 300_000)
EXCLUDED_SENSOR = 20


@dataclass(frozen=True)
class FrequencyScore:
    repetition_index: int
    score: float
    energy_score: float
    valid_paths: int
    top_paths: int
    median_best_lag: float


def frequency_index_map(index_frequency_vs_repetition: np.ndarray) -> dict[int, list[int]]:
    mapping = np.asarray(index_frequency_vs_repetition, dtype=np.float64)
    if mapping.shape != (2, 15):
        raise ValueError(f"expected frequency index mapping shape (2,15), got {mapping.shape}")
    indices = np.rint(mapping[0]).astype(np.int64)
    frequencies = np.rint(mapping[1]).astype(np.int64)
    if indices.min() >= 1 and indices.max() <= 18:
        indices = indices - 1
    if indices.min() < 0 or indices.max() >= 18 or len(np.unique(indices)) != 15:
        raise ValueError("frequency repetition indices are invalid")
    # Official COPV A2 metadata mapping: Signal_Frequency_Burst declares the
    # fourth excitation as 260 kHz, while Index_FrequencyvsRepetition labels
    # its three contiguous repetitions (indices 10--12 in MATLAB convention)
    # as 240 kHz. This is a schema-label inconsistency, not a frequency change.
    if (
        np.count_nonzero(frequencies == 260_000) == 0
        and np.count_nonzero(frequencies == 240_000) == 3
        and sorted(indices[frequencies == 240_000].tolist()) == [9, 10, 11]
    ):
        frequencies = frequencies.copy()
        frequencies[frequencies == 240_000] = 260_000
    result: dict[int, list[int]] = {}
    for frequency in FREQUENCIES_HZ:
        selected = sorted(indices[frequencies == frequency].tolist())
        if len(selected) != 3:
            raise ValueError(f"frequency {frequency} Hz has {len(selected)} repetitions, expected 3")
        result[frequency] = selected
    if sum(len(value) for value in result.values()) != 15:
        raise ValueError("frequency mapping does not contain exactly 15 burst repetitions")
    return result


def valid_path_mask(channels: np.ndarray, excluded_sensor: int = EXCLUDED_SENSOR) -> np.ndarray:
    array = np.asarray(channels, dtype=np.float64)
    if array.shape == (2, 600):
        array = array.T
    if array.shape != (600, 2):
        raise ValueError(f"expected channel shape (600,2), got {array.shape}")
    rounded = np.rint(array).astype(np.int64)
    if not np.allclose(array, rounded):
        raise ValueError("channel identifiers are not integral")
    senders, receivers = rounded[:, 0], rounded[:, 1]
    if senders.min() < 1 or receivers.min() < 1 or senders.max() > 25 or receivers.max() > 25:
        raise ValueError("channel identifiers fall outside sensors 1..25")
    if np.any(senders == receivers):
        raise ValueError("self paths are present in the 600-path map")
    pairs = set(zip(senders.tolist(), receivers.tolist()))
    if len(pairs) != 600:
        raise ValueError("channel map contains duplicate directed paths")
    mask = (senders != excluded_sensor) & (receivers != excluded_sensor)
    if int(mask.sum()) != 552:
        raise ValueError(f"sensor-{excluded_sensor} mask leaves {int(mask.sum())} paths, expected 552")
    return mask


def bandpass_waveforms(waveforms: np.ndarray, sampling_frequency: float, carrier_frequency: float) -> np.ndarray:
    data = np.asarray(waveforms, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("waveforms must have shape (paths, samples)")
    if not np.isfinite(data).all():
        raise ValueError("waveforms contain non-finite values")
    fs = float(sampling_frequency)
    fc = float(carrier_frequency)
    low = 0.7 * fc
    high = min(1.3 * fc, 0.9 * (fs / 2.0))
    if not (0.0 < low < high < fs / 2.0):
        raise ValueError(f"incompatible bandpass for fs={fs}, fc={fc}: low={low}, high={high}")
    dc_count = max(1, int(math.ceil(data.shape[-1] * 0.05)))
    centered = data - np.median(data[:, :dc_count], axis=1, keepdims=True)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, centered, axis=-1)
    return np.asarray(filtered, dtype=np.float32)


def build_frequency_template(
    raw_dataset,
    repetition_indices: Iterable[int],
    path_mask: np.ndarray,
    sampling_frequency: float,
    carrier_frequency: float,
) -> np.ndarray:
    selected_paths = np.flatnonzero(np.asarray(path_mask, dtype=bool))
    rows = []
    for repetition in repetition_indices:
        raw = np.asarray(raw_dataset[int(repetition), selected_paths, :], dtype=np.float64)
        rows.append(bandpass_waveforms(raw, sampling_frequency, carrier_frequency))
    if len(rows) != 3:
        raise ValueError("template requires exactly three repetitions")
    return np.median(np.stack(rows, axis=0), axis=0).astype(np.float32)


def _linear_cross_correlation(x: torch.Tensor, b: torch.Tensor, nfft: int) -> torch.Tensor:
    return torch.fft.irfft(
        torch.fft.rfft(x, n=nfft, dim=-1) * torch.conj(torch.fft.rfft(b, n=nfft, dim=-1)),
        n=nfft,
        dim=-1,
    )


def lag_gain_path_scores(
    target: np.ndarray,
    template: np.ndarray,
    max_lag: int,
    *,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return minimum normalized LS residual, RMS energy change, and best lag per path."""
    x_np = np.asarray(target, dtype=np.float32)
    b_np = np.asarray(template, dtype=np.float32)
    if x_np.shape != b_np.shape or x_np.ndim != 2:
        raise ValueError("target and template must be equal 2D arrays")
    if max_lag < 0 or max_lag >= x_np.shape[1] - 16:
        raise ValueError("max_lag is invalid for waveform length")
    target_device = torch.device(device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=target_device)
    b = torch.as_tensor(b_np, dtype=torch.float32, device=target_device)
    paths, samples = x.shape
    nfft = 1 << int(math.ceil(math.log2(2 * samples - 1)))
    correlation = _linear_cross_correlation(x, b, nfft)
    x_prefix = torch.nn.functional.pad(torch.cumsum(x.square(), dim=1), (1, 0))
    b_prefix = torch.nn.functional.pad(torch.cumsum(b.square(), dim=1), (1, 0))
    eps = torch.finfo(torch.float32).eps
    candidates = []
    lags = list(range(-max_lag, max_lag + 1))
    for lag in lags:
        if lag >= 0:
            dot = correlation[:, lag]
            x_norm = x_prefix[:, samples] - x_prefix[:, lag]
            b_norm = b_prefix[:, samples - lag]
        else:
            shift = -lag
            dot = correlation[:, nfft - shift]
            x_norm = x_prefix[:, samples - shift]
            b_norm = b_prefix[:, samples] - b_prefix[:, shift]
        safe_b = torch.clamp(b_norm, min=eps)
        residual_sq = torch.clamp(x_norm - dot.square() / safe_b, min=0.0)
        di = torch.sqrt(residual_sq / safe_b)
        di = torch.where(b_norm > eps, di, torch.full_like(di, torch.inf))
        candidates.append(di)
    stacked = torch.stack(candidates, dim=1)
    best_values, best_indices = torch.min(stacked, dim=1)
    best_lags = torch.as_tensor(lags, device=target_device)[best_indices]

    x_energy = torch.sqrt(torch.mean(x.square(), dim=1))
    b_energy = torch.sqrt(torch.mean(b.square(), dim=1))
    energy_change = torch.abs(x_energy / torch.clamp(b_energy, min=eps) - 1.0)
    return (
        best_values.detach().cpu().numpy().astype(np.float64),
        energy_change.detach().cpu().numpy().astype(np.float64),
        best_lags.detach().cpu().numpy().astype(np.int64),
    )


def aggregate_top_fraction(values: np.ndarray, fraction: float = 0.05) -> tuple[float, int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("no finite path values")
    count = max(1, int(math.ceil(fraction * finite.size)))
    top = np.partition(finite, finite.size - count)[-count:]
    return float(np.mean(top)), count


def score_frequency_repetitions(
    raw_dataset,
    repetition_indices: Iterable[int],
    path_mask: np.ndarray,
    sampling_frequency: float,
    carrier_frequency: float,
    template: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> list[FrequencyScore]:
    selected_paths = np.flatnonzero(np.asarray(path_mask, dtype=bool))
    max_lag = int(math.ceil(float(sampling_frequency) / float(carrier_frequency)))
    results = []
    for order, repetition in enumerate(repetition_indices):
        raw = np.asarray(raw_dataset[int(repetition), selected_paths, :], dtype=np.float64)
        filtered = bandpass_waveforms(raw, sampling_frequency, carrier_frequency)
        path_score, energy_score, lags = lag_gain_path_scores(
            filtered, template, max_lag, device=device
        )
        score, top_count = aggregate_top_fraction(path_score, 0.05)
        energy, _ = aggregate_top_fraction(energy_score, 0.05)
        results.append(FrequencyScore(
            repetition_index=order,
            score=score,
            energy_score=energy,
            valid_paths=int(path_score.size),
            top_paths=top_count,
            median_best_lag=float(np.median(lags)),
        ))
    if len(results) != 3:
        raise ValueError("target frequency requires exactly three repetitions")
    return results


def confirm_two_of_three(alarms: Iterable[bool]) -> bool:
    values = list(bool(value) for value in alarms)
    if len(values) != 3:
        raise ValueError("repeat confirmation requires exactly three values")
    return sum(values) >= 2


def fuse_five_frequencies(confirmed: Iterable[bool | None]) -> str:
    values = list(confirmed)
    if len(values) != 5:
        raise ValueError("frequency fusion requires five values")
    available = [bool(value) for value in values if value is not None]
    if len(available) < 4:
        return "abstain"
    return "alarm" if sum(available) >= 3 else "no_alarm"
