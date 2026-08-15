"""Dual-branch event representation for guided-wave change detection."""
from __future__ import annotations

import torch
from torch import nn

from src.envwave.model import EnvWaveSSL


class EventDualBranchSSL(nn.Module):
    """Encode normalized waveform shape while retaining explicit amplitude/context."""

    def __init__(self, dim: int = 192, env_dim: int = 7, layers: int = 4, heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.shape_backbone = EnvWaveSSL(dim=dim, env_dim=env_dim, layers=layers, heads=heads, dropout=dropout)
        self.amplitude_encoder = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.environment_encoder = nn.Sequential(
            nn.LayerNorm(env_dim), nn.Linear(env_dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.fusion = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.amplitude_predictor = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 8)
        )

    def forward(self, shape_wave: torch.Tensor, log_amplitude: torch.Tensor, environment: torch.Tensor):
        shape_result = self.shape_backbone(shape_wave)
        z_shape = shape_result["z_damage"]
        z_amplitude = self.amplitude_encoder(log_amplitude)
        z_environment = self.environment_encoder(environment)
        z_fused = self.fusion(torch.cat([z_shape, z_amplitude, z_environment], dim=1))
        amplitude_prediction = self.amplitude_predictor(torch.cat([z_shape, z_environment], dim=1))
        return {
            **shape_result,
            "z_shape": z_shape,
            "z_amplitude": z_amplitude,
            "z_environment_context": z_environment,
            "z_fused": z_fused,
            "amplitude_prediction": amplitude_prediction,
        }


def variance_covariance_regularizer(embedding: torch.Tensor) -> torch.Tensor:
    """Small VICReg-style anti-collapse term without negative pairs."""
    if embedding.shape[0] < 2:
        return embedding.new_zeros(())
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = torch.relu(1.0 - std).mean()
    covariance = centered.T @ centered / max(embedding.shape[0] - 1, 1)
    off_diagonal = covariance * (1.0 - torch.eye(covariance.shape[0], device=covariance.device))
    return variance + off_diagonal.square().sum() / embedding.shape[1]
