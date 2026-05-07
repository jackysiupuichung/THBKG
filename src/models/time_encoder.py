"""Learnable Fourier time encoder for per-pair entry year conditioning."""

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    def __init__(self, d_time: int, t_min: float, t_max: float):
        super().__init__()
        assert d_time > 0 and d_time % 2 == 0, "d_time must be positive and even"
        self.d_time = d_time
        self.register_buffer("t_min", torch.tensor(float(t_min)))
        self.register_buffer("t_max", torch.tensor(float(t_max)))
        self.linear_freq = nn.Linear(1, d_time // 2)
        self.proj = nn.Linear(d_time, d_time)

    def forward(self, t_entry: torch.Tensor) -> torch.Tensor:
        denom = (self.t_max - self.t_min).clamp_min(1e-8)
        t_norm = ((t_entry.float() - self.t_min) / denom).clamp(0.0, 1.0)
        freqs = self.linear_freq(t_norm.unsqueeze(-1))
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return self.proj(emb)
