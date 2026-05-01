"""Residual velocity U-Net for AA-HAD depth completion — Stage 6.

Predicts the per-pixel velocity field ``v_θ(x_t, t | cond)`` used by
the conditional flow matcher in
:mod:`lidar_anchored_depth.models.flow_matching`. The conditioning
tensor concatenates RGB, DA3 ``d̃``, the closed-form AA-HAD initial
metric depth, the sparse LiDAR depth map, and the sparse LiDAR mask:

    cond[B, 0:3, H, W]   : RGB / 255
    cond[B,   3, H, W]   : d̃   (DA3 raw output, normalised)
    cond[B,   4, H, W]   : (a · d̃ + b) / Z_NORM  — AA-HAD initial z
    cond[B,   5, H, W]   : sparse LiDAR depth / Z_NORM (0 outside)
    cond[B,   6, H, W]   : 1 where LiDAR hits, 0 elsewhere

The network is a small 4-level U-Net with FiLM time conditioning:

    8 → 32 → 64 → 128 → 256 (bottleneck) → 128 → 64 → 32 → 1

~5 M parameters. Easy to fit on one A100 with batch 16+ at 384×384.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

try:
    import torch
    from torch import Tensor, nn

    _TORCH_OK = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    _TORCH_OK = False

if TYPE_CHECKING:
    import torch as _torch  # noqa: F401


COND_CHANNELS = 7
"""Number of conditioning channels expected by the network."""


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError("residual_unet requires torch>=2.1")


if _TORCH_OK:

    class _SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            if dim % 2 != 0:
                raise ValueError("time embedding dim must be even")
            self.dim = dim

        def forward(self, t: "Tensor") -> "Tensor":
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10_000.0)
                * torch.arange(half, device=t.device, dtype=t.dtype)
                / half
            )
            args = t[:, None] * freqs[None]
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    class _FiLMBlock(nn.Module):
        """Two 3x3 convs + GroupNorm + SiLU with FiLM conditioning."""

        def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
            super().__init__()
            groups = min(8, out_ch)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.gn1 = nn.GroupNorm(groups, out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.gn2 = nn.GroupNorm(groups, out_ch)
            self.act = nn.SiLU()
            self.time_proj = nn.Linear(time_dim, 2 * out_ch)
            if in_ch != out_ch:
                self.skip = nn.Conv2d(in_ch, out_ch, 1)
            else:
                self.skip = nn.Identity()

        def forward(self, x: "Tensor", t_emb: "Tensor") -> "Tensor":
            h = self.act(self.gn1(self.conv1(x)))
            scale_shift = self.time_proj(t_emb)
            scale, shift = scale_shift.chunk(2, dim=-1)
            h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            h = self.act(self.gn2(self.conv2(h)))
            return h + self.skip(x)

    class ResidualVelocityUNet(nn.Module):
        """Velocity-field U-Net for residual depth completion.

        Parameters
        ----------
        cond_channels : conditioning input channels (default ``7``).
        base : base channel count (default ``32``); doubles per level.
        time_dim : time embedding dimension (default ``128``).
        """

        def __init__(
            self,
            cond_channels: int = COND_CHANNELS,
            base: int = 32,
            time_dim: int = 128,
        ) -> None:
            super().__init__()
            in_ch = 1 + cond_channels  # x_t (1) + cond (7) = 8

            self.time_emb = nn.Sequential(
                _SinusoidalTimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

            self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
            self.down1 = _FiLMBlock(base, base * 2, time_dim)
            self.down2 = _FiLMBlock(base * 2, base * 4, time_dim)
            self.down3 = _FiLMBlock(base * 4, base * 8, time_dim)
            self.bottleneck = _FiLMBlock(base * 8, base * 8, time_dim)

            # Decoder takes upsampled features + skip from the encoder.
            self.up3 = _FiLMBlock(base * 8 + base * 4, base * 4, time_dim)
            self.up2 = _FiLMBlock(base * 4 + base * 2, base * 2, time_dim)
            self.up1 = _FiLMBlock(base * 2 + base, base, time_dim)

            self.out_conv = nn.Sequential(
                nn.GroupNorm(min(8, base), base),
                nn.SiLU(),
                nn.Conv2d(base, 1, 3, padding=1),
            )

        def forward(self, x_t: "Tensor", t: "Tensor", cond: "Tensor") -> "Tensor":
            if x_t.shape[1] != 1:
                raise ValueError(f"x_t must have 1 channel, got {x_t.shape[1]}")
            if cond.shape[1] != self.in_conv.in_channels - 1:
                raise ValueError(
                    f"cond must have {self.in_conv.in_channels - 1} channels, "
                    f"got {cond.shape[1]}"
                )

            t_emb = self.time_emb(t)

            h = torch.cat([x_t, cond], dim=1)
            h0 = self.in_conv(h)
            h1 = self.down1(
                nn.functional.avg_pool2d(h0, 2), t_emb,
            )
            h2 = self.down2(
                nn.functional.avg_pool2d(h1, 2), t_emb,
            )
            h3 = self.down3(
                nn.functional.avg_pool2d(h2, 2), t_emb,
            )
            b = self.bottleneck(h3, t_emb)

            u3 = nn.functional.interpolate(
                b, scale_factor=2.0, mode="bilinear", align_corners=False,
            )
            u3 = self.up3(torch.cat([u3, h2], dim=1), t_emb)
            u2 = nn.functional.interpolate(
                u3, scale_factor=2.0, mode="bilinear", align_corners=False,
            )
            u2 = self.up2(torch.cat([u2, h1], dim=1), t_emb)
            u1 = nn.functional.interpolate(
                u2, scale_factor=2.0, mode="bilinear", align_corners=False,
            )
            u1 = self.up1(torch.cat([u1, h0], dim=1), t_emb)

            return self.out_conv(u1)

else:  # pragma: no cover

    class ResidualVelocityUNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


__all__ = [
    "COND_CHANNELS",
    "ResidualVelocityUNet",
]
