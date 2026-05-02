"""Per-point velocity field for 3-D depth-completion flow matching — Stage 7B.

The model takes a *prior* point cloud (Stage 1-5 closed-form output)
together with per-point conditioning features and a noisy displacement
state ``x_t``, and predicts the per-point velocity ``v(x_t, t | cond)``
that the rectified flow matcher integrates from ``x_0 ~ N(0, I)``
to the displacement target ``x_1 = Δxyz``.

Backbone: a small **DGCNN-style EdgeConv** stack (Wang et al., ToG
2019) — 4 layers of K-NN edge features with FiLM time conditioning.
Each layer aggregates per-point features over the K nearest neighbours
in feature space, giving the model enough geometric context without
needing a sparse 3-D conv library (MinkowskiEngine / spconv) that is
notoriously hard to install. The architecture is a drop-in
replacement target: we can swap in MinkUNet later if results justify
the extra dependency.

Inputs (per batch):
    prior_xyz   (B, N, 3)  centred + normalised XYZ
    prior_rgb   (B, N, 3)  in [0, 1]
    prior_feat  (B, N, F)  per-point conditioning features
    x_t         (B, N, 3)  noisy state at time t (in DISP_SCALE units)
    t           (B,)       in [0, 1]

Output:
    v           (B, N, 3)  velocity field
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


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError("point_cloud_unet requires torch>=2.1")


if _TORCH_OK:

    def _knn_indices(x: "Tensor", k: int, chunk: int | None = None) -> "Tensor":
        """Pairwise-distance K-NN.

        ``x`` is ``(B, N, C)``. Returns ``(B, N, k)`` int64 indices.
        For large ``N`` use ``chunk`` to compute distances row-block by
        row-block, capping peak memory at ``B * chunk * N * 4`` bytes
        instead of ``B * N * N * 4``.
        """
        B, N, C = x.shape
        sq = (x * x).sum(dim=-1, keepdim=True)  # (B, N, 1)
        if chunk is None or chunk >= N:
            inner = 2 * torch.bmm(x, x.transpose(1, 2))
            d = sq + sq.transpose(1, 2) - inner
            return d.topk(k=k, dim=-1, largest=False).indices
        out_chunks: list[Tensor] = []
        for i in range(0, N, chunk):
            x_c = x[:, i:i + chunk, :]
            sq_c = sq[:, i:i + chunk, :]
            inner = 2 * torch.bmm(x_c, x.transpose(1, 2))
            d_c = sq_c + sq.transpose(1, 2) - inner  # (B, chunk, N)
            out_chunks.append(d_c.topk(k=k, dim=-1, largest=False).indices)
        return torch.cat(out_chunks, dim=1)

    def _gather_neighbours(x: "Tensor", idx: "Tensor") -> "Tensor":
        """Gather neighbour features without materialising ``(B, N, N, C)``.

        ``x`` ``(B, N, C)``, ``idx`` ``(B, N, k)`` -> ``(B, N, k, C)``.
        Implemented as ``torch.gather`` along dim 1 with a flattened
        index tensor so peak memory is only ``B * N * k * C`` instead
        of ``B * N * N * C`` (the previous implementation blew up the
        memory because ``expand`` is materialised by ``torch.gather``
        on the expanded axis).
        """
        B, N, C = x.shape
        k = idx.shape[-1]
        idx_flat = idx.reshape(B, N * k).unsqueeze(-1).expand(-1, -1, C)
        gathered = torch.gather(x, 1, idx_flat)
        return gathered.reshape(B, N, k, C)

    class _SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            if dim % 2 != 0:
                raise ValueError("dim must be even")
            self.dim = dim

        def forward(self, t: "Tensor") -> "Tensor":
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=t.device, dtype=t.dtype)
                / half
            )
            args = t[:, None] * freqs[None]
            return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    class _EdgeConv(nn.Module):
        """EdgeConv layer with FiLM time-conditioning."""

        def __init__(
            self, in_ch: int, out_ch: int, k: int, time_dim: int,
            knn_chunk: int | None = None,
        ) -> None:
            super().__init__()
            self.k = k
            self.knn_chunk = knn_chunk
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_ch, out_ch),
                nn.GELU(),
                nn.Linear(out_ch, out_ch),
            )
            self.norm = nn.LayerNorm(out_ch)
            self.time_proj = nn.Linear(time_dim, 2 * out_ch)

        def forward(self, x: "Tensor", t_emb: "Tensor") -> "Tensor":
            # x: (B, N, C); KNN in feature space
            idx = _knn_indices(x, self.k, chunk=self.knn_chunk)
            nb = _gather_neighbours(x, idx)               # (B, N, k, C)
            cur = x.unsqueeze(2).expand_as(nb)
            edge = torch.cat([cur, nb - cur], dim=-1)
            f = self.mlp(edge).max(dim=2).values          # (B, N, out_ch)
            f = self.norm(f)
            scale_shift = self.time_proj(t_emb)
            scale, shift = scale_shift.chunk(2, dim=-1)
            f = f * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            return f

    class PointCloudVelocityNet(nn.Module):
        """Per-point flow-matching velocity field."""

        def __init__(
            self,
            prior_feat_dim: int,
            *,
            base: int = 64,
            time_dim: int = 128,
            k: int = 16,
            knn_chunk: int | None = 2048,
        ) -> None:
            super().__init__()
            in_ch = 3 + 3 + prior_feat_dim + 3  # xyz + rgb + feat + x_t
            self.time_emb = nn.Sequential(
                _SinusoidalTimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            self.in_proj = nn.Linear(in_ch, base)
            self.layer1 = _EdgeConv(base, base, k=k, time_dim=time_dim, knn_chunk=knn_chunk)
            self.layer2 = _EdgeConv(base, base * 2, k=k, time_dim=time_dim, knn_chunk=knn_chunk)
            self.layer3 = _EdgeConv(base * 2, base * 2, k=k, time_dim=time_dim, knn_chunk=knn_chunk)
            self.layer4 = _EdgeConv(base * 2, base, k=k, time_dim=time_dim, knn_chunk=knn_chunk)
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(base),
                nn.GELU(),
                nn.Linear(base, base),
                nn.GELU(),
                nn.Linear(base, 3),
            )

        def forward(
            self,
            prior_xyz: "Tensor",
            prior_rgb: "Tensor",
            prior_feat: "Tensor",
            x_t: "Tensor",
            t: "Tensor",
        ) -> "Tensor":
            B, N, _ = prior_xyz.shape
            h = torch.cat([prior_xyz, prior_rgb, prior_feat, x_t], dim=-1)
            h = self.in_proj(h)
            t_emb = self.time_emb(t)
            h = self.layer1(h, t_emb) + h
            h = self.layer2(h, t_emb)
            h = self.layer3(h, t_emb) + h
            h = self.layer4(h, t_emb)
            return self.out_mlp(h)


else:  # pragma: no cover

    class PointCloudVelocityNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


__all__ = ["PointCloudVelocityNet"]
