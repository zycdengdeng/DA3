"""Conditional Flow Matching helpers — Stage 6.

Implements rectified flow matching for the AA-HAD residual head:

    x_0 ~ N(0, I)                            (base noise)
    x_1 = Δz_target                          (residual to AA-HAD init)
    x_t = (1 - t) x_0 + t x_1                (linear path)
    u_t(x_t | x_0, x_1) = x_1 - x_0          (constant velocity)

Training loss (per pixel)
    L_CFM = || v_θ(x_t, t | c) - (x_1 - x_0) ||²

Inference (single-step or N-step Euler integration)
    x_0 ~ N(0, I)
    for k in range(N):
        x ← x + (1 / N) · v_θ(x, k / N | c)
    Δz ≈ x

References
----------
Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer
Data with Rectified Flow", ICLR 2023.
Tong et al., torchcfm.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor

    _TORCH_OK = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    _TORCH_OK = False


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError("flow_matching requires torch>=2.1")


@dataclass
class FlowMatchingBatch:
    """Output of :meth:`RectifiedFlowMatcher.sample_train_pair`.

    Attributes
    ----------
    x_t : (B, 1, H, W) noisy state at time ``t``.
    t : (B,) the time variable in [0, 1].
    u_target : (B, 1, H, W) the constant velocity ``x_1 - x_0`` for
        the linear path. The network learns to output this.
    """

    x_t: "Tensor"
    t: "Tensor"
    u_target: "Tensor"


class RectifiedFlowMatcher:
    """Linear-path conditional flow matcher (a.k.a. rectified flow).

    Trains a velocity field ``v_θ(x_t, t | c)`` to transport noise to
    the data distribution along a straight line. The optional ``sigma``
    knob lets the conditional flow be a thin Gaussian tube around the
    linear path (Lipman 2023, eq. 23) instead of pointwise; left at
    ``0`` it reduces to plain rectified flow.
    """

    def __init__(self, sigma: float = 0.0) -> None:
        _require_torch()
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        self.sigma = float(sigma)

    def sample_train_pair(
        self, x_1: "Tensor", x_0: "Tensor | None" = None,
    ) -> FlowMatchingBatch:
        """Sample ``t`` uniformly, build ``x_t`` and the target velocity.

        Parameters
        ----------
        x_1 : (B, 1, H, W) the data sample (residual Δz).
        x_0 : optional (B, 1, H, W) base noise; sampled from ``N(0, I)``
            if ``None``.
        """
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        B = x_1.shape[0]
        t = torch.rand(B, device=x_1.device, dtype=x_1.dtype)
        t_view = t.view(B, *([1] * (x_1.dim() - 1)))
        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        if self.sigma > 0.0:
            x_t = x_t + self.sigma * torch.randn_like(x_t) * (
                t_view * (1.0 - t_view)
            )
        u_target = x_1 - x_0
        return FlowMatchingBatch(x_t=x_t, t=t, u_target=u_target)

    def sample(
        self,
        velocity_fn,
        cond: "Tensor",
        *,
        n_steps: int = 4,
        x_init: "Tensor | None" = None,
        target_shape: tuple[int, int, int, int] | None = None,
    ) -> "Tensor":
        """Integrate the learnt velocity field from ``t=0`` to ``t=1``.

        Parameters
        ----------
        velocity_fn : callable ``(x_t, t, cond) -> v_t``.
        cond : (B, C, H, W) conditioning tensor (kept fixed across steps).
        n_steps : number of Euler steps. ``1`` = single-step rectified;
            ``4-10`` typically saturates.
        x_init : optional starting state (defaults to ``N(0, I)``).
        target_shape : ``(B, 1, H, W)`` of the target; required when
            ``x_init`` is ``None``.
        """
        if x_init is None:
            if target_shape is None:
                raise ValueError("either x_init or target_shape must be given")
            x_init = torch.randn(target_shape, device=cond.device, dtype=cond.dtype)
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        x = x_init
        dt = 1.0 / n_steps
        with torch.no_grad():
            for k in range(n_steps):
                t_now = torch.full(
                    (x.shape[0],), float(k * dt),
                    device=x.device, dtype=x.dtype,
                )
                v = velocity_fn(x, t_now, cond)
                x = x + dt * v
        return x


def masked_cfm_loss(
    v_pred: "Tensor",
    u_target: "Tensor",
    valid: "Tensor | None" = None,
    *,
    valid_weight: float = 1.0,
    invalid_weight: float = 0.05,
) -> "Tensor":
    """Per-pixel CFM loss with optional weighting.

    With sparse LiDAR supervision the target velocity is well-defined
    only at LiDAR pixels (where we know ``Δz``). At non-LiDAR pixels we
    set ``Δz = 0`` (i.e. trust the AA-HAD prior) and down-weight the
    loss with ``invalid_weight`` so the network is gently pulled
    towards "stay near the closed-form prior" rather than rigidly
    constrained.
    """
    diff = (v_pred - u_target).pow(2)
    if valid is None:
        return diff.mean()
    w = torch.where(
        valid.bool(),
        torch.full_like(diff, float(valid_weight)),
        torch.full_like(diff, float(invalid_weight)),
    )
    return (diff * w).sum() / w.sum().clamp_min(1.0)


__all__ = [
    "FlowMatchingBatch",
    "RectifiedFlowMatcher",
    "masked_cfm_loss",
]
