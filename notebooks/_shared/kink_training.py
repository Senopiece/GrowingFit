"""Conventional ReLU MLP and alternating kink-aware fitting / repulsion.

This module only defines the architecture, objectives and optimizer loop.
It does not construct data or run experiments.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Iterator

import torch
from torch import nn


class KinkMLP(nn.Module):
    """Linear(1, width) -> ReLU -> Linear(width, 1); no free kink parameters."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive.")
        self.hidden = nn.Linear(1, width)
        self.activation = nn.ReLU()
        self.output = nn.Linear(width, 1)
        # Match the zero output bias used by the existing ordinary MLP.
        nn.init.zeros_(self.output.bias)

    @property
    def width(self) -> int:
        return self.hidden.out_features

    @property
    def trainable_width(self) -> int:
        return self.width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(x)))


@dataclass(frozen=True)
class KinkTrainingConfig:
    sigma_fit: float = 0.15
    lambda_fit: float = 1.0
    sigma_rep: float = 0.15
    lr_fit: float = 0.01
    lr_rep: float = 0.001
    min_abs_weight: float = 1e-4
    max_repulsion_step: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "sigma_fit", "sigma_rep", "lr_fit", "lr_rep",
            "min_abs_weight", "max_repulsion_step",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isfinite(self.lambda_fit) or self.lambda_fit < 0:
            raise ValueError("lambda_fit must be finite and nonnegative.")


def extract_kinks(
    hidden: nn.Linear,
    x_min: float,
    x_max: float,
    min_abs_weight: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable in-domain positions and their hidden-neuron mask.

    Division is performed only with safe denominators, even for rejected rows.
    The mask intentionally has no gradient; retained coordinates are exactly -b/w.
    """
    w = hidden.weight[:, 0]
    b = hidden.bias
    stable = torch.isfinite(w) & torch.isfinite(b) & (w.abs() >= min_abs_weight)
    safe_w = torch.where(stable, w, torch.ones_like(w))
    safe_b = torch.where(stable, b, torch.zeros_like(b))
    positions = -safe_b / safe_w
    valid = stable & torch.isfinite(positions) & (positions >= x_min) & (positions <= x_max)
    return positions[valid], valid


def kink_sample_weights(
    x: torch.Tensor, kinks: torch.Tensor, sigma_fit: float, lambda_fit: float
) -> torch.Tensor:
    """Detached q(x) = 1 + lambda * sum_i exp(-(x-t_i)^2 / (2*sigma^2))."""
    with torch.no_grad():
        distances = (x.reshape(-1, 1) - kinks.detach().reshape(1, -1)) / sigma_fit
        density = torch.exp(-0.5 * distances.square()).sum(dim=1, keepdim=True)
        return (1.0 + lambda_fit * density).detach()


def kink_weighted_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Normalize by sum(q), never differentiate the fitting weights."""
    q = weights.detach().reshape(-1)
    squared_errors = (prediction.reshape(-1) - target.reshape(-1)).square()
    return (q * squared_errors).sum() / q.sum()


def kink_repulsion_loss(kinks: torch.Tensor, sigma_rep: float) -> torch.Tensor:
    """Unnormalized Gaussian sum over i < j, excluding self-interactions.

    The gradient is exactly zero for coincident kinks. No jitter or extra
    objective is silently introduced to break that symmetry.
    """
    if kinks.numel() < 2:
        return kinks.sum() * 0.0
    pairs = torch.triu_indices(kinks.numel(), kinks.numel(), offset=1, device=kinks.device)
    differences = (kinks[pairs[0]] - kinks[pairs[1]]) / (2.0 * sigma_rep)
    return torch.exp(-differences.square()).sum()


@dataclass(frozen=True)
class KinkTrainingStep:
    iteration: int
    weighted_fit_loss: float  # Before the fitting update.
    repulsion_loss: float  # After fitting, before the repulsion update.
    active_kinks: int  # Used in this iteration's repulsion substep.


class KinkAwareTrainer:
    """Adam for fitting, independent preconditioned SGD for hidden-only repulsion.

    For t=-b/w, J=dt/d(w,b)=(-t/w, -1/w). Its squared norm is
    (1+t^2)/w^2. Multiplying each neuron's two repulsion gradients by
    w^2/(1+t^2) cancels the small-w sensitivity: to first order,
    delta_t = -lr_rep * dL_rep/dt, independent of the neuron's scale.

    A per-neuron trust bound also limits the exact change of the ratio -b/w.
    After SGD, a positive radial retraction restores the norm of each (w,b)
    pair without changing its new kink coordinate. No output weights change.
    """

    def __init__(
        self,
        model: KinkMLP,
        *,
        x_min: float,
        x_max: float,
        config: KinkTrainingConfig = KinkTrainingConfig(),
    ) -> None:
        if not isfinite(x_min) or not isfinite(x_max) or x_min >= x_max:
            raise ValueError("Require finite x_min < x_max.")
        self.model = model
        self.config = config
        self.x_min = x_min
        self.x_max = x_max
        self.iteration = 0
        self.model.requires_grad_(True)
        self.fit_optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_fit)
        # Momentum / weight decay would move excluded neurons even with zero
        # current gradients, and adaptive Adam scaling would undo the metric.
        self.repulsion_optimizer = torch.optim.SGD(
            model.hidden.parameters(), lr=config.lr_rep, momentum=0.0, weight_decay=0.0
        )

    def _kinks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return extract_kinks(
            self.model.hidden, self.x_min, self.x_max, self.config.min_abs_weight
        )

    @torch.no_grad()
    def _precondition_repulsion(self, kinks: torch.Tensor, valid: torch.Tensor) -> None:
        w = self.model.hidden.weight[:, 0]
        t = kinks.detach()
        g = kinks.grad.detach()
        metric_denominator = 1.0 + t.square()

        # For s=lr_rep*g, the un-retracted parameter update gives exactly
        # delta_t = -s / (1 + s*t/(1+t^2)). This bound keeps its denominator
        # away from zero and ensures |delta_t| <= max_repulsion_step.
        max_delta = self.config.max_repulsion_step
        max_s = max_delta / (1.0 + max_delta * t.abs() / metric_denominator)
        proposed_s = self.config.lr_rep * g.abs()
        tiny = torch.finfo(w.dtype).tiny
        trust_scale = (max_s / proposed_s.clamp_min(tiny)).clamp(max=1.0)

        multiplier = torch.zeros_like(w)
        multiplier[valid] = w[valid].square() / metric_denominator * trust_scale
        self.model.hidden.weight.grad.mul_(multiplier.unsqueeze(1))
        self.model.hidden.bias.grad.mul_(multiplier)

    @torch.no_grad()
    def _restore_hidden_norms(self, old_norms: torch.Tensor, valid: torch.Tensor) -> None:
        w = self.model.hidden.weight[:, 0]
        b = self.model.hidden.bias
        new_norms = torch.hypot(w[valid], b[valid])
        scale = old_norms[valid] / new_norms.clamp_min(torch.finfo(w.dtype).tiny)
        w[valid] = w[valid] * scale
        b[valid] = b[valid] * scale

    def step(self, x: torch.Tensor, y: torch.Tensor) -> KinkTrainingStep:
        self.model.train()

        # A: q is detached. Every weight and bias participates in fitting.
        self.fit_optimizer.zero_grad(set_to_none=True)
        prediction = self.model(x)
        with torch.no_grad():
            fit_kinks, _ = self._kinks()
        weights = kink_sample_weights(
            x, fit_kinks, self.config.sigma_fit, self.config.lambda_fit
        )
        fit_loss = kink_weighted_mse(prediction, y, weights)
        fit_loss.backward()
        self.fit_optimizer.step()

        # B: clear fitting gradients (including the output layer), then derive
        # fresh, differentiable kinks from the already updated hidden layer.
        self.fit_optimizer.zero_grad(set_to_none=True)
        self.repulsion_optimizer.zero_grad(set_to_none=True)
        rep_kinks, valid = self._kinks()
        rep_loss = kink_repulsion_loss(rep_kinks, self.config.sigma_rep)
        if rep_kinks.numel() >= 2:
            rep_kinks.retain_grad()
            rep_loss.backward()
            self._precondition_repulsion(rep_kinks, valid)
            with torch.no_grad():
                old_norms = torch.hypot(
                    self.model.hidden.weight[:, 0], self.model.hidden.bias
                )
            self.repulsion_optimizer.step()
            self._restore_hidden_norms(old_norms, valid)

        self.iteration += 1
        return KinkTrainingStep(
            iteration=self.iteration,
            weighted_fit_loss=fit_loss.detach().item(),
            repulsion_loss=rep_loss.detach().item(),
            active_kinks=rep_kinks.numel(),
        )

    def train(
        self, x: torch.Tensor, y: torch.Tensor, iterations: int
    ) -> Iterator[KinkTrainingStep]:
        for _ in range(iterations):
            yield self.step(x, y)
