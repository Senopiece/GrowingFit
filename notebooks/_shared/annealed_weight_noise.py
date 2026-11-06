"""Annealed Gaussian KL with exactly matched, input-conditional weight noise."""

from dataclasses import dataclass
from math import cos, isfinite, pi
from typing import Iterator, Literal

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


@dataclass(frozen=True)
class NoiseSchedule:
    """Cosine decay over noisy iterations; the final iteration has sigma=0.

    sigma_before_final is the last positive level, not a hidden variance floor.
    The same actual sigma is used for the target and predictive distributions.
    """

    sigma_start: float = 1.0
    sigma_before_final: float = 0.05

    def __post_init__(self) -> None:
        if not all(isfinite(v) and v > 0 for v in (
            self.sigma_start, self.sigma_before_final
        )):
            raise ValueError("Noise levels must be finite and positive.")
        if self.sigma_before_final > self.sigma_start:
            raise ValueError("Require sigma_before_final <= sigma_start.")

    def __call__(self, iteration: int, total_iterations: int) -> float:
        if total_iterations < 1 or not 1 <= iteration <= total_iterations:
            raise ValueError("Require 1 <= iteration <= total_iterations.")
        if iteration == total_iterations:
            return 0.0
        progress = (iteration - 1) / max(total_iterations - 2, 1)
        cosine_decay = 0.5 * (1.0 + cos(pi * progress))
        return self.sigma_before_final + (
            self.sigma_start - self.sigma_before_final
        ) * cosine_decay


class MatchedWeightNoiseMLP(nn.Module):
    """Linear -> ReLU -> Linear with conditional noise in the output parameters.

    For a(x)=[ReLU(Wx+b), 1] and theta=[v, b_out], sample
        theta_tilde(x) = theta + sigma / ||a(x)|| * epsilon,
        epsilon ~ N(0, I).
    Then a(x) @ theta_tilde(x) has mean f(x) and variance exactly sigma^2.
    Noise is applied to the augmented output weights, not added to f(x).

    The normalization depends on x. This defines pointwise conditional
    parameter noise, not a single shared random parameter vector for all x.
    The constant feature also handles inputs with all hidden activations zero.
    """

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive.")
        self.hidden = nn.Linear(1, width)
        self.activation = nn.ReLU()
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.bias)

    @property
    def width(self) -> int:
        return self.hidden.out_features

    @property
    def trainable_width(self) -> int:
        return self.width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(x)))

    def _conditional_parameters(
        self, x: torch.Tensor, sigma: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.activation(self.hidden(x))
        features = torch.cat((h, torch.ones_like(h[:, :1])), dim=1)
        weights = torch.cat((self.output.weight.reshape(-1), self.output.bias))
        parameter_std = sigma / torch.linalg.vector_norm(features, dim=1, keepdim=True)
        return features, weights, parameter_std

    def output_distribution(self, x: torch.Tensor, sigma: float) -> Normal:
        """Exact marginal of the perturbed output weights; no MC approximation."""
        if not isfinite(sigma) or sigma <= 0:
            raise ValueError("A nondegenerate Gaussian requires sigma > 0.")
        features, weights, parameter_std = self._conditional_parameters(x, sigma)
        mean = features @ weights.unsqueeze(1)
        variance = (features * parameter_std).square().sum(dim=1, keepdim=True)
        return Normal(mean, variance.sqrt())

    def sample_noisy_forward(
        self,
        x: torch.Tensor,
        sigma: float,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw independent perturbed output parameters for each input point."""
        if not isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative.")
        if sigma == 0:
            return self(x)
        features, weights, parameter_std = self._conditional_parameters(x, sigma)
        epsilon = torch.randn(
            features.shape, device=features.device, dtype=features.dtype, generator=generator
        )
        noisy_weights = weights.unsqueeze(0) + parameter_std * epsilon
        return (features * noisy_weights).sum(dim=1, keepdim=True)


def matched_gaussian_kl(
    model: MatchedWeightNoiseMLP, x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Mean KL(N(y, sigma^2) || predictive law induced by noisy weights).

    Under this exact variance matching, KL equals MSE(f(x), y)/(2*sigma^2).
    Integrating the weight noise analytically introduces no extra regularizer.
    """
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("KL requires a strictly positive sigma.")
    target = Normal(y, torch.full_like(y, sigma))
    prediction = model.output_distribution(x, sigma)
    return kl_divergence(target, prediction).mean()


@dataclass(frozen=True)
class NoiseTrainingStep:
    iteration: int
    sigma: float
    objective: Literal["KL", "MSE"]
    loss: float  # Objective used for the update; rendering evaluates separately.


class AnnealedWeightNoiseTrainer:
    def __init__(
        self,
        model: MatchedWeightNoiseMLP,
        *,
        learning_rate: float = 0.01,
        schedule: NoiseSchedule = NoiseSchedule(),
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.model.requires_grad_(True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.iteration = 0

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, sigma: float
    ) -> tuple[torch.Tensor, Literal["KL", "MSE"]]:
        if sigma == 0:
            # KL between unequal Dirac distributions is infinite: explicitly
            # use the user's deterministic MSE objective for the last update.
            return nn.functional.mse_loss(self.model(x), y), "MSE"
        return matched_gaussian_kl(self.model, x, y, sigma), "KL"

    def step(
        self, x: torch.Tensor, y: torch.Tensor, total_iterations: int
    ) -> NoiseTrainingStep:
        iteration = self.iteration + 1
        sigma = self.schedule(iteration, total_iterations)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, objective = self.loss(x, y, sigma)
        loss.backward()
        self.optimizer.step()
        self.iteration = iteration
        return NoiseTrainingStep(iteration, sigma, objective, loss.detach().item())

    def train(
        self, x: torch.Tensor, y: torch.Tensor, total_iterations: int
    ) -> Iterator[NoiseTrainingStep]:
        while self.iteration < total_iterations:
            yield self.step(x, y, total_iterations)
