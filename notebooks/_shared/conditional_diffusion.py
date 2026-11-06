"""Conditional x0-prediction MLP with two inputs and an eight-step DDIM sampler."""

from dataclasses import dataclass
from math import isfinite, pi

import torch
from torch import nn


class DiffusionMLP(nn.Module):
    """[x, y_t] -> Linear(2, width) -> ReLU -> Linear(width, 1) predicts y_0.

    No timestep embedding is introduced: the architecture gains exactly one
    scalar input. For the deterministic conditional target y_0=f(x), x0
    prediction remains unambiguous without knowing the sampled timestep.
    """

    def __init__(self, width: int = 512) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive.")
        self.hidden = nn.Linear(2, width)
        self.activation = nn.ReLU()
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.bias)

    @property
    def width(self) -> int:
        return self.hidden.out_features

    @property
    def trainable_width(self) -> int:
        return self.width

    def forward(self, x: torch.Tensor, noisy_y: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((x, noisy_y), dim=1)
        return self.output(self.activation(self.hidden(inputs)))


@dataclass(frozen=True)
class DiffusionConfig:
    denoise_steps: int = 8
    cosine_offset: float = 0.008

    def __post_init__(self) -> None:
        if self.denoise_steps < 1:
            raise ValueError("denoise_steps must be positive.")
        if not isfinite(self.cosine_offset) or not 0 <= self.cosine_offset < 1:
            raise ValueError("cosine_offset must be finite and in [0, 1).")


class ConditionalDiffusion:
    """Cosine forward marginals and deterministic DDIM with x0 prediction.

    alpha_bar[0]=1, alpha_bar[T]=0: the final forward marginal is exactly
    standard Gaussian. x0 prediction avoids division by sqrt(alpha_bar[T]).
    The schedule has no trainable parameters and is shared read-only.
    """

    def __init__(self, config: DiffusionConfig = DiffusionConfig()) -> None:
        self.config = config
        time = torch.linspace(0.0, 1.0, config.denoise_steps + 1, dtype=torch.float64)
        angles = (time + config.cosine_offset) / (1.0 + config.cosine_offset) * (pi / 2)
        curve = torch.cos(angles).square()
        self.alpha_bar = curve / curve[0]
        self.alpha_bar[0] = 1.0
        self.alpha_bar[-1] = 0.0

    def _table(self, reference: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(device=reference.device, dtype=reference.dtype)

    def add_noise(
        self, clean_y: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._table(clean_y)[timesteps].reshape(-1, 1)
        return alpha.sqrt() * clean_y + (1.0 - alpha).sqrt() * noise

    @torch.inference_mode()
    def sample(
        self,
        model: DiffusionMLP,
        x: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Exactly T model calls, starting from noise; never reads a target y."""
        if initial_noise is None:
            y = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        else:
            if initial_noise.shape != x.shape:
                raise ValueError("initial_noise must have the same (N, 1) shape as x.")
            y = initial_noise.to(device=x.device, dtype=x.dtype).clone()
        alpha_bar = self._table(x)
        for timestep in range(self.config.denoise_steps, 0, -1):
            alpha = alpha_bar[timestep]
            previous_alpha = alpha_bar[timestep - 1]
            clean_prediction = model(x, y)
            noise_scale = (1.0 - alpha).sqrt().clamp_min(torch.finfo(y.dtype).eps)
            predicted_noise = (y - alpha.sqrt() * clean_prediction) / noise_scale
            # DDIM eta=0. The final previous_alpha is exactly 1, so the
            # last result is clean_prediction without any residual noise.
            y = (
                previous_alpha.sqrt() * clean_prediction
                + (1.0 - previous_alpha).sqrt() * predicted_noise
            )
        return y


@dataclass(frozen=True)
class DiffusionTrainingStep:
    iteration: int
    denoising_loss: float


class DiffusionTrainer:
    def __init__(
        self,
        model: DiffusionMLP,
        diffusion: ConditionalDiffusion,
        *,
        learning_rate: float = 0.01,
        noise_seed: int = 1042,
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.model.requires_grad_(True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.generator = torch.Generator(device=next(model.parameters()).device)
        self.generator.manual_seed(noise_seed)
        self.iteration = 0

    def step(self, x: torch.Tensor, clean_y: torch.Tensor) -> DiffusionTrainingStep:
        self.model.train()
        # Each point independently samples one of the T noise levels. Training
        # needs one forward/backward pass, not an unrolled sampling trajectory.
        timesteps = torch.randint(
            1, self.diffusion.config.denoise_steps + 1, (x.shape[0],),
            device=x.device, generator=self.generator,
        )
        noise = torch.randn(
            clean_y.shape, device=clean_y.device, dtype=clean_y.dtype, generator=self.generator
        )
        noisy_y = self.diffusion.add_noise(clean_y, timesteps, noise)
        self.optimizer.zero_grad(set_to_none=True)
        clean_prediction = self.model(x, noisy_y)
        loss = nn.functional.mse_loss(clean_prediction, clean_y)
        loss.backward()
        self.optimizer.step()
        self.iteration += 1
        return DiffusionTrainingStep(self.iteration, loss.detach().item())
