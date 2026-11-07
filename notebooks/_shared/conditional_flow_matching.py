"""Conditional flow matching with endpoint-residual prediction and Euler sampling."""

from dataclasses import dataclass

import torch
from torch import nn

from .experiment import MLP


class FlowMatchingMLP(MLP):
    """Same two-input architecture; predicts the endpoint residual y_1-y_t.

    For deterministic y_1=f(x), residual_theta(x,y_t) approximates f(x)-y_t.
    The endpoint is y_t+residual_theta; the velocity is residual_theta/(1-t).
    Thus no third input or timestep embedding is required for this experiment.
    """

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__(hidden_size, input_size=2)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.cat((x, state), dim=1))


@dataclass(frozen=True)
class FlowMatchingConfig:
    eval_sampling_steps: int = 8
    train_sampling_steps: int | None = None

    def __post_init__(self) -> None:
        if self.eval_sampling_steps < 1:
            raise ValueError("eval_sampling_steps must be positive.")
        if self.train_sampling_steps is not None and self.train_sampling_steps < 1:
            raise ValueError("train_sampling_steps must be positive or None.")


class ConditionalFlowMatching:
    """Straight Gaussian-to-data path, integrated with uniform Euler steps.

    Training draws t uniformly from [0, 1) when train_sampling_steps is None,
    or from {0, 1/T, ..., (T-1)/T} for a specified training grid size T.
    The evaluation Euler grid is configured independently. MSE(residual,
    y_1-y_t) equals (1-t)^2 times the corresponding velocity-matching loss,
    so this is the endpoint-weighted CFM objective, not uniform velocity MSE.
    Time is not a model input.
    Both training and sampling exclude t=1, avoiding a zero denominator.
    """

    def __init__(self, config: FlowMatchingConfig = FlowMatchingConfig()) -> None:
        self.config = config

    @staticmethod
    def interpolate(
        noise: torch.Tensor, target: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        return (1.0 - time) * noise + time * target

    @staticmethod
    def velocity(
        model: nn.Module, x: torch.Tensor, state: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # time must be strictly less than one.
        return model(x, state) / (1.0 - time)

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Exactly T model calls; the sampler never receives target values."""
        state_shape = (x.shape[0], 1)
        if initial_noise is None:
            state = torch.randn(state_shape, device=x.device, dtype=x.dtype, generator=generator)
        else:
            if initial_noise.shape != state_shape:
                raise ValueError("initial_noise must have shape (N, 1), matching the condition batch size.")
            state = initial_noise.to(device=x.device, dtype=x.dtype).clone()
        steps = self.config.eval_sampling_steps
        dt = 1.0 / steps
        for index in range(steps):
            # 1-t=(T-index)/T is evaluated directly to avoid cancellation.
            remaining_time = (steps - index) / steps
            endpoint_residual = model(x, state)
            velocity = endpoint_residual / remaining_time
            state = state + dt * velocity
        return state


@dataclass(frozen=True)
class FlowMatchingTrainingStep:
    iteration: int
    residual_loss: float

    @property
    def endpoint_loss(self) -> float:
        # MSE on residuals equals MSE on reconstructed endpoints.
        return self.residual_loss


class FlowMatchingTrainer:
    def __init__(
        self,
        model: nn.Module,
        flow: ConditionalFlowMatching,
        *,
        learning_rate: float = 0.01,
        noise_seed: int = 1042,
    ) -> None:
        self.model = model
        self.flow = flow
        self.model.requires_grad_(True)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.generator = torch.Generator(device=next(model.parameters()).device)
        self.generator.manual_seed(noise_seed)
        self.iteration = 0

    def step(self, x: torch.Tensor, target: torch.Tensor) -> FlowMatchingTrainingStep:
        self.model.train()
        steps = self.flow.config.train_sampling_steps
        if steps is None:
            time = torch.rand(
                (x.shape[0], 1), device=x.device, dtype=x.dtype, generator=self.generator
            )
        else:
            index = torch.randint(
                0, steps, (x.shape[0], 1), device=x.device, generator=self.generator
            )
            remaining_time = (steps - index).to(dtype=x.dtype) / steps
            time = 1.0 - remaining_time
        noise = torch.randn(
            target.shape, device=target.device, dtype=target.dtype, generator=self.generator
        )
        state = self.flow.interpolate(noise, target, time)
        self.optimizer.zero_grad(set_to_none=True)
        predicted_residual = self.model(x, state)
        target_residual = target - state
        loss = nn.functional.mse_loss(predicted_residual, target_residual)
        loss.backward()
        self.optimizer.step()
        self.iteration += 1
        return FlowMatchingTrainingStep(self.iteration, loss.detach().item())
