"""Fixed-width ReLU MLP, training configuration and renderer snapshots."""

from dataclasses import dataclass
from threading import RLock
from typing import Iterator

import numpy as np
import torch
from torch import nn


def target_function(x: torch.Tensor) -> torch.Tensor:
    """Smooth target with several frequencies and a nonlinear trend."""
    return torch.sin(1.7 * x) + 0.35 * torch.sin(5.3 * x) + 0.15 * x**2 - 0.3 * x


@dataclass(frozen=True)
class Config:
    hidden_size: int = 64
    total_steps: int = 20_000
    learning_rate: float = 0.01
    seed: int = 42
    n_train: int = 256
    n_grid: int = 601
    x_min: float = -3.0
    x_max: float = 3.0

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive.")
        if self.n_train < 2 or self.n_grid < 2 or self.x_min >= self.x_max:
            raise ValueError("At least two points and x_min < x_max are required.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")


class MLP(nn.Module):
    """Linear(input_size, hidden_size) -> ReLU -> Linear(hidden_size, 1)."""

    def __init__(self, hidden_size: int = 64, *, input_size: int = 1) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(x)))


@dataclass(frozen=True)
class TrainingProgress:
    step: int


@dataclass(frozen=True)
class Snapshot:
    step: int
    train_mse: float
    grid_mse: float
    prediction: np.ndarray


class Experiment:
    """Training publishes progress; the renderer evaluates a model copy."""

    prediction_name = "MLP"
    evaluation_label = "MSE"

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.x_train = torch.linspace(config.x_min, config.x_max, config.n_train).unsqueeze(1)
        self.x_grid = torch.linspace(config.x_min, config.x_max, config.n_grid).unsqueeze(1)
        self.y_train = target_function(self.x_train)
        self.y_grid = target_function(self.x_grid)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = self._make_model()
        self.history: list[Snapshot] = []
        self.model_lock = RLock()
        self.started = False
        self.revision = -1
        self._record_progress(0)

    def _make_model(self) -> MLP:
        return MLP(self.config.hidden_size)

    def make_figure(self, snapshot: Snapshot):
        from .visualization import _make_figure

        return _make_figure(self, snapshot)

    def _begin(self) -> None:
        with self.model_lock:
            if self.started:
                raise RuntimeError("Create a new Experiment to run training again.")
            self.started = True

    def _record_progress(self, step: int) -> TrainingProgress:
        """Call under model_lock to publish matching progress and parameters."""
        self.revision += 1
        self.progress = TrainingProgress(step=step)
        return self.progress

    @torch.inference_mode()
    def evaluate_model(self, model: nn.Module, progress: TrainingProgress) -> Snapshot:
        model.eval()
        train_prediction = model(self.x_train)
        grid_prediction = model(self.x_grid)
        return Snapshot(
            step=progress.step,
            train_mse=torch.mean((train_prediction - self.y_train) ** 2).item(),
            grid_mse=torch.mean((grid_prediction - self.y_grid) ** 2).item(),
            prediction=grid_prediction.squeeze(1).numpy().copy(),
        )

    def run(self) -> Iterator[TrainingProgress]:
        self._begin()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        yield self.progress
        for step in range(1, self.config.total_steps + 1):
            with self.model_lock:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(self.model(self.x_train), self.y_train)
                loss.backward()
                optimizer.step()
                progress = self._record_progress(step)
            yield progress
