"""Conditional diffusion connected to the shared background training / renderer."""

from dataclasses import replace
from typing import Iterator

import plotly.graph_objects as go
import torch

from .conditional_diffusion import (
    ConditionalDiffusion, DiffusionConfig, DiffusionMLP, DiffusionTrainer, DiffusionTrainingStep,
)
from .experiment import Config, Experiment, Snapshot, TrainingProgress
from .visualization import _make_figure


class DiffusionExperiment(Experiment):
    def __init__(self, config: Config, diffusion_config: DiffusionConfig = DiffusionConfig()) -> None:
        config = replace(
            config, group_sizes=(sum(config.group_sizes),), steps_per_stage=config.total_steps
        )
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = DiffusionMLP(config.group_sizes[0])
        self.diffusion = ConditionalDiffusion(diffusion_config)
        self.trainer = DiffusionTrainer(
            self.model, self.diffusion,
            learning_rate=config.learning_rate, noise_seed=config.seed + 1000,
        )
        self.diffusion_history: list[DiffusionTrainingStep] = []
        # Keep evaluation's initial latent fixed across frames. Changes in the
        # curves then come from the model, not freshly sampled evaluation noise.
        generator = torch.Generator(device=self.x_grid.device)
        generator.manual_seed(config.seed + 20000)
        self.eval_x = torch.cat((self.x_train, self.x_grid), dim=0)
        self.eval_noise = torch.randn(
            self.eval_x.shape, device=self.eval_x.device, dtype=self.eval_x.dtype, generator=generator
        )

    def run(self) -> Iterator[TrainingProgress]:
        self._begin()
        yield self.progress
        for _ in range(self.config.total_steps):
            with self.model_lock:
                result = self.trainer.step(self.x_train, self.y_train)
                self.diffusion_history.append(result)
                progress = self._record_progress(
                    result.iteration, result.iteration, "Обучение denoiser: предсказание чистого y₀"
                )
            yield progress

    @torch.inference_mode()
    def evaluate_model(self, model: DiffusionMLP, progress: TrainingProgress) -> Snapshot:
        """The renderer performs all eight reverse steps against a model copy."""
        model.eval()
        prediction = self.diffusion.sample(model, self.eval_x, initial_noise=self.eval_noise)
        train_prediction = prediction[: self.config.n_train]
        grid_prediction = prediction[self.config.n_train :]
        return Snapshot(
            step=progress.step,
            stage_step=progress.stage_step,
            width=progress.width,
            trainable_width=progress.trainable_width,
            frozen_width=progress.frozen_width,
            event=progress.event,
            train_mse=torch.mean((train_prediction - self.y_train) ** 2).item(),
            grid_mse=torch.mean((grid_prediction - self.y_grid) ** 2).item(),
            prediction=grid_prediction.squeeze(1).numpy().copy(),
        )

    def make_render_figure(self, model: DiffusionMLP, snapshot: Snapshot, metadata) -> go.Figure:
        steps = self.diffusion.config.denoise_steps
        figure = _make_figure(
            self, snapshot,
            [item.step for item in self.history],
            [max(item.train_mse, 1e-15) for item in self.history],
        )
        figure.layout.annotations[0].text = f"MSE после {steps} шагов denoising = {snapshot.train_mse:.6f}"
        figure.layout.annotations[1].text = (
            f"MLP[x, y_t] · {snapshot.width} нейронов · {steps} DDIM-шагов · "
            f"обновление {snapshot.step}/{self.config.total_steps}"
        )
        figure.data[2].name = f"Diffusion MLP · {steps} шагов"
        return figure
