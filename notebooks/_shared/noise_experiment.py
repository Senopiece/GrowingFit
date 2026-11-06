"""Обучение с шумом отдельно от evaluation и подготовки запрошенных кадров."""

from dataclasses import replace
from typing import Iterator

import plotly.graph_objects as go
import torch

from .annealed_weight_noise import (
    AnnealedWeightNoiseTrainer, MatchedWeightNoiseMLP, NoiseSchedule, NoiseTrainingStep,
    matched_gaussian_kl,
)
from .experiment import Config, Experiment, Snapshot, TrainingProgress
from .visualization import _make_figure, train_live


class AnnealedNoiseExperiment(Experiment):
    def __init__(self, config: Config, schedule: NoiseSchedule) -> None:
        config = replace(
            config, group_sizes=(sum(config.group_sizes),), steps_per_stage=config.total_steps
        )
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = MatchedWeightNoiseMLP(config.group_sizes[0])
        self.trainer = AnnealedWeightNoiseTrainer(
            self.model, learning_rate=config.learning_rate, schedule=schedule
        )
        self.noise_history: list[NoiseTrainingStep] = []
        self.render_metrics: list[NoiseTrainingStep] = []
        self.preview_generator = torch.Generator(device=self.x_grid.device)
        self.preview_generator.manual_seed(config.seed + 10000)

    def run(self) -> Iterator[TrainingProgress]:
        self._begin()
        yield self.progress
        for _ in range(self.config.total_steps):
            with self.model_lock:
                result = self.trainer.step(self.x_train, self.y_train, self.config.total_steps)
                self.noise_history.append(result)
                progress = self._record_progress(
                    result.iteration, result.iteration,
                    f"{result.objective}; sigma={result.sigma:.5f}",
                )
            yield progress

    def render_metadata(self) -> dict:
        sigma = (
            self.noise_history[-1].sigma if self.noise_history
            else self.trainer.schedule(1, self.config.total_steps)
        )
        return {"sigma": sigma, "objective": "MSE" if sigma == 0 else "KL"}

    @torch.inference_mode()
    def make_render_figure(
        self, model: MatchedWeightNoiseMLP, snapshot: Snapshot, metadata: dict
    ) -> go.Figure:
        """Only the renderer calls this; all evaluation uses its model copy."""
        sigma = metadata["sigma"]
        objective = metadata["objective"]
        loss = (
            snapshot.train_mse if sigma == 0
            else matched_gaussian_kl(model, self.x_train, self.y_train, sigma).item()
        )
        metrics = NoiseTrainingStep(snapshot.step, sigma, objective, loss)
        self.render_metrics.append(metrics)
        figure = _make_figure(
            self, snapshot,
            [item.iteration for item in self.render_metrics],
            [max(item.loss, 1e-15) if item.objective == "KL" else None for item in self.render_metrics],
        )
        figure.data[0].name = "KL при σ > 0"
        figure.data[0].showlegend = True
        figure.layout.annotations[0].text = (
            f"{objective} = {loss:.6f} · σ = {sigma:.4f} · MSE среднего = {snapshot.train_mse:.6f}"
        )
        figure.update_yaxes(title_text="KL / финальный MSE", row=1, col=1)
        deterministic = [item for item in self.render_metrics if item.objective == "MSE"]
        figure.add_trace(
            go.Scatter(
                x=[item.iteration for item in deterministic],
                y=[max(item.loss, 1e-15) for item in deterministic],
                name="MSE при σ = 0", mode="markers", marker=dict(color="#dc2626", size=9),
            ), row=1, col=1,
        )
        target_noise = torch.randn(
            self.y_grid.shape, generator=self.preview_generator,
            device=self.y_grid.device, dtype=self.y_grid.dtype,
        ) if sigma > 0 else torch.zeros_like(self.y_grid)
        target_sample = self.y_grid + sigma * target_noise
        prediction_sample = model.sample_noisy_forward(
            self.x_grid, sigma, generator=self.preview_generator
        )
        x = self.x_grid.squeeze(1).tolist()
        for name, values, color in (
            ("Шумный таргет", target_sample, "#94a3b8"),
            ("Выход при шумных весах", prediction_sample, "#f59e0b"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=x, y=values.squeeze(1).tolist(), name=name, mode="markers",
                    marker=dict(size=3, color=color, opacity=0.4),
                ), row=2, col=1,
            )
        return figure


async def train_noise_live(
    experiment: AnnealedNoiseExperiment, *, target_fps: float = 10.0,
    step_delay: float | None = None,
) -> None:
    await train_live(experiment, target_fps=target_fps, step_delay=step_delay)
