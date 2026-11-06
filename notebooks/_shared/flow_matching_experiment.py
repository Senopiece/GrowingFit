"""Conditional flow matching connected to background training and pull rendering."""

from typing import Iterator

import torch

from .conditional_flow_matching import (
    ConditionalFlowMatching, FlowMatchingConfig, FlowMatchingMLP,
    FlowMatchingTrainer, FlowMatchingTrainingStep,
)
from .experiment import Config, Experiment, Snapshot, TrainingProgress


class FlowMatchingExperiment(Experiment):
    def __init__(self, config: Config, flow_config: FlowMatchingConfig = FlowMatchingConfig()) -> None:
        super().__init__(config)
        self.flow = ConditionalFlowMatching(flow_config)
        self.trainer = FlowMatchingTrainer(
            self.model, self.flow,
            learning_rate=config.learning_rate, noise_seed=config.seed + 1000,
        )
        self.flow_history: list[FlowMatchingTrainingStep] = []
        self.eval_x = torch.cat((self.x_train, self.x_grid), dim=0)
        self.eval_noise = torch.zeros_like(self.eval_x)

    def _make_model(self) -> FlowMatchingMLP:
        return FlowMatchingMLP(self.config.hidden_size)

    @property
    def prediction_name(self) -> str:
        return f"CFM endpoint residual · {self.flow.config.eval_sampling_steps}-step Euler"

    @property
    def evaluation_label(self) -> str:
        return f"MSE after {self.flow.config.eval_sampling_steps}-step Euler sampling"

    def run(self) -> Iterator[TrainingProgress]:
        self._begin()
        yield self.progress
        for _ in range(self.config.total_steps):
            with self.model_lock:
                result = self.trainer.step(self.x_train, self.y_train)
                self.flow_history.append(result)
                progress = self._record_progress(result.iteration)
            yield progress

    @torch.inference_mode()
    def evaluate_model(self, model: FlowMatchingMLP, progress: TrainingProgress) -> Snapshot:
        model.eval()
        prediction = self.flow.sample(model, self.eval_x, initial_noise=self.eval_noise)
        train_prediction = prediction[: self.config.n_train]
        grid_prediction = prediction[self.config.n_train :]
        return Snapshot(
            step=progress.step,
            train_mse=torch.mean((train_prediction - self.y_train) ** 2).item(),
            grid_mse=torch.mean((grid_prediction - self.y_grid) ** 2).item(),
            prediction=grid_prediction.squeeze(1).numpy().copy(),
        )

