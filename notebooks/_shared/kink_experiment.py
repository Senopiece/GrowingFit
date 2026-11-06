"""Адаптер kink-aware обучения к данным и живым графикам существующих ноутбуков."""

from dataclasses import replace
from typing import Iterator

import torch

from .experiment import Config, Experiment, TrainingProgress
from .kink_training import KinkAwareTrainer, KinkMLP, KinkTrainingConfig, KinkTrainingStep


class KinkAwareExperiment(Experiment):
    def __init__(self, config: Config, training_config: KinkTrainingConfig) -> None:
        # One full-width stage; retain the total budget of a growing config too.
        config = replace(
            config, group_sizes=(sum(config.group_sizes),), steps_per_stage=config.total_steps
        )
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = KinkMLP(config.group_sizes[0])
        self.trainer = KinkAwareTrainer(
            self.model, x_min=config.x_min, x_max=config.x_max, config=training_config
        )
        self.kink_history: list[KinkTrainingStep] = []

    def run(self) -> Iterator[TrainingProgress]:
        self._begin()
        yield self.progress
        for _ in range(self.config.total_steps):
            # The renderer can only capture complete A+B iterations.
            with self.model_lock:
                result = self.trainer.step(self.x_train, self.y_train)
                self.kink_history.append(result)
                progress = self._record_progress(
                    result.iteration, result.iteration,
                    f"Kink-aware: weighted fit + repulsion; активных изломов {result.active_kinks}",
                )
            yield progress
