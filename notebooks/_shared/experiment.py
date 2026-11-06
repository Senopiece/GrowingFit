"""Однослойная ReLU-сеть: добавляем группы нейронов и замораживаем обученные."""

from dataclasses import dataclass
from itertools import accumulate
from threading import RLock
from typing import Iterator

import numpy as np
import torch
from torch import nn


def target_function(x: torch.Tensor) -> torch.Tensor:
    """Гладкая функция с несколькими частотами и нелинейным трендом."""
    return torch.sin(1.7 * x) + 0.35 * torch.sin(5.3 * x) + 0.15 * x**2 - 0.3 * x


@dataclass(frozen=True)
class Config:
    seed: int = 42
    growth_seed: int = 123
    group_sizes: tuple[int, ...] = (1, 1, 2, 4, 8, 16, 32, 64)
    steps_per_stage: int = 200
    learning_rate: float = 0.01
    n_train: int = 256
    n_grid: int = 601
    x_min: float = -3.0
    x_max: float = 3.0

    @property
    def stages(self) -> tuple[tuple[int, int], ...]:
        return tuple((width, self.steps_per_stage) for width in accumulate(self.group_sizes))

    @property
    def total_steps(self) -> int:
        return sum(n_steps for _, n_steps in self.stages)

    def __post_init__(self) -> None:
        if not self.group_sizes or any(size < 1 for size in self.group_sizes):
            raise ValueError("group_sizes должен содержать положительные размеры групп.")
        if self.steps_per_stage < 1:
            raise ValueError("Число шагов каждого этапа должно быть положительным.")
        if self.n_train < 2 or self.n_grid < 2 or self.x_min >= self.x_max:
            raise ValueError("Нужны как минимум две точки и x_min < x_max.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate должен быть конечным и положительным.")


class _NeuronGroup(nn.Module):
    """Группа нейронов одного общего скрытого слоя, без отдельного выходного bias."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(1, width)
        self.activation = nn.ReLU()
        self.output = nn.Linear(width, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(x)))


class MLP(nn.Module):
    """Сумма групп эквивалентна одному скрытому ReLU-слою с линейным выходом.

    Отдельные группы Parameter позволяют полностью замораживать старые нейроны,
    включая входящие веса, hidden bias и связи с выходом.
    """

    def __init__(self, width: int = 1) -> None:
        super().__init__()
        self.groups = nn.ModuleList([_NeuronGroup(width)])
        self.output_bias = nn.Parameter(torch.zeros(1))

    @property
    def width(self) -> int:
        return sum(group.hidden.out_features for group in self.groups)

    @property
    def trainable_width(self) -> int:
        return sum(
            group.hidden.out_features for group in self.groups
            if any(parameter.requires_grad for parameter in group.parameters())
        )

    def freeze(self) -> None:
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.output_bias.expand(x.shape[0], 1)
        for group in self.groups:
            prediction = prediction + group(x)
        return prediction


def grow_hidden_layer(model: MLP, extra: int = 4, seed: int = 123) -> MLP:
    """Замораживает старые параметры и добавляет новую обучаемую группу на месте.

    Старые группы и общий выходной bias остаются без изменений. Новые входящие
    и исходящие веса случайны. Adam на следующем этапе получает только их.
    """
    if extra < 1:
        raise ValueError("extra должен быть положительным.")
    model.freeze()
    reference = model.output_bias
    # Инициализация воспроизводима и не меняет внешний генератор случайных чисел.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        group = _NeuronGroup(extra).to(device=reference.device, dtype=reference.dtype)
    group.train(model.training)
    model.groups.append(group)
    return model


@dataclass(frozen=True)
class TrainingProgress:
    step: int
    stage_step: int
    width: int
    trainable_width: int
    frozen_width: int
    event: str


@dataclass(frozen=True)
class Snapshot:
    step: int
    stage_step: int
    width: int
    trainable_width: int
    frozen_width: int
    event: str
    train_mse: float
    grid_mse: float
    prediction: np.ndarray


class Experiment:
    """Обучение публикует прогресс; evaluation делает рендерер по запросу."""

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.x_train = torch.linspace(config.x_min, config.x_max, config.n_train).unsqueeze(1)
        self.x_grid = torch.linspace(config.x_min, config.x_max, config.n_grid).unsqueeze(1)
        self.y_train = target_function(self.x_train)
        self.y_grid = target_function(self.x_grid)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = MLP(config.group_sizes[0])
        self.history: list[Snapshot] = []
        self.model_lock = RLock()
        self.started = False
        self.revision = -1
        self._record_progress(0, 0, f"Инициализация: {self.model.width} нейронов")

    def _begin(self) -> None:
        with self.model_lock:
            if self.started:
                raise RuntimeError("Для повторного запуска создайте новый Experiment.")
            self.started = True

    def _record_progress(self, step: int, stage_step: int, event: str) -> TrainingProgress:
        """Вызывать под model_lock: прогресс и параметры образуют одно состояние."""
        self.revision += 1
        self.progress = TrainingProgress(
            step=step,
            stage_step=stage_step,
            width=self.model.width,
            trainable_width=self.model.trainable_width,
            frozen_width=self.model.width - self.model.trainable_width,
            event=event,
        )
        return self.progress

    def render_metadata(self):
        """Дополнительные данные текущей итерации; читать под model_lock."""
        return None

    @torch.inference_mode()
    def evaluate_model(self, model: nn.Module, progress: TrainingProgress) -> Snapshot:
        """Evaluation отдельной копии модели в потоке рендерера."""
        model.eval()
        train_prediction = model(self.x_train)
        grid_prediction = model(self.x_grid)
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

    def run(self) -> Iterator[TrainingProgress]:
        self._begin()
        yield self.progress
        step = 0
        for stage, (width, n_steps) in enumerate(self.config.stages):
            if stage > 0:
                with self.model_lock:
                    old_width = self.model.width
                    self.model = grow_hidden_layer(
                        self.model, extra=width - old_width, seed=self.config.growth_seed + stage - 1
                    )
                    progress = self._record_progress(
                        step, 0, f"Рост {old_width} → {width}: старые нейроны заморожены"
                    )
                yield progress
            optimizer = torch.optim.Adam(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                lr=self.config.learning_rate,
            )
            for stage_step in range(1, n_steps + 1):
                with self.model_lock:
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(self.model(self.x_train), self.y_train)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    if stage_step == n_steps:
                        self.model.freeze()
                        event = f"Этап завершён: все {width} нейронов заморожены"
                    else:
                        event = f"Обучение новых: {self.model.trainable_width}; всего: {width}"
                    progress = self._record_progress(step, stage_step, event)
                yield progress
