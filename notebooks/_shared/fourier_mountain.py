"""Fourier-conditioned CFM and standard MSE fitting for an eroded 2D height field."""

from dataclasses import asdict, dataclass
from math import isfinite, pi
from pathlib import Path
from threading import RLock

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import nn

from .conditional_flow_matching import ConditionalFlowMatching, FlowMatchingConfig, FlowMatchingTrainer
from .experiment import Experiment, Snapshot, TrainingProgress
from .mountains import MountainTerrain, TerrainConfig


@dataclass(frozen=True)
class MountainFitConfig:
    hidden_size: int = 512
    frequencies: tuple[float, ...] = (1, 2, 4, 8, 16, 32)
    directions: int = 4
    total_steps: int = 32_000
    batch_size: int = 1024
    learning_rate: float = 0.001
    seed: int = 42
    preview_resolution: int = 96
    inference_batch_size: int = 2048
    device: str = "cpu"

    def __post_init__(self) -> None:
        sizes = (self.hidden_size, self.directions, self.total_steps, self.batch_size,
                 self.preview_resolution, self.inference_batch_size)
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("Network, batch, grid, and training sizes must be positive integers.")
        if self.preview_resolution < 2:
            raise ValueError("preview_resolution must be at least 2.")
        if not self.frequencies or any(not isfinite(value) or value <= 0 for value in self.frequencies):
            raise ValueError("Fourier frequencies must be finite and positive.")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")


class CoordinateFourierFeatures(nn.Module):
    """Fixed oriented sin/cos bands plus normalized raw coordinates."""

    def __init__(self, config: MountainFitConfig, terrain: MountainTerrain) -> None:
        super().__init__()
        angles = torch.arange(config.directions, dtype=torch.float32) * pi / config.directions
        directions = torch.stack((angles.cos(), angles.sin()), dim=1)
        frequencies = torch.tensor(config.frequencies, dtype=torch.float32)
        self.register_buffer("projection", (frequencies[:, None, None] * directions).reshape(-1, 2))
        domain = terrain.config
        low = torch.tensor([domain.x_min, domain.y_min], dtype=torch.float32)
        high = torch.tensor([domain.x_max, domain.y_max], dtype=torch.float32)
        self.register_buffer("center", (low + high) / 2)
        self.register_buffer("half_extent", (high - low) / 2)
        self.output_size = 2 + 2 * len(config.frequencies) * config.directions

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        normalized = (coords - self.center) / self.half_extent
        phase = pi * normalized @ self.projection.T
        return torch.cat((normalized, phase.sin(), phase.cos()), dim=1)


class ResidualHead(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(feature_size + 1, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(torch.cat((features, state), dim=1))))


class FourierMountainMLP(nn.Module):
    """[Fourier(x,y), noisy height] -> one hidden ReLU layer -> height residual."""

    def __init__(self, config: MountainFitConfig, terrain: MountainTerrain) -> None:
        super().__init__()
        self.features = CoordinateFourierFeatures(config, terrain)
        self.head = ResidualHead(self.features.output_size, config.hidden_size)

    def forward(self, coords: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(coords), state)


class FourierStandardMLP(nn.Module):
    """Fourier(x,y) -> one hidden ReLU layer -> normalized height."""

    def __init__(self, config: MountainFitConfig, terrain: MountainTerrain) -> None:
        super().__init__()
        self.features = CoordinateFourierFeatures(config, terrain)
        self.head = nn.Sequential(
            nn.Linear(self.features.output_size, config.hidden_size), nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(coords))


@torch.inference_mode()
def sample_heights(model, coords, flow, mean, std, batch_size):
    """Cache coordinate features once per chunk across all Euler evaluations."""
    model.eval()
    parts = []
    for chunk in coords.split(batch_size):
        features = model.features(chunk)
        noise = torch.zeros((len(chunk), 1), device=chunk.device, dtype=chunk.dtype)
        normalized = flow.sample(model.head, features, initial_noise=noise)
        parts.append(normalized * std + mean)
    return torch.cat(parts)


class MountainExperiment(Experiment):
    method = "CFM"

    def __init__(
        self, terrain: MountainTerrain, config: MountainFitConfig = MountainFitConfig(),
        flow_config: FlowMatchingConfig = FlowMatchingConfig(eval_sampling_steps=64),
    ) -> None:
        self.config, self.terrain, self.flow_config = config, terrain, flow_config
        self.device = torch.device(config.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = self._make_mountain_model().to(self.device)
        self.flow = ConditionalFlowMatching(flow_config)
        xx, yy = np.meshgrid(terrain.x, terrain.y)
        self.coords = torch.from_numpy(np.stack((xx.ravel(), yy.ravel()), axis=1)).float().to(self.device)
        raw = torch.from_numpy(terrain.eroded_height.reshape(-1, 1).copy()).to(self.device)
        self.height_mean = raw.mean().item()
        self.height_std = raw.std(unbiased=False).clamp_min(1e-8).item()
        self.targets = (raw - self.height_mean) / self.height_std
        # Ordinary no_grad tensors can be used as inputs during backpropagation.
        with torch.no_grad():
            self.encoded_coords = self.model.features(self.coords)
        self._configure_training()
        self.batch_generator = torch.Generator(device=self.device).manual_seed(config.seed + 2000)
        self.preview_size = min(config.preview_resolution, terrain.config.resolution)
        axis = np.linspace(0, terrain.config.resolution - 1, self.preview_size, dtype=np.int64)
        self.preview_x, self.preview_y = terrain.x[axis], terrain.y[axis]
        flat_indices = (axis[:, None] * terrain.config.resolution + axis[None, :]).ravel()
        indices = torch.as_tensor(flat_indices, device=self.device)
        self.preview_coords, self.preview_target = self.coords[indices], raw[indices]
        self.history: list[Snapshot] = []
        self.training_losses: list[dict] = []
        self.model_lock = RLock()
        self.started, self.revision = False, -1
        self._record_progress(0)

    def _make_mountain_model(self):
        return FourierMountainMLP(self.config, self.terrain)

    def _configure_training(self) -> None:
        self.trainer = FlowMatchingTrainer(
            self.model.head, self.flow,
            learning_rate=self.config.learning_rate, noise_seed=self.config.seed + 1000,
        )

    def _train_batch(self, indices) -> float:
        return self.trainer.step(self.encoded_coords[indices], self.targets[indices]).residual_loss

    def _predict_model(self, model, coords):
        return sample_heights(
            model, coords, self.flow, self.height_mean, self.height_std,
            self.config.inference_batch_size,
        )

    @property
    def prediction_title(self) -> str:
        return f"CFM prediction | {self.flow_config.eval_sampling_steps} Euler steps"

    def run(self):
        self._begin()
        yield self.progress
        for step in range(1, self.config.total_steps + 1):
            with self.model_lock:
                indices = torch.randint(
                    len(self.coords), (self.config.batch_size,), device=self.device,
                    generator=self.batch_generator,
                )
                loss = self._train_batch(indices)
                self.training_losses.append({"step": step, "loss": loss})
                progress = self._record_progress(step)
            yield progress

    @torch.inference_mode()
    def evaluate_model(self, model: nn.Module, progress: TrainingProgress) -> Snapshot:
        predicted = self._predict_model(model, self.preview_coords)
        mse = ((predicted - self.preview_target) ** 2).mean().item()
        return Snapshot(
            step=progress.step, train_mse=mse, grid_mse=mse,
            prediction=predicted.reshape(self.preview_size, self.preview_size).cpu().numpy().copy(),
        )

    def make_figure(self, snapshot: Snapshot) -> go.Figure:
        target = self.preview_target.reshape(self.preview_size, self.preview_size).cpu().numpy()
        error = snapshot.prediction - target
        figure = make_subplots(
            rows=2, cols=3,
            specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
            row_heights=[0.3, 0.7], vertical_spacing=0.18,
            subplot_titles=(
                f"Preview grid MSE = {snapshot.grid_mse:.6g} | step {snapshot.step:,}/{self.config.total_steps:,}",
                "Eroded target", self.prediction_title, "Prediction - target",
            ),
        )
        figure.add_trace(go.Scatter(
            x=[item.step for item in self.history],
            y=[max(item.grid_mse, 1e-15) for item in self.history],
            mode="lines", line=dict(color="#2563eb"), showlegend=False,
        ), row=1, col=1)
        low, high = float(target.min()), float(target.max())
        error_limit = max(float(np.abs(error).max()), 1e-8)
        for column, values in enumerate((target, snapshot.prediction), start=1):
            figure.add_trace(go.Heatmap(
                x=self.preview_x.tolist(), y=self.preview_y.tolist(), z=values.tolist(),
                colorscale="Earth", zmin=low, zmax=high, showscale=False,
                zsmooth=False,
            ), row=2, col=column)
        figure.add_trace(go.Heatmap(
            x=self.preview_x.tolist(), y=self.preview_y.tolist(), z=error.tolist(),
            colorscale="RdBu_r", zmin=-error_limit, zmax=error_limit,
            colorbar=dict(title="Height error", len=0.6, y=0.3),
        ), row=2, col=3)
        for column in range(1, 4):
            figure.update_xaxes(
                title_text="x", range=[float(self.preview_x[0]), float(self.preview_x[-1])],
                autorange=False, constrain="domain", row=2, col=column,
            )
            figure.update_yaxes(
                title_text="y", range=[float(self.preview_y[0]), float(self.preview_y[-1])],
                autorange=False, constrain="domain", scaleanchor=f"x{column + 1}", scaleratio=1,
                row=2, col=column,
            )
        figure.update_xaxes(title_text="Training step", range=[0, self.config.total_steps], row=1, col=1)
        figure.update_yaxes(title_text="Height MSE", type="log", row=1, col=1)
        figure.update_layout(
            template="plotly_white", autosize=True, height=850,
            margin=dict(l=65, r=100, t=65, b=65), font=dict(size=12),
        )
        return figure

    @torch.inference_mode()
    def predict_full_grid(self) -> np.ndarray:
        with self.model_lock:
            prediction = self._predict_model(self.model, self.coords)
        return prediction.reshape(self.terrain.eroded_height.shape).cpu().numpy().copy()

    def save(self, path: Path) -> None:
        """Save model, Fourier buffers, target terrain, normalization, and configuration."""
        with self.model_lock:
            payload = {
                "format_version": 1, "method": self.method, "fit_config": asdict(self.config),
                "terrain_config": asdict(self.terrain.config),
                "flow_config": asdict(self.flow_config) if self.method == "CFM" else None,
                "completed_steps": self.progress.step,
                "height_mean": self.height_mean, "height_std": self.height_std,
                "state_dict": {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()},
                "terrain": {key: torch.from_numpy(getattr(self.terrain, key).copy())
                            for key in ("x", "y", "base_height", "eroded_height")},
                "training_losses": list(self.training_losses),
            }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)


class StandardMountainExperiment(MountainExperiment):
    method = "Standard MLP"

    def _make_mountain_model(self):
        return FourierStandardMLP(self.config, self.terrain)

    def _configure_training(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _train_batch(self, indices) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model.head(self.encoded_coords[indices])
        loss = nn.functional.mse_loss(prediction, self.targets[indices])
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    @torch.inference_mode()
    def _predict_model(self, model, coords):
        model.eval()
        predictions = [
            model(chunk) * self.height_std + self.height_mean
            for chunk in coords.split(self.config.inference_batch_size)
        ]
        return torch.cat(predictions)

    @property
    def prediction_title(self) -> str:
        return "Standard MLP prediction"


def comparison_heatmaps(terrain, cfm_prediction, standard_prediction, cfm_steps):
    """Compare both full-grid fits on shared height and signed-error scales."""
    target = terrain.eroded_height
    errors = (cfm_prediction - target, standard_prediction - target)
    limit = max(float(np.abs(errors[0]).max()), float(np.abs(errors[1]).max()), 1e-8)
    figure = make_subplots(
        rows=2, cols=3, specs=[[{}, {}, {}], [None, {}, {}]], vertical_spacing=0.18,
        subplot_titles=("Eroded target", f"CFM | {cfm_steps} Euler steps", "Standard MLP",
                        "CFM - target", "Standard MLP - target"),
    )
    panels = ((1, 1, target), (1, 2, cfm_prediction), (1, 3, standard_prediction),
              (2, 2, errors[0]), (2, 3, errors[1]))
    for axis, (row, column, values) in enumerate(panels, start=1):
        figure.add_trace(go.Heatmap(
            x=terrain.x.tolist(), y=terrain.y.tolist(), z=values.tolist(),
            coloraxis="coloraxis" if row == 1 else "coloraxis2", zsmooth=False,
        ), row=row, col=column)
        figure.update_xaxes(
            title_text="x", range=[float(terrain.x[0]), float(terrain.x[-1])],
            autorange=False, constrain="domain", row=row, col=column,
        )
        figure.update_yaxes(
            title_text="y", range=[float(terrain.y[0]), float(terrain.y[-1])],
            autorange=False, constrain="domain", scaleratio=1,
            scaleanchor="x" if axis == 1 else f"x{axis}", row=row, col=column,
        )
    figure.update_layout(
        template="plotly_white", autosize=True, height=900,
        margin=dict(l=65, r=110, t=65, b=65),
        coloraxis=dict(colorscale="Earth", cmin=float(target.min()), cmax=float(target.max()),
                       colorbar=dict(title="Height", len=0.35, y=0.8)),
        coloraxis2=dict(colorscale="RdBu_r", cmin=-limit, cmax=limit,
                        colorbar=dict(title="Height error", len=0.35, y=0.2)),
    )
    return figure


def surface_comparison(terrain: MountainTerrain, prediction: np.ndarray | None = None) -> go.Figure:
    """Use identical height and color scales for the two surfaces."""
    first = terrain.base_height if prediction is None else terrain.eroded_height
    second = terrain.eroded_height if prediction is None else prediction
    titles = ("Mountain field", "Eroded target") if prediction is None else ("Eroded target", "CFM prediction")
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]], subplot_titles=titles)
    low = float(min(first.min(), second.min()))
    high = float(max(first.max(), second.max()))
    for column, height in enumerate((first, second), start=1):
        figure.add_trace(go.Surface(
            x=terrain.x.tolist(), y=terrain.y.tolist(), z=height.tolist(), colorscale="Earth",
            cmin=low, cmax=high, showscale=False,
        ), row=1, col=column)
    figure.update_scenes(
        xaxis_title="x", yaxis_title="y", zaxis=dict(title="Height", range=[low, high]),
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.6),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    )
    figure.update_layout(template="plotly_white", autosize=True, height=650, margin=dict(l=10, r=10, t=55, b=10))
    return figure


def load_mountain_checkpoint(path: Path, *, device: str = "cpu") -> MountainExperiment:
    """Restore a saved model and its exact target terrain for inference."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported mountain checkpoint format.")
    terrain = MountainTerrain(
        config=TerrainConfig(**payload["terrain_config"]),
        **{key: tensor.numpy().copy() for key, tensor in payload["terrain"].items()},
    )
    fit_values = {**payload["fit_config"], "device": device}
    experiment_class = {
        "CFM": MountainExperiment, "Standard MLP": StandardMountainExperiment,
    }[payload.get("method", "CFM")]
    experiment = experiment_class(
        terrain, MountainFitConfig(**fit_values),
        FlowMatchingConfig(**(payload.get("flow_config") or {})),
    )
    experiment.model.load_state_dict(payload["state_dict"], strict=True)
    with torch.no_grad():
        experiment.encoded_coords = experiment.model.features(experiment.coords)
    experiment.height_mean = payload["height_mean"]
    experiment.height_std = payload["height_std"]
    experiment.targets = (
        torch.from_numpy(terrain.eroded_height.reshape(-1, 1).copy()).to(experiment.device)
        - experiment.height_mean
    ) / experiment.height_std
    experiment.training_losses = payload.get("training_losses", payload.get("residual_losses", []))
    experiment._record_progress(payload["completed_steps"])
    experiment.started = True
    return experiment
