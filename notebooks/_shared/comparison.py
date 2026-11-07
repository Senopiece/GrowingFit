"""Training checkpoints and repeatable evaluation for the comparison notebook."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .conditional_flow_matching import (
    ConditionalFlowMatching, FlowMatchingConfig, FlowMatchingMLP,
)
from .experiment import Config, Experiment, MLP, target_function


METHODS = ("Standard MLP", "CFM")
METRIC_COLUMNS = (
    "training_steps", "hidden_size", "method", "eval_sampling_steps", "variant", "seed",
    "train_mse", "grid_mse", "training_seconds", "checkpoint",
)


def checkpoint_path(
    directory: Path, method: str, hidden_size: int, seed: int, training_steps: int
) -> Path:
    names = {"Standard MLP": "standard_mlp", "CFM": "cfm"}
    return (
        Path(directory) / names[method] / f"width_{hidden_size}"
        / f"seed_{seed}" / f"step_{training_steps}.pt"
    )


def save_checkpoint(
    path: Path, experiment: Experiment, method: str, training_seconds: float
) -> None:
    """Persist parameters at the current training step; publish the file atomically."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}.")
    if not 1 <= experiment.progress.step <= experiment.config.total_steps:
        raise ValueError("A checkpoint must represent a completed optimizer step.")
    payload = {
        "format_version": 2,
        "method": method,
        "prediction_type": "endpoint_residual" if method == "CFM" else "target",
        "config": asdict(experiment.config),
        "flow_config": asdict(experiment.flow.config) if method == "CFM" else None,
        "completed_steps": experiment.progress.step,
        "training_seconds": float(training_seconds),
        "torch_version": str(torch.__version__),
        "state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in experiment.model.state_dict().items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, *, method: str, config: Config, flow_config: FlowMatchingConfig,
    training_steps: int | None = None,
) -> dict:
    """Validate a specific training checkpoint, independently of evaluation settings."""
    if training_steps is None:
        training_steps = config.total_steps
    if not 1 <= training_steps <= config.total_steps:
        raise ValueError("training_steps must lie between 1 and config.total_steps.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    saved_config = Config(**payload["config"])
    saved_training = asdict(saved_config)
    requested_training = asdict(config)
    # The evaluation grid can change without changing the learned parameters.
    saved_training.pop("n_grid")
    requested_training.pop("n_grid")
    prediction_type = "endpoint_residual" if method == "CFM" else "target"
    compatible = (
        payload.get("format_version") in (1, 2)
        and payload.get("method") == method
        and payload.get("prediction_type") == prediction_type
        and payload.get("completed_steps") == training_steps
        and saved_training == requested_training
    )
    if method == "CFM":
        saved_flow = payload.get("flow_config") or {}
        compatible = compatible and (
            saved_flow.get("train_sampling_steps") == flow_config.train_sampling_steps
        )
    if not compatible:
        raise ValueError(
            f"Checkpoint training settings differ: {path}. "
            "Restore those settings or choose a different checkpoint_dir."
        )
    return payload


def variant_label(method: str, eval_sampling_steps: int) -> str:
    return method if method == "Standard MLP" else f"CFM {eval_sampling_steps}"


@torch.inference_mode()
def evaluate_checkpoint(
    payload: dict,
    path: Path,
    *,
    eval_sampling_steps: tuple[int, ...],
    n_grid: int,
) -> list[dict]:
    """Reload parameters into a fresh model; evaluate CFM from zero initial noise."""
    if not eval_sampling_steps or any(type(step) is not int or step < 1 for step in eval_sampling_steps):
        raise ValueError("eval_sampling_steps must contain positive integers.")
    if len(set(eval_sampling_steps)) != len(eval_sampling_steps):
        raise ValueError("eval_sampling_steps must not contain duplicates.")
    config = replace(Config(**payload["config"]), n_grid=n_grid)
    method = payload["method"]
    # Reconstructing a model must not change the caller's random state.
    with torch.random.fork_rng(devices=[]):
        model = FlowMatchingMLP(config.hidden_size) if method == "CFM" else MLP(config.hidden_size)
    dtype = next(iter(payload["state_dict"].values())).dtype
    model.to(dtype=dtype)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    x_train = torch.linspace(config.x_min, config.x_max, config.n_train, dtype=dtype).unsqueeze(1)
    x_grid = torch.linspace(config.x_min, config.x_max, config.n_grid, dtype=dtype).unsqueeze(1)
    x = torch.cat((x_train, x_grid))
    target = target_function(x)
    noise = torch.zeros_like(x)
    settings = eval_sampling_steps if method == "CFM" else (0,)
    rows = []
    for steps in settings:
        if method == "CFM":
            flow_config = replace(FlowMatchingConfig(**payload["flow_config"]), eval_sampling_steps=steps)
            prediction = ConditionalFlowMatching(flow_config).sample(model, x, initial_noise=noise)
        else:
            prediction = model(x)
        squared_error = (prediction - target) ** 2
        train_mse = squared_error[:config.n_train].mean().item()
        grid_mse = squared_error[config.n_train:].mean().item()
        if not np.isfinite([train_mse, grid_mse]).all():
            raise FloatingPointError(f"Nonfinite MSE: {path}, evaluation steps {steps}.")
        rows.append({
            "training_steps": payload["completed_steps"],
            "hidden_size": config.hidden_size,
            "method": method,
            # Zero denotes direct prediction, not an Euler sampler.
            "eval_sampling_steps": steps,
            "variant": variant_label(method, steps),
            "seed": config.seed,
            "train_mse": train_mse,
            "grid_mse": grid_mse,
            "training_seconds": payload["training_seconds"],
            "checkpoint": str(path),
        })
    return rows
