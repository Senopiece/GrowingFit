"""Shared extrapolation for the standard MLP and conditional flow matching."""

from copy import deepcopy
from math import isfinite

from IPython.display import display
import plotly.graph_objects as go
import torch

from .experiment import Experiment, target_function


def make_extrapolation_figure(
    experiment: Experiment, padding_fraction: float = 0.10
) -> go.Figure:
    """Copy the model and plot its fit with padding on both sides of the domain.

    Iterative models start with zero noise at every point, including
    the padding outside the training domain.
    """
    if not isfinite(padding_fraction) or padding_fraction < 0:
        raise ValueError("padding_fraction must be finite and nonnegative.")
    config = experiment.config
    padding = (config.x_max - config.x_min) * padding_fraction
    n_pad = max(1, round((config.n_grid - 1) * padding_fraction)) if padding > 0 else 0
    with experiment.model_lock:
        model = deepcopy(experiment.model)
    model.eval()
    reference = next(model.parameters())
    tensor_options = dict(device=reference.device, dtype=reference.dtype)
    left_x = torch.linspace(config.x_min - padding, config.x_min, n_pad + 1, **tensor_options)[:-1, None]
    right_x = torch.linspace(config.x_max, config.x_max + padding, n_pad + 1, **tensor_options)[1:, None]
    x = torch.cat((left_x, experiment.x_grid.to(**tensor_options), right_x), dim=0)

    if hasattr(experiment, "flow"):
        sampler = experiment.flow
        name = "Flow matching endpoint residual"
        method = f"{sampler.config.eval_sampling_steps}-step Euler sampling"
    else:
        sampler = None
        name = "MLP"
        method = "direct prediction"

    with torch.inference_mode():
        if sampler is None:
            prediction = model(x)
        else:
            initial_noise = torch.zeros_like(x)
            prediction = sampler.sample(model, x, initial_noise=initial_noise)
        target = target_function(x)

    plot_x = x.squeeze(1).cpu().tolist()
    figure = go.Figure([
        go.Scatter(
            x=plot_x, y=target.squeeze(1).cpu().tolist(), name="Target function",
            mode="lines", line=dict(color="#64748b", width=3),
        ),
        go.Scatter(
            x=plot_x, y=prediction.squeeze(1).cpu().tolist(), name=name,
            mode="lines", line=dict(color="#2563eb", width=3),
        ),
    ])
    for x0, x1 in ((config.x_min - padding, config.x_min), (config.x_max, config.x_max + padding)):
        figure.add_vrect(
            x0=x0, x1=x1, fillcolor="#f59e0b", opacity=0.12, line_width=0, layer="below"
        )
    for boundary in (config.x_min, config.x_max):
        figure.add_vline(x=boundary, line_dash="dash", line_color="#f59e0b")
    figure.update_layout(
        title=f"Extrapolation · {padding_fraction:.0%} padding on each side · {method}",
        template="plotly_white", autosize=True, height=550,
        margin=dict(l=65, r=25, t=60, b=65), legend=dict(orientation="h", y=-0.15),
        xaxis=dict(title="x", range=[config.x_min - padding, config.x_max + padding]),
        yaxis=dict(title="f(x)"),
    )
    return figure


def show_extrapolation(experiment: Experiment, padding_fraction: float = 0.10) -> None:
    figure = make_extrapolation_figure(experiment, padding_fraction)
    payload = figure.to_plotly_json()
    payload["config"] = dict(responsive=True, displayModeBar=False)
    display({"application/vnd.plotly.v1+json": payload}, raw=True)
