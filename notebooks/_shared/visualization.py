"""Two live Plotly charts: loss above, function approximation below."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .experiment import Experiment, Snapshot


def _make_figure(experiment: Experiment, snapshot: Snapshot) -> go.Figure:
    """Build a complete frame without partial updates to individual curves."""
    config = experiment.config
    total_steps = config.total_steps
    x = experiment.x_grid.squeeze(1).tolist()
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.35, 0.65],
        vertical_spacing=0.16,
        subplot_titles=(
            f"{experiment.evaluation_label} = {snapshot.train_mse:.6f}",
            f"{experiment.prediction_name} · {config.hidden_size} neurons · "
            f"step {snapshot.step}/{total_steps}",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=[item.step for item in experiment.history],
            y=[max(item.train_mse, 1e-15) for item in experiment.history],
            name="MSE", showlegend=False,
            mode="lines", line=dict(color="#2563eb", width=2),
        ),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x, y=experiment.y_grid.squeeze(1).tolist(), name="Target function",
            mode="lines", line=dict(color="#64748b", width=3),
        ),
        row=2, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x, y=snapshot.prediction.reshape(-1).tolist(), name=experiment.prediction_name,
            mode="lines", line=dict(color="#2563eb", width=3),
        ),
        row=2, col=1,
    )
    figure.update_xaxes(title_text="Step", range=[0, total_steps], autorange=False, row=1, col=1)
    figure.update_yaxes(title_text="MSE", type="log", row=1, col=1)
    figure.update_xaxes(
        title_text="x", type="linear", range=[config.x_min, config.x_max],
        autorange=False, row=2, col=1,
    )
    figure.update_yaxes(title_text="f(x)", type="linear", row=2, col=1)
    figure.update_layout(
        template="plotly_white", autosize=True, height=800,
        margin=dict(l=65, r=25, t=55, b=65),
        legend=dict(orientation="h", x=0, y=-0.08),
        font=dict(size=14),
    )
    return figure


async def train_live(
    experiment: Experiment, *, target_fps: float = 10.0
) -> None:
    """Train in the background and render the latest model with an FPS limit."""
    from .pull_render import run_live

    await run_live(experiment, target_fps=target_fps)
