"""Два живых графика Plotly: лосс сверху, аппроксимация снизу."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .experiment import Experiment, Snapshot


def _make_figure(
    experiment: Experiment, snapshot: Snapshot, steps: list[int], losses: list[float]
) -> go.Figure:
    """Собирает целый кадр, без частичных обновлений отдельных кривых."""
    config = experiment.config
    total_steps = config.total_steps
    colors = ("#2563eb", "#0891b2", "#16a34a", "#d97706", "#9333ea", "#db2777", "#0d9488", "#475569")
    stage_index = next(i for i, (width, _) in enumerate(config.stages) if width == snapshot.width)
    fit_color = colors[stage_index % len(colors)]
    x = experiment.x_grid.squeeze(1).tolist()
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.35, 0.65],
        vertical_spacing=0.16,
        subplot_titles=(
            f"Лосс · MSE = {snapshot.train_mse:.6f}",
            f"Фит · всего {snapshot.width} · обучается {snapshot.trainable_width} · "
            f"заморожено {snapshot.frozen_width} · шаг {snapshot.step}/{total_steps}",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=list(steps), y=list(losses), name="MSE", showlegend=False,
            mode="lines", line=dict(color="#2563eb", width=2),
        ),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x, y=experiment.y_grid.squeeze(1).tolist(), name="Целевая функция",
            mode="lines", line=dict(color="#64748b", width=3),
        ),
        row=2, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=x, y=snapshot.prediction.reshape(-1).tolist(), name=f"MLP · {snapshot.width} нейронов",
            mode="lines", line=dict(color=fit_color, width=3),
        ),
        row=2, col=1,
    )
    boundary = 0
    for (width, n_steps), (next_width, _) in zip(config.stages, config.stages[1:]):
        boundary += n_steps
        figure.add_vline(
            x=boundary, line_dash="dash", line_color="#a78bfa",
            annotation_text=f"{width} → {next_width}", row=1, col=1,
        )
    figure.update_xaxes(title_text="Шаг", range=[0, total_steps], autorange=False, row=1, col=1)
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
    experiment: Experiment, *, target_fps: float = 10.0, step_delay: float | None = None
) -> None:
    """Фоновое обучение и browser-pull отрисовка только актуальной модели.

    step_delay для старых ячеек — интервал отрисовки, не пауза Adam.
    При step_delay=0 используется target_fps; обучение не тормозится таймером.
    """
    from .pull_render import run_live

    await run_live(experiment, target_fps=target_fps, step_delay=step_delay)
