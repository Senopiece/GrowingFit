"""One training worker, on-demand evaluation worker, one browser frame in flight."""

import asyncio
from copy import deepcopy
from functools import lru_cache
from math import isfinite
from pathlib import Path
from threading import Event, Thread

import anywidget
from IPython.display import display
from plotly.offline import get_plotlyjs
import traitlets

from .experiment import Experiment
from .visualization import _make_figure


@lru_cache(maxsize=1)
def _plotly_source() -> str:
    # Use the JS shipped with the installed Plotly version; no CDN dependency.
    return get_plotlyjs()


class _LiveSession:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.stop = Event()
        self.finished = Event()
        self.snapshot_requested = Event()
        self.snapshot_copied = Event()
        self.error: Exception | None = None
        self.started = False
        self.last_figure: dict | None = None
        self.thread = Thread(target=self._train, name="growingfit-training", daemon=True)

    def start(self) -> None:
        if not self.started:
            self.started = True
            self.thread.start()

    def _train(self) -> None:
        try:
            # No rendering, grid evaluation, frame messages or timer here.
            for _ in self.experiment.run():
                if self.stop.is_set():
                    break
                # Give an awaiting snapshot priority between iterations. This
                # pause ends after the short model copy, before evaluation.
                if self.snapshot_requested.is_set():
                    self.snapshot_copied.wait()
        except Exception as error:
            self.error = error
        finally:
            with self.experiment.model_lock:
                self.finished.set()

    def _capture(self, last_revision: int):
        experiment = self.experiment
        self.snapshot_copied.clear()
        self.snapshot_requested.set()
        try:
            with experiment.model_lock:
                revision = experiment.revision
                done = self.finished.is_set()
                if self.error is not None:
                    raise self.error
                if revision == last_revision:
                    return revision, done, None, None, None
                model = deepcopy(experiment.model)
                progress = experiment.progress
                metadata = deepcopy(experiment.render_metadata())
            return revision, done, model, progress, metadata
        finally:
            self.snapshot_requested.clear()
            self.snapshot_copied.set()

    def render_current(self, last_revision: int) -> dict:
        """Called only in the evaluation worker, never in the training thread."""
        experiment = self.experiment
        revision, done, model, progress, metadata = self._capture(last_revision)
        if model is None:
            return {"revision": revision, "done": done, "figure": None}

        snapshot = experiment.evaluate_model(model, progress)
        experiment.history.append(snapshot)
        custom_renderer = getattr(experiment, "make_render_figure", None)
        if custom_renderer is None:
            figure = _make_figure(
                experiment, snapshot,
                [item.step for item in experiment.history],
                [max(item.train_mse, 1e-15) for item in experiment.history],
            )
        else:
            figure = custom_renderer(model, snapshot, metadata)
        figure.update_layout(datarevision=revision, uirevision="growingfit-live")
        payload = figure.to_plotly_json()
        payload["config"] = {"responsive": True, "displayModeBar": False}
        self.last_figure = payload
        return {"revision": revision, "done": done, "figure": payload}


class _PullPlot(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("pull_plot.js")
    plotly_source = traitlets.Unicode().tag(sync=True)
    target_fps = traitlets.Float(10.0).tag(sync=True)
    final_frame = traitlets.Dict().tag(sync=True)

    def __init__(self, session: _LiveSession, target_fps: float) -> None:
        super().__init__(plotly_source=_plotly_source(), target_fps=target_fps)
        self.session = session
        self.loop = asyncio.get_running_loop()
        self.first_request = asyncio.Event()
        self.render_finished = asyncio.Event()
        self.render_error: Exception | None = None
        self.pending: asyncio.Task | None = None
        self.on_msg(self._on_message)

    def _on_message(self, widget, content: dict, buffers) -> None:
        kind = content.get("type")
        if kind == "pull":
            self.first_request.set()
            if self.pending is not None and not self.pending.done():
                self.send({
                    "type": "frame", "request_id": content.get("request_id"),
                    "error": "A frame request is already in progress.",
                })
                return
            self.pending = self.loop.create_task(self._serve_frame(content))
        elif kind == "painted":
            if content.get("done"):
                self.render_finished.set()
            else:
                # Start only after the initial graph is actually drawn.
                self.session.start()
        elif kind in ("failed", "closed"):
            if kind == "failed":
                self.render_error = RuntimeError(content.get("error", "Plot rendering failed."))
            self.first_request.set()
            self.render_finished.set()

    async def _serve_frame(self, content: dict) -> None:
        try:
            frame = await asyncio.to_thread(
                self.session.render_current, int(content.get("last_revision", -1))
            )
            if frame["done"] and self.session.last_figure is not None:
                # Persist only the final frame, not every intermediate state.
                self.final_frame = self.session.last_figure
            self.send({"type": "frame", "request_id": content.get("request_id"), **frame})
        except Exception as error:
            self.render_error = error
            self.send({
                "type": "frame", "request_id": content.get("request_id"), "error": str(error),
            })
            self.render_finished.set()


async def run_live(
    experiment: Experiment, *, target_fps: float = 10.0, step_delay: float | None = None
) -> None:
    if step_delay is not None:
        if not isfinite(step_delay) or step_delay < 0:
            raise ValueError("step_delay must be finite and nonnegative.")
        if step_delay > 0:
            target_fps = 1.0 / step_delay
    if not isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive.")
    if experiment.started:
        raise RuntimeError("Для повторного запуска создайте новый Experiment.")

    session = _LiveSession(experiment)
    widget = _PullPlot(session, target_fps)
    experiment.live_widget = widget
    display(widget)
    try:
        try:
            await asyncio.wait_for(widget.first_request.wait(), timeout=60)
        except TimeoutError as error:
            raise RuntimeError("График не подключился к ядру: нужна поддержка anywidget.") from error
        # Finishing Adam alone is insufficient: wait for the browser to paint
        # the final version. There is no queue of earlier frames to drain.
        await widget.render_finished.wait()
        if session.error is not None:
            raise session.error
        if widget.render_error is not None:
            raise widget.render_error
    finally:
        session.stop.set()
        session.snapshot_copied.set()
        if session.started:
            await asyncio.to_thread(session.thread.join)
        if widget.pending is not None:
            await asyncio.gather(widget.pending, return_exceptions=True)
        widget.on_msg(widget._on_message, remove=True)
