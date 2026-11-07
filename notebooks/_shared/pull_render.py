"""Background training and recoverable, browser-driven rendering of latest state."""

import asyncio
from collections import deque
from copy import deepcopy
from functools import lru_cache
from math import isfinite
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from time import perf_counter

import anywidget
from IPython.display import display
from plotly.offline import get_plotlyjs
import traitlets

from .experiment import Experiment
from .render_worker import RenderJob, RenderWorker


@lru_cache(maxsize=1)
def _plotly_source() -> str:
    return get_plotlyjs()


class _LiveSession:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.stop = Event()
        self.finished = Event()
        self.loop = asyncio.get_running_loop()
        self.training_finished = asyncio.Event()
        self.capture_condition = Condition()
        self.snapshot_requested = False
        self.render_lock = Lock()
        self.error: Exception | None = None
        self.started = False
        self.last_figure: dict | None = None
        self.last_figure_revision = -1
        self.last_figure_step = 0
        self.render_stage = "idle"
        self.thread = Thread(target=self._train, name="growingfit-training", daemon=True)

    def start(self) -> None:
        if not self.started:
            self.started = True
            self.thread.start()

    def request_stop(self) -> None:
        self.stop.set()
        with self.capture_condition:
            self.capture_condition.notify_all()

    def _train(self) -> None:
        try:
            for _ in self.experiment.run():
                if self.stop.is_set():
                    break
                # Check the predicate and wait under one Condition, avoiding
                # the lost-notification risks of separately reset Events.
                with self.capture_condition:
                    while self.snapshot_requested and not self.stop.is_set():
                        self.capture_condition.wait(timeout=0.25)
        except Exception as error:
            self.error = error
        finally:
            with self.experiment.model_lock:
                self.finished.set()
            self.loop.call_soon_threadsafe(self.training_finished.set)

    def _capture(self, last_revision: int):
        experiment = self.experiment
        with self.capture_condition:
            self.snapshot_requested = True
        try:
            with experiment.model_lock:
                revision = experiment.revision
                done = self.finished.is_set()
                if self.error is not None:
                    raise self.error
                if revision == last_revision or revision == self.last_figure_revision:
                    return revision, done, None, None
                model = deepcopy(experiment.model)
                progress = experiment.progress
            return revision, done, model, progress
        finally:
            with self.capture_condition:
                self.snapshot_requested = False
                self.capture_condition.notify_all()

    def render_current(self, last_revision: int) -> dict:
        # A final-frame request and a browser request can overlap, but evaluation
        # and history updates remain serialized, with no queue of training states.
        with self.render_lock:
            started = perf_counter()
            self.render_stage = "snapshot"
            revision, done, model, progress = self._capture(last_revision)
            captured = perf_counter()
            if revision == last_revision:
                self.render_stage = "idle"
                return {"revision": revision, "step": self.last_figure_step, "done": done, "figure": None}
            if model is None:
                self.render_stage = "idle"
                return {"revision": revision, "step": self.last_figure_step, "done": done, "figure": self.last_figure}
            experiment = self.experiment
            self.render_stage = "evaluation"
            snapshot = experiment.evaluate_model(model, progress)
            evaluated = perf_counter()
            experiment.history.append(snapshot)
            self.render_stage = "figure"
            figure = experiment.make_figure(snapshot)
            figure.update_layout(datarevision=revision, uirevision="growingfit-live")
            payload = figure.to_plotly_json()
            # Scatter styles are explicit; do not resend defaults for dozens
            # of unrelated trace types in Plotly's template on every frame.
            template = payload.get("layout", {}).get("template")
            if isinstance(template, dict):
                template.pop("data", None)
            payload["config"] = {"responsive": True, "displayModeBar": False}
            self.last_figure = payload
            self.last_figure_revision = revision
            self.last_figure_step = progress.step
            self.render_stage = "idle"
            return {
                "revision": revision, "step": progress.step, "done": done, "figure": payload,
                "timings": {
                    "snapshot_ms": (captured - started) * 1000,
                    "evaluation_ms": (evaluated - captured) * 1000,
                    "figure_ms": (perf_counter() - evaluated) * 1000,
                },
            }


class _PullPlot(anywidget.AnyWidget):
    _esm = Path(__file__).with_name("pull_plot.js")
    plotly_source = traitlets.Unicode().tag(sync=True)
    target_fps = traitlets.Float(10.0).tag(sync=True)
    final_frame = traitlets.Dict().tag(sync=True)
    final_revision = traitlets.Int(-1).tag(sync=True)
    request = traitlets.Dict().tag(sync=True)
    response = traitlets.Dict().tag(sync=True)

    def __init__(self, session: _LiveSession, target_fps: float) -> None:
        super().__init__(plotly_source=_plotly_source(), target_fps=target_fps)
        self.session = session
        self.loop = asyncio.get_running_loop()
        self.first_request = asyncio.Event()
        self.render_finished = asyncio.Event()
        self.render_error: Exception | None = None
        self.last_response: dict | None = None
        self.delivery = 0
        self.render_stats = deque(maxlen=256)
        self.final_waiter: asyncio.Future | None = None
        self.worker = RenderWorker(self.loop, session.render_current, self._deliver_frame)
        self.observe(self._on_request, names="request")
        self.on_msg(self._on_message)

    @property
    def pending(self) -> dict | None:
        return self.worker.describe()

    def close(self) -> None:
        worker = getattr(self, "worker", None)
        if worker is not None:
            worker.stop()
        super().close()

    def _publish_response(self, response: dict) -> None:
        self.delivery += 1
        # A state update remains readable even if a view misses its event.
        # The delivery number also makes retransmission a new state update.
        self.response = {**response, "delivery": self.delivery}

    def _on_request(self, change: dict) -> None:
        if change["new"]:
            # Comm callbacks may arrive on a different kernel/subshell loop.
            # Marshal before touching asyncio Events or scheduling a response.
            self.loop.call_soon_threadsafe(self._handle_pull, dict(change["new"]), perf_counter())

    def _handle_pull(self, content: dict, received_at: float | None = None) -> None:
        self.first_request.set()
        client_id = content.get("client_id", "legacy")
        request_id = content.get("request_id")
        self.render_stats.append({
            "type": "server_request", "request_id": request_id, "attempt": content.get("attempt", 1),
            "dispatch_ms": (perf_counter() - received_at) * 1000 if received_at else 0,
            "stage": self.session.render_stage, "pending": self.pending,
        })
        if self.session.finished.is_set() and self.final_frame:
            response = {
                "type": "frame", "client_id": client_id, "request_id": request_id,
                "revision": self.final_revision, "step": self.session.last_figure_step,
                "done": True, "figure": self.final_frame,
            }
            self.last_response = response
            self._publish_response(response)
            return
        if self.last_response is not None and (
            self.last_response["client_id"], self.last_response["request_id"]
        ) == (client_id, request_id):
            self._publish_response(self.last_response)
            return
        if not self.session.finished.is_set():
            self.session.start()
        if not self.worker.submit_request(content):
            self._publish_response({"type": "pending", "client_id": client_id, "request_id": request_id})

    def publish_final(self, frame: dict) -> None:
        figure = frame.get("figure") or self.session.last_figure
        if figure is not None:
            with self.hold_sync():
                self.final_revision = frame["revision"]
                self.final_frame = figure

    def _on_message(self, widget, content: dict, buffers) -> None:
        self.loop.call_soon_threadsafe(self._handle_message, dict(content), perf_counter())

    def _handle_message(self, content: dict, received_at: float) -> None:
        kind = content.get("type")
        if kind in ("pull", "resync"):
            self._handle_pull(content, received_at)
        elif kind == "painted":
            self.render_stats.append(dict(content))
            self.render_error = None
            if content.get("done"):
                self.render_finished.set()
        elif kind == "render_retry":
            self.render_stats.append(dict(content))
        elif kind == "failed":
            self.render_stats.append({
                **content, "server_stage": self.session.render_stage, "server_pending": self.pending,
                "dispatch_ms": (perf_counter() - received_at) * 1000,
            })
            self.render_error = RuntimeError(content.get("error", "Plot rendering failed."))
            self.first_request.set()
            self.render_finished.set()
        # A view can be disposed/recreated by VS Code without user cancellation.
        # 'detached'/'closed' are not signals to stop the training worker.

    async def prepare_final(self) -> dict:
        if self.final_waiter is None:
            self.final_waiter = self.loop.create_future()
            self.worker.submit_final(self.final_waiter)
        return await asyncio.shield(self.final_waiter)

    def _deliver_frame(self, job: RenderJob, frame: dict | None, error: Exception | None) -> None:
        """Runs on the widget's event loop, woken by call_soon_threadsafe."""
        self.render_stats.append({
            "type": "server_delivery", "request": job.request, "final": job.waiter is not None,
            "elapsed_ms": (perf_counter() - job.started_at) * 1000,
            "stage": self.session.render_stage, "error": str(error) if error else None,
        })
        if job.waiter is not None:
            if error is not None:
                if not job.waiter.done():
                    job.waiter.set_exception(error)
            else:
                self.publish_final(frame)
                if not job.waiter.done():
                    job.waiter.set_result(frame)
            return
        content = job.request
        response = {
            "type": "frame", "client_id": content.get("client_id", "legacy"),
            "request_id": content.get("request_id"),
        }
        if error is None:
            if frame["done"]:
                self.publish_final(frame)
            response.update(frame)
        else:
            self.render_error = error
            response["error"] = str(error)
            self.render_finished.set()
        self.last_response = response
        self._publish_response(response)


async def run_live(
    experiment: Experiment, *, target_fps: float = 10.0
) -> None:
    if not isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive.")
    if experiment.started:
        raise RuntimeError("Create a new Experiment to run training again.")

    session = _LiveSession(experiment)
    widget = _PullPlot(session, target_fps)
    experiment.live_widget = widget
    display(widget)
    try:
        try:
            await asyncio.wait_for(widget.first_request.wait(), timeout=60)
        except TimeoutError as error:
            raise RuntimeError("The plot could not connect to the kernel: anywidget support is required.") from error
        if not session.started and widget.render_error is not None:
            raise widget.render_error
        await session.training_finished.wait()
        if session.error is not None:
            raise session.error
        # Prepare/persist the final state independently of browser liveness.
        await widget.prepare_final()
        widget.worker.stop()
        try:
            await asyncio.wait_for(widget.render_finished.wait(), timeout=3)
        except TimeoutError:
            pass
        if widget.render_error is not None:
            raise widget.render_error
    finally:
        session.request_stop()
        if session.started:
            await session.training_finished.wait()
        # Keep the comm handler: a recreated/late view can still pull the final
        # model. A missing paint acknowledgement never stalls the cell forever.


async def show_current(experiment: Experiment, *, target_fps: float = 10.0) -> None:
    """Redraw the existing model without starting/restarting optimization."""
    session = _LiveSession(experiment)
    session.finished.set()
    session.training_finished.set()
    widget = _PullPlot(session, target_fps)
    await widget.prepare_final()
    widget.worker.stop()
    experiment.live_widget = widget
    display(widget)
