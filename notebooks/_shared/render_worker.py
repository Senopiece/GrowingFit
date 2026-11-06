"""Dedicated renderer thread: no dependence on asyncio's default executor."""

import asyncio
from dataclasses import dataclass
from threading import Condition, Thread
from time import perf_counter
from typing import Callable


@dataclass
class RenderJob:
    request: dict | None = None
    waiter: asyncio.Future | None = None
    started_at: float = 0.0


class RenderWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop, render: Callable, deliver: Callable) -> None:
        self.loop = loop
        self.render = render
        self.deliver = deliver
        self.condition = Condition()
        self.active: RenderJob | None = None
        self.request: RenderJob | None = None
        self.final: RenderJob | None = None
        self.stopped = False
        self.thread = Thread(target=self._run, name="growingfit-renderer", daemon=True)
        self.thread.start()

    def submit_request(self, request: dict) -> bool:
        with self.condition:
            if self.stopped or self.active is not None or self.request is not None:
                return False
            self.request = RenderJob(request=dict(request))
            self.condition.notify()
            return True

    def submit_final(self, waiter: asyncio.Future) -> None:
        with self.condition:
            if self.stopped:
                raise RuntimeError("Renderer is closed.")
            self.final = RenderJob(waiter=waiter)
            self.condition.notify()

    def describe(self) -> dict | None:
        with self.condition:
            job = self.active or self.request or self.final
            if job is None:
                return None
            return {
                "request": job.request,
                "final": job.waiter is not None,
                "elapsed_ms": (perf_counter() - job.started_at) * 1000 if job.started_at else 0,
            }

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def _run(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or (
                    self.active is None and (self.final is not None or self.request is not None)
                ))
                if self.stopped:
                    return
                if self.final is not None:
                    job, self.final = self.final, None
                else:
                    job, self.request = self.request, None
                self.active = job
                job.started_at = perf_counter()
            frame, error = None, None
            try:
                last_revision = int(job.request.get("last_revision", -1)) if job.request else -1
                frame = self.render(last_revision)
            except Exception as caught:
                error = caught
            # Thread-safe dispatch wakes the notebook's loop even when it is
            # otherwise idle awaiting completion of the training worker.
            try:
                self.loop.call_soon_threadsafe(self._complete, job, frame, error)
            except RuntimeError:
                return  # Kernel/event loop is shutting down.

    def _complete(self, job: RenderJob, frame: dict | None, error: Exception | None) -> None:
        try:
            self.deliver(job, frame, error)
        finally:
            with self.condition:
                self.active = None
                self.condition.notify_all()
