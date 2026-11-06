// Request the latest model through custom comm messages, without widget state
// throttling (which waits for kernel idle during an awaited training cell).
// Only one evaluation and one drawing are outstanding; skip intermediate states.
let plotlyPromise;

function getPlotly(source) {
  if (!plotlyPromise) {
    const moduleSource = "const module = { exports: {} }; const exports = module.exports;\n" +
      source + "\nexport default module.exports;\n";
    const url = URL.createObjectURL(new Blob([moduleSource], { type: "text/javascript" }));
    plotlyPromise = import(url).then((module) => module.default).catch((error) => {
      plotlyPromise = undefined;
      throw error;
    }).finally(() => URL.revokeObjectURL(url));
  }
  return plotlyPromise;
}

function withTimeout(promise, milliseconds, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), milliseconds);
    Promise.resolve(promise).then(
      (value) => { clearTimeout(timer); resolve(value); },
      (error) => { clearTimeout(timer); reject(error); },
    );
  });
}

function waitForPaint() {
  return new Promise((resolve) => {
    let animation;
    let timer;
    const finish = () => {
      clearTimeout(timer);
      cancelAnimationFrame(animation);
      resolve();
    };
    timer = setTimeout(finish, 100);
    animation = requestAnimationFrame(finish);
  });
}

export default {
  render({ model, el }) {
    let plot = document.createElement("div");
    plot.style.width = "100%";
    plot.style.minHeight = "800px";
    el.style.width = "100%";
    el.style.position = "relative";
    el.replaceChildren(plot);

    let plotly;
    let stopped = false;
    let completed = false;
    let running = false;
    let notice = null;
    let requestId = 0;
    let lastRevision = -1;
    let pending = null;
    let latestFinal = null;
    let staging = null;
    let delayTimer;
    let finishDelay;
    let lastDrawMs = 100;
    let droppedDraws = 0;
    const clientId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

    const settle = (frame) => {
      if (!pending) return;
      clearTimeout(pending.timer);
      const { resolve, startedAt, attempt } = pending;
      pending = null;
      resolve(frame ? {
        ...frame, request_ms: performance.now() - startedAt, retries: Math.max(0, attempt - 1),
      } : null);
    };
    const acceptFrame = (frame) => {
      if (frame?.type !== "frame" || !pending || frame.client_id !== clientId ||
          frame.request_id !== pending.id) return;
      settle(frame);
    };
    const onResponse = () => acceptFrame(model.get("response"));
    const onFinal = () => {
      const figure = model.get("final_frame");
      if (!figure || !Object.keys(figure).length) return;
      latestFinal = { figure, revision: model.get("final_revision") ?? -1, done: true };
      if (pending) settle(latestFinal);
      else if (!running && !stopped && !completed) void run();
    };
    model.on("change:response", onResponse);
    model.on("change:final_frame", onFinal);
    // Also accept a directly delivered response from an older widget.
    model.on("msg:custom", acceptFrame);

    const pull = () => new Promise((resolve, reject) => {
      const id = ++requestId;
      const resend = () => {
        if (stopped || !pending || pending.id !== id) return;
        // Recover an already received state even if its change event was missed.
        onResponse();
        if (!pending) return;
        if (performance.now() - pending.startedAt >= 10000) {
          pending = null;
          reject(new Error("No renderer response within 10 seconds; see live_widget.render_stats for details."));
          return;
        }
        pending.attempt++;
        const request = {
          client_id: clientId, request_id: id, last_revision: lastRevision, attempt: pending.attempt,
        };
        const retryDelay = Math.max(250, Math.min(1000, 3000 / model.get("target_fps")));
        pending.timer = setTimeout(resend, retryDelay);
        // save_changes() buffers subsequent updates until its previous comm
        // gets an idle status. An awaited live-training cell can stay busy
        // throughout the run, so every pull must use the unthrottled channel.
        model.send({ type: "pull", ...request });
      };
      pending = { id, resolve, reject, startedAt: performance.now(), attempt: 0, timer: null, resend };
      resend();
    });

    const delay = (milliseconds) => new Promise((resolve) => {
      finishDelay = resolve;
      delayTimer = setTimeout(() => { finishDelay = null; resolve(); }, milliseconds);
    });

    const discardPlot = (node) => {
      if (!node) return;
      node.remove();
      if (plotly) {
        try { plotly.purge(node); } catch { /* detached/unfinished Plotly instance */ }
      }
    };

    const draw = async (figure) => {
      const startedAt = performance.now();
      const candidate = document.createElement("div");
      candidate.style.cssText = "position:absolute;left:0;top:0;width:100%;visibility:hidden;pointer-events:none";
      staging = candidate;
      el.appendChild(candidate);
      try {
        // Plotly mutates data/layout. Keep synchronized transport state immutable.
        const copy = structuredClone(figure);
        // Build in a fresh div; a stalled old Plotly.react cannot block the
        // next graph. Keep the visible graph until the replacement is ready.
        const budget = Math.max(1000, Math.min(5000, 8 * lastDrawMs));
        await withTimeout(
          plotly.newPlot(candidate, copy.data, copy.layout, copy.config),
          budget, "Plotly frame timed out; requesting the latest model.",
        );
        if (stopped) { discardPlot(candidate); return null; }
        candidate.style.position = "";
        candidate.style.left = "";
        candidate.style.top = "";
        candidate.style.visibility = "";
        candidate.style.pointerEvents = "";
        const previous = plot;
        plot = candidate;
        staging = null;
        discardPlot(previous);
        await waitForPaint();
        lastDrawMs = performance.now() - startedAt;
        droppedDraws = 0;
        notice?.remove();
        notice = null;
        return lastDrawMs;
      } catch (error) {
        if (staging === candidate) staging = null;
        discardPlot(candidate);
        model.send({
          type: "render_retry", stage: "plotly", revision: lastRevision,
          elapsed_ms: performance.now() - startedAt, error: String(error?.message ?? error),
        });
        if (++droppedDraws >= 3) throw error;
        return null;
      }
    };

    const onVisibility = () => {
      if (!document.hidden && pending) {
        clearTimeout(pending.timer);
        pending.resend();
      } else if (!document.hidden && !running && !completed && !stopped) {
        void run();
      }
    };
    document.addEventListener("visibilitychange", onVisibility);

    const run = async () => {
      if (running || stopped || completed) return;
      running = true;
      let stage = "load";
      try {
        plotly = await withTimeout(
          getPlotly(model.get("plotly_source")), 30000, "Could not load Plotly within 30 seconds.",
        );
        if (stopped) return;
        onFinal();
        while (!stopped) {
          const startedAt = performance.now();
          stage = "request";
          const frame = latestFinal ?? await pull();
          latestFinal = null;
          if (stopped || !frame) break;
          if (frame.error) throw new Error(frame.error);
          stage = "draw";
          const drawMs = frame.figure ? await draw(frame.figure) : 0;
          if (stopped) break;
          if (drawMs === null) continue; // Discard delayed state, fetch latest.
          lastRevision = frame.revision;
          model.send({
            type: "painted", client_id: clientId, revision: lastRevision, step: frame.step ?? null,
            done: frame.done, request_ms: frame.request_ms ?? 0, draw_ms: drawMs,
            total_ms: performance.now() - startedAt, retries: frame.retries ?? 0,
            server: frame.timings ?? {},
          });
          if (frame.done) { completed = true; break; }
          stage = "fps_wait";
          const remaining = 1000 / model.get("target_fps") - (performance.now() - startedAt);
          if (remaining > 0) await delay(remaining);
        }
      } catch (error) {
        if (!stopped) {
          const message = String(error?.message ?? error);
          notice?.remove();
          notice = document.createElement("div");
          notice.textContent = "Plot stopped: " + message;
          el.appendChild(notice);
          model.send({ type: "failed", client_id: clientId, stage, revision: lastRevision, error: message });
        }
      } finally {
        running = false;
      }
    };
    void run();

    return () => {
      stopped = true;
      clearTimeout(delayTimer);
      finishDelay?.();
      settle(null);
      model.off("change:response", onResponse);
      model.off("change:final_frame", onFinal);
      model.off("msg:custom", acceptFrame);
      document.removeEventListener("visibilitychange", onVisibility);
      if (!completed) model.send({ type: "detached", client_id: clientId });
      discardPlot(staging);
      discardPlot(plot);
    };
  },
};
