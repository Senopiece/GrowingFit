// Browser-driven rendering: exactly one pull -> evaluation -> Plotly -> paint
// cycle at a time. Training never pushes intermediate frames to this view.

let plotlyPromise;

function getPlotly(source) {
  if (!plotlyPromise) {
    // Plotly's bundled UMD build can be loaded as a local ES module without
    // eval(), a CDN, or a frontend/backend Plotly version mismatch.
    const moduleSource =
      "const module = { exports: {} }; const exports = module.exports;\n" +
      source + "\nexport default module.exports;\n";
    const url = URL.createObjectURL(new Blob([moduleSource], { type: "text/javascript" }));
    plotlyPromise = import(url)
      .then((module) => module.default)
      .catch((error) => {
        plotlyPromise = undefined;
        throw error;
      })
      .finally(() => URL.revokeObjectURL(url));
  }
  return plotlyPromise;
}

export default {
  render({ model, el }) {
    const plot = document.createElement("div");
    plot.style.width = "100%";
    plot.style.minHeight = "800px";
    el.style.width = "100%";
    el.replaceChildren(plot);

    let stopped = false;
    let completed = false;
    let plotly;
    let requestId = 0;
    let lastRevision = -1;
    let pending = null;
    let delayTimer = null;
    let finishDelay = null;

    const onMessage = (message) => {
      if (message.type !== "frame" || !pending || message.request_id !== pending.id) return;
      const resolve = pending.resolve;
      pending = null;
      resolve(message);
    };
    model.on("msg:custom", onMessage);

    const pull = () => new Promise((resolve) => {
      const id = ++requestId;
      pending = { id, resolve };
      model.send({ type: "pull", request_id: id, last_revision: lastRevision });
    });

    const delay = (milliseconds) => new Promise((resolve) => {
      finishDelay = resolve;
      delayTimer = setTimeout(() => {
        delayTimer = null;
        finishDelay = null;
        resolve();
      }, milliseconds);
    });

    const draw = async (figure) => {
      await plotly.react(plot, figure.data, figure.layout, figure.config);
      await new Promise((resolve) => requestAnimationFrame(resolve));
    };

    const run = async () => {
      try {
        plotly = await getPlotly(model.get("plotly_source"));
        if (stopped) return;
        const saved = model.get("final_frame");
        if (saved && Object.keys(saved).length) {
          await draw(saved);
          completed = true;
          return;
        }
        while (!stopped) {
          // The frame budget includes fetching, Python evaluation and drawing.
          const startedAt = performance.now();
          const frame = await pull();
          if (stopped || !frame) break;
          if (frame.error) throw new Error(frame.error);
          if (frame.figure) await draw(frame.figure);
          if (stopped) break;
          lastRevision = frame.revision;
          model.send({ type: "painted", revision: lastRevision, done: frame.done });
          if (frame.done) {
            completed = true;
            break;
          }
          const interval = 1000 / model.get("target_fps");
          const remaining = interval - (performance.now() - startedAt);
          if (remaining > 0) await delay(remaining);
          // If drawing took longer than the budget, fetch the latest model
          // immediately. Never try to catch up on skipped intermediate states.
        }
      } catch (error) {
        if (!stopped) {
          const message = String(error?.message ?? error);
          plot.textContent = "Не удалось отрисовать график: " + message;
          model.send({ type: "failed", error: message });
        }
      }
    };
    void run();

    return () => {
      stopped = true;
      if (delayTimer !== null) clearTimeout(delayTimer);
      finishDelay?.();
      pending?.resolve(null);
      pending = null;
      model.off("msg:custom", onMessage);
      if (!completed) model.send({ type: "closed" });
      if (plotly) plotly.purge(plot);
    };
  },
};
