# GrowingFit

Python 3.12 + uv. Three notebooks explore approximations of the same function:

- `notebooks/01_standard_mlp.ipynb` — a standard MLP trained with MSE.
- `notebooks/02_cfm.ipynb` — conditional flow matching with endpoint residual prediction.
- `notebooks/03_cfm_vs_standard.ipynb` — 100 runs per method at each hidden width
  (32, 64, and 128), for 600 networks total. Final grid MSE statistics and paired
  box plots with highlighted arithmetic means are reported separately for each width.
  Both methods share the width, update count, learning rate, data, and seed IDs.
  Completed runs are saved to `notebooks/outputs/cfm_vs_standard.csv` after each model.
  Rerunning resumes missing width/method/seed combinations. Earlier single-width
  results are preserved using their saved configuration to identify the width.

```powershell
uv sync
uv run jupyter lab notebooks/01_standard_mlp.ipynb
```

In VS Code, select the `.venv\Scripts\python.exe` kernel. After changing shared code,
restart the kernel and run all cells.

Both models have one hidden ReLU layer. All parameters are trained jointly.
The shared `Config` specifies `hidden_size`, `total_steps`, `learning_rate`, `seed`,
training and evaluation grid sizes, and domain bounds. The target function in
`notebooks/_shared/experiment.py` is:
`sin(1.7x) + 0.35 sin(5.3x) + 0.15x² − 0.3x`.

The standard MLP receives `x` and predicts `f(x)` directly. CFM receives `[x, y_t]`
and predicts the residual `f(x) − y_t`, where `y_t=(1−t)z+t*f(x)`, `z ~ N(0,1)`.
CFM minimizes residual MSE. The network has no separate time input:
for a deterministic target, `(x, y_t)` uniquely determines the residual.

The `FlowMatchingConfig` settings are independent:

- `eval_sampling_steps` — the number of Euler steps for evaluation and extrapolation.
- `train_sampling_steps=None` — continuous time `t ~ Uniform(0, 1)`.
- `train_sampling_steps=T` — time sampled from `{0, 1/T, …, (T−1)/T}`.
  With `T=1`, training only sees `t=0`.

Sampling reconstructs velocity as `residual/(1−t)`.
The sampler never accesses the target; the final network call occurs before `t=1`.
CFM plots and shared extrapolation start at `y=0` for every `x`.
Gaussian noise remains random during training.

Plotly shows MSE above the function approximation, using the full available width.
Training runs in a separate thread. The renderer copies the latest model,
evaluates it, and prepares a frame only when the browser requests one.
The next request is sent after drawing finishes; `target_fps` limits
the request frequency. Intermediate states are skipped, with no frame queue.
The final model is displayed when training finishes.

Frame requests use direct comm messages to avoid waiting for kernel `idle`
during `ipywidgets` state synchronization. The latest response is stored in
widget state. Plotly.js comes from the installed package, without a CDN.
`experiment.history` contains only the evaluation snapshots prepared by the renderer.

Shared code lives in `notebooks/_shared/`: `experiment.py`,
`conditional_flow_matching.py`, `flow_matching_experiment.py`,
`visualization.py`, `pull_render.py`, `render_worker.py`, `pull_plot.js`
and `extrapolation.py`. `show_extrapolation(experiment, padding_fraction=0.10)`
plots the fit with additional padding on both sides of the training domain.
