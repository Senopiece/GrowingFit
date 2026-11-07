"""Mountain and erosion definitions copied from smooth_chaos/noise.ipynb."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numba as nb
import torch
import torch.nn as nn


class Saw(nn.Module):
    def forward(self, x):
        return (1 / (torch.abs(2 * torch.remainder(x, 1.0) - 1.0) + 1.0)) - 0.5


class MountainGen(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.noise = nn.Sequential(
            nn.Linear(2, hidden),
            Saw(),
            # nn.Linear(hidden, 1, bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.5)
            if m.bias is not None:
                nn.init.normal_(m.bias, 0.0, 0.5)

    def forward(self, coords):
        z = self.noise(coords).mean(dim=1, keepdim=True)
        return z  # / (1 + torch.norm(coords, dim=-1, keepdim=True) ** 2)


@nb.njit
def sample_height_and_gradient(heightmap, x_pos, y_pos):
    x0 = int(x_pos)
    y0 = int(y_pos)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = x_pos - x0
    sy = y_pos - y0

    h00 = heightmap[y0, x0]
    h10 = heightmap[y0, x1]
    h01 = heightmap[y1, x0]
    h11 = heightmap[y1, x1]

    h0 = h00 * (1.0 - sx) + h10 * sx
    h1 = h01 * (1.0 - sx) + h11 * sx
    h = h0 * (1.0 - sy) + h1 * sy

    grad_x = (h10 - h00) * (1.0 - sy) + (h11 - h01) * sy
    grad_y = (h01 - h00) * (1.0 - sx) + (h11 - h10) * sx
    return h, grad_x, grad_y


@nb.njit
def deposit_bilinear(heightmap, x_pos, y_pos, amount):
    x0 = int(x_pos)
    y0 = int(y_pos)
    x1 = x0 + 1
    y1 = y0 + 1

    sx = x_pos - x0
    sy = y_pos - y0

    w00 = (1.0 - sx) * (1.0 - sy)
    w10 = sx * (1.0 - sy)
    w01 = (1.0 - sx) * sy
    w11 = sx * sy

    heightmap[y0, x0] += amount * w00
    heightmap[y0, x1] += amount * w10
    heightmap[y1, x0] += amount * w01
    heightmap[y1, x1] += amount * w11


@nb.njit
def erode_with_brush(heightmap, x_pos, y_pos, amount, radius):
    center_x = int(x_pos)
    center_y = int(y_pos)
    h = heightmap.shape[0]
    w = heightmap.shape[1]

    total_weight = 0.0
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            xi = center_x + ox
            yi = center_y + oy
            if xi <= 0 or xi >= w - 1 or yi <= 0 or yi >= h - 1:
                continue
            dist = (ox * ox + oy * oy) ** 0.5
            if dist > radius:
                continue
            total_weight += radius + 1.0 - dist

    if total_weight <= 0.0:
        return 0.0

    removed = 0.0
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            xi = center_x + ox
            yi = center_y + oy
            if xi <= 0 or xi >= w - 1 or yi <= 0 or yi >= h - 1:
                continue
            dist = (ox * ox + oy * oy) ** 0.5
            if dist > radius:
                continue

            weight = radius + 1.0 - dist
            share = amount * (weight / total_weight)

            local_floor = 1e18
            for ny in range(yi - 1, yi + 2):
                for nx in range(xi - 1, xi + 2):
                    if ny == yi and nx == xi:
                        continue
                    v = heightmap[ny, nx]
                    if v < local_floor:
                        local_floor = v

            max_remove = heightmap[yi, xi] - local_floor
            if max_remove < 0.0:
                max_remove = 0.0

            delta = share
            if delta > max_remove:
                delta = max_remove

            heightmap[yi, xi] -= delta
            removed += delta

    return removed


@nb.njit
def diffuse_step(heightmap, alpha):
    src = heightmap.copy()
    h = heightmap.shape[0]
    w = heightmap.shape[1]

    for yi in range(1, h - 1):
        for xi in range(1, w - 1):
            lap = (
                src[yi - 1, xi]
                + src[yi + 1, xi]
                + src[yi, xi - 1]
                + src[yi, xi + 1]
                - 4.0 * src[yi, xi]
            )
            heightmap[yi, xi] = src[yi, xi] + alpha * lap


@nb.njit
def hydraulic_erosion_numba(
    orig_heightmap,
    start_x,
    start_y,
    max_steps,
    inertia,
    capacity,
    min_slope,
    deposition,
    erosion,
    evaporation,
    gravity,
    max_deposit_step,
    max_erode_step,
    brush_radius,
    diffuse_every,
    diffuse_alpha,
):
    hm = orig_heightmap.copy()
    h = hm.shape[0]
    w = hm.shape[1]
    iterations = start_x.shape[0]

    for drop_idx in range(iterations):
        x_pos = start_x[drop_idx]
        y_pos = start_y[drop_idx]

        dir_x = 0.0
        dir_y = 0.0
        # speed = 1.0
        water = capacity
        sediment = 0.0

        for _ in range(max_steps):
            if x_pos < 1.0 or x_pos >= w - 2.0 or y_pos < 1.0 or y_pos >= h - 2.0:
                break

            h_now, grad_x, grad_y = sample_height_and_gradient(
                orig_heightmap, x_pos, y_pos
            )

            dir_x = dir_x * inertia - grad_x * (1.0 - inertia)
            dir_y = dir_y * inertia - grad_y * (1.0 - inertia)

            norm = (dir_x * dir_x + dir_y * dir_y) ** 0.5
            if norm < 1e-12:
                break

            dir_x /= norm
            dir_y /= norm

            new_x = x_pos + dir_x
            new_y = y_pos + dir_y
            if new_x < 1.0 or new_x >= w - 2.0 or new_y < 1.0 or new_y >= h - 2.0:
                break

            h_new, _, _ = sample_height_and_gradient(orig_heightmap, new_x, new_y)
            delta_h = h_new - h_now

            # seems like slope does not do much
            slope = -delta_h
            if slope < min_slope:
                slope = min_slope
            sediment_capacity = water * slope  # slope * speed * water

            if delta_h > 0.0 or sediment > sediment_capacity:
                # target = (
                #     delta_h
                #     if delta_h > 0.0
                #     else (sediment - sediment_capacity) * deposition
                # )
                # deposit_amount = sediment
                # if target < deposit_amount:
                #     deposit_amount = target
                # if max_deposit_step < deposit_amount:
                #     deposit_amount = max_deposit_step

                # if deposit_amount > 0.0:
                #     deposit_bilinear(hm, x_pos, y_pos, deposit_amount)
                #     sediment -= deposit_amount
                pass
            else:
                # erode_amount = (sediment_capacity - sediment) * erosion
                erode_amount = sediment_capacity * erosion
                # downhill_limit = -delta_h
                # if downhill_limit < erode_amount:
                #     erode_amount = downhill_limit
                if max_erode_step < erode_amount:
                    erode_amount = max_erode_step

                if erode_amount > 0.0:
                    lifted = erode_with_brush(
                        hm, x_pos, y_pos, erode_amount, brush_radius
                    )
                    # sediment += lifted

            # speed_sq = speed * speed + (-delta_h) * gravity
            # if speed_sq < 0.05:
            #     speed_sq = 0.05
            # speed = speed_sq**0.5
            # if speed > 3.0:
            #     speed = 3.0

            water *= 1.0 - evaporation
            x_pos = new_x
            y_pos = new_y

            if water < 0.02:
                break

        # if diffuse_every > 0 and ((drop_idx + 1) % diffuse_every == 0):
        #     diffuse_step(hm, diffuse_alpha)

    return hm


def hydraulic_erosion(
    heightmap,
    iterations=128 * 128,
    max_steps=8,
    erosion=0.2,
    max_erode_step=0.02,
    capacity=1.6,
    evaporation=0.01,
    brush_radius=1,
    min_slope=0.01,
    inertia=0.0,  # removed
    gravity=4.0,  # removed
    deposition=0.0,  # removed
    max_deposit_step=0.0,  # removed
    diffuse_every=6000,  # removed
    diffuse_alpha=0.01,  # removed
    seed=None,
):
    rng = np.random.default_rng(seed)
    h, w = heightmap.shape
    start_x = rng.uniform(1.0, w - 2.0, size=iterations).astype(np.float64)
    start_y = rng.uniform(1.0, h - 2.0, size=iterations).astype(np.float64)

    hm64 = np.asarray(heightmap, dtype=np.float64)
    eroded = hydraulic_erosion_numba(
        hm64,
        start_x,
        start_y,
        max_steps,
        inertia,
        capacity,
        min_slope,
        deposition,
        erosion,
        evaporation,
        gravity,
        max_deposit_step,
        max_erode_step,
        int(brush_radius),
        int(diffuse_every),
        diffuse_alpha,
    )
    return eroded.astype(heightmap.dtype, copy=False)


@dataclass(frozen=True)
class TerrainConfig:
    source: str = "generated"
    resolution: int = 128
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    mountain_seed: int = 2003
    mountain_hidden_size: int = 64
    erosion_seed: int | None = None
    erosion_droplets: int = 128 * 128
    erosion_max_steps: int = 8
    erosion: float = 0.2
    max_erode_step: float = 0.02
    capacity: float = 1.6
    evaporation: float = 0.01
    brush_radius: int = 1
    min_slope: float = 0.01
    inertia: float = 0.0
    gravity: float = 4.0
    deposition: float = 0.0
    max_deposit_step: float = 0.0
    diffuse_every: int = 6000
    diffuse_alpha: float = 0.01

    def __post_init__(self) -> None:
        if self.source not in ("generated", "reference"):
            raise ValueError("source must be 'generated' or 'reference'.")
        if self.resolution < 8 or self.mountain_hidden_size < 1:
            raise ValueError("resolution must be at least 8; mountain_hidden_size must be positive.")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Both coordinate domains must have positive extent.")
        if self.erosion_droplets < 0 or self.erosion_max_steps < 1 or self.brush_radius < 1:
            raise ValueError("Invalid droplet count, erosion step count, or brush radius.")
        if not 0 <= self.evaporation < 1 or not 0 <= self.inertia <= 1:
            raise ValueError("evaporation must be in [0, 1); inertia must be in [0, 1].")
        if any(not np.isfinite(value) or value < 0 for value in (
            self.erosion, self.max_erode_step, self.capacity, self.min_slope,
        )):
            raise ValueError("Erosion magnitudes must be finite and nonnegative.")


@dataclass
class MountainTerrain:
    config: TerrainConfig
    x: np.ndarray
    y: np.ndarray
    base_height: np.ndarray
    eroded_height: np.ndarray


def load_reference_terrain(config: TerrainConfig | None = None) -> MountainTerrain:
    """Load the hydraulic heightmap saved in noise.ipynb and its recovered input."""
    if config is None:
        config = TerrainConfig(source="reference")
    path = Path(__file__).with_name("data") / "noise_hydraulic_terrain.npz"
    with np.load(path, allow_pickle=False) as data:
        x, y = data["x"].copy(), data["y"].copy()
        base, eroded = data["base_height"].copy(), data["eroded_height"].copy()
    if (config.x_min, config.x_max, config.y_min, config.y_max) != (x[0], x[-1], y[0], y[-1]):
        raise ValueError("The reference terrain domain is [-1, 1] x [-1, 1].")
    if config.resolution != len(x):
        def resize(height):
            tensor = torch.from_numpy(height)[None, None]
            return torch.nn.functional.interpolate(
                tensor, size=(config.resolution, config.resolution), mode="bilinear", align_corners=True,
            )[0, 0].numpy().copy()
        base, eroded = resize(base), resize(eroded)
        x = np.linspace(config.x_min, config.x_max, config.resolution)
        y = np.linspace(config.y_min, config.y_max, config.resolution)
    return MountainTerrain(config, x, y, base, eroded)


def generate_terrain(config: TerrainConfig = TerrainConfig()) -> MountainTerrain:
    """Sample the raw mountain field and apply hydraulic_erosion with its settings."""
    if config.source == "reference":
        return load_reference_terrain(config)
    x = np.linspace(config.x_min, config.x_max, config.resolution)
    y = np.linspace(config.y_min, config.y_max, config.resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(config.mountain_seed)
        model = MountainGen(config.mountain_hidden_size).eval()
        inputs = torch.from_numpy(grid).float()
        height = model(inputs).detach().cpu().numpy()[:, 0].reshape(config.resolution, config.resolution)
    eroded = hydraulic_erosion(
        height,
        iterations=config.erosion_droplets,
        max_steps=config.erosion_max_steps,
        erosion=config.erosion,
        max_erode_step=config.max_erode_step,
        capacity=config.capacity,
        evaporation=config.evaporation,
        brush_radius=config.brush_radius,
        min_slope=config.min_slope,
        inertia=config.inertia,
        gravity=config.gravity,
        deposition=config.deposition,
        max_deposit_step=config.max_deposit_step,
        diffuse_every=config.diffuse_every,
        diffuse_alpha=config.diffuse_alpha,
        seed=config.erosion_seed,
    )
    return MountainTerrain(config, x, y, height, eroded)
