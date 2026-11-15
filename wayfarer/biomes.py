"""Biomes: what lives where, read from `biomes.toml`.

Two separate questions decide whether something grows at a spot:

- Which room is this? The room's theme gives a table of layers (grass, reeds,
  bushes, rocks, trees) and how much of each it has.
- What is this spot like? Wet or dry, shaded, steep, under trees, how far above
  the water. These mean the same in every room; each layer in the table says
  how it responds to them.

Rooms do not end at a line. `theme_weights` gives every cell a share of each
theme, blended across room borders over a distance the table sets: short along
a wall, long through a door. Every layer is worked out per theme and mixed by
those shares, so grass thins out into a desert rather than stopping.

The table's format is documented at the top of `biomes.toml`.
"""
import os
import tomllib
import numpy as np
from scipy import ndimage

THEMES = ["field", "lake", "forest", "desert", "ash", "moor"]
# Conditions a layer may respond to with a [none, full] pair, on the world grid.
GRID_CONDITIONS = ("wet", "shade", "ridge", "high", "canopy", "edge", "slope")


def load(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "biomes.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)


def ramp(x, pair):
    """0 at pair[0], 1 at pair[1], straight between; works either way round."""
    lo, hi = float(pair[0]), float(pair[1])
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def theme_weights(W, PL, table):
    """Share of each theme at every cell: dict theme -> field, summing to 1."""
    n, dx = W["n"], W["dx"]
    themes = W["rooms"]["theme"]
    label = W["label"]
    # Sea and anything unlabelled takes the nearest room.
    if (label < 0).any():
        _, (ii, jj) = ndimage.distance_transform_edt(label < 0, return_indices=True)
        label = label[ii, jj]
    cfg = table.get("blend", {})
    wall, door = float(cfg.get("wall", 50)), float(cfg.get("door", 220))
    # A border with no high ground on it is open country: the rooms run into each other.
    open_ = float(cfg.get("open", 400))
    kinds = W["rooms"].get("kinds", {})
    reach = float(cfg.get("door_reach", 140))
    pairs = cfg.get("pairs", {})

    # Blend width at every border cell, then spread to the cells around it.
    width = np.zeros((n, n))
    border = np.zeros((n, n), bool)
    doors = np.array(PL["sites"]["door"]) if len(PL["sites"]["door"]) else np.zeros((0, 2))
    for axis in (0, 1):
        a = label
        b = np.roll(label, -1, axis)
        diff = a != b
        if axis == 0:
            diff[-1, :] = False
        else:
            diff[:, -1] = False
        ii, jj = np.nonzero(diff)
        if len(ii) == 0:
            continue
        w = np.full(len(ii), wall)
        if len(doors):
            x, y = (jj + 0.5) * dx, (ii + 0.5) * dx
            d = np.min(np.hypot(x[:, None] - doors[None, :, 0], y[:, None] - doors[None, :, 1]), 1)
            w = np.where(d < reach, door, w)
        for k in np.unique(a[ii, jj] * 1000 + b[ii, jj]):
            ra, rb = sorted((int(k) // 1000, int(k) % 1000))
            if kinds.get((ra, rb), "range") not in ("range", "scarp"):
                sel = (a[ii, jj] * 1000 + b[ii, jj]) == k
                w[sel] = np.maximum(w[sel], open_)
            ta, tb = sorted((themes[int(k) // 1000], themes[int(k) % 1000]))
            if f"{ta}-{tb}" in pairs:
                sel = (a[ii, jj] * 1000 + b[ii, jj]) == k
                # A pair's width is for the open stretch at a door; a wall stays a wall.
                w[sel] = np.where(w[sel] == door, float(pairs[f"{ta}-{tb}"]), w[sel])
        width[ii, jj] = np.maximum(width[ii, jj], w)
        border[ii, jj] = True
    if border.any():
        _, (bi, bj) = ndimage.distance_transform_edt(~border, return_indices=True)
        width = ndimage.gaussian_filter(width[bi, bj], 30.0 / dx)
    else:
        width[:] = wall

    # Blur each theme's rooms at a few widths and pick between them per cell.
    levels = np.array([12.0, 25.0, 50.0, 100.0, 200.0, 400.0])
    pos = np.clip(np.interp(np.log(np.maximum(width, levels[0])), np.log(levels), np.arange(len(levels))), 0, len(levels) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(levels) - 2)
    t = pos - lo
    theme_of = np.array([THEMES.index(th) for th in themes])[label]
    out = {}
    for k, name in enumerate(THEMES):
        ind = (theme_of == k).astype(float)
        if not ind.any():
            out[name] = np.zeros((n, n))
            continue
        # A blend width covers about 2.5 standard deviations of the blur.
        stack = np.stack([ndimage.gaussian_filter(ind, lv / 2.5 / dx) for lv in levels])
        out[name] = np.take_along_axis(stack, lo[None], 0)[0] * (1 - t) + np.take_along_axis(stack, lo[None] + 1, 0)[0] * t
    total = sum(out.values())
    return {k: v / np.maximum(total, 1e-6) for k, v in out.items()}


def mix(weights, table, layer, key, default=0.0):
    """A number from each theme's layer, mixed by theme share."""
    return sum(w * float(table.get(t, {}).get(layer, {}).get(key, default)) for t, w in weights.items())


def density(weights, table, layer, cond, key="density"):
    """Amount of a layer at every cell: each theme's amount times its responses
    to the grid conditions in `cond`, mixed by theme share."""
    out = 0.0
    for t, w in weights.items():
        spec = table.get(t, {}).get(layer)
        if not spec or float(spec.get(key, 0.0)) <= 0.0:
            continue
        d = float(spec[key]) * w
        for c in GRID_CONDITIONS:
            if c in spec and c in cond:
                d = d * ramp(cond[c], spec[c])
        out = out + d
    return out if isinstance(out, np.ndarray) else np.zeros_like(next(iter(weights.values())))


def water_rule(weights, table, layer, key_lake, key_sea, by_lake, default, amount="density"):
    """A [low, high] pair in metres above water at every cell, mixed between the
    themes that have the layer. `by_lake` says whether a cell's nearest water is
    a lake (else the sea). Returns (low, high) fields."""
    lo, hi, tot = 0.0, 0.0, 0.0
    for t, w in weights.items():
        spec = table.get(t, {}).get(layer, {})
        share = w * max(float(spec.get(amount, 0.0)), 0.02)
        pl = spec.get(key_lake, default)
        ps = spec.get(key_sea, pl) if key_sea else pl
        lo = lo + share * np.where(by_lake, pl[0], ps[0])
        hi = hi + share * np.where(by_lake, pl[1], ps[1])
        tot = tot + share
    return lo / tot, hi / tot
