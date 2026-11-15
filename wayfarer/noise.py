"""Small seeded value-noise helpers."""
import numpy as np
from scipy import ndimage


def value_noise(n, cells, rng):
    """Smooth noise in [-1, 1] on an n x n grid with `cells` lattice cells across."""
    g = rng.uniform(-1, 1, (cells + 3, cells + 3))
    z = ndimage.zoom(g, n / cells, order=3)
    o = int(n / cells)
    return z[o:o + n, o:o + n]


def fbm(n, cells, octaves, rng, gain=0.5):
    out = np.zeros((n, n))
    amp, tot = 1.0, 0.0
    for _ in range(octaves):
        out += amp * value_noise(n, cells, rng)
        tot += amp
        amp *= gain
        cells *= 2
        if cells > n // 2:
            break
    return out / tot


def smoothstep(x, a, b):
    """0 below a, 1 above b, a smooth S between."""
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)
