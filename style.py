"""Stage 1c: regional character. Deliberate departures from the eroded surface.

Each one is a known exaggeration that still reads natural:
flat valley floors (room for roads and towns), stepped mesas in the desert,
a young cone for the volcano, then finer grain when the grid is refined.
"""
import numpy as np
from numba import njit
from scipy import ndimage
from erode import fill, route
from noise import fbm


@njit(cache=True)
def _stream_base(z, rcv, order, area, a_min):
    """For every cell: height and size of the stream it drains into."""
    n = z.size
    base = z.copy()
    sa = area.copy()
    for q in range(n):
        i = order[q]
        r = rcv[i]
        if r != i and area[i] < a_min:
            base[i] = base[r]
            sa[i] = sa[r]
    return base, sa


def flow(z, sea, dx, seed=0):
    zf = fill(z, sea, 1e-3)
    jit = np.random.default_rng(seed + 3).uniform(0, 0.6, z.shape + (8,))
    rcv, dist, order, area = route(zf, sea, dx, jit)
    return zf, rcv, order, area


def finish(lay, z, log=print):
    f, dx, n = lay.f, lay.dx, lay.n
    rng = np.random.default_rng(lay.seed + 5)
    sea = ~f["land"]
    X, Y = lay.grid()

    # 1. valley floors: flatten ground that sits just above a real stream
    zf, rcv, order, area = flow(z, sea, dx)
    base, sa = _stream_base(zf.ravel(), rcv, order, area, 2.5e5)
    base, sa = base.reshape(n, n), sa.reshape(n, n)
    hand = np.maximum(zf - base, 0)
    h0 = np.clip(5 + 9 * np.log10(np.maximum(sa, 2.5e5) / 2.5e5), 5, 26)
    z = base + hand - 0.8 * h0 * (1 - np.exp(-hand / h0))
    z = np.where(sea, 0.0, z)

    # 2. volcano: a young cone sits on top of the eroded pedestal
    v = lay.pts["volcano"]
    r = np.hypot(X - v[0], Y - v[1])
    th = np.arctan2(Y - v[1], X - v[0])
    R, Hc, rc = 1450.0, 900.0, 110.0
    q = np.clip(1 - r / R, 0, 1)
    gul = sum(np.cos(k * th + rng.uniform(0, 6.3)) * a for k, a in [(7, 1.0), (13, 0.7), (23, 0.5)])
    cone = Hc * q ** 1.9 + 14 * gul * q * (1 - q) * 4 * (r > rc * 1.5)
    rim = Hc * (1 - rc / R) ** 1.9
    crater = rim - 70 * (1 - np.clip(r / rc, 0, 1) ** 2)
    cone = np.where(r < rc, crater, cone)
    k = 25.0
    z = np.where(f["land"] | (r < R * 0.9), np.maximum(z, cone) + 0 * k, z)
    land = f["land"] | ((r < R * 0.9) & (cone > 1.0))
    lake = f["lake"] & ~(r < R)
    z[lake] = lay.lake_z

    # 3. refine the grid x2 and add grain where the ground is steep
    N = n * 2
    up = lambda a, o=1: ndimage.zoom(a.astype(np.float64), 2, order=o)
    Z = up(z, 3)
    land2 = up(land) > 0.5
    lake2 = up(lake) > 0.5
    gy, gx = np.gradient(Z, dx / 2)
    sl = np.hypot(gx, gy)
    grain = fbm(N, 40, 5, rng, gain=0.6)
    Z = Z + np.clip(sl / 0.7, 0, 1) * 7 * grain * (1 + 1.5 * up(f["alpine"]))

    # desert: flat-topped buttes with cliff edges stand on the low floor
    wd = np.clip((up(f["desert"]) - 0.30) / 0.25, 0, 1)
    bn = fbm(N, 15, 4, rng)
    edge = np.clip((bn - 0.05) / 0.03, 0, 1)
    edge2 = np.clip((bn - 0.17) / 0.025, 0, 1)
    butte = (60 + 30 * fbm(N, 3, 2, rng)) * edge + 55 * edge2
    Z = Z + wd * (butte - 0.6 * np.clip(Z - 60, 0, None) * (1 - edge))
    W_butte = wd * edge
    crag = 1 - np.abs(fbm(N, 22, 4, rng, gain=0.55)) * 2.2                  # sharp-crested noise
    Z = Z + up(f["alpine"]) * np.clip((Z - 380) / 450, 0, 1) * 55 * (crag - 0.45)
    Z = np.where(land2, np.maximum(Z, 0.3), -6.0)
    Z[lake2] = lay.lake_z
    shore = ndimage.binary_dilation(lake2, iterations=2) & ~lake2
    Z[shore] = np.maximum(Z[shore], lay.lake_z + 0.4)

    W = dict(z=Z, land=land2, lake=lake2, dx=dx / 2, n=N, lake_z=lay.lake_z, pts=lay.pts,
             w=lay.w, nn=lay.nn, seed=lay.seed, butte=W_butte)
    for kf in ("alpine", "east", "ash", "desert", "basin"):
        W[kf] = up(f[kf])
    zf, rcv, order, area = flow(Z, ~land2, W["dx"])
    W["area"] = area.reshape(N, N)
    gy, gx = np.gradient(Z, W["dx"])
    W["slope"] = np.hypot(gx, gy)
    log(f"  finished surface {N}x{N} at {W['dx']:.0f} m, peak {Z.max():.0f} m")
    return W
