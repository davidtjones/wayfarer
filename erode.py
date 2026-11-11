"""Stage 1b: land-forming. Turns the design fields into land that reads natural.

Rivers cut down in proportion to the water they carry, slopes fail above a
stable angle, soil creeps. Run long enough, this gives branching valleys and
ridges whose heights follow the designed uplift. The lake and sea are pinned,
so the designed water stays where the brief put it.
"""
import heapq
import numpy as np
from numba import njit

DI = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
DJ = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
DD = np.array([2 ** 0.5, 1, 2 ** 0.5, 1, 1, 2 ** 0.5, 1, 2 ** 0.5])


@njit(cache=True)
def fill(z, base, eps):
    """Priority flood: raise pits so every land cell drains to a base cell."""
    n, m = z.shape
    out = z.copy()
    closed = np.zeros((n, m), np.bool_)
    heap = [(0.0, 0)]
    heap.pop()
    for i in range(n):
        for j in range(m):
            if base[i, j] or i == 0 or j == 0 or i == n - 1 or j == m - 1:
                closed[i, j] = True
                heapq.heappush(heap, (out[i, j], i * m + j))
    while len(heap) > 0:
        zc, idx = heapq.heappop(heap)
        i, j = idx // m, idx % m
        for k in range(8):
            a, b = i + DI[k], j + DJ[k]
            if a < 0 or b < 0 or a >= n or b >= m or closed[a, b]:
                continue
            closed[a, b] = True
            if out[a, b] < zc + eps * DD[k]:
                out[a, b] = zc + eps * DD[k]
            heapq.heappush(heap, (out[a, b], a * m + b))
    return out


@njit(cache=True)
def route(z, base, dx, jit):
    """Steepest-descent receiver per cell, upstream->downstream order, drainage area."""
    n, m = z.shape
    rcv = np.arange(n * m)
    dist = np.ones(n * m) * dx
    for i in range(n):
        for j in range(m):
            if base[i, j]:
                continue
            best = 0.0
            for k in range(8):
                a, b = i + DI[k], j + DJ[k]
                if a < 0 or b < 0 or a >= n or b >= m:
                    continue
                s = (z[i, j] - z[a, b]) / DD[k]
                if s > 0:
                    s *= 1 + jit[i, j, k]    # breaks the grid's 8-direction bias
                if s > best:
                    best = s
                    rcv[i * m + j] = a * m + b
                    dist[i * m + j] = DD[k] * dx
    order = np.argsort(z.ravel())            # low to high: receivers come first
    area = np.ones(n * m) * dx * dx
    for q in range(n * m - 1, -1, -1):
        i = order[q]
        if rcv[i] != i:
            area[rcv[i]] += area[i]
    return rcv, dist, order, area


@njit(cache=True)
def _step(z, base, pinned, U, K, soil, talus, jit, dx, dt, mexp, sweeps, sw, samp, period):
    n, m = z.shape
    for i in range(n):
        for j in range(m):
            if not base[i, j] and not pinned[i, j]:
                z[i, j] += U[i, j] * dt
    z[:] = fill(z, base, 1e-3)
    rcv, dist, order, area = route(z, base, dx, jit)
    zf = z.ravel()
    Kf = K.ravel()
    pf = pinned.ravel()
    swf = sw.ravel()
    sampf = samp.ravel()
    for q in range(n * m):
        i = order[q]
        r = rcv[i]
        if r == i or pf[i]:
            continue
        ch = min(area[i] / 3000.0, 1.0)          # below this, slopes not channels
        hard = 0.5 + 0.5 * np.sin(6.2832 * (zf[i] + swf[i]) / period)
        hard = hard * hard * sampf[i]                     # 0 soft bed .. 1 hard bed
        F = Kf[i] * (1 - 0.7 * hard) * dt * ch * ch * area[i] ** mexp / dist[i]
        zf[i] = (zf[i] + F * zf[r]) / (1 + F)
    # slopes above the stable angle shed material downhill
    for _ in range(sweeps):
        d = np.zeros((n, m))
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                if base[i, j] or pinned[i, j]:
                    continue
                hard = 0.5 + 0.5 * np.sin(6.2832 * (z[i, j] + sw[i, j]) / period)
                tl = talus[i, j] * (1 + 1.6 * hard * hard * samp[i, j])
                for k in range(8):
                    a, b = i + DI[k], j + DJ[k]
                    ex = z[i, j] - z[a, b] - tl * DD[k] * dx
                    if ex > 0:
                        d[i, j] -= 0.08 * ex
                        if not base[a, b] and not pinned[a, b]:
                            d[a, b] += 0.08 * ex
        z += d
    # soil creep rounds the gentle country, leaves the hard country sharp
    s = z.copy()
    for i in range(1, n - 1):
        for j in range(1, m - 1):
            if base[i, j] or pinned[i, j]:
                continue
            acc = 0.0
            for k in range(8):
                acc += z[i + DI[k], j + DJ[k]]
            s[i, j] = z[i, j] + soil[i, j] * (acc / 8 - z[i, j])
    z[:] = s
    return area.reshape(n, m)


def form(lay, steps=140, dt=1.0e4, mexp=0.4, log=None):
    f = lay.f
    base = ~f["land"]
    pinned = f["lake"].copy()
    z = np.where(base, 0.0, f["z0"]).astype(np.float64)
    jit = np.random.default_rng(lay.seed + 77).uniform(0, 0.6, z.shape + (8,))
    pin_z = f.get("pin_z", lay.lake_z)            # one level, or a level per cell
    pz = pin_z[pinned] if isinstance(pin_z, np.ndarray) else pin_z
    z[pinned] = pz
    area = None
    zero = np.zeros_like(z)
    sw, samp = f.get("strata_warp", zero), f.get("strata_amp", zero)   # optional layered rock
    period = float(f.get("strata_period", 60.0))
    for s in range(steps):
        area = _step(z, base, pinned, f["U"], f["K"], f["soil"], f["talus"], jit,
                     lay.dx, dt, mexp, 3, sw, samp, period)
        z[pinned] = pz
        z[base] = 0.0
        if log and s % 20 == 0:
            log(f"  step {s:3d}  max {z.max():6.0f} m  mean land {z[f['land']].mean():5.0f} m")
    return z, area
