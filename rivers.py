"""Rivers: where the finished land already sends its water, cut a bed and lay a
water surface in it.

1. Course. Route flow over the finished surface. A river is every line that
   gathers enough ground, plus each lake's outlet down to the sea and each
   lake's biggest feeder, so a lake always has water coming in and going out.
2. Level. The water level follows the ground along the course and never rises
   downstream. It starts from a lake's surface where it leaves one.
3. Bed. The course is smoothed into a line, and the ground is shaped by distance
   from that line: a shallow bed under the water, a bank rising from it. Low
   ground beside the river is lifted to a small levee so the water has a shore
   on both sides. Where the land drops faster than a river can run, the bed is
   cut back into a gorge.

Returns the new ground, the river cells, and the water level over a ribbon a
little wider than the river (-1000 elsewhere). The game draws the ribbon; the
ground hides it wherever it is not river.
"""
import numpy as np
from numba import njit
from scipy import ndimage
from style import flow

DRY = -1000.0


@njit(cache=True)
def _course(rcv, order, area, land, lake, a_min, a_feed):
    n = rcv.size
    don = np.full(n, -1)                       # each cell's biggest donor
    for i in range(n):
        r = rcv[i]
        if r != i and (don[r] < 0 or area[i] > area[don[r]]):
            don[r] = i
    chan = (area >= a_min) & land
    feed = np.full(n, -1)                      # biggest inflow per lake cell's lake is found by the caller
    for i in range(n):
        r = rcv[i]
        if r == i:
            continue
        if lake[i] and not lake[r]:            # outlet: follow it to the sea
            j = i
            while rcv[j] != j and land[j]:
                chan[j] = True
                j = rcv[j]
        if lake[r] and not lake[i]:
            feed[i] = r
    return don, chan, feed


@njit(cache=True)
def _upstream(start, don, area, chan, lake, a_feed):
    j = start
    while j >= 0 and area[j] >= a_feed:
        chan[j] = True
        j = don[j]


@njit(cache=True)
def _profile(rcv, order, don, chan, lake, L, px, py, wd, iters):
    n = rcv.size
    for q in range(n - 1, -1, -1):             # high to low: never rise downstream
        i = order[q]
        r = rcv[i]
        if chan[i] and r != i and not lake[r] and L[r] > L[i]:
            L[r] = L[i]
    for _ in range(iters):
        L2, x2, y2, w2 = L.copy(), px.copy(), py.copy(), wd.copy()
        for i in range(n):
            if not chan[i] or lake[i]:
                continue
            r, d = rcv[i], don[i]
            if r == i or d < 0 or not chan[d] or not chan[r] or lake[d] or lake[r]:
                continue                        # heads, mouths and lake joins stay put
            L2[i] = 0.25 * L[d] + 0.5 * L[i] + 0.25 * L[r]
            x2[i] = 0.25 * px[d] + 0.5 * px[i] + 0.25 * px[r]
            y2[i] = 0.25 * py[d] + 0.5 * py[i] + 0.25 * py[r]
            w2[i] = 0.25 * wd[d] + 0.5 * wd[i] + 0.25 * wd[r]
        L[:], px[:], py[:], wd[:] = L2, x2, y2, w2
        for q in range(n - 1, -1, -1):
            i = order[q]
            r = rcv[i]
            if chan[i] and r != i and not lake[r] and L[r] > L[i]:
                L[r] = L[i]


@njit(cache=True)
def _grade(rcv, order, chan, lake, L, m, dx, smax):
    """No stretch falls faster than `smax`: above a drop the river has cut back
    into a gorge, as a real one does, instead of going over a cliff."""
    for q in range(rcv.size):                    # low to high: the cell below is settled
        i = order[q]
        r = rcv[i]
        if not chan[i] or lake[i] or r == i:
            continue
        step = dx if (abs(r - i) == 1 or abs(r - i) == m) else dx * 1.4142
        if L[i] > L[r] + smax * step:
            L[i] = L[r] + smax * step


@njit(cache=True)
def _run(rcv, order, chan, lake, dx):
    """Metres of river above each channel cell. A lake counts as a long river."""
    run = np.where(lake, 1e4, 0.0)
    for q in range(rcv.size - 1, -1, -1):
        i = order[q]
        r = rcv[i]
        if (chan[i] or lake[i]) and r != i and run[r] < run[i] + dx:
            run[r] = run[i] + dx
    return run


@njit(cache=True)
def _splat(rcv, chan, lake, L, px, py, wd, run, n, m, dx, reach):
    dist = np.full((n, m), 1e9)
    lev = np.full((n, m), DRY)
    wid = np.zeros((n, m))
    src = np.zeros((n, m))
    k = int(reach / dx) + 2
    for i in range(rcv.size):
        r = rcv[i]
        if not chan[i] or r == i or (lake[i] and lake[r]):
            continue
        ax, ay, bx, by = px[i], py[i], px[r], py[r]
        ex, ey = bx - ax, by - ay
        ee = ex * ex + ey * ey + 1e-9
        ci, cj = int(ay / dx), int(ax / dx)
        for a in range(max(ci - k, 0), min(ci + k + 1, n)):
            for b in range(max(cj - k, 0), min(cj + k + 1, m)):
                x, y = (b + 0.5) * dx, (a + 0.5) * dx
                t = ((x - ax) * ex + (y - ay) * ey) / ee
                t = min(max(t, 0.0), 1.0)
                d = np.hypot(x - ax - t * ex, y - ay - t * ey)
                if d < dist[a, b]:
                    dist[a, b] = d
                    lev[a, b] = L[i] + t * (L[r] - L[i])
                    wid[a, b] = wd[i] + t * (wd[r] - wd[i])
                    src[a, b] = run[i] + t * (run[r] - run[i])
    return dist, lev, wid, src


def _ss(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def carve(Z, land, lake, lake_z, dx, seed, a_min=6e5, a_feed=1.2e5, log=print):
    n, m = Z.shape
    zf, rcv, order, area = flow(Z, ~land, dx, seed)
    landf, lakef = land.ravel(), lake.ravel()
    don, chan, feed = _course(rcv, order, area, landf, lakef, a_min, a_feed)
    # each lake's biggest feeder, followed up to where it is only a trickle
    lab, k = ndimage.label(lake)
    labf = lab.ravel()
    src = np.nonzero(feed >= 0)[0]
    for q in range(1, k + 1):
        s = src[labf[feed[src]] == q]
        if len(s):
            _upstream(int(s[np.argmax(area[s])]), don, area, chan, lakef, a_feed)
    chan &= landf
    ii, jj = np.divmod(np.arange(n * m), m)
    px, py = (jj + 0.5) * dx, (ii + 0.5) * dx
    L = np.where(lakef, lake_z.ravel(), zf.ravel()).astype(np.float64)
    L[~landf] = 0.0
    wd = np.clip(10 + 9 * np.sqrt(area / 1e6), 12, 30)
    chan_s = chan | ~landf                      # a mouth's sea cell ends the line
    _profile(rcv, order, don, chan_s, lakef, L, px, py, wd, 14)
    _grade(rcv, order, chan, lakef, L, m, dx, 0.18)
    reach = 15 + 60.0
    run = _run(rcv, order, chan, lakef, dx)
    dist, lev, wid, src = _splat(rcv, chan, lakef, L, px, py, wd, run, n, m, dx, reach)
    half = wid / 2
    depth = np.clip(0.5 + 0.05 * wid, 1.0, 1.8)
    # A river does not begin full width. Near its head the same bed holds less
    # water, so the stream starts as a trickle in the bottom and widens.
    water = lev - 1.05 * depth * (1 - _ss(src, 0, 260)) ** 1.5
    out = dist - half                            # metres outside the water's edge
    u = np.clip(dist / np.maximum(half, 1), 0, 1)
    near = (dist < 1e8) & land & ~lake
    bed = lev - depth * (1 - u * u)
    bank = lev + 0.28 * np.maximum(out, 0)       # the cut bank rises 1 in 3.5
    cut = np.where(out < 0, bed, bank)
    low = np.where(out < 0, bed, lev + np.minimum(0.25 * np.maximum(out, 0), 0.7))
    Zn = np.maximum(np.minimum(Z, cut), low)
    fade = 1 - _ss(out, 14, reach - 16)
    Zn = Z + (Zn - Z) * fade
    Z2 = np.where(near, Zn, Z)
    river = near & (out < 0)
    ribbon = near & (out < 15)
    Z2[ribbon] = np.maximum(Z2[ribbon], low[ribbon])
    # At the mouth the river slips just under the sea's surface, so the two
    # waters meet without an edge. The bed there is below sea level already.
    sea_d = ndimage.distance_transform_edt(land) * dx
    water = water - 0.5 * (1 - _ss(sea_d, 0, 70))
    level = np.where(ribbon, water, DRY)
    river &= water > Z2 + 0.05                   # the dry bed above a head is not river
    km = chan.sum() * dx * 1.2 / 1000
    drop = np.abs(L[chan] - L[rcv[chan]]) / dx
    log(f"  rivers: {km:.1f} km, {k} lakes, steepest fall {drop.max() * 100:.0f}%, "
        f"{(drop > 0.25).mean() * 100:.0f}% of length steeper than 1 in 4")
    return Z2, river, level, dict(chan=chan.reshape(n, m), area=area.reshape(n, m), rcv=rcv, L=L.reshape(n, m))
