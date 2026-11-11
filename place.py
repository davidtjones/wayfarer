"""Stage 1d: place the journey on the finished land, by what the player can see.

Nothing here edits terrain to force a view. The big shapes were designed so the
views should exist; this step finds the exact route, overlook and landmark sites
on the real surface, then measures whether the brief was met.
"""
import heapq
import numpy as np
from numba import njit, prange
from scipy import ndimage

EYE = 1.7


@njit(cache=True)
def _bil(H, inv, x, y):
    n = H.shape[0]
    fx = min(max(x * inv - 0.5, 0.0), n - 1.001)
    fy = min(max(y * inv - 0.5, 0.0), n - 1.001)
    i, j = int(fy), int(fx)
    a, b = fx - j, fy - i
    return (H[i, j] * (1 - a) + H[i, j + 1] * a) * (1 - b) + (H[i + 1, j] * (1 - a) + H[i + 1, j + 1] * a) * b


@njit(cache=True)
def sees(H, dx, ex, ey, ez, tx, ty, tz):
    d = np.sqrt((tx - ex) ** 2 + (ty - ey) ** 2)
    k = int(d / (dx * 0.6)) + 1
    inv = 1.0 / dx
    for s in range(1, k):
        u = s / k
        if _bil(H, inv, ex + (tx - ex) * u, ey + (ty - ey) * u) > ez + (tz - ez) * u:
            return False
    return True


@njit(parallel=True, cache=True)
def sees_many(H, dx, eye, T):
    out = np.zeros(T.shape[0], np.bool_)
    for i in prange(T.shape[0]):
        out[i] = sees(H, dx, eye[0], eye[1], eye[2], T[i, 0], T[i, 1], T[i, 2])
    return out


@njit(cache=True)
def _dijkstra(z, block, extra, dx, si, sj, ti, tj):
    n, m = z.shape
    dist = np.full(n * m, 1e18)
    prev = np.full(n * m, -1)
    DI = (-1, -1, -1, 0, 0, 1, 1, 1)
    DJ = (-1, 0, 1, -1, 1, -1, 0, 1)
    heap = [(0.0, si * m + sj)]
    dist[si * m + sj] = 0
    while len(heap) > 0:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == ti * m + tj:
            break
        i, j = u // m, u % m
        for k in range(8):
            a, b = i + DI[k], j + DJ[k]
            if a < 0 or b < 0 or a >= n or b >= m or block[a, b]:
                continue
            L = dx * np.sqrt(DI[k] ** 2 + DJ[k] ** 2)
            g = abs(z[a, b] - z[i, j]) / L
            c = L * (1 + (g / 0.14) ** 2) + extra[a, b]
            if d + c < dist[a * m + b]:
                dist[a * m + b] = d + c
                prev[a * m + b] = u
                heapq.heappush(heap, (d + c, a * m + b))
    path = [ti * m + tj]
    while path[-1] != si * m + sj and prev[path[-1]] >= 0:
        path.append(prev[path[-1]])
    return np.array(path[::-1])


def find_route(W, a, b, step=20.0):
    """Easiest walking line from a to b: gentle grades, no swimming."""
    s = 2
    z = W["z"][::s, ::s]
    dx = W["dx"] * s
    block = (~W["land"] | W["lake"])[::s, ::s]
    river = (W["area"][::s, ::s] > 6e5) & ~block
    extra = np.where(river, 120.0, 0.0)
    ij = lambda p: (int(p[1] / dx), int(p[0] / dx))
    block = block.copy()
    for p in (a, b):
        block[ij(p)] = False
    path = _dijkstra(z, block, extra, dx, *ij(a), *ij(b))
    m = z.shape[1]
    xy = np.stack([(path % m + 0.5) * dx, (path // m + 0.5) * dx], 1)
    for _ in range(3):                                   # relax the staircase
        xy[1:-1] = 0.25 * xy[:-2] + 0.5 * xy[1:-1] + 0.25 * xy[2:]
    seg = np.hypot(*np.diff(xy, axis=0).T)
    d = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, d[-1], step)
    return np.stack([np.interp(t, d, xy[:, 0]), np.interp(t, d, xy[:, 1])], 1)


def ground(W, p):
    return _bil(W["z"], 1.0 / W["dx"], p[0], p[1])


def eye(W, p, h=EYE):
    return np.array([p[0], p[1], ground(W, p) + h])


def targets(W, mask, k=900, lift=0.5, seed=0):
    ii, jj = np.nonzero(mask)
    sel = np.random.default_rng(seed).choice(len(ii), min(k, len(ii)), replace=False)
    dx = W["dx"]
    x, y = (jj[sel] + 0.5) * dx, (ii[sel] + 0.5) * dx
    return np.stack([x, y, W["z"][ii[sel], jj[sel]] + lift], 1)


def frac_seen(W, p, T):
    return float(sees_many(W["z"], W["dx"], eye(W, p), T).mean())


def can_see(W, p, q, top):
    e = eye(W, p)
    return bool(sees(W["z"], W["dx"], e[0], e[1], e[2], q[0], q[1], ground(W, q) + top))


def sees_cone(W, p, v):
    """True if any point of the crater rim shows: the cone reads as a cone."""
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        if can_see(W, p, v + 105 * np.array([np.cos(a), np.sin(a)]), 6):
            return True
    return False


def bearing(p, q):
    return float(np.degrees(np.arctan2(q[0] - p[0], q[1] - p[1])) % 360)


def snap(W, p, radius, score):
    """Best cell within radius of p under a score array."""
    dx, n = W["dx"], W["n"]
    i0, j0, r = int(p[1] / dx), int(p[0] / dx), int(radius / dx)
    sl = (slice(max(i0 - r, 0), min(i0 + r + 1, n)), slice(max(j0 - r, 0), min(j0 + r + 1, n)))
    sub = score[sl]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    return np.array([(sl[1].start + k[1] + 0.5) * dx, (sl[0].start + k[0] + 0.5) * dx])


def place(W, log=print):
    P, dx, z = W["pts"], W["dx"], W["z"]
    walk = W["land"] & ~W["lake"]
    S = {}                                                # sites
    R = {}                                                # report
    prom = z - ndimage.uniform_filter(z, int(400 / dx))
    flat = np.where(walk & (W["slope"] < 0.25), 1.0, -1e9)

    # fixed anchors that follow water and summits
    S["horn"] = snap(W, P["horn"], 700, z)
    S["volcano"] = snap(W, P["volcano"], 200, z)
    S["island"] = snap(W, P["island"], 150, np.where(walk, z, -1e9))
    S["town"] = snap(W, P["lake_w"] + W["nn"] * 90, 220, flat - np.abs(z - W["lake_z"] - 3))
    lakeT = targets(W, W["lake"])
    rng = np.random.default_rng(W["seed"])
    valley = np.where(walk, np.minimum(W["area"], 3e5), -1e9)
    S["start"] = snap(W, P["start"], 110, valley)
    lip = snap(W, P["bend"], 110, valley)                 # where the walk-out valley meets the scarp
    S["oasis"] = snap(W, P["desert"], 400, flat - z * 0.002)

    route = np.concatenate([find_route(W, S["start"], lip), find_route(W, lip, S["town"])])
    prof = np.array([frac_seen(W, p, lakeT) for p in route])
    R["route_m"] = float(len(route) * 20)

    # the reveal: first place the lake opens up, then the best standing spot beside it
    hi = np.nonzero(prof > 0.30)[0]
    r0 = int(hi[0]) if len(hi) else int(np.argmax(prof))
    best, bp = -1, route[r0]
    for k in range(max(r0 - 3, 0), min(r0 + 14, len(route))):
        for _ in range(60):
            q = route[k] + rng.uniform(-260, 260, 2)
            i, j = int(q[1] / dx), int(q[0] / dx)
            if not walk[i, j] or W["slope"][i, j] > 0.45 or z[i, j] > ground(W, route[k]) + 45:
                continue                                  # a short side climb at most
            sc = frac_seen(W, q, lakeT) - 0.0008 * np.hypot(*(q - route[k])) \
                + 0.25 * can_see(W, q, S["horn"], 12) + 0.1 * can_see(W, q, S["island"], 30) \
                + 0.1 * can_see(W, q, S["town"], 20) - 0.4 * (W["area"][i, j] > 4e4) + 0.004 * min(prom[i, j], 40)
            if sc > best:
                best, bp = sc, q
    S["overlook"] = bp
    r1 = int(np.argmin(np.hypot(*(route - bp).T)))
    R["overlook_at_frac"] = r1 / len(route)
    R["lake_seen_before"] = float(prof[:max(r0 - 2, 1)].mean())
    R["lake_seen_first_15pct"] = float(prof[:max(int(0.15 * len(route)), 1)].max())
    R["lake_seen_at_overlook"] = frac_seen(W, bp, lakeT)
    R["view_bearing"] = bearing(bp, P["lake"])

    # keep: a knoll on the north shore, seen from the overlook, stacked under the horn from town
    Xc = (np.arange(W["n"]) + 0.5) * dx
    X, Y = np.meshgrid(Xc, Xc)
    dv = (X - P["lake"][0]) * W["nn"][0] + (Y - P["lake"][1]) * W["nn"][1]
    lake_d = ndimage.distance_transform_edt(~W["lake"]) * dx
    cand = walk & (dv > 0) & (lake_d > 120) & (prom > 6) & (W["slope"] < 0.5) & (z < W["lake_z"] + 170) \
        & (np.hypot(X - P["crag"][0], Y - P["crag"][1]) < 650)
    ii, jj = np.nonzero(cand)
    sel = rng.choice(len(ii), min(500, len(ii)), replace=False)
    hb = bearing(S["town"], S["horn"])
    best, kp = -1e9, P["crag"]
    for i, j in zip(ii[sel], jj[sel]):
        q = np.array([X[i, j], Y[i, j]])
        d = abs((bearing(S["town"], q) - hb + 180) % 360 - 180)
        sc = 2.0 * can_see(W, bp, q, 80) + 1.5 * can_see(W, S["town"], q, 80) \
            + 1.5 * np.exp(-(d / 8) ** 2) + prom[i, j] / 60
        if sc > best:
            best, kp = sc, q
    S["keep"] = kp

    # lighthouse: west coast headland, on the far side of town from the overlook
    sea_d = ndimage.distance_transform_edt(W["land"]) * dx
    cand = walk & (sea_d < 90) & (np.hypot(X - P["coast_w"][0], Y - P["coast_w"][1]) < 1300)
    ii, jj = np.nonzero(cand)
    sel = rng.choice(len(ii), min(400, len(ii)), replace=False)
    best, lp = -1e9, P["coast_w"]
    for i, j in zip(ii[sel], jj[sel]):
        q = np.array([X[i, j], Y[i, j]])
        sc = 2.0 * can_see(W, bp, q, 55) + can_see(W, S["town"], q, 55) + z[i, j] / 40
        if sc > best:
            best, lp = sc, q
    S["lighthouse"] = lp

    # second leg: town -> south pass -> desert. The pass is where the desert opens up.
    leg2 = find_route(W, S["town"], S["oasis"])
    zs = np.array([ground(W, p) for p in leg2])
    pk = int(np.argmax(zs))
    desT = targets(W, (W["desert"] > 0.55) & walk, lift=1.0, seed=1)
    prof2 = np.array([frac_seen(W, p, desT) for p in leg2])
    lo, hi2 = max(pk - 2, 0), min(pk + 12, len(leg2))
    S["pass"] = leg2[lo + int(np.argmax(prof2[lo:hi2]))]      # the shrine marks where the desert opens
    leg3 = find_route(W, S["town"], S["keep"])

    tops = dict(horn=12, volcano=12, island=30, town=20, keep=80, lighthouse=55)
    R["from_overlook"] = {k: can_see(W, bp, S[k], t) for k, t in tops.items()}
    R["from_town"] = {k: can_see(W, S["town"], S[k], t) for k, t in tops.items() if k != "town"}
    R["from_overlook"]["volcano"] = sees_cone(W, bp, S["volcano"])
    around = [S["town"] + 150 * np.array(d) for d in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]
    R["from_town"]["volcano"] = any(sees_cone(W, q, S["volcano"]) for q in around
                                    if walk[int(q[1] / dx), int(q[0] / dx)])
    R["bearings_from_overlook"] = {k: (round(bearing(bp, S[k])), round(float(np.hypot(*(S[k] - bp)))))
                                   for k in tops}
    R["keep_horn_stack_deg"] = abs((bearing(S["town"], S["keep"]) - hb + 180) % 360 - 180)
    R["desert_seen_from_town"] = frac_seen(W, S["town"], desT)
    R["desert_seen_at_pass"] = float(prof2[lo:hi2].max())
    R["volcano_seen_on_route_frac"] = float(np.mean([sees_cone(W, p, S["volcano"]) for p in route[::4]]))
    R["horn_seen_on_route_frac"] = float(np.mean([can_see(W, p, S["horn"], 12) for p in route[::4]]))
    lw = np.nonzero(walk)
    sel = rng.choice(len(lw[0]), 1500, replace=False)
    pts = np.stack([X[lw[0][sel], lw[1][sel]], Y[lw[0][sel], lw[1][sel]]], 1)
    R["horn_seen_from_land_frac"] = float(np.mean([can_see(W, p, S["horn"], 12) for p in pts]))
    R["heights"] = {k: round(float(ground(W, p))) for k, p in S.items()}

    seen = R["from_overlook"]
    R["checks"] = {
        "start is blind to the lake (first 15% of route sees < 2%)": R["lake_seen_first_15pct"] < 0.02,
        "lake hidden before the reveal (< 5% on average)": R["lake_seen_before"] < 0.05,
        "overlook shows at least half the lake": R["lake_seen_at_overlook"] >= 0.5,
        "reveal comes in the first half of the walk": R["overlook_at_frac"] < 0.5,
        "view faces the sunset (within 30 deg of west)": abs(R["view_bearing"] - 270) < 30,
        "overlook shows town, island and 4+ landmarks": seen["town"] and seen["island"]
            and sum(seen[k] for k in ("horn", "island", "town", "keep", "lighthouse")) >= 4,
        "volcano held back at the overlook": not seen["volcano"],
        "volcano appears from town": R["from_town"]["volcano"],
        "keep stacked under the horn from town (< 12 deg)": R["keep_horn_stack_deg"] < 12
            and R["from_town"]["keep"] and R["from_town"]["horn"],
        "desert hidden from town (< 5%)": R["desert_seen_from_town"] < 0.05,
        "desert opens at the pass (> 30%)": R["desert_seen_at_pass"] > 0.3,
        "walk start->town takes 5-12 min at 6 m/s": 5 <= R["route_m"] / 360 <= 12,
    }
    return dict(sites=S, route=route, leg2=leg2, leg3=leg3, lake_profile=prof,
                desert_profile=prof2, report=R, reveal_index=r1)
