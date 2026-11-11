"""Rooms and doors: a world as a graph of basins.

A room is a basin with a floor height and a theme. A wall is the high ground on
the border between two rooms. A door is a gap in a wall, placed anywhere along
that border. The sea counts as one more neighbour: a coast is either open
(beach) or a cliff with coves.

This module only designs causes (where land is pushed up, where rock is hard,
where water is pinned). `erode.form` turns them into land.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from layout import Layout
from noise import fbm

THEMES = {
    #          floor relief  K mult  soil  talus  wall mult
    "field":  dict(relief=10, k=1.3, soil=0.50, talus=0.80, wall=1.0),
    "lake":   dict(relief=8,  k=1.3, soil=0.50, talus=0.85, wall=1.0),
    "forest": dict(relief=22, k=1.0, soil=0.40, talus=0.90, wall=1.0),
    "desert": dict(relief=6,  k=0.8, soil=0.15, talus=1.30, wall=1.1),
    "ash":    dict(relief=12, k=0.9, soil=0.20, talus=1.00, wall=0.9),
    "moor":   dict(relief=26, k=0.7, soil=0.12, talus=1.20, wall=1.2),
}


@dataclass
class Cfg:
    rooms: int = 7            # how many rooms on the island
    wall: float = 160.0       # typical wall height above the floors, metres
    loops: float = 0.35       # chance that an adjacent pair beyond the spanning tree also gets a door
    hub: float = 0.0          # >0: one central room this many times the typical radius
    step: float = 38.0        # floor rise per room going inland
    door_w: float = 150.0     # door half width, metres
    saddle: float = 0.0       # how much wall is left standing in a door (0 = cut to the floor)
    cliff: float = 0.4        # share of coastal rooms with a cliffed coast
    width: float = 1.1        # wall half width as a multiple of its height
    uplift: float = 2.0e-6    # how hard walls are pushed up, per metre of designed height
    peak: float = 250.0       # extra height of the one summit
    strata: float = 1.0       # 0 = plain rock, 1 = layered rock on walls and desert
    period: float = 70.0      # thickness of one hard/soft pair of beds, metres
    L: float = 8192.0
    n: int = 512


def _island(n, L, rng):
    c = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(c, c)
    th = np.arctan2(Y - L / 2, X - L / 2)
    r = np.hypot(X - L / 2, Y - L / 2)
    k, ph = rng.uniform(-1, 1, 6), rng.uniform(0, 2 * np.pi, 6)
    R = 0.445 * L * (1 + sum(0.045 * k[i] * np.cos((i + 2) * th + ph[i]) for i in range(6)))
    land = (r + 0.027 * L * fbm(n, 6, 4, rng)) < R
    land = ndimage.binary_opening(land, iterations=2)
    lab, _ = ndimage.label(land)
    return X, Y, lab == lab[n // 2, n // 2]


def make_rooms(seed, cfg=Cfg()):
    rng = np.random.default_rng(seed)
    n, L = cfg.n, cfg.L
    lay = Layout(seed=seed, n=n, dx=L / n)
    dx = lay.dx
    X, Y, land = _island(n, L, rng)
    coast_d = ndimage.distance_transform_edt(land) * dx
    N = cfg.rooms

    # --- room sites: uneven on purpose --------------------------------------
    area = land.sum() * dx * dx
    dmin = 0.80 * np.sqrt(area / N)
    ii, jj = np.nonzero(coast_d > min(450, 0.35 * dmin))
    sites = []
    if cfg.hub > 0:
        sites.append(np.array([L / 2, L / 2]) + rng.normal(0, 150, 2))
    tries = 0
    while len(sites) < N:
        k = rng.integers(len(ii))
        p = np.array([X[ii[k], jj[k]], Y[ii[k], jj[k]]])
        need = dmin * (1 + (0.5 * cfg.hub if cfg.hub > 0 else 0) * (len(sites) > 0))
        if all(np.hypot(*(p - q)) > (need if (cfg.hub > 0 and i == 0) else dmin) for i, q in enumerate(sites)):
            sites.append(p)
        tries += 1
        if tries % 400 == 0:
            dmin *= 0.93
    sites = np.array(sites)
    wt = (rng.uniform(0, 0.42, N) * dmin) ** 2
    if cfg.hub > 0:
        wt[0] = (cfg.hub * 0.75 * dmin) ** 2
    Xw = X + 330 * fbm(n, 5, 3, rng) + 110 * fbm(n, 14, 2, rng)
    Yw = Y + 330 * fbm(n, 5, 3, rng) + 110 * fbm(n, 14, 2, rng)
    pd = np.stack([(Xw - s[0]) ** 2 + (Yw - s[1]) ** 2 - w for s, w in zip(sites, wt)])
    label = np.argmin(pd, 0)
    # every room is one connected piece; stray bits join whoever surrounds them
    keep = np.zeros((n, n), bool)
    for i in range(N):
        lab, k = ndimage.label((label == i) & land)
        if k:
            big = 1 + np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, k + 1)))
            keep |= lab == big
    _, (ni, nj) = ndimage.distance_transform_edt(~keep, return_indices=True)
    label = np.where(land, label[ni, nj], -1)
    sites = np.array([[X[label == i].mean(), Y[label == i].mean()] if (label == i).any() else sites[i]
                      for i in range(N)])

    # --- borders, adjacency --------------------------------------------------
    pair_len = {}
    bcell = {}                                              # (a, b) -> border cells (i, j)
    for di, dj in ((0, 1), (1, 0)):
        a = label[:n - di, :n - dj]
        b = label[di:, dj:]
        m = (a != b) & (a >= 0) & (b >= 0)
        ai, aj = np.nonzero(m)
        for i, j in zip(ai, aj):
            key = (min(a[i, j], b[i, j]), max(a[i, j], b[i, j]))
            bcell.setdefault(key, []).append((i, j))
    for key, v in bcell.items():
        pair_len[key] = len(v) * dx
    adj = [k for k, v in pair_len.items() if v > 350]
    border = np.zeros((n, n), bool)
    for v in bcell.values():
        for i, j in v:
            border[i, j] = True
    wall_d, (wi, wj) = ndimage.distance_transform_edt(~border, return_indices=True)
    wall_d *= dx
    # junctions: where three rooms meet
    mx = ndimage.maximum_filter(label, 3)
    mn = ndimage.minimum_filter(np.where(label < 0, 99, label), 3)
    md = ndimage.median_filter(label, 3)
    junc = land & (mx != mn) & (md != mx) & (md != mn) & (ndimage.minimum_filter(label, 3) >= 0)
    junc_d = ndimage.distance_transform_edt(~junc) * dx if junc.any() else np.full((n, n), 1e9)

    coastal = np.array([(ndimage.binary_dilation(label == i, iterations=2) & ~land).sum() * dx > 500
                        for i in range(N)])

    pk = None
    if junc.any():
        ji, jj2 = np.nonzero(junc)
        k = np.argmax(coast_d[ji, jj2])
        pk = np.array([X[ji[k], jj2[k]], Y[ji[k], jj2[k]]])
    pk_d = np.hypot(X - pk[0], Y - pk[1]) if pk is not None else np.full((n, n), 1e9)

    # --- doors: spanning tree first, then some loops -------------------------
    rk = {k: rng.uniform() for k in adj}
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    doors = []
    for k in sorted(adj, key=lambda k: rk[k]):
        ra, rb = find(k[0]), find(k[1])
        if ra != rb:
            parent[ra] = rb
            doors.append(k)
    for k in adj:
        if k not in doors and (rng.uniform() < cfg.loops or (cfg.hub > 0 and 0 in k)):
            doors.append(k)                                 # a hub opens onto every neighbour

    # floor heights rise going inland along the door graph, so water always has a way down
    depth = np.where(coastal, 0, 99)
    for _ in range(N):
        for a, b in doors:
            depth[a] = min(depth[a], depth[b] + 1)
            depth[b] = min(depth[b], depth[a] + 1)
    cliffed = coastal & (rng.uniform(0, 1, N) < cfg.cliff)
    h = 16 + cfg.step * depth + rng.uniform(0, 0.45 * cfg.step, N) + np.where(cliffed, rng.uniform(45, 80, N), 0)

    # themes: the highest room is moor, one mid room holds the volcano, the rest are dealt out
    order = list(np.argsort(-h))
    theme = [None] * N
    theme[order[0]] = "moor"
    pool = ["lake", "desert", "ash", "forest", "field", "lake", "forest", "desert", "field", "moor"]
    if cfg.hub > 0:
        theme[0] = "field"
    k = 0
    for i in order[::-1]:
        if theme[i] is None:
            theme[i] = pool[k % len(pool)]
            k += 1
    T = lambda key: np.array([THEMES[theme[i]][key] for i in range(N)])

    # door points: anywhere along the shared border, away from junctions
    D = []
    for a, b in doors:
        cells = np.array(bcell[(a, b)])
        ok = junc_d[cells[:, 0], cells[:, 1]] > min(380, 0.3 * pair_len[(a, b)])
        ok &= coast_d[cells[:, 0], cells[:, 1]] > 300
        if (ok & (pk_d[cells[:, 0], cells[:, 1]] > 950)).any():
            ok &= pk_d[cells[:, 0], cells[:, 1]] > 950
        c = cells[ok] if ok.any() else cells[[np.argmax(junc_d[cells[:, 0], cells[:, 1]])]]
        i, j = c[rng.integers(len(c))]
        D.append(dict(a=int(a), b=int(b), p=np.array([X[i, j], Y[i, j]]), kind="door",
                      w=cfg.door_w * rng.uniform(0.8, 1.25)))
    # sea doors: one cove per cliffed room
    for r in np.nonzero(cliffed)[0]:
        m = (label == r) & (coast_d < dx * 1.6)
        ci, cj = np.nonzero(m)
        if len(ci):
            k = rng.integers(len(ci))
            D.append(dict(a=int(r), b=-1, p=np.array([X[ci[k], cj[k]], Y[ci[k], cj[k]]]), kind="cove",
                          w=cfg.door_w * 1.3))

    # --- the designed surface ------------------------------------------------
    lab0 = np.maximum(label, 0)
    floor = ndimage.gaussian_filter(h[lab0], 4.0)
    relief = ndimage.gaussian_filter(T("relief")[lab0], 3.0)
    floor = floor + relief * fbm(n, 9, 4, rng)
    # walls are ranges, not fences: each border has its own height, and the band
    # swells, thins and wanders off the border line along its length
    hw = {k: cfg.wall * rng.uniform(0.75, 1.3) * 0.5 * (THEMES[theme[k[0]]]["wall"] + THEMES[theme[k[1]]]["wall"])
          for k in bcell}
    Hb = np.zeros((n, n))
    for k, v in bcell.items():
        v = np.array(v)
        Hb[v[:, 0], v[:, 1]] = hw[k]
    Hw = ndimage.gaussian_filter(Hb[wi, wj], 2.0) * (0.75 + 0.55 * fbm(n, 6, 3, rng))
    ww = np.maximum(110.0, cfg.width * Hw * (0.75 + 0.6 * fbm(n, 7, 3, rng)))
    dw = np.abs(wall_d + 0.45 * ww * fbm(n, 8, 3, rng))
    prof = np.exp(-(dw / ww) ** 2)
    jn = np.exp(-(junc_d / 330) ** 2)
    gate = np.ones((n, n))
    sill = np.zeros((n, n))
    for d in D:
        r2 = (X - d["p"][0]) ** 2 + (Y - d["p"][1]) ** 2
        if d["kind"] == "door":
            gate *= cfg.saddle + (1 - cfg.saddle) * (1 - np.exp(-r2 / d["w"] ** 2))
            if abs(h[d["a"]] - h[d["b"]]) > 12:
                sill = np.maximum(sill, np.exp(-r2 / (1.3 * d["w"]) ** 2))
    wall = Hw * prof * (1 + 0.7 * jn) * gate
    # the peak: the most inland junction becomes the one summit seen from everywhere
    if pk is not None:
        pr = np.hypot(Xw - pk[0], Yw - pk[1])
        ang = np.arctan2(Yw - pk[1], Xw - pk[0])
        arms = 1 + 0.4 * np.cos(3 * ang + rng.uniform(0, 6.3))
        wall += cfg.peak * np.exp(-(pr / 520) ** 2) * arms * np.clip(gate + 0.3, 0, 1)
    # coast: beaches taper to the sea, cliffed rooms stand until the shore, except at coves
    cl = ndimage.gaussian_filter(cliffed[lab0].astype(float), 3.0)
    cove = np.zeros((n, n))
    for d in D:
        if d["kind"] == "cove":
            cove = np.maximum(cove, np.exp(-((X - d["p"][0]) ** 2 + (Y - d["p"][1]) ** 2) / d["w"] ** 2))
    cl = cl * (1 - cove)
    taper = np.clip(coast_d / 300, 0.06, 1) ** 1.2
    taper = taper * (1 - cl) + np.clip(coast_d / 70, 0.25, 1) * cl
    Tz = (floor + wall) * taper

    # lakes: the low part of a lake room's floor, so the shore follows the ground
    lake = np.zeros((n, n), bool)
    pin_z = np.zeros((n, n))
    bowl = fbm(n, 9, 4, rng)
    for i in range(N):
        if theme[i] != "lake":
            continue
        m = label == i
        wd = np.where(m, np.minimum(wall_d - 1.6 * ww, coast_d - 420), 0)
        dep = wd / max(wd.max(), 1) + 0.6 * bowl
        ok = m & (wd > 0)
        if ok.sum() < 60:
            continue
        lk = ok & (dep > np.quantile(dep[ok], 0.6))
        lab, k = ndimage.label(lk)
        if k == 0:
            continue
        big = 1 + np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, k + 1)))
        lk = lab == big
        lvl = max(h[i] - 5, 6)
        lake |= lk
        pin_z[lk] = lvl
        near = ndimage.binary_dilation(lk, iterations=6)
        Tz = np.where(near & ~lk, np.maximum(Tz, lvl + 1.5), Tz)

    # --- causes ---------------------------------------------------------------
    wn = wall / max(cfg.wall, 1.0)
    U = (5.0e-5 + cfg.uplift * wall) * (0.8 + 0.4 * fbm(n, 5, 4, rng)) * np.clip(coast_d / 260, 0.3, 1)
    hardw = np.clip(prof * gate * 1.2, 0, 1)
    K = 4.0e-5 * ndimage.gaussian_filter(T("k")[lab0], 3.0)
    K = K * (1 - 0.6 * hardw)                                      # walls are harder rock
    K = K * (1 - 0.80 * sill)                                      # a hard sill holds each upper room up
    K = K * (1 - 0.8 * cl * np.exp(-(coast_d / 200) ** 2))        # sea cliffs
    K = K * np.exp(0.5 * fbm(n, 12, 4, rng) + 0.25 * rng.normal(size=(n, n)))
    soil = np.clip(ndimage.gaussian_filter(T("soil")[lab0], 3.0) * (1 - 0.85 * hardw), 0.04, 0.5)
    talus = ndimage.gaussian_filter(T("talus")[lab0], 3.0) * (1 + 0.25 * hardw)
    z0 = ndimage.gaussian_filter(floor + 0.6 * wall, 2.0) * taper + 5 * fbm(n, 24, 3, rng)
    for i in range(N):
        if theme[i] == "lake":
            near = ndimage.binary_dilation(lake & (label == i), iterations=6) & ~lake
            z0 = np.where(near, np.maximum(z0, pin_z.max(where=lake & (label == i), initial=0) + 1.5), z0)
    # layered rock, gently tilted: gives cliffs and benches on walls and in the desert
    tilt = rng.uniform(0, 6.3)
    strata_warp = 0.05 * (X * np.cos(tilt) + Y * np.sin(tilt)) + 25 * fbm(n, 6, 3, rng)
    des = ndimage.gaussian_filter(np.array([t == "desert" for t in theme], float)[lab0], 3.0)
    strata_amp = np.clip(hardw + 0.9 * des, 0, 1) * cfg.strata

    lay.pts = dict(peak=pk)
    lay.w, lay.nn = np.array([-1.0, 0.0]), np.array([0.0, 1.0])
    lay.f = dict(land=land, lake=lake, pin_z=pin_z, U=U, K=K, soil=soil, talus=talus, z0=z0,
                 coast_d=coast_d, label=label, wall_d=wall_d, prof=prof * gate, design=Tz, wall=wall,
                 strata_warp=strata_warp, strata_amp=strata_amp, strata_period=cfg.period)
    lay.rooms = dict(n=N, sites=sites, h=h, theme=theme, depth=depth, coastal=coastal, cliffed=cliffed,
                     adj=adj, doors=D, cfg=cfg, pair_len=pair_len)
    return lay
