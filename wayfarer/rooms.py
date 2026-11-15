"""Rooms and doors: a world as a graph of basins.

A room is a region with a floor and a theme. Floors follow a lay of the land:
a random tilt with a highland or two, so the island is not flat between its
high ground. A few rooms in a row are mountain country: one massif with valleys
through it, a place you travel through to reach the rooms beyond. Other borders
are a scarp, hills, a ribbon of water, rocky knolls, or open. A door is where a
road crosses a border. The sea counts as one more neighbour: a coast is either
open (beach) or a cliff with coves.

With `belt=0` the older scheme applies instead: mountain ranges run along
chains of borders, with doors as gaps in them.

This module only designs causes (where land is pushed up, where rock is hard,
where water is pinned). `erode.form` turns them into land.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from .noise import fbm


@dataclass
class Layout:
    """A designed world before the land forms: grid, named points, design fields."""
    seed: int
    n: int
    dx: float
    pts: dict = field(default_factory=dict)     # named design points, metres (x east, y north)
    w: np.ndarray = None                        # unit vector west
    nn: np.ndarray = None                       # unit vector north
    lake_z: float = 14.0                        # water level where no per-cell level is pinned
    f: dict = field(default_factory=dict)       # design fields, n x n

    def grid(self):
        c = (np.arange(self.n) + 0.5) * self.dx
        return np.meshgrid(c, c)                # X, Y with row = y


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
    wall: float = 190.0       # typical height of the high ground between rooms, metres
    loops: float = 0.35       # chance that an adjacent pair beyond the spanning tree also gets a door
    hub: float = 0.0          # >0: one central room this many times the typical radius
    lift: float = 170.0       # height range of the lay of the land, metres; 0 = floors rise by `step` going inland
    step: float = 38.0        # lift=0 only: floor rise per room going inland
    door_w: float = 150.0     # door half width, metres
    saddle: float = 0.0       # how much wall is left standing in a door (0 = cut to the floor)
    cliff: float = 0.4        # share of coastal rooms with a cliffed coast
    width: float = 1.1        # fence only: wall half width as a multiple of its height
    mass: float = 1.0         # 1 = walls are ranges with width of their own, 0 = thin fences
    span: float = 520.0       # typical half width of the high ground on a border, metres
    belt: int = 3             # this many rooms in a row are mountain country, a place you travel through (0: ranges along borders instead)
    massif: float = 300.0     # designed height of mountain country above its valley floors, metres
    ranges: int = 2           # how many mountain chains; 0 = any border may be a range (a cellular look)
    reach: float = 1.0        # length of the main chain as a multiple of the island's size
    hills: float = 0.35       # share of soft borders that are low hills
    water: float = 0.25       # share of soft borders that are a long lake with a dry gap
    broken: float = 0.25      # share of soft borders that are a belt of rocky knolls
    scarp: float = 0.4        # share of borders without a door that are one-sided escarpments
    rim: float = 50.0         # height of the unbroken cliff band on a border without a door, metres
    uplift: float = 2.0e-7    # how hard walls are pushed up, per metre of designed height (fences used 2e-6)
    peak: float = 200.0       # extra height of the one summit
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


def _seg(X, Y, a, b):
    ab = b - a
    t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / (ab @ ab + 1e-9), 0, 1)
    return np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1])


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
    Xw = X + 600 * fbm(n, 4, 3, rng) + 190 * fbm(n, 11, 2, rng)       # borders bend, they are not ruled lines
    Yw = Y + 600 * fbm(n, 4, 3, rng) + 190 * fbm(n, 11, 2, rng)
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

    # --- ranges: mountains come in chains, not in cells ------------------------
    # A chain is a run of borders that carries on through each junction without a
    # sharp turn. It ends at the sea or dies out inland. Borders off a chain get
    # something softer than mountains.
    chains = (cfg.ranges > 0 or cfg.belt > 0) and cfg.mass > 0
    geo = {}
    for k, v in bcell.items():
        v = np.array(v)
        if len(v) < 4:
            continue
        p = np.stack([X[v[:, 0], v[:, 1]], Y[v[:, 0], v[:, 1]]], 1)
        c = p.mean(0)
        t = (p - c) @ np.linalg.svd(p - c, full_matrices=False)[2][0]
        ends = []
        for q in (t.argmin(), t.argmax()):
            i, j = v[q]
            third = [x for x in label[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4].ravel() if x not in k]
            if coast_d[i, j] < 220 or not third or min(third) < 0:
                ends.append((p[q], None))                   # this end is at the sea
            else:
                ends.append((p[q], frozenset((*k, max(set(third), key=third.count)))))
        geo[k] = dict(c=c, ends=ends)
    # The lay of the land: one broad random height map for the whole island. It
    # tilts one way and has a highland or two. Room floors follow it, and so do the
    # chains, so mountains stand on high country and each island differs.
    ang = rng.uniform(0, 2 * np.pi)
    G = 0.5 + 0.45 * ((X - L / 2) * np.cos(ang) + (Y - L / 2) * np.sin(ang)) / (0.45 * L)
    li, lj = np.nonzero(coast_d > 900)
    for _ in range(rng.integers(1, 3)):
        q = rng.integers(len(li))
        G += rng.uniform(0.4, 0.9) * np.exp(-((X - X[li[q], lj[q]]) ** 2 + (Y - Y[li[q], lj[q]]) ** 2)
                                            / rng.uniform(1300, 2200) ** 2)
    G += 0.25 * fbm(n, 3, 2, rng)
    # stretch it so the lowest room sits near the sea and the highest uses the full lift
    gm = np.array([G[label == i].mean() for i in range(N)])
    G = np.clip(0.06 + 0.86 * (G - gm.min()) / np.ptp(gm), -0.05, 1.15)

    # Mountain country: a row of rooms on the highest ground. Each is a whole
    # massif with valleys through it, not a strip on a border.
    mount = np.zeros(N, bool)
    if cfg.belt > 0 and cfg.mass > 0:
        nb = {i: set() for i in range(N)}
        for a, b in adj:
            nb[a].add(b)
            nb[b].add(a)
        tip = [int(np.argmax(gm))] * 2
        mount[tip[0]] = True
        while mount.sum() < cfg.belt:
            opts = [(gm[q], e, q) for e in (0, 1) for q in nb[tip[e]]
                    if not mount[q] and sum(mount[r] for r in nb[q]) == 1]
            if not opts:
                break
            _, e, q = max(opts)
            mount[q] = True
            tip[e] = q

    def grow(k0, target, taken, used0):
        used = set(used0)
        chain, total, fe = [k0], pair_len[k0], []
        for end in geo[k0]["ends"]:
            cur, (pt, J) = k0, end
            while J is not None:
                used.add(J)
                d0 = pt - geo[cur]["c"]
                d0 = d0 / np.hypot(*d0)
                best, bdot, nxt = None, 0.15, None
                for q in geo:
                    if q in taken or q in chain or not set(q) < J:
                        continue
                    near = [e for e in geo[q]["ends"] if e[1] == J]
                    far = [e for e in geo[q]["ends"] if e[1] != J]
                    if not near or not far or far[0][1] in used:
                        continue
                    d1 = geo[q]["c"] - near[0][0]
                    d1 = d1 / np.hypot(*d1)
                    if d0 @ d1 > bdot:
                        best, bdot, nxt = q, d0 @ d1, far[0]
                if best is None or total > target:
                    fe.append(pt)                           # the chain dies out inland here
                    break
                chain.append(best)
                total += pair_len[best]
                cur, (pt, J) = best, nxt
        return chain, fe

    size = np.sqrt(land.sum()) * dx

    def score(chain):                                       # long, through the interior, on high country
        v = np.concatenate([np.array(bcell[k]) for k in chain])
        seen = {r for k in ranged for r in k}                # a second chain keeps clear of the first one's rooms
        apart = 1 - 0.7 * np.mean([r in seen for k in chain for r in k])
        return (apart * min(sum(pair_len[k] for k in chain), 1.2 * size)
                * (0.4 + coast_d[v[:, 0], v[:, 1]].mean() / coast_d.max())
                * (0.5 + G[v[:, 0], v[:, 1]].mean()))

    ranged, free_ends, used = set(), [], set()
    for ci in range(cfg.ranges if chains and cfg.belt == 0 else 0):
        cand = [k for k in geo if k not in ranged and pair_len[k] > 700
                and not any(e[1] in used for e in geo[k]["ends"])]
        if not cand:
            break
        target = size * cfg.reach * (1.0 if ci == 0 else 0.45)
        opts = [grow(k, target, ranged, used) for k in cand]
        chain, fe = max(opts, key=lambda o: score(o[0]) * rng.uniform(0.85, 1.15))
        ranged |= set(chain)
        free_ends += fe
        for k in chain:
            used |= {e[1] for e in geo[k]["ends"] if e[1] is not None}

    pk = None
    if mount.any():
        # the one summit stands deep inside mountain country, in its lowest room
        r0 = min(np.nonzero(mount)[0], key=lambda i: gm[i])
        dmm = ndimage.distance_transform_edt(mount[np.maximum(label, 0)] & land) * (label == r0)
        i, j = np.unravel_index(np.argmax(dmm), dmm.shape)
        pk = np.array([X[i, j], Y[i, j]])
    elif ranged:
        # the one summit stands on a chain, away from where it dies out
        v = np.concatenate([np.array(bcell[k]) for k in ranged])
        sc = coast_d[v[:, 0], v[:, 1]].copy()
        for p in free_ends:
            sc[np.hypot(X[v[:, 0], v[:, 1]] - p[0], Y[v[:, 0], v[:, 1]] - p[1]) < 900] *= 0.2
        i, j = v[np.argmax(sc)]
        pk = np.array([X[i, j], Y[i, j]])
    elif junc.any():
        ji, jj2 = np.nonzero(junc)
        k = np.argmax(coast_d[ji, jj2])
        pk = np.array([X[ji[k], jj2[k]], Y[ji[k], jj2[k]]])
    pk_d = np.hypot(X - pk[0], Y - pk[1]) if pk is not None else np.full((n, n), 1e9)

    # --- doors: spanning tree first, then some loops -------------------------
    # doors follow the land: a chain is crossed only where the door graph needs it
    rk = {k: rng.uniform() + (1.0 if k in ranged else 0.0) for k in adj}
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
        p_loop = cfg.loops * ((0.4 if k in ranged else 2.0) if chains else 1.0)
        if k not in doors and (rng.uniform() < p_loop or (cfg.hub > 0 and 0 in k)):
            doors.append(k)                                 # a hub opens onto every neighbour

    # floor heights rise going inland along the door graph, so water always has a way down
    depth = np.where(coastal, 0, 99)
    for _ in range(N):
        for a, b in doors:
            depth[a] = min(depth[a], depth[b] + 1)
            depth[b] = min(depth[b], depth[a] + 1)
    base = 16 + cfg.lift * G
    if cfg.lift > 0:
        # floors follow the lay of the land; high country that reaches the sea ends in cliffs
        h = np.array([base[label == i].mean() for i in range(N)]) + rng.uniform(-6, 6, N)
        hc = np.array([base[(label == i) & (coast_d < 450)].mean() if coastal[i] else 0.0 for i in range(N)])
        cliffed = coastal & (hc > 16 + 0.42 * cfg.lift)
        h = h + 30.0 * mount
    else:
        cliffed = coastal & (rng.uniform(0, 1, N) < cfg.cliff)
        h = 16 + cfg.step * depth + rng.uniform(0, 0.45 * cfg.step, N) + np.where(cliffed, rng.uniform(45, 80, N), 0)

    # themes: the volcano stands on the highest room, moor is next, the rest are dealt out from the lowest up
    order = list(np.argsort(-h))
    theme = [None] * N
    if mount.any():                                        # the volcano is a peak of the mountain country
        mo = [i for i in order if mount[i]]
        theme[mo[0]] = "ash"
        for i in mo[1:]:
            theme[i] = "moor"
    else:
        theme[order[0]] = "ash"
        theme[order[1]] = "moor"
    pool = ["lake", "forest", "lake", "desert", "field", "forest", "desert", "field", "moor"]   # lakes go to the lowest rooms
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
    floor = h[lab0]
    if cfg.lift > 0:
        # inside a room the floor still leans with the land
        floor = floor + 0.6 * (base - floor)
    floor = ndimage.gaussian_filter(floor, 4.0)
    relief = ndimage.gaussian_filter(T("relief")[lab0], 8.0)
    floor = floor + relief * fbm(n, 9, 4, rng)
    # Each border has its own height. What stands on it is either a fence (mass=0,
    # the first attempt: one thin crest on the border line) or a landform (mass=1).
    hw = {k: cfg.wall * rng.uniform(0.75, 1.3) * 0.5 * (THEMES[theme[k[0]]]["wall"] + THEMES[theme[k[1]]]["wall"])
          for k in bcell}
    # Landforms, one per border. A border with a door is open anyway, so it may be
    # soft. A border without one has to stop a walker.
    #   range  a belt of mountains: foothills, a rock core, a broad top with summits
    #   hills  low rounded high ground under soil (only where there is a door)
    #   scarp  one-sided: the upper room climbs gently to an edge, then drops as a cliff
    #   water  a long lake lies on the border; the door is the dry gap in it
    #   broken a belt of rocky knolls to thread through
    # With chains, only chain borders are ranges, a border without a door off the
    # chain is a scarp, and a border with a door off the chain is one of the soft kinds.
    #   open   nothing on the line: one side is mountain country, which is its own barrier
    RANGE, HILLS, SCARP, WATER, BROKEN, OPEN = range(6)
    doored = {tuple(sorted(k)) for k in doors}
    kind = {}
    for k in bcell:
        r = rng.uniform()
        if chains:
            wet = pair_len[k] > 900 and abs(h[k[0]] - h[k[1]]) < 22
            kind[k] = (OPEN if mount[k[0]] or mount[k[1]] else
                       RANGE if k in ranged else SCARP if k not in doored else
                       WATER if r < cfg.water and wet else
                       HILLS if r < cfg.water + cfg.hills else
                       BROKEN if r < cfg.water + cfg.hills + cfg.broken else SCARP)
        elif k in doored:
            kind[k] = HILLS if r < cfg.hills else SCARP if r < cfg.hills + 0.5 * cfg.scarp else RANGE
        else:
            kind[k] = SCARP if r < cfg.scarp else RANGE
    Hb, Sb = np.zeros((n, n)), np.zeros((n, n))
    Kb, Ub = np.zeros((n, n), int), np.zeros((n, n), int)
    Cb = np.zeros((n, n))                                  # 1 on borders that must stop a walker
    Lb = np.zeros((n, n))                                  # floor height of the lower room of the pair
    for k, v in bcell.items():
        v = np.array(v)
        Hb[v[:, 0], v[:, 1]] = hw[k]
        Sb[v[:, 0], v[:, 1]] = cfg.span * np.exp(rng.normal(0.0, 0.38))   # most are modest, a few are massifs
        Kb[v[:, 0], v[:, 1]] = kind[k]
        Lb[v[:, 0], v[:, 1]] = min(h[k[0]], h[k[1]])
        Cb[v[:, 0], v[:, 1]] = 0.0 if k in doored or kind[k] == OPEN else 1.0
        Ub[v[:, 0], v[:, 1]] = k[0] if h[k[0]] > h[k[1]] else k[1]      # the upper room of the pair
    jn = np.exp(-(junc_d / 330) ** 2)
    ss = lambda x, lo, hi: (lambda t: t * t * (3 - 2 * t))(np.clip((x - lo) / (hi - lo), 0, 1))
    if cfg.mass > 0:
        H0 = ndimage.gaussian_filter(Hb[wi, wj], 2.0)
        ww = ndimage.gaussian_filter(Sb[wi, wj], 3.0) * (0.75 + 0.6 * fbm(n, 5, 3, rng))
        ww = np.maximum(ww * (1 + 0.38 * fbm(n, 20, 3, rng)), 240.0)     # the foot is scalloped: spurs and bays
        wander = fbm(n, 7, 3, rng)
        along = 0.55 + 0.9 * np.clip(0.5 + 0.9 * fbm(n, 6, 3, rng), 0, 1)   # summits and saddles
        ridged = 1 - 2 * np.abs(fbm(n, 16, 4, rng, gain=0.55))
        # range
        s_in = np.clip(1 - np.abs(wall_d + 0.35 * ww * wander) / ww, 0, 1)   # 0 at the foot, 1 on the axis
        core_r = ss(s_in, 0.30, 0.66)
        knoll = np.clip(0.5 + 1.1 * fbm(n, 24, 3, rng), 0, 1)             # foothills are knobby, not a ramp
        prof_r = 0.22 * ss(s_in, 0.0, 0.5) * (0.35 + 1.3 * knoll) + 0.78 * core_r
        wall_r = H0 * along * (1 + 0.5 * ridged * core_r) * prof_r
        # A closed border carries one unbroken band of cliff, the same height at a
        # saddle as under a summit: rimrock on a range, the face itself on a scarp.
        closed = ndimage.gaussian_filter(Cb[wi, wj], 2.0)
        wall_r = wall_r + closed * cfg.rim * ss(s_in, 0.40, 0.40 + 34.0 / ww)
        # where a chain dies out inland it sinks into the floor over a long nose
        endt = np.ones((n, n))
        for p in free_ends:
            endt *= 0.1 + 0.9 * ss(np.hypot(X - p[0], Y - p[1]), 150.0, 1500.0)
        wall_r, core_r, prof_r = wall_r * endt, core_r * endt, prof_r * endt
        # hills: wider, lower, rounded, no bare rock
        s_h = np.clip(1 - np.abs(wall_d + 0.5 * ww * wander) / (1.25 * ww), 0, 1)
        prof_h = ss(s_h, 0.0, 0.85)
        wall_h = 0.38 * H0 * along * (1 + 0.3 * ridged) * prof_h
        # scarp: signed distance from the edge, positive on the upper side
        up_side = label == Ub[wi, wj]
        sd = np.where(up_side, 1.0, -1.0) * wall_d + 0.22 * ww * wander
        back = 1.9 * ww                                        # how far the dip slope runs back
        dip = np.clip(1 - sd / back, 0, 1) ** 1.6
        face = ss(sd, -0.24 * ww, 0.0)
        rim_s = closed * 1.3 * cfg.rim * ss(sd, -40.0, 0.0) * np.clip(1 - sd / back, 0, 1)
        prof_s = np.where(sd > 0, dip, face)
        core_s = np.where(sd > 0, np.clip(1 - sd / (0.25 * ww), 0, 1), face)   # the cap and the cliff
        wall_s = 0.85 * H0 * (0.75 + 0.5 * (along - 0.55)) * prof_s + rim_s
        # broken ground: separate rocky knolls, with level ways between them
        tor = np.clip(2.4 * fbm(n, 30, 3, rng) - 0.15, 0, 1) ** 1.5
        prof_b = tor * ss(s_h, 0.0, 0.6)
        wall_b = 0.3 * H0 * prof_b
        # blend where borders of different kinds meet (a water border has no high ground)
        kc = Kb[wi, wj]
        wk = [ndimage.gaussian_filter((kc == q).astype(float), 8.0) for q in range(6)]
        tot = sum(wk) + 1e-9
        Hw = np.ones((n, n))
        prof = (wk[0] * wall_r + wk[1] * wall_h + wk[2] * wall_s + wk[4] * wall_b) / tot   # metres, before doors
        core = (wk[0] * core_r + wk[2] * core_s + 0.8 * wk[4] * prof_b) / tot
        soft = wk[1] / tot
        shape = np.clip((wk[0] * prof_r + wk[1] * prof_h + wk[2] * prof_s + wk[4] * prof_b) / tot, 0, 1)
        rangew = wk[0] / tot if chains else np.ones((n, n))
        if mount.any():
            # Mountain country fills its rooms: foothills at the edge, high ground
            # inside. Every door has a valley running in to the heart of its room,
            # so the way through climbs, crosses and comes down again.
            dm = ndimage.distance_transform_edt(mount[lab0] & land) * dx
            rise = ss(dm * (1 + 0.5 * fbm(n, 6, 3, rng)), 0.0, 1100.0)
            Xm, Ym = X + 260 * fbm(n, 9, 3, rng), Y + 260 * fbm(n, 9, 3, rng)
            way = np.zeros((n, n))
            for d in D:
                for r in (d["a"], d["b"]):
                    if d["kind"] == "door" and mount[r]:
                        way = np.maximum(way, np.exp(-(_seg(Xm, Ym, d["p"], sites[r]) / 190.0) ** 2))
            body = (0.3 * ss(dm, 0.0, 500.0) * (0.4 + 1.2 * knoll)
                    + 0.7 * rise * (0.55 + 0.9 * np.clip(0.5 + fbm(n, 5, 3, rng), 0, 1)) * (1 + 0.5 * ridged))
            prof = prof + cfg.massif * body * (1 - 0.94 * way)
            core = np.maximum(core, ss(rise, 0.25, 0.7) * (1 - way))
            shape = np.maximum(shape, np.clip(body, 0, 1) * (1 - way))
    else:
        Hw = ndimage.gaussian_filter(Hb[wi, wj], 2.0) * (0.75 + 0.55 * fbm(n, 6, 3, rng))
        ww = np.maximum(110.0, cfg.width * Hw * (0.75 + 0.6 * fbm(n, 7, 3, rng)))
        dw = np.abs(wall_d + 0.45 * ww * fbm(n, 8, 3, rng))
        prof = np.exp(-(dw / ww) ** 2)
        core = shape = prof
        ridged = soft = closed = np.zeros((n, n))
        rangew = np.ones((n, n))
    gate = np.ones((n, n))
    sill = np.zeros((n, n))
    for d in D:
        r2 = (X - d["p"][0]) ** 2 + (Y - d["p"][1]) ** 2
        if d["kind"] != "door":
            continue
        if cfg.mass > 0:
            # a pass: a valley cut through the whole belt, widening to both mouths
            cells = np.array(bcell[(d["a"], d["b"])])
            pc = np.stack([X[cells[:, 0], cells[:, 1]], Y[cells[:, 0], cells[:, 1]]], 1) - d["p"]
            pc = pc[np.hypot(pc[:, 0], pc[:, 1]) < 420]
            tng = np.linalg.svd(pc - pc.mean(0), full_matrices=False)[2][0] if len(pc) > 3 else np.array([1.0, 0.0])
            u = (X - d["p"][0]) * tng[0] + (Y - d["p"][1]) * tng[1]
            v = -(X - d["p"][0]) * tng[1] + (Y - d["p"][1]) * tng[0]
            sp = ww[int(d["p"][1] / dx), int(d["p"][0] / dx)]
            g = np.exp(-(u / (d["w"] * (1 + 1.2 * (v / sp) ** 2))) ** 2) * (1 - ss(np.abs(v) / sp, 1.3, 2.0))
        else:
            g = np.exp(-r2 / d["w"] ** 2)
        gate *= cfg.saddle + (1 - cfg.saddle) * (1 - g)
        if abs(h[d["a"]] - h[d["b"]]) > 12:
            sill = np.maximum(sill, np.exp(-r2 / (1.3 * d["w"]) ** 2))
    wall = Hw * prof * (1 + 0.7 * jn * rangew) * gate
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
    # a closed border meets the sea as a headland, not a beach you can walk round on
    cl = np.maximum(cl, np.clip(closed * shape * 2.0, 0, 1) * (1 - cove))
    taper = taper * (1 - cl) + np.clip(coast_d / 70, 0.25, 1) * cl
    Tz = (floor + wall) * taper

    # lakes: water lies where it would collect, in the low part of a lake room, well
    # back from the room's edge. The ground around is eased to the water level, so
    # the lake sits in a hollow and never hangs over a drop.
    ribbons = []
    lake = np.zeros((n, n), bool)
    pin_z = np.zeros((n, n))
    bowl = fbm(n, 9, 4, rng)
    for i in range(N):
        if theme[i] != "lake":
            continue
        m = label == i
        # the higher the water stands above the next room or the sea, the further back it lies
        back = 9.0 * np.maximum(h[i] - Lb[wi, wj], 0)
        wd = np.where(m, np.minimum(wall_d - np.maximum((1.1 if cfg.mass > 0 else 1.6) * ww, 450.0) - back,
                                    coast_d - 420 - 9.0 * max(h[i] - 10, 0)), 0)
        dep = wd / max(wd.max(), 1) + 0.6 * bowl - (floor - h[i]) / 25.0
        ok = m & (wd > 0)
        if ok.sum() < 60:
            continue
        lk = ok & (dep > np.quantile(dep[ok], 0.6))
        lab, k = ndimage.label(lk)
        if k == 0:
            continue
        big = 1 + np.argmax(ndimage.sum(np.ones_like(lab), lab, range(1, k + 1)))
        lk = lab == big
        lvl = max(np.percentile(floor[lk], 15) - 2, 6)
        bank = ss(1 - ndimage.distance_transform_edt(~lk) * dx / 380.0, 0.0, 1.0)
        Tz = Tz * (1 - bank) + (lvl + 2) * bank
        ribbons.append((lvl, bank))
        lake |= lk
        pin_z[lk] = lvl
        near = ndimage.binary_dilation(lk, iterations=6)
        Tz = np.where(near & ~lk, np.maximum(Tz, lvl + 1.5), Tz)

    # water borders: a long lake lies on the border, and the door is the dry gap in it
    if cfg.mass > 0 and WATER in kind.values():
        dgap = np.zeros((n, n))
        for d in D:
            if d["kind"] == "door":
                dgap = np.maximum(dgap, np.exp(-((X - d["p"][0]) ** 2 + (Y - d["p"][1]) ** 2) / (1.7 * d["w"]) ** 2))
        rw = 60 + 150 * np.clip(0.5 + fbm(n, 10, 3, rng), 0, 1)        # half width, metres
        for k, q in kind.items():
            if q != WATER:
                continue
            v = np.array(bcell[k])
            m = np.ones((n, n), bool)
            m[v[:, 0], v[:, 1]] = False
            d_b = ndimage.distance_transform_edt(m) * dx
            lvl = max(min(h[k[0]], h[k[1]]) - 4, 6)
            # like any lake, it keeps back from the sea in proportion to how high it stands
            lk = land & (d_b < rw) & (junc_d > 420) & (coast_d > 380 + 9.0 * lvl) & (dgap < 0.35)
            if lk.sum() < 40:
                continue
            bank = ss(1 - d_b / 320.0, 0.0, 1.0)                        # both floors fall gently to the shore
            Tz = Tz * (1 - bank) + (lvl + 2) * bank
            lake |= lk
            pin_z[lk] = lvl
            ribbons.append((lvl, bank))

    # --- causes ---------------------------------------------------------------
    wn = wall / max(cfg.wall, 1.0)
    U = (5.0e-5 + cfg.uplift * wall) * (0.8 + 0.4 * fbm(n, 5, 4, rng)) * np.clip(coast_d / 260, 0.3, 1)
    hardw = np.clip(core * gate * 1.2, 0, 1)                      # the rock core; foothills keep their soil
    K = 4.0e-5 * ndimage.gaussian_filter(T("k")[lab0], 8.0)
    K = K * (1 - 0.6 * hardw)                                      # walls are harder rock
    K = K * (1 - 0.80 * sill)                                      # a hard sill holds each upper room up
    K = K * (1 - 0.8 * cl * np.exp(-(coast_d / 200) ** 2))        # sea cliffs
    K = K * np.exp(0.5 * fbm(n, 12, 4, rng) + 0.25 * rng.normal(size=(n, n)))
    soil = np.clip(ndimage.gaussian_filter(T("soil")[lab0], 8.0) * (1 - 0.85 * hardw), 0.04, 0.5)
    talus = ndimage.gaussian_filter(T("talus")[lab0], 8.0) * (1 + 0.25 * hardw)
    z0 = ndimage.gaussian_filter(floor + 0.6 * wall * (1 + 0.3 * ridged), 2.0) * taper + 5 * fbm(n, 24, 3, rng)
    for i in range(N):
        if theme[i] == "lake":
            near = ndimage.binary_dilation(lake & (label == i), iterations=6) & ~lake
            z0 = np.where(near, np.maximum(z0, pin_z.max(where=lake & (label == i), initial=0) + 1.5), z0)
    for lvl, bank in ribbons:
        z0 = z0 * (1 - bank) + (lvl + 2) * bank
        K = K * (1 - 0.85 * bank)                               # a lake is held up by a collar of hard rock
    # layered rock, gently tilted: gives cliffs and benches on walls and in the desert
    tilt = rng.uniform(0, 6.3)
    strata_warp = 0.05 * (X * np.cos(tilt) + Y * np.sin(tilt)) + 25 * fbm(n, 6, 3, rng)
    des = ndimage.gaussian_filter(np.array([t == "desert" for t in theme], float)[lab0], 3.0)
    strata_amp = np.clip(hardw + 0.9 * des, 0, 1) * cfg.strata

    lay.pts = dict(peak=pk)
    lay.w, lay.nn = np.array([-1.0, 0.0]), np.array([0.0, 1.0])
    lay.f = dict(land=land, lake=lake, pin_z=pin_z, U=U, K=K, soil=soil, talus=talus, z0=z0,
                 coast_d=coast_d, label=label, wall_d=wall_d, prof=shape * gate, design=Tz, wall=wall,
                 strata_warp=strata_warp, strata_amp=strata_amp, strata_period=cfg.period)
    lay.rooms = dict(n=N, sites=sites, h=h, theme=theme, depth=depth, coastal=coastal, cliffed=cliffed,
                     adj=adj, doors=D, cfg=cfg, pair_len=pair_len,
                     kinds={k: ("range", "hills", "scarp", "water", "broken", "open")[q] for k, q in kind.items()},
                     free_ends=free_ends, mount=mount)
    return lay
