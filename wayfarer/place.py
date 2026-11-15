"""Place settlements, landmarks and roads in a rooms world, then measure whether
the rooms and doors behave as designed. All of it is read off the finished land.

The questions, written down before any result:
  contain   from a room's floor, how much of each other room's floor can you see?
  reveal    walking through a door, does the next room open at the door and not before?
  leak      can you walk over a wall that has no door?  is any door blocked?
  landmark  is each room's landmark seen from its own doors, and the peak from everywhere?
  floor     how much of each room is usable floor, and how much is wall?
"""
import heapq
import numpy as np
from numba import njit, prange
from scipy import ndimage

EYE = 1.7            # eye height, metres
WALK = 0.75          # steeper than this (about 37 degrees) cannot be walked


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
    # a ford costs a detour's worth, and a road keeps off the bank beside the water
    wet = ndimage.maximum_filter(W["river"], s)[::s, ::s] & ~block
    bank = (W["river_z"] > -999)[::s, ::s] & ~block
    extra = np.where(wet, 160.0, np.where(bank, 40.0, 0.0))
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


def can_see(W, p, q, top):
    e = eye(W, p)
    return bool(sees(W["z"], W["dx"], e[0], e[1], e[2], q[0], q[1], ground(W, q) + top))


def snap(W, p, radius, score):
    """Best cell within radius of p under a score array."""
    dx, n = W["dx"], W["n"]
    i0, j0, r = int(p[1] / dx), int(p[0] / dx), int(radius / dx)
    sl = (slice(max(i0 - r, 0), min(i0 + r + 1, n)), slice(max(j0 - r, 0), min(j0 + r + 1, n)))
    sub = score[sl]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    return np.array([(sl[1].start + k[1] + 0.5) * dx, (sl[0].start + k[0] + 0.5) * dx])



def _pick(mask, k, rng):
    ii, jj = np.nonzero(mask)
    if len(ii) == 0:
        return np.zeros((0, 2), int)
    s = rng.choice(len(ii), min(k, len(ii)), replace=False)
    return np.stack([ii[s], jj[s]], 1)


def place(W, log=print):
    R, dx, z, n = W["rooms"], W["dx"], W["z"], W["n"]
    N = R["n"]
    rng = np.random.default_rng(W["seed"] + 21)
    lab = W["label"]
    walk = W["land"] & ~W["lake"]
    c = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(c, c)
    xy = lambda ij: np.array([(ij[1] + 0.5) * dx, (ij[0] + 0.5) * dx])
    prom = z - ndimage.uniform_filter(z, int(300 / dx))
    lake_d = ndimage.distance_transform_edt(~W["lake"]) * dx
    riv_d = ndimage.distance_transform_edt(~W["river"]) * dx
    sea_d = ndimage.distance_transform_edt(W["land"]) * dx

    floor = np.zeros((n, n), bool)
    S = dict(town=[], landmark=[], door=[], peak=None)
    for i in range(N):
        m = (lab == i) & walk
        zmed = np.median(z[m & (W["wallw"] < 0.15)]) if (m & (W["wallw"] < 0.15)).any() else np.median(z[m])
        fl = m & (W["slope"] < 0.30) & (z < zmed + 45)
        l2, k = ndimage.label(fl)
        if k:
            fl = l2 == 1 + np.argmax(ndimage.sum(np.ones_like(l2), l2, range(1, k + 1)))
        floor |= fl
        cen = np.array([X[fl].mean(), Y[fl].mean()])
        # settlement: flat, near water, toward the middle of the floor
        sc = np.where(fl & (W["slope"] < 0.12) & (sea_d > 220), 0.0, -1e9)
        sc = sc - 0.004 * np.hypot(X - cen[0], Y - cen[1]) - 0.004 * np.minimum(np.minimum(lake_d, riv_d), 400) \
            - 3.0 * (np.minimum(lake_d, riv_d) < 25)
        S["town"].append(xy(np.unravel_index(np.argmax(sc), sc.shape)))
    W["floor"] = floor

    # roads: one per door, settlement to settlement through the door
    roads = []
    for d in R["doors"]:
        if d["kind"] != "door":
            roads.append(None)
            continue
        a, b = S["town"][d["a"]], S["town"][d["b"]]
        p = snap(W, d["p"], 1.2 * d["w"], np.where(walk, -z, -1e9))        # lowest ground in the gap
        try:
            roads.append(np.concatenate([find_route(W, a, p), find_route(W, p, b)]))
        except Exception:
            roads.append(None)
        d["site"] = p
    S["door"] = [d.get("site", d["p"]) for d in R["doors"]]
    grade_roads(W, [r for r in roads if r is not None])

    # landmark: a knoll in the room that the room's doors and settlement can see
    for i in range(N):
        fl = floor & (lab == i)
        cand = (lab == i) & walk & (W["slope"] < 0.5) & (prom > np.percentile(prom[fl], 85) if fl.any() else True)
        cand &= np.hypot(X - S["town"][i][0], Y - S["town"][i][1]) > 250
        cs = _pick(cand, 60, rng)
        eyes = [S["town"][i]] + [d["site"] for d in R["doors"] if d["kind"] == "door" and i in (d["a"], d["b"])]
        best, bp = -1, S["town"][i] + 300
        for ij in cs:
            q = xy(ij)
            s = np.mean([can_see(W, e, q, 30.0) for e in eyes]) + 0.002 * prom[tuple(ij)]
            if s > best:
                best, bp = s, q
        S["landmark"].append(bp)
    S["peak"] = xy(np.unravel_index(np.argmax(z), z.shape))
    return dict(sites=S, roads=roads)


def grade_roads(W, roads, half=10.0, fade=38.0):
    """Roads are built: cut and fill along each road so its long profile is smooth.
    This is the only place the finished land is edited, and only under a road."""
    z, dx, n = W["z"], W["dx"], W["n"]
    num = np.zeros_like(z)
    den = np.zeros_like(z)
    r = int(fade / dx) + 1
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    for line in roads:
        zs = np.array([ground(W, p) for p in line])
        k = np.ones(9) / 9                                   # 180 m running mean along the road
        zs = np.convolve(np.pad(zs, 4, mode="edge"), k, mode="valid")
        for p, zp in zip(line, zs):
            i, j = int(p[1] / dx), int(p[0] / dx)
            if i < r or j < r or i >= n - r or j >= n - r:
                continue
            d = np.hypot((j + ox + 0.5) * dx - p[0], (i + oy + 0.5) * dx - p[1])
            w = np.clip((fade - d) / (fade - half), 0, 1) ** 2
            num[i - r:i + r + 1, j - r:j + r + 1] += w * zp
            den[i - r:i + r + 1, j - r:j + r + 1] += w
    m = (den > 0) & W["land"] & ~W["lake"]
    a = np.clip(den, 0, 1)
    # a road leaves the river's bed and banks as they are, and comes down to a ford
    off = ndimage.distance_transform_edt(W["river_z"] < -999) * dx
    a = a * np.clip(off / 30.0, 0, 1) ** 2
    z[m] = z[m] * (1 - a[m]) + (num[m] / den[m]) * a[m]
    gy, gx = np.gradient(z, dx)
    W["slope"] = np.hypot(gx, gy)


def measure(W, PL, log=print):
    R, dx, z, n = W["rooms"], W["dx"], W["z"], W["n"]
    N, S = R["n"], PL["sites"]
    rng = np.random.default_rng(W["seed"] + 31)
    lab, floor = W["label"], W["floor"]
    walk = W["land"] & ~W["lake"]
    M = {}

    # --- contain: floor-to-floor visibility between rooms --------------------
    view, targ, tid = [], [], []
    for i in range(N):
        for ij in _pick(floor & (lab == i), 14, rng):
            view.append((i, (ij[1] + 0.5) * dx, (ij[0] + 0.5) * dx, z[tuple(ij)] + EYE))
        for ij in _pick((floor | W["lake"]) & (lab == i), 220, rng):
            targ.append(((ij[1] + 0.5) * dx, (ij[0] + 0.5) * dx, z[tuple(ij)] + 1.0))
            tid.append(i)
    # any ground of the room, walls included: "can I see that area at all?"
    for i in range(N):
        for ij in _pick((lab == i) & walk, 220, rng):
            targ.append(((ij[1] + 0.5) * dx, (ij[0] + 0.5) * dx, z[tuple(ij)] + 1.0))
            tid.append(N + i)
    LMK = np.array([[*p, ground(W, p) + 50.0] for p in S["landmark"]])
    T, tid = np.array(targ), np.array(tid)
    V = np.zeros((N, N))
    VL = np.zeros((N, N))
    VM = np.zeros((N, N))
    cnt = np.zeros(N)
    pk = np.array([[*S["peak"], z.max() + 12]])
    peak_seen = []
    for (i, x, y, ez) in view:
        s = sees_many(z, dx, np.array([x, y, ez]), T)
        for j in range(N):
            if (tid == j).any():
                V[i, j] += s[tid == j].mean()
            if (tid == N + j).any():
                VL[i, j] += s[tid == N + j].mean()
        VM[i] += sees_many(z, dx, np.array([x, y, ez]), LMK)
        cnt[i] += 1
        peak_seen.append(bool(sees_many(z, dx, np.array([x, y, ez]), pk)[0]))
    V /= np.maximum(cnt[:, None], 1)
    VL /= np.maximum(cnt[:, None], 1)
    VM /= np.maximum(cnt[:, None], 1)
    door_pairs = {(d["a"], d["b"]) for d in R["doors"] if d["kind"] == "door"}
    adj = set(R["adj"])
    cls = lambda i, j: "self" if i == j else "door" if (min(i, j), max(i, j)) in door_pairs else \
        "wall" if (min(i, j), max(i, j)) in adj else "far"
    for k in ("self", "door", "wall", "far"):
        v = [V[i, j] for i in range(N) for j in range(N) if cls(i, j) == k]
        M[f"see_{k}"] = float(np.mean(v)) if v else None
        v = [VL[i, j] for i in range(N) for j in range(N) if cls(i, j) == k]
        M[f"land_{k}"] = float(np.mean(v)) if v else None
        v = [VM[i, j] for i in range(N) for j in range(N) if cls(i, j) == k]
        M[f"mark_{k}"] = float(np.mean(v)) if v else None
    # from an average floor spot: how many other rooms show any ground, how many landmarks show
    M["rooms_in_view"] = float(np.mean([(VL[i] > 0.02).sum() - 1 for i in range(N)]))
    M["marks_in_view"] = float(np.mean([VM[i].sum() for i in range(N)]))
    M["peak_seen"] = float(np.mean(peak_seen))
    M["V"] = V.round(3).tolist()

    # --- leak / blocked: can you walk where you should and only there --------
    ok = walk & (W["slope"] < WALK)
    leaks = blocked = 0
    for (a, b) in R["adj"]:
        m = ok & ((lab == a) | (lab == b))
        l2, _ = ndimage.label(m, structure=np.ones((3, 3)))
        la = np.bincount(l2[floor & (lab == a)]).argmax() if (floor & (lab == a)).any() else 0
        lb = np.bincount(l2[floor & (lab == b)]).argmax() if (floor & (lab == b)).any() else -1
        joined = la == lb and la != 0
        if (a, b) not in door_pairs and joined:
            leaks += 1
    for d, road in zip(R["doors"], PL["roads"]):
        if d["kind"] != "door":
            continue
        if road is None:
            blocked += 1
            continue
        zs = np.array([ground(W, q) for q in road])
        if np.abs(zs[2:] - zs[:-2]).max() / 40.0 > WALK:    # steepest 40 m of the road
            blocked += 1
    M["doors"] = len(door_pairs)
    M["walls"] = len(adj) - len(door_pairs)
    M["blocked"], M["leaks"] = blocked, leaks

    # --- reveal: what the road through each door shows, and when -------------
    rev = []
    for d, road in zip(R["doors"], PL["roads"]):
        if road is None:
            continue
        for (src, dst, line) in ((d["a"], d["b"], road), (d["b"], d["a"], road[::-1])):
            Tj = T[tid == dst]
            li = lab[(line[:, 1] / dx).astype(int), (line[:, 0] / dx).astype(int)]
            cross = np.nonzero(li == dst)[0]
            if len(cross) == 0 or len(Tj) == 0:
                continue
            c0 = int(cross[0])
            prof = np.array([sees_many(z, dx, eye(W, p), Tj).mean() for p in line[max(c0 - 40, 0):c0 + 26]])
            k0 = min(c0, 40)
            before = float(prof[:max(k0 - 7, 1)].max())            # up to 150 m short of the door
            after = float(prof[k0 + 5:k0 + 16].max()) if len(prof) > k0 + 5 else float(prof[-1])
            lm = bool(np.any([can_see(W, p, S["landmark"][dst], 30.0) for p in line[c0:c0 + 12]]))
            rev.append(dict(src=src, dst=dst, before=before, after=after, landmark=lm))
    M["reveal"] = rev
    if rev:
        M["reveal_before"] = float(np.mean([r["before"] for r in rev]))
        M["reveal_after"] = float(np.mean([r["after"] for r in rev]))
        M["reveal_good"] = float(np.mean([(r["after"] > 0.25) and (r["before"] < 0.10) for r in rev]))
        M["reveal_landmark"] = float(np.mean([r["landmark"] for r in rev]))

    # --- exits: standing in a room's settlement, how many of its doors can you pick out?
    seen = []
    for i in range(N):
        ds = [d["site"] for d in R["doors"] if d["kind"] == "door" and i in (d["a"], d["b"])]
        seen += [can_see(W, S["town"][i], q, 10.0) for q in ds]
    M["doors_seen"] = float(np.mean(seen)) if seen else None

    # --- floor: usable ground per room ---------------------------------------
    fr = [float((floor & (lab == i)).sum() / max((lab == i).sum(), 1)) for i in range(N)]
    M["floor_frac"] = float(np.mean(fr))
    M["room_km2"] = float(np.mean([(lab == i).sum() for i in range(N)]) * dx * dx / 1e6)
    M["room_area_km2"] = [float((lab == i).sum() * dx * dx / 1e6) for i in range(N)]
    slots = lambda L_: 0 if L_ <= 350 else 1 + int(max(L_ - 1200, 0) // 600)   # doors a border has room for
    M["door_slots"] = [sum(slots(R["pair_len"][k]) for k in R["adj"] if i in k) for i in range(N)]
    M["neighbours"] = [sum(1 for k in R["adj"] if i in k) for i in range(N)]
    M["doors_per_room"] = [sum(1 for d in R["doors"] if i in (d["a"], d["b"])) for i in range(N)]
    M["peak_m"] = float(z.max())
    return M
