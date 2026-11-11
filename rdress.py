"""Dress a rooms world for the preview renderer: ground cover by room theme,
trees, water, snow, and simple oversized structures at the placed sites."""
import numpy as np
from scipy import ndimage
from noise import fbm
from dress import _ss


def dress(W, PL, up=2):
    rng = np.random.default_rng(W["seed"] + 11)
    N, dx = W["n"] * up, W["dx"] / up
    zo = lambda a, o=1: ndimage.zoom(np.asarray(a, np.float64), up, order=o)
    z = zo(W["z"])
    land, lake = zo(W["land"]) > 0.5, zo(W["lake"]) > 0.5
    lz = ndimage.zoom(W["lake_zmap"], up, order=0)
    gy, gx = np.gradient(z, dx)
    rough = np.clip(np.hypot(gx, gy) / 0.8, 0.08, 1)
    z = z + rough * (3.0 * fbm(N, 200, 3, rng) + 0.5 * rng.normal(size=(N, N))) * land
    z[lake] = lz[lake]
    z[~land] = -6.0
    gy, gx = np.gradient(ndimage.gaussian_filter(z, 1.5), dx)
    slope = np.hypot(gx, gy)
    des_w, ash_w, moor_w, for_w, fld_w = (zo(W[k]) for k in ("desert", "ash", "moor", "forest", "field"))
    c = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(c, c)
    n1, n2, n3 = fbm(N, 10, 4, rng), fbm(N, 48, 3, rng), rng.uniform(0, 1, (N, N))
    ash = _ss(ash_w + 0.2 * n1, 0.45, 0.7)
    desert = _ss(des_w + 0.15 * n1, 0.45, 0.65) * (1 - ash)
    moor = _ss(moor_w + 0.2 * n1, 0.45, 0.7)
    north = -gy / (slope + 1e-6)

    area = W["area"]
    riv = np.zeros_like(area, bool)
    for a_min, grow in ((4e5, 0), (3e6, 1)):
        m = area > a_min
        riv |= ndimage.binary_dilation(m, iterations=grow) if grow else m
    riv = (zo(riv) > 0.55) & land & ~lake

    dry = _ss(n1 + 0.5 * fld_w, -0.2, 0.6)
    grass = np.stack([0.06 + 0.08 * dry, 0.11 + 0.035 * dry, 0.03 + 0.02 * dry], -1)
    tundra = np.array([0.17, 0.15, 0.085])
    hi = np.maximum(_ss(z, 330, 480), 0.85 * moor)[..., None]
    grass = grass * (1 - hi) + tundra * hi
    rock = np.array([0.30, 0.28, 0.26]) * (0.8 + 0.4 * n3[..., None])
    alb = grass.copy()
    w = _ss(slope + 0.25 * n2, 0.85, 1.15)[..., None]
    alb = alb * (1 - w) + rock * w
    band = 0.5 + 0.5 * np.sin((z + 30 * n1) / 7.0)
    strata = np.array([0.36, 0.13, 0.06]) * (1 - band[..., None]) + np.array([0.50, 0.33, 0.19]) * band[..., None]
    sand = np.array([0.36, 0.21, 0.10]) * (0.9 + 0.2 * n2[..., None])
    st = _ss(slope, 0.35, 0.7)[..., None]
    dcol = sand * (1 - st) + strata * st
    alb = alb * (1 - desert[..., None]) + dcol * desert[..., None]
    acol = np.array([0.075, 0.07, 0.068]) * (0.7 + 0.6 * n3[..., None])
    alb = alb * (1 - ash[..., None]) + acol * ash[..., None]
    snowline = 520 + 60 * n1 - 80 * north
    snow = _ss(z, snowline, snowline + 70) * (1 - _ss(slope + 0.2 * n2, 0.9, 1.3))
    alb = alb * (1 - snow[..., None]) + np.array([0.86, 0.89, 0.94]) * snow[..., None]
    beach = land & ~lake & ((z < 2.2) | ((z < lz.max() + 0) & (z < ndimage.maximum_filter(lz, 11) + 1.2)
                                         & ndimage.binary_dilation(lake, iterations=5)))
    alb[beach] = np.array([0.36, 0.31, 0.22])
    emis = np.zeros((N, N, 3))
    if "volcano" in W["pts"]:
        vr = np.hypot(X - W["pts"]["volcano"][0], Y - W["pts"]["volcano"][1])
        emis[vr < W["pts"]["volcano_r"] * 0.8] = np.array([5.0, 1.1, 0.12])

    treeline = 400 + 50 * n1
    dens = np.clip(0.18 + 0.75 * for_w - 0.12 * fld_w - 0.6 * moor, 0.02, 0.95)
    forest = land & ~lake & ~beach & ~riv & (slope < 1.0) & (z < treeline) & (desert < 0.3) & (ash < 0.25) \
        & (_ss(n1 * 0.5 + 0.5, 1 - dens - 0.15, 1 - dens + 0.15) > 0.5) & (n3 < 0.85)
    road_d = np.full((N, N), 1e9)
    for line in PL["roads"]:
        if line is None:
            continue
        m = np.ones((N, N), bool)
        m[(line[:, 1] / dx).astype(int), (line[:, 0] / dx).astype(int)] = False
        road_d = np.minimum(road_d, ndimage.distance_transform_edt(m) * dx)
    clear = road_d < 22
    S = PL["sites"]
    for p in S["town"]:
        clear |= np.hypot(X - p[0], Y - p[1]) < 150
    for p in S["landmark"]:
        clear |= np.hypot(X - p[0], Y - p[1]) < 45
    forest &= ~clear
    n4 = rng.uniform(0, 1, (N, N))
    tree = forest & (ndimage.maximum_filter(n4, 3) == n4)
    pine = np.clip(_ss(z, 120, 300) + 0.5 * moor, 0, 1)
    th = (11 + 9 * n3 + 6 * pine) * tree
    leaf = np.array([0.055, 0.14, 0.04])[None, None] * (0.7 + 0.7 * n3[..., None])
    needle = np.array([0.03, 0.085, 0.045])[None, None] * (0.7 + 0.6 * n3[..., None])
    tcol = leaf * (1 - pine[..., None]) + needle * pine[..., None]
    alb[forest] = np.array([0.045, 0.075, 0.03]) * (0.8 + 0.4 * n2[forest][..., None])
    alb[tree] = tcol[tree] * 1.25
    road = road_d < 3.5
    alb[road & land & ~lake] = np.array([0.24, 0.19, 0.13])
    H = z + th

    stone, roof = np.array([0.42, 0.40, 0.37]), np.array([0.33, 0.13, 0.07])
    tcolr = dict(field=stone, lake=np.array([0.80, 0.80, 0.76]), forest=stone * 0.8,
                 desert=np.array([0.62, 0.40, 0.24]), ash=np.array([0.16, 0.14, 0.14]), moor=stone * 1.1)

    def block(p, hx, hy, top, col, rot=0.0):
        u = (X - p[0]) * np.cos(rot) + (Y - p[1]) * np.sin(rot)
        v = -(X - p[0]) * np.sin(rot) + (Y - p[1]) * np.cos(rot)
        m = (np.abs(u) < hx) & (np.abs(v) < hy)
        H[m] = np.maximum(H[m], top)
        alb[m] = col

    def drum(p, r, top, col):
        m = np.hypot(X - p[0], Y - p[1]) < r
        H[m] = np.maximum(H[m], top)
        alb[m] = col

    g_at = lambda p: z[int(p[1] / dx), int(p[0] / dx)]
    R = W["rooms"]
    for i, p in enumerate(S["town"]):
        col = tcolr[R["theme"][i]]
        for _ in range(34):
            q = p + rng.normal(0, 60, 2)
            ii, jj = int(q[1] / dx), int(q[0] / dx)
            if lake[ii, jj] or riv[ii, jj] or not land[ii, jj] or road[ii, jj]:
                continue
            rot = rng.uniform(0, 3.14)
            hx, hy = rng.uniform(7, 11), rng.uniform(5, 8)
            block(q, hx, hy, z[ii, jj] + 7, col * 1.1, rot)
            block(q, hx * 0.6, hy * 0.5, z[ii, jj] + 12, roof, rot)
        drum(p + np.array([18.0, 12.0]), 6, g_at(p) + 30, col)
    for i, p in enumerate(S["landmark"]):
        col, g = tcolr[R["theme"][i]], g_at(p)
        drum(p, 15, g + 12, col * 0.85)
        drum(p, 10, g + 44, col)
        drum(p, 6, g + 56, roof if R["theme"][i] not in ("ash", "desert") else col * 0.7)
    for d, p in zip(R["doors"], S["door"]):
        if d["kind"] != "door":
            continue
        g = g_at(p)
        for a in (0.0, np.pi):
            drum(p + 14 * np.array([np.cos(a), np.sin(a)]), 2.6, g + 11, stone)

    water = (lake | ~land | riv) & (H <= z + 0.01)
    H[~land] = 0.0
    alb[~land] = np.array([0.02, 0.07, 0.11])
    alb[lake | riv] = np.array([0.03, 0.09, 0.12])
    return dict(H=H.astype(np.float32), ground=z.astype(np.float32), alb=alb.astype(np.float32),
                emis=emis.astype(np.float32), water=water, dx=dx, n=N, L=N * dx)
