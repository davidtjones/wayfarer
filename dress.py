"""Dress the land for viewing: ground cover, trees, water, snow, built landmarks.

Rules only: what grows or lies where follows slope, height, water and region.
Trees and buildings are raised into the render surface so they cast shadows and
make silhouettes. This is a viewing aid for judging the world, not game assets.
"""
import numpy as np
from scipy import ndimage
from noise import fbm


def _ss(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def dress(W, PL, up=2):
    rng = np.random.default_rng(W["seed"] + 11)
    N, dx = W["n"] * up, W["dx"] / up
    zo = lambda a, o=1: ndimage.zoom(np.asarray(a, np.float64), up, order=o)
    z = zo(W["z"])
    land, lake = zo(W["land"]) > 0.5, zo(W["lake"]) > 0.5
    gy, gx = np.gradient(z, dx)
    rough = np.clip(np.hypot(gx, gy) / 0.8, 0.08, 1)
    z = z + rough * (3.0 * fbm(N, 200, 3, rng) + 0.5 * rng.normal(size=(N, N))) * land   # outcrops and tussocks
    z[lake] = W["lake_z"]
    z[~land] = -6.0
    gy, gx = np.gradient(ndimage.gaussian_filter(z, 1.5), dx)
    slope = np.hypot(gx, gy)
    alpine, east, ash_w, des_w, basin = (zo(W[k]) for k in ("alpine", "east", "ash", "desert", "basin"))
    c = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(c, c)
    n1, n2, n3 = fbm(N, 10, 4, rng), fbm(N, 48, 3, rng), rng.uniform(0, 1, (N, N))
    vr = np.hypot(X - PL["sites"]["volcano"][0], Y - PL["sites"]["volcano"][1])
    ash = np.maximum(_ss(ash_w + 0.25 * n1, 0.4, 0.75), _ss(1450 - vr + 350 * n1, 0, 500))
    desert = _ss(des_w, 0.32, 0.5) * (1 - ash)
    north = -gy / (slope + 1e-6)                           # +1 on north-facing ground

    # rivers: wider where they carry more
    area = W["area"]
    riv = np.zeros_like(area, bool)
    for a_min, grow in ((2.5e5, 0), (1.5e6, 1), (6e6, 2)):
        m = area > a_min
        riv |= ndimage.binary_dilation(m, iterations=grow) if grow else m
    riv = (zo(riv) > 0.55) & land & ~lake

    # --- ground colour ------------------------------------------------------
    dry = _ss(n1, -0.2, 0.5)
    grass = np.stack([0.06 + 0.08 * dry, 0.11 + 0.035 * dry, 0.03 + 0.02 * dry], -1)
    tundra = np.array([0.16, 0.15, 0.08])
    hi = _ss(z, 380, 560)[..., None]
    grass = grass * (1 - hi) + tundra * hi
    rock = np.array([0.30, 0.28, 0.26]) * (0.8 + 0.4 * n3[..., None])
    alb = grass.copy()
    w = _ss(slope + 0.25 * n2, 0.95, 1.25)[..., None]
    alb = alb * (1 - w) + rock * w
    band = 0.5 + 0.5 * np.sin((z + 30 * n1) / 7.0)
    strata = np.array([0.36, 0.13, 0.06]) * (1 - band[..., None]) + np.array([0.50, 0.33, 0.19]) * band[..., None]
    sand = np.array([0.36, 0.21, 0.10]) * (0.9 + 0.2 * n2[..., None])
    dcol = sand * (1 - _ss(slope, 0.35, 0.7)[..., None]) + strata * _ss(slope, 0.35, 0.7)[..., None]
    top = (zo(W["butte"]) > 0.9)[..., None] * (1 - _ss(slope, 0.2, 0.5)[..., None])
    dcol = dcol * (1 - top) + np.array([0.33, 0.17, 0.09]) * top
    alb = alb * (1 - desert[..., None]) + dcol * desert[..., None]
    acol = np.array([0.075, 0.07, 0.068]) * (0.7 + 0.6 * n3[..., None])
    alb = alb * (1 - ash[..., None]) + acol * ash[..., None]
    snowline = 640 + 70 * n1 - 90 * north
    snow = _ss(z, snowline, snowline + 70) * (1 - _ss(slope + 0.2 * n2, 0.85, 1.2)) * (1 - ash)
    alb = alb * (1 - snow[..., None]) + np.array([0.86, 0.89, 0.94]) * snow[..., None]
    beach = land & ~lake & ((z < 2.2) | ((z < W["lake_z"] + 1.2) & (ndimage.binary_dilation(lake, iterations=5))))
    alb[beach] = np.array([0.36, 0.31, 0.22])
    emis = np.zeros((N, N, 3))
    lava = (vr < 95) | ((ash > 0.5) & riv & (vr < 900) & (vr > 60))
    emis[lava] = np.array([5.0, 1.1, 0.12])
    riv &= ~(ash > 0.5)

    # --- trees: raised into the surface, so forests have edges and shadows --
    treeline = 470 + 60 * n1
    moist = _ss(n1 + 0.4 * basin - 0.5 * east * (north < -0.2), -0.35, 0.1)
    forest = land & ~lake & ~beach & ~riv & (slope < 1.0) & (slope > 0.04) & (z < treeline) \
        & (moist > 0.5) & (desert < 0.3) & (ash < 0.25) & (n3 < 0.8)
    road_d = np.full((N, N), 1e9)
    for line in (PL["route"], PL["leg2"], PL["leg3"]):
        m = np.ones((N, N), bool)
        m[(line[:, 1] / dx).astype(int), (line[:, 0] / dx).astype(int)] = False
        road_d = np.minimum(road_d, ndimage.distance_transform_edt(m) * dx)
    clear = road_d < 26
    for k, r in (("overlook", 120), ("town", 170), ("keep", 70), ("start", 50), ("pass", 80),
                 ("island", 60), ("lighthouse", 50), ("oasis", 80)):
        p = PL["sites"][k]
        clear |= np.hypot(X - p[0], Y - p[1]) < r
    forest &= ~clear
    pine = _ss(z, 150, 330) * 0.7 + 0.5 * east
    n4 = rng.uniform(0, 1, (N, N))
    tree = forest & (ndimage.maximum_filter(n4, 3) == n4)   # one tree per ~12 m, never two touching
    th = (11 + 9 * n3 + 6 * np.clip(pine, 0, 1)) * tree
    leaf = np.array([0.055, 0.14, 0.04])[None, None] * (0.7 + 0.7 * n3[..., None])
    needle = np.array([0.03, 0.085, 0.045])[None, None] * (0.7 + 0.6 * n3[..., None])
    tcol = leaf * (1 - np.clip(pine, 0, 1)[..., None]) + needle * np.clip(pine, 0, 1)[..., None]
    alb[forest] = np.array([0.045, 0.075, 0.03]) * (0.8 + 0.4 * n2[forest][..., None])   # shaded floor
    alb[tree] = tcol[tree] * 1.25

    road = road_d < 3.5
    alb[road & land & ~lake] = np.array([0.24, 0.19, 0.13])
    H = z + th

    # --- built landmarks: oversized on purpose, like game landmarks ---------
    stone, roof, white = np.array([0.42, 0.40, 0.37]), np.array([0.33, 0.13, 0.07]), np.array([0.85, 0.85, 0.82])

    def pad(p, r):
        d = np.hypot(X - p[0], Y - p[1])
        g = z[int(p[1] / dx), int(p[0] / dx)]
        k = _ss(d, r * 1.6, r * 0.8) * (land & ~lake)
        H[:] = H * (1 - k) + (g + (H - z) * 0) * k
        return g

    def block(p, hx, hy, top, col, rot=0.0):
        u = (X - p[0]) * np.cos(rot) + (Y - p[1]) * np.sin(rot)
        v = -(X - p[0]) * np.sin(rot) + (Y - p[1]) * np.cos(rot)
        m = (np.abs(u) < hx) & (np.abs(v) < hy)
        H[m] = np.maximum(H[m], top)
        alb[m] = col
        emis[m] = 0

    def drum(p, r, top, col):
        m = np.hypot(X - p[0], Y - p[1]) < r
        H[m] = np.maximum(H[m], top)
        alb[m] = col

    S = PL["sites"]
    g = pad(S["keep"], 75)
    for a in np.linspace(0, 2 * np.pi, 5)[:-1] + 0.4:
        drum(S["keep"] + 62 * np.array([np.cos(a), np.sin(a)]), 11, g + 34, stone)
    block(S["keep"], 50, 50, g + 18, stone, 0.4)
    block(S["keep"], 40, 40, g + 8, stone * 0.8, 0.4)
    block(S["keep"], 18, 14, g + 46, stone, 0.4)
    drum(S["keep"], 13, g + 88, stone)
    drum(S["keep"], 8, g + 100, roof)
    td = np.hypot(X - S["town"][0], Y - S["town"][1])
    fields = land & ~lake & ~riv & (td < 420) & (slope < 0.22) & (H <= z + 0.01)
    patch = np.floor(X / 46) + 7 * np.floor(Y / 38)
    fcol = np.stack([0.24 + 0.10 * np.sin(patch * 1.7), 0.21 + 0.05 * np.sin(patch * 2.3), 0.07 + 0 * X], -1)
    alb[fields] = fcol[fields]
    g = pad(S["town"], 150)
    for _ in range(70):
        q = S["town"] + rng.normal(0, 85, 2)
        if lake[int(q[1] / dx), int(q[0] / dx)] or riv[int(q[1] / dx), int(q[0] / dx)]:
            continue
        gq = z[int(q[1] / dx), int(q[0] / dx)]
        rot = rng.uniform(0, 3.14)
        hx, hy = rng.uniform(7, 11), rng.uniform(5, 8)
        block(q, hx, hy, gq + 7, np.array([0.50, 0.44, 0.34]), rot)
        block(q, hx * 0.6, hy * 0.5, gq + 12, roof, rot)
    block(S["town"], 16, 10, g + 16, stone)
    block(S["town"], 11, 6, g + 23, roof)
    drum(S["town"] + np.array([20, 0]), 6, g + 40, stone)
    g = pad(S["island"], 30)
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        if rng.uniform() < 0.8:
            drum(S["island"] + 24 * np.array([np.cos(a), np.sin(a)]), 4.5, g + rng.uniform(10, 22), stone)
    drum(S["island"], 8, g + 46, stone)
    g = z[int(S["overlook"][1] / dx), int(S["overlook"][0] / dx)]
    for a in np.linspace(0, 2 * np.pi, 8)[:-1]:
        drum(S["overlook"] - W["w"] * 34 + 13 * np.array([np.cos(a), np.sin(a)]), 2.2, g + 7.5, stone * 0.9)
    g = pad(S["lighthouse"], 16)
    drum(S["lighthouse"], 10, g + 48, white)
    drum(S["lighthouse"], 6.5, g + 60, white)
    emis[np.hypot(X - S["lighthouse"][0], Y - S["lighthouse"][1]) < 6.5] = np.array([8.0, 6.0, 2.5])
    g = pad(S["pass"], 20)
    for k, r in enumerate((13, 9, 5)):
        block(S["pass"] + W["nn"] * 0 + np.array([22.0, 0]), r, r, g + 4 * (k + 1), stone)
    g = pad(S["oasis"], 40)
    for a in np.linspace(0, 2 * np.pi, 7)[:-1]:
        drum(S["oasis"] + 28 * np.array([np.cos(a), np.sin(a)]), 4, g + rng.uniform(9, 20), np.array([0.62, 0.40, 0.24]))

    water = (lake | ~land | riv) & (H <= z + 0.01)
    H[~land] = 0.0
    alb[~land] = np.array([0.02, 0.07, 0.11])
    alb[lake | riv] = np.array([0.03, 0.09, 0.12])
    return dict(H=H.astype(np.float32), ground=z.astype(np.float32), alb=alb.astype(np.float32),
                emis=emis.astype(np.float32), water=water, dx=dx, n=N, L=N * dx)
