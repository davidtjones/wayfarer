"""Stage 1a: brief -> macro layout.

The brief names beats and relationships, never coordinates. This module turns it
into a seeded macro design: where the lake, the valleys, the massifs and the
themed regions sit, plus the design fields the land-forming step consumes
(uplift, rock hardness, soil mobility, pinned water).

Everything here is big-shape design (500 m and up). Nothing here is meant to
look natural yet.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from noise import fbm

BRIEF = {
    "world_km": 8.2,
    "sunset": "west",
    "journey": [
        ("start", "enclosed high side valley, no view of the basin"),
        ("overlook", "valley bend: the long lake opens to the west, toward sunset"),
        ("town", "at the lake outlet, the far end of that view"),
    ],
    "landmarks": {
        "horn": "one dominant snow peak, north of the lake, seen from most of the world",
        "volcano": "cone in the south-east, hidden from the first overlook",
        "keep": "on a crag on the north shore, stacked under the horn as seen from town",
        "island": "ruin on an island in the lake, on the axis of the first view",
        "lighthouse": "on the west coast, beyond the town, against the sunset",
    },
    "regions": ["alpine", "pine highland", "lake basin", "ashland", "mesa desert", "coast marsh"],
    "held_back": {"desert": "only revealed from the south pass", "volcano": "revealed on the lake"},
}


@dataclass
class Layout:
    seed: int
    n: int
    dx: float
    pts: dict = field(default_factory=dict)     # named design points, metres (x east, y north)
    w: np.ndarray = None                        # unit vector down-lake (west-ish)
    nn: np.ndarray = None                       # unit vector north-ish
    lake_z: float = 14.0
    f: dict = field(default_factory=dict)       # design fields, n x n

    def grid(self):
        c = (np.arange(self.n) + 0.5) * self.dx
        return np.meshgrid(c, c)                # X, Y with row = y


def _gauss(X, Y, p, su, sv, u, power=2.0):
    """Anisotropic bump centred on p; su along unit vector u, sv across it."""
    v = np.array([-u[1], u[0]])
    du = (X - p[0]) * u[0] + (Y - p[1]) * u[1]
    dv = (X - p[0]) * v[0] + (Y - p[1]) * v[1]
    return np.exp(-(np.abs(du / su) ** power + np.abs(dv / sv) ** power))


def _seg_dist(X, Y, a, b):
    """Distance to segment ab and the 0..1 position along it."""
    ab = b - a
    t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / (ab @ ab), 0, 1)
    return np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1])), t


def make_layout(seed, n=512, L=8192.0):
    rng = np.random.default_rng(seed)
    lay = Layout(seed=seed, n=n, dx=L / n)
    X, Y = lay.grid()
    mid = np.array([L / 2, L / 2])
    P = lay.pts

    # --- the spine of the brief: lake axis points at the sunset -------------
    ang = np.radians(180 + rng.uniform(-14, 14))
    w = np.array([np.cos(ang), np.sin(ang)])
    nn = np.array([-w[1], w[0]])
    if nn[1] < 0:
        nn = -nn
    lay.w, lay.nn = w, nn
    c = mid + np.array([rng.uniform(100, 350), rng.uniform(-200, 200)])
    a, b = rng.uniform(900, 1050), rng.uniform(400, 480)
    side = rng.choice([-1.0, 1.0])
    P["lake"] = c
    P["lake_e"] = c - w * a
    P["lake_w"] = c + w * a
    P["bend"] = P["lake_e"] - w * rng.uniform(760, 860) + nn * rng.uniform(-60, 60)      # the scarp lip
    P["start"] = P["bend"] - w * rng.uniform(260, 380) + nn * side * rng.uniform(800, 950)
    P["coast_w"] = P["lake_w"] + w * rng.uniform(950, 1200)
    P["horn"] = c + nn * rng.uniform(2000, 2250) + w * rng.uniform(-300, 500)
    P["crag"] = c + nn * (b + rng.uniform(330, 420)) + w * rng.uniform(100, 500)
    P["volcano"] = c - nn * rng.uniform(2150, 2400) - w * rng.uniform(1500, 1900)
    P["desert"] = c - nn * rng.uniform(2250, 2500) + w * rng.uniform(500, 1100)
    P["pass"] = c - nn * rng.uniform(1200, 1300) + w * rng.uniform(700, 1300)
    P["island"] = c + w * rng.uniform(-250, 250) + nn * rng.uniform(-60, 60)

    # --- coast: an island with a western bay that brings the sea near town --
    th = np.arctan2(Y - mid[1], X - mid[0])
    r = np.hypot(X - mid[0], Y - mid[1])
    k = rng.uniform(-1, 1, 6)
    ph = rng.uniform(0, 2 * np.pi, 6)
    R = 3650 * (1 + sum(0.035 * k[i] * np.cos((i + 2) * th + ph[i]) for i in range(6)))
    cw = P["coast_w"] - mid
    dth = np.angle(np.exp(1j * (th - np.arctan2(cw[1], cw[0]))))
    R = R - np.maximum(3650 - np.hypot(*cw), 0) * np.exp(-(dth / 0.5) ** 2)
    land = (r + 220 * fbm(n, 6, 4, rng)) < R
    land = ndimage.binary_opening(land, iterations=2)
    lab, _ = ndimage.label(land)
    land = lab == lab[int(c[1] / lay.dx), int(c[0] / lay.dx)]
    coast_d = ndimage.distance_transform_edt(land) * lay.dx

    # --- lake: long ellipse on the axis, with one island --------------------
    du = (X - c[0]) * w[0] + (Y - c[1]) * w[1]
    dv = (X - c[0]) * nn[0] + (Y - c[1]) * nn[1]
    e = np.hypot(du / a, dv / b) + 0.55 * fbm(n, 7, 3, rng)
    isl = np.hypot((X - P["island"][0]) / 95, (Y - P["island"][1]) / 70)
    lake = (e < 1.0) & (isl > 1.0) & land

    # --- themed regions as soft weights -------------------------------------
    alpine = _gauss(X, Y, P["horn"] + nn * 300, 1900, 1050, w)
    east = _gauss(X, Y, P["bend"] - w * 900, 1500, 1700, w)
    ash = _gauss(X, Y, P["volcano"], 1250, 1250, w)
    desert = _gauss(X, Y, P["desert"], 1500, 1150, w, power=3)
    basin = _gauss(X, Y, c, a * 1.5, b * 2.3, w, power=3)

    # --- uplift: where the land wants to be high ----------------------------
    # (drawn on a bent copy of the grid, so no massif is a clean ellipse or line)
    Xs, Ys = X, Y
    X = Xs + 420 * fbm(n, 4, 3, rng)
    Y = Ys + 420 * fbm(n, 4, 3, rng)
    wob = 0.75 + 0.5 * fbm(n, 5, 4, rng)
    U = 1.8e-4 * np.ones((n, n))
    ha_ = np.arctan2(Y - P["horn"][1], X - P["horn"][0])
    arms = 1 + 0.55 * np.cos(3 * ha_ + rng.uniform(0, 6.3)) + 0.25 * np.cos(5 * ha_ + rng.uniform(0, 6.3))
    U += 1.5e-3 * _gauss(X, Y, P["horn"], 420, 420, w) * arms                # the horn, with ridges
    U += 5.5e-4 * _gauss(X, Y, P["horn"] + nn * 350, 1700, 700, w)           # its range
    U += 9.0e-4 * _gauss(X, Y, P["bend"] - w * 700, 1500, 1500, w, power=3)  # east highland
    rim = _gauss(X, Y, c - nn * 1250 + w * 1100, 1150, 360, w, power=3)      # high only in the west half
    U += 6.0e-4 * rim * (1 - 0.6 * _gauss(X, Y, P["pass"], 230, 800, w))    # south rim + pass
    U += 4.5e-4 * _gauss(X, Y, P["crag"], 170, 130, w)                       # crag for the keep
    U += 3.0e-4 * _gauss(X, Y, P["volcano"], 800, 800, w)
    U *= 1 - 0.55 * desert                                                   # desert floor stays low
    U *= wob * 0.5
    X, Y = Xs, Ys
    U *= 1 - 0.8 * basin                                                     # keep the bowl low
    out_d, _ = _seg_dist(X, Y, P["lake_w"], P["coast_w"])
    U *= 1 - 0.7 * np.exp(-(out_d / 500) ** 2)                               # low outlet plain
    pass_d, _ = _seg_dist(X, Y, P["pass"] + nn * 600, P["pass"] - nn * 700)
    U *= 1 - 0.8 * np.exp(-(pass_d / 280) ** 2)                              # the one gap in the south rim
    gap_d, _ = _seg_dist(X, Y, c - w * 500 - nn * 300, P["volcano"])
    U *= 1 - 0.75 * np.exp(-(gap_d / 420) ** 2)                              # window to the volcano
    U *= np.maximum(np.clip(coast_d / 260, 0.3, 1), rim > 0.35)   # (the rim meets the sea as cliffs)
    #                                   # land tapers to shore

    # --- rock hardness and soil mobility: regional character ----------------
    K = 4.0e-5 * (1 - 0.45 * alpine) * (1 + 0.5 * basin) * (1 - 0.2 * desert)
    # hard-rock band east of the lake: streams stall on it, so the upland behind
    # stays perched and its lip is a sharp convex edge (hidden before, open after)
    hub = P["lake_e"] + w * 600
    hr = np.hypot(X - hub[0], Y - hub[1])
    ha = np.abs(np.angle(np.exp(1j * (np.arctan2(Y - hub[1], X - hub[0]) - np.arctan2(-w[1], -w[0])))))
    scarp = np.exp(-((hr - 1150) / 130) ** 2) * np.clip((1.25 - ha) / 0.3, 0, 1)
    K = K * (1 - 0.94 * scarp)
    K = K * np.exp(0.55 * fbm(n, 12, 4, rng) + 0.25 * rng.normal(size=(n, n)))   # uneven rock
    soil = np.clip(0.50 - 0.42 * np.maximum(alpine, 0.8 * east) - 0.35 * desert, 0.05, 0.5)
    talus = 0.85 + alpine * (0.35 + 0.5 * fbm(n, 14, 3, rng)) + 0.5 * desert                             # max stable slope

    # --- first surface: uplift sketch with the designed valleys cut in ------
    z0 = lay.lake_z + 15 + 2.2e5 * ndimage.gaussian_filter(U, 6) + 6 * fbm(n, 24, 3, rng)
    for (p, q, zp, zq, wd) in [
        (P["start"], P["bend"], 62, 45, 90),          # upper valley: the walk-out
        (P["bend"], P["lake_e"], 45, 2, 130),         # lower valley: the view corridor
        (P["lake_w"], P["coast_w"], -2, -lay.lake_z + 2, 120),  # outlet river
    ]:
        d, t = _seg_dist(X, Y, p, q)
        line = lay.lake_z + zp + t * (zq - zp) + 0.30 * np.maximum(d - wd * 0.3, 0)
        z0 = np.where(d < 900, np.minimum(z0, line), z0)

    lay.f = dict(land=land, lake=lake, U=U, K=K, soil=soil, talus=talus, z0=z0,
                 scarp=scarp, alpine=alpine, east=east, ash=ash, desert=desert, basin=basin, coast_d=coast_d)
    return lay
