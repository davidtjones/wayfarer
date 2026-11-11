"""Write a rooms world as a game3d world directory (see game3d/src/world.rs).

    pixi run python export_game3d.py <seed> <name> [key=value ...]

Writes ../../game3d/assets/terrain/<name>/ : terrain.raster, lake.raster, tint.png,
world.txt (structures, placeholder trees, shots) and report.json (the measurements).
"""
import json
import os
import struct
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from rrun import build, parse
from rmap import diag
from place import ground
from dress import _ss
from noise import fbm

ONLY = [int(v) for a in sys.argv if a.startswith("--doors=") and a != "--doors=hub" for v in a[8:].split(",")]
HUB = "--doors=hub" in sys.argv                         # only the doors of room 0
PAD = 6144.0          # half side of the exported square; the island is 4096, the rest is sea
STEP = 8.0


def _raster(path, a):
    a = np.ascontiguousarray(a, "<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<8sIIff", b"RASTER01", a.shape[1], a.shape[0], STEP, 0.0))
        f.write(a.tobytes())


def export(W, PL, M, out):
    os.makedirs(out, exist_ok=True)
    R, S = W["rooms"], PL["sites"]
    n, dx = W["n"], W["dx"]
    L = n * dx
    half = L / 2
    rng = np.random.default_rng(W["seed"] + 41)
    gx = lambda p: float(p[0] - half)                       # game X east
    gz = lambda p: float(half - p[1])                       # game Z south

    # --- heights on the game grid: nodes every 8 m, north row first ----------
    z = W["z"].copy()
    lake, lz = W["lake"], W["lake_zmap"]
    z[lake] = lz[lake] - 2.5 - 4 * np.clip(ndimage.distance_transform_edt(lake)[lake] * dx / 60, 0, 1)
    sea_d = ndimage.distance_transform_edt(~W["land"]) * dx
    z[~W["land"]] = -1.5 - 30 * np.clip(sea_d[~W["land"]] / 400, 0, 1)
    z = np.minimum(z, 1000.0)
    m = int(2 * PAD / STEP) + 1
    c = np.arange(m) * STEP - PAD                           # game coordinate of each node
    col = (c + half) / dx - 0.5                             # array column for game x
    row = (half - c) / dx - 0.5                             # array row for game z
    rr, cc = np.meshgrid(row, col, indexing="ij")
    hz = ndimage.map_coordinates(z, [rr, cc], order=1, mode="constant", cval=-31.5)
    _raster(f"{out}/terrain.raster", hz)
    lvl = np.where(lake, lz, -1000.0)
    lvl = ndimage.maximum_filter(lvl, 5)
    _raster(f"{out}/lake.raster", ndimage.map_coordinates(lvl, [rr, cc], order=0, mode="constant", cval=-1000.0))

    # --- ground colour by room theme -----------------------------------------
    n1, n2 = fbm(n, 10, 4, rng), fbm(n, 60, 3, rng)
    rgba = np.zeros((n, n, 4))

    def lay(colour, a):
        a = np.clip(a, 0, 1)[..., None]
        rgba[..., :3] = rgba[..., :3] * (1 - a) + np.array(colour) * a
        rgba[..., 3:] = rgba[..., 3:] * (1 - a) + a

    zz, slope = W["z"], W["slope"]
    # grass is only drawn as blades near the camera, so the ground itself must be green
    green = np.array([0.36, 0.50, 0.20])[None, None] * (0.9 + 0.25 * n1[..., None]) \
        + np.array([0.10, 0.04, -0.02]) * _ss(W["field"], 0.3, 0.7)[..., None]
    rgba[..., :3] = green
    rgba[..., 3] = 0.88
    lay((0.20, 0.31, 0.14), 0.75 * _ss(W["forest"] + 0.2 * n1, 0.4, 0.7))
    lay((0.42, 0.40, 0.24), 0.85 * _ss(W["moor"] + 0.2 * n1, 0.45, 0.7))
    lay((0.42, 0.40, 0.25), 0.8 * _ss(zz, 300, 440))
    band = 0.5 + 0.5 * np.sin((zz + 25 * n1) / 6.0)
    sand = np.array([0.80, 0.60, 0.38])[None, None] * (0.92 + 0.16 * n2[..., None])
    strata = np.array([0.62, 0.33, 0.20]) * (1 - band[..., None]) + np.array([0.78, 0.58, 0.40]) * band[..., None]
    st = _ss(slope, 0.4, 0.75)[..., None]
    des = _ss(W["desert"] + 0.15 * n1, 0.45, 0.65)
    dcol = sand * (1 - st) + strata * st
    rgba[..., :3] = rgba[..., :3] * (1 - des[..., None]) + dcol * des[..., None]
    rgba[..., 3] = rgba[..., 3] * (1 - des) + 1.0 * des
    lay((0.20, 0.19, 0.19), 0.97 * _ss(W["ash"] + 0.2 * n1, 0.45, 0.7))
    gy_, _ = np.gradient(zz, dx)
    snowline = 520 + 60 * n1 + 80 * gy_ / (slope + 1e-6)
    lay((0.93, 0.95, 0.98), 0.97 * _ss(zz, snowline, snowline + 60) * (1 - _ss(slope, 1.0, 1.4)))
    lake_near = ndimage.binary_dilation(lake, iterations=3)
    lay((0.70, 0.64, 0.48), 0.9 * ((zz < 2.5) | (lake_near & ~lake)))
    road = np.zeros((n, n), bool)
    for line in PL["roads"]:
        if line is not None:
            road[(line[:, 1] / dx).astype(int), (line[:, 0] / dx).astype(int)] = True
    lay((0.46, 0.38, 0.27), 0.85 * ndimage.gaussian_filter(road.astype(float), 0.7) * 2.2)
    img = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)[::-1]            # north row first
    Image.fromarray(img, "RGBA").save(f"{out}/tint.png")

    # --- structures, trees, shots --------------------------------------------
    lines = [f"extent {half}", "sea 0"]
    stone, roofc = (0.55, 0.53, 0.50), (0.42, 0.17, 0.10)
    tcol = dict(field=stone, lake=(0.86, 0.86, 0.82), forest=(0.44, 0.42, 0.38), desert=(0.72, 0.48, 0.30),
                ash=(0.17, 0.15, 0.15), moor=(0.60, 0.58, 0.54))
    P = lambda p, dy=0.0: f"{gx(p):.1f} {ground(W, p) + dy:.1f} {gz(p):.1f}"
    C = lambda c: " ".join(f"{v:.2f}" for v in c)
    for i, p in enumerate(S["town"]):
        col_ = tcol[R["theme"][i]]
        for _ in range(26):
            q = p + rng.normal(0, 55, 2)
            ii, jj = int(q[1] / dx), int(q[0] / dx)
            if lake[ii, jj] or not W["land"][ii, jj] or road[ii, jj] or slope[ii, jj] > 0.3:
                continue
            hx, hy, rot = rng.uniform(4.5, 7.5), rng.uniform(3.5, 5), rng.uniform(0, 3.14)
            lines.append(f"box {P(q, -1.0)} {hx:.1f} 3.5 {hy:.1f} {rot:.2f} {C(col_)}")
            lines.append(f"box {P(q, 5.5)} {hx * 1.08:.1f} 1.3 {hy * 0.7:.1f} {rot:.2f} {C(roofc)}")
        t = p + np.array([16.0, 10.0])
        lines.append(f"cyl {P(t, -1)} 4.5 26 {C(col_)}")
        lines.append(f"cone {P(t, 25)} 5.5 9 {C(roofc)}")
    for i, p in enumerate(S["landmark"]):
        col_ = tcol[R["theme"][i]]
        lines.append(f"cyl {P(p, -2)} 15 14 {C(tuple(0.85 * v for v in col_))}")
        lines.append(f"cyl {P(p, 10)} 9.5 40 {C(col_)}")
        lines.append(f"cone {P(p, 50)} 11.5 20 {C(roofc if R['theme'][i] not in ('ash', 'desert') else col_)}")
    for d, p in zip(R["doors"], S["door"]):
        if d["kind"] == "door":
            for a in (0.0, np.pi):
                q = p + 13 * np.array([np.cos(a), np.sin(a)])
                lines.append(f"cyl {P(q, -1)} 2.4 11 {C(stone)}")
    if "volcano" in W["pts"]:
        v = W["pts"]["volcano"]
        lines.append(f"cyl {P(v, 3)} {W['pts']['volcano_r'] * 0.85:.1f} 2 1.00 0.35 0.05")

    town_d = np.min([np.hypot(*(np.indices((n, n))[::-1] * dx + dx / 2 - np.array(p)[:, None, None])) for p in S["town"]], 0)
    dens = np.clip(0.02 + 0.30 * W["forest"] + 0.03 * W["field"], 0, 0.4) * (W["desert"] < 0.3) * (W["ash"] < 0.25) \
        * (1 - 0.9 * _ss(W["moor"], 0.4, 0.7)) * _ss(n1, -0.5, 0.1)
    ok = W["land"] & ~lake & ~lake_near & (zz > 3) & (zz < 400) & (slope < 0.8) & (town_d > 110) \
        & ~ndimage.binary_dilation(road, iterations=2)
    ti, tj = np.nonzero(ok & (rng.uniform(0, 1, (n, n)) < dens))
    for i, j in zip(ti, tj):
        q = np.array([(j + rng.uniform(0.1, 0.9)) * dx, (i + rng.uniform(0.1, 0.9)) * dx])
        if np.min(np.hypot(*(np.array(S["town"]) - q).T)) > 260 or rng.uniform() < 0.3:
            lines.append(f"tree {P(q, -0.3)} {rng.uniform(9, 19):.1f}")

    def shot(name, eye_xy, look_xy, up=7.0, pitch=-1.0, fov=70.0, sun=None, eye_z=None, reach=None):
        f = np.array(look_xy[:2], float) - np.array(eye_xy[:2], float)
        f /= np.linalg.norm(f) + 1e-9
        t = np.tan(np.radians(pitch))
        d = np.array([f[0], t, -f[1]])
        d /= np.linalg.norm(d)
        ey = ground(W, eye_xy) + up if eye_z is None else eye_z
        head = np.degrees(np.arctan2(d[0], d[2]))
        az, el = sun if sun else (head + 180 + 60, 30.0)
        s = f"shot {name} {gx(eye_xy):.1f} {ey:.1f} {gz(eye_xy):.1f} {d[0]:.4f} {d[1]:.4f} {d[2]:.4f} {fov} {az % 360:.1f} {el}"
        lines.append(s + (f" {reach:.0f}" if reach else ""))

    lab = W["label"]
    shots = []
    shot("00_overview", (half, -2300.0), (half, half), eye_z=4300.0, pitch=-40.0, fov=58.0, sun=(35.0, 42.0), reach=6500)
    for k, (d, road_) in enumerate(zip(R["doors"], PL["roads"])):
        if road_ is None or (ONLY and k not in ONLY) or (HUB and 0 not in (d["a"], d["b"])):
            continue
        li = lab[(road_[:, 1] / dx).astype(int), (road_[:, 0] / dx).astype(int)]
        cross = np.nonzero(li == d["b"])[0]
        if len(cross) == 0:
            continue
        c0 = int(cross[0])
        f = road_[min(c0 + 12, len(road_) - 1)] - road_[max(c0 - 12, 0)]
        sun = ((np.degrees(np.arctan2(f[0], -f[1])) + 180 + 60) % 360, 30.0)
        ta, tb = R["theme"][d["a"]], R["theme"][d["b"]]
        for tag, off in (("1before", -9), ("2door", 0), ("3after", 9)):
            i = int(np.clip(c0 + off, 0, len(road_) - 14))
            name = f"door{k:02d}_{ta}{d['a']}_to_{tb}{d['b']}_{tag}"
            shot(name, road_[i], road_[min(i + 12, len(road_) - 1)], sun=sun)
            shots.append(name)
    for i, p in enumerate(S["town"]):
        v = S["landmark"][i] - p
        e = p - 210 * v / (np.linalg.norm(v) + 1e-9)
        ii, jj = int(np.clip(e[1] / dx, 0, n - 1)), int(np.clip(e[0] / dx, 0, n - 1))
        if not W["land"][ii, jj] or lake[ii, jj]:
            e = p
        shot(f"room{i}_{R['theme'][i]}_town", e, S["landmark"][i], up=14.0, pitch=0.0)
    open(f"{out}/world.txt", "w").write("\n".join(lines) + "\n")
    diag(W, PL).save(f"{out}/map.png")
    Mj = dict(M)
    Mj["themes"] = R["theme"]
    Mj["floor_heights"] = [float(v) for v in R["h"]]
    json.dump(Mj, open(f"{out}/report.json", "w"), indent=1)
    print(f"  wrote {out}: {m}x{m} raster, {len(ti)} trees, {sum(l.startswith('shot') for l in lines)} shots, "
          f"heights {hz.min():.0f}..{hz.max():.0f} m")


if __name__ == "__main__":
    seed, name = int(sys.argv[1]), sys.argv[2]
    W, PL, M = build(seed, parse([a for a in sys.argv[3:] if "=" in a and not a.startswith("--")]))
    here = os.path.dirname(os.path.abspath(__file__))
    export(W, PL, M, os.path.join(here, "..", "..", "game3d", "assets", "terrain", name))
