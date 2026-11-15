"""Write a rooms world as a game3d world directory (see game3d/src/world/mod.rs).

    pixi run export <seed> <name> [key=value ...] [--doors=1,4 | --doors=hub]

Writes <game3d>/assets/terrain/<name>/ (workspace.py says how game3d is found):
terrain.raster, lake.raster, flow_x.raster and flow_z.raster (river current),
tint.png, cover.png, paths.png, world.txt (structures, trees, shots) and
report.json (the measurements).

cover.png says what the ground is made of, one channel per material, same grid
as tint.png: R loose sand (desert floor, beaches, lake shores), G meadow (where
grass grows), B forest floor, A bare ground (town yards, ash). tint.png stays the
colour; cover.png picks the texture and decides where grass is planted.

paths.png is the path network (roads, footpaths, trails) as a distance map at
2 m a pixel; see paths.py for its channels.

What grows where (grass, reeds, bushes, rocks, trees) is set per room theme in
biomes.toml and blended across room borders; see biomes.py. Besides cover.png's
grass channel that gives life.png (R reeds, G rocks), under.png (R ferns,
G flowers, B heather, A scrub) and shore.png (how far
above the water grass starts and where reeds stand), for the layers the game
scatters itself.

A tree line is `tree x y z height species`, species 0 spruce, 1 pine, 2 oak,
3 birch, 4 dead snag, 5 bush.
"""
import json
import os
import struct
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from . import biomes
from .build import build, parse
from .noise import fbm, smoothstep
from .paths import PIXEL, network, path_map
from .place import ground
from .preview import diag
from .shots import lake_shots, river_shots, wall_shots
from .trees import plant
from .workspace import world_dir

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
    # Rivers: a sloping level over a ribbon wider than the water; the ground hides the rest.
    lvl = np.where(lvl > -999, lvl, W["river_z"])
    _raster(f"{out}/lake.raster", ndimage.map_coordinates(lvl, [rr, cc], order=0, mode="constant", cval=-1000.0))
    # Which way a river runs and how fast, m/s along game x and z; zero on lakes.
    still = ndimage.maximum_filter(np.where(lake, 1.0, 0.0), 5) > 0
    for axis, sign, name in ((0, 1.0, "flow_x"), (1, -1.0, "flow_z")):
        v = np.where(still, 0.0, sign * W["river_flow"][..., axis])
        _raster(f"{out}/{name}.raster", ndimage.map_coordinates(v, [rr, cc], order=0, mode="constant", cval=0.0))
    lake = lake | W["river"]                                # from here on, any fresh water

    # --- ground colour by room theme -----------------------------------------
    n1, n2 = fbm(n, 10, 4, rng), fbm(n, 60, 3, rng)
    rgba = np.zeros((n, n, 4))

    def lay(colour, a):
        a = np.clip(a, 0, 1)[..., None]
        rgba[..., :3] = rgba[..., :3] * (1 - a) + np.array(colour) * a
        rgba[..., 3:] = rgba[..., 3:] * (1 - a) + a

    zz, slope = W["z"], W["slope"]
    # --- paths: roads, footpaths, trails --------------------------------------
    ways = network(W, PL, np.random.default_rng(W["seed"] + 59))
    pmap, pdist, phalf = path_map(W, ways)
    Image.fromarray(pmap[::-1], "RGBA").save(f"{out}/paths.png")
    road = np.zeros((n, n), bool)                          # roads and footpaths, on the world grid
    for w_ in ways:
        if w_["kind"] < 2:
            road[np.clip((w_["line"][:, 1] / dx).astype(int), 0, n - 1),
                 np.clip((w_["line"][:, 0] / dx).astype(int), 0, n - 1)] = True
    # --- biomes: what lives where (biomes.toml) --------------------------------
    table = biomes.load()
    weights = biomes.theme_weights(W, PL, table)
    trees, canopy, site_ = plant(W, PL, lake, road, np.random.default_rng(W["seed"] + 97), weights, table)
    # No trunk stands on a path or crowds its edge.
    ti = np.clip((trees[:, 1] / PIXEL).astype(int), 0, pdist.shape[0] - 1)
    tj = np.clip((trees[:, 0] / PIXEL).astype(int), 0, pdist.shape[1] - 1)
    trees = trees[pdist[ti, tj] > phalf[ti, tj] + 1.6]
    woods = ndimage.gaussian_filter(canopy, 2.0)
    # grass is only drawn as blades near the camera, so the ground itself must be green
    green = np.array([0.36, 0.50, 0.20])[None, None] * (0.9 + 0.25 * n1[..., None]) \
        + np.array([0.10, 0.04, -0.02]) * smoothstep(W["field"], 0.3, 0.7)[..., None]
    rgba[..., :3] = green
    rgba[..., 3] = 0.88
    lay((0.20, 0.31, 0.14), 0.75 * smoothstep(woods, 0.2, 0.7))
    lay((0.42, 0.40, 0.24), 0.85 * smoothstep(W["moor"] + 0.35 * n1, 0.25, 0.8))
    lay((0.42, 0.40, 0.25), 0.8 * smoothstep(zz, 300, 440))
    band = 0.5 + 0.5 * np.sin((zz + 25 * n1) / 6.0)
    sand = np.array([0.80, 0.60, 0.38])[None, None] * (0.92 + 0.16 * n2[..., None])
    strata = np.array([0.62, 0.33, 0.20]) * (1 - band[..., None]) + np.array([0.78, 0.58, 0.40]) * band[..., None]
    st = smoothstep(slope, 0.4, 0.75)[..., None]
    des = smoothstep(W["desert"] + 0.35 * n1, 0.25, 0.8)
    dcol = sand * (1 - st) + strata * st
    rgba[..., :3] = rgba[..., :3] * (1 - des[..., None]) + dcol * des[..., None]
    rgba[..., 3] = rgba[..., 3] * (1 - des) + 1.0 * des
    ash = smoothstep(W["ash"] + 0.35 * n1, 0.25, 0.8)
    lay((0.20, 0.19, 0.19), 0.97 * ash)
    gy_, _ = np.gradient(zz, dx)
    snowline = 520 + 60 * n1 + 80 * gy_ / (slope + 1e-6)
    snow = smoothstep(zz, snowline, snowline + 60) * (1 - smoothstep(slope, 1.0, 1.4))
    lay((0.93, 0.95, 0.98), 0.97 * snow)
    lake_near = ndimage.binary_dilation(W["lake"], iterations=3) | ndimage.binary_dilation(W["river"])
    shore = ((zz < 2.5) | (lake_near & ~lake)).astype(float)
    lay((0.70, 0.64, 0.48), 0.9 * shore)
    # Trodden ground in a town: bare earth between the houses. Paths themselves
    # are in paths.png, which is fine enough to draw them; this grid is not.
    town_d = np.min([np.hypot(*(np.indices((n, n))[::-1] * dx + dx / 2 - np.array(p)[:, None, None]))
                     for p in S["town"]], 0)
    path = smoothstep(town_d + 40 * n1, 95, 40) * 0.8
    lay((0.46, 0.38, 0.27), 0.7 * path)
    img = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)[::-1]            # north row first
    Image.fromarray(img, "RGBA").save(f"{out}/tint.png")

    # --- ground material cover -----------------------------------------------
    soft = lambda a: ndimage.gaussian_filter(np.clip(a, 0, 1), 0.8)
    c_sand = soft(np.maximum(des * (1 - st[..., 0]), shore))
    c_bare = soft(np.maximum(path, ash))
    c_forest = soft(smoothstep(woods, 0.25, 0.7) * (1 - c_sand) * (1 - c_bare))
    # Grass, reeds and rocks come from the biome table; snow and town yards are bare.
    site_["slope"] = slope
    bare = 1 - np.maximum(snow, path)
    c_grass = soft(np.clip(biomes.density(weights, table, "grass", site_), 0, 1) * bare)
    cover = np.stack([c_sand, c_grass, c_forest, c_bare], -1)
    Image.fromarray((np.clip(cover, 0, 1) * 255).astype(np.uint8)[::-1], "RGBA").save(f"{out}/cover.png")

    # --- life.png, shore.png: the layers the game scatters itself ----------------
    # life: R reeds, G rocks (densities 0..1). shore: where grass and reeds stand
    # relative to the water, in metres: R/G grass none/full above water (0..4 m),
    # B/A reed band lowest/highest about the waterline (-1..1 m).
    sea_d2 = ndimage.distance_transform_edt(W["land"]) * dx
    lake_d2 = ndimage.distance_transform_edt(~lake) * dx
    by_lake = lake_d2 < sea_d2
    g_lo, g_hi = biomes.water_rule(weights, table, "grass", "above_lake", "above_sea", by_lake, [0.2, 0.7])
    r_lo, r_hi = biomes.water_rule(weights, table, "reeds", "band", None, by_lake, [-0.3, 0.3])
    reeds = np.clip(biomes.density(weights, table, "reeds", site_), 0, 1)
    rocks = np.clip(biomes.density(weights, table, "rocks", site_), 0, 1) * (1 - 0.9 * path)
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    life = np.stack([u8(soft(reeds)), u8(soft(rocks)), np.zeros((n, n), np.uint8), np.full((n, n), 255, np.uint8)], -1)
    Image.fromarray(life[::-1], "RGBA").save(f"{out}/life.png")
    # under.png: undergrowth densities, R ferns, G flowers, B heather, A scrub.
    # Small plants need soil that stays put: none on paths, snow or steep ground.
    holds = bare * np.clip((0.75 - slope) / 0.3, 0, 1)
    under = [soft(np.clip(biomes.density(weights, table, k, site_), 0, 1) * holds)
             for k in ("ferns", "flowers", "heather", "scrub")]
    Image.fromarray(np.stack([u8(a) for a in under], -1)[::-1], "RGBA").save(f"{out}/under.png")
    shore_map = np.stack([u8(g_lo / 4), u8(g_hi / 4), u8((r_lo + 1) / 2), u8((r_hi + 1) / 2)], -1)
    Image.fromarray(shore_map[::-1], "RGBA").save(f"{out}/shore.png")

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
            if lake[ii, jj] or W["river_z"][ii, jj] > -999 or not W["land"][ii, jj] or road[ii, jj] or slope[ii, jj] > 0.3:
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

    # Conifers in wet, shaded and high stands, broadleaves on open low ground,
    # birch at edges and by water: see trees.py.
    for x, y, h, sp in trees:
        lines.append(f"tree {P((x, y), -0.3)} {h:.1f} {int(sp)}")

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
    # Art checks: a stand of trees from the air, and a cliff face from its foot.
    dens = ndimage.gaussian_filter((canopy > 0.5).astype(float), 40)
    fi, fj = np.unravel_index(np.argmax(dens), dens.shape)
    fp = np.array([(fj + 0.5) * dx, (fi + 0.5) * dx])
    shot("art_forest_air", fp + np.array([-330.0, -330.0]), fp, up=190.0, pitch=-28.0, fov=60.0)
    shot("art_forest_edge", fp + np.array([-420.0, 60.0]), fp, up=30.0, pitch=-6.0, fov=65.0)
    shot("art_forest_in", fp + np.array([-40.0, 10.0]), fp, up=2.0, pitch=2.0, fov=75.0)
    # Undergrowth and bushes: a low look across wherever each is thickest.
    def thickest(name, dens, back=14.0, up=2.6, pitch=-14.0):
        d = ndimage.gaussian_filter(np.where(W["land"] & ~lake & (slope < 0.3) & (zz > 8), dens, 0.0), 5)
        if d.max() <= 0.02:
            return
        i, j = np.unravel_index(np.argmax(d), d.shape)
        p = np.array([(j + 0.5) * dx, (i + 0.5) * dx])
        shot(name, p + np.array([-back, -back * 0.4]), p, up=up, pitch=pitch, fov=70.0)
    for name_, dens_ in zip(("ferns", "flowers", "heather", "scrub"), under):
        thickest(f"art_under_{name_}", dens_)
    bush_xy = trees[trees[:, 3] == 5][:, :2] if len(trees) else np.zeros((0, 2))
    if len(bush_xy):
        bi = np.clip((bush_xy[:, 1] / dx).astype(int), 0, n - 1)
        bj = np.clip((bush_xy[:, 0] / dx).astype(int), 0, n - 1)
        bushes_ = np.zeros((n, n))
        np.add.at(bushes_, (bi, bj), 1.0)
        thickest("art_bushes", bushes_, back=24.0, up=3.0, pitch=-7.0)
    wall = ndimage.gaussian_filter(((slope > 1.1) & W["land"] & (W["desert"] < 0.2) & (W["ash"] < 0.2)).astype(float), 12)
    wi, wj = np.unravel_index(np.argmax(wall), wall.shape)
    wp_ = np.array([(wj + 0.5) * dx, (wi + 0.5) * dx])
    gy0, gx0 = np.gradient(ndimage.gaussian_filter(zz, 20), dx)
    down = -np.array([gx0[wi, wj], gy0[wi, wj]])
    down /= np.linalg.norm(down) + 1e-9
    shot("art_cliff_far", wp_ + 380 * down, wp_, up=25.0, pitch=8.0, fov=60.0)
    shot("art_cliff_near", wp_ + 230 * down, wp_, up=10.0, pitch=12.0, fov=70.0)
    shot("art_cliff_wide", wp_ + 900 * down, wp_, up=60.0, pitch=2.0, fov=55.0)
    # Water's edge: a lake bank where reeds stand, and a sea beach.
    def waterside(name, mask, toward, up=1.8):
        ii_, jj_ = np.nonzero(mask)
        if len(ii_) == 0:
            return
        xy_ = np.stack([(jj_ + 0.5) * dx, (ii_ + 0.5) * dx], 1)
        td_ = np.min(np.hypot(xy_[:, None, 0] - towns_[None, :, 0], xy_[:, None, 1] - towns_[None, :, 1]), 1)
        k = int(np.argmin(np.where(td_ > 220, td_, 1e9)))    # near a town, but clear of its houses
        gy_w, gx_w = np.gradient(ndimage.gaussian_filter(toward, 3), dx)
        g = np.array([gx_w[ii_[k], jj_[k]], gy_w[ii_[k], jj_[k]]])
        g /= np.linalg.norm(g) + 1e-9                         # points away from the water
        along_ = np.array([-g[1], g[0]])
        shot(name, xy_[k] + 9 * g, xy_[k] + 40 * along_ - 10 * g, up=up, pitch=-8.0, fov=72.0)
    towns_ = np.array(S["town"])
    waterside("art_lake_bank", (lake_d2 > 5) & (lake_d2 < 12) & (reeds > 0.45) & W["land"] & ~lake, lake_d2)
    waterside("art_sea_beach", (sea_d2 > 8) & (sea_d2 < 20) & (zz > 1.0) & (zz < 3.0) & (slope < 0.25)
              & (c_grass > 0.6) & W["land"] & ~lake, sea_d2)
    # Paths: a paved road leaving town, an unpaved stretch, a footpath, a trail.
    def along(w_, at, name, up=2.2):
        i = int(np.clip(at, 0, len(w_["line"]) - 61))
        shot(name, w_["line"][i], w_["line"][i + 60], up=up, pitch=-9.0, fov=70.0)
    by_kind = lambda k: sorted((w_ for w_ in ways if w_["kind"] == k), key=lambda w_: -len(w_["line"]))
    if by_kind(0):
        r0 = by_kind(0)[0]
        along(r0, 240, "art_road_paved")
        along(r0, np.argmin(r0["paved"][600:-600]) + 600 if len(r0["line"]) > 1400 else len(r0["line"]) // 2, "art_road_dirt")
    if by_kind(1):
        along(by_kind(1)[0], len(by_kind(1)[0]["line"]) // 2, "art_footpath", up=1.8)
    if by_kind(2):
        along(by_kind(2)[0], len(by_kind(2)[0]["line"]) // 2, "art_trail", up=1.7)
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
    wall_shots(W, shot)
    lake_shots(W, shot)
    river_shots(W, shot)
    for i, p in enumerate(S["town"]):
        v = S["landmark"][i] - p
        e = p - 210 * v / (np.linalg.norm(v) + 1e-9)
        ii, jj = int(np.clip(e[1] / dx, 0, n - 1)), int(np.clip(e[0] / dx, 0, n - 1))
        if not W["land"][ii, jj] or lake[ii, jj]:
            e = p
        shot(f"room{i}_{R['theme'][i]}_town", e, S["landmark"][i], up=14.0, pitch=0.0)
        if R["theme"][i] == "field":
            # From rooftop height, looking down across the meadow: where grass ends shows here.
            shot(f"art_high_{i}", e, S["landmark"][i], up=38.0, pitch=-16.0, fov=70.0)
    dbg = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
    for w_ in ways:
        dbg[np.clip((w_["line"][:, 1] / dx).astype(int), 0, n - 1), np.clip((w_["line"][:, 0] / dx).astype(int), 0, n - 1)] = \
            [(255, 240, 200), (230, 190, 120), (200, 120, 90)][w_["kind"]]
    dbg[(trees[:, 1] / dx).astype(int), (trees[:, 0] / dx).astype(int)] = \
        np.array([[20, 60, 30], [40, 90, 70], [120, 150, 30], [200, 220, 90], [90, 70, 60], [150, 60, 140]], np.uint8)[trees[:, 3].astype(int)]
    Image.fromarray(dbg[::-1]).save(f"{out}/trees_map.png")
    open(f"{out}/world.txt", "w").write("\n".join(lines) + "\n")
    diag(W, PL).save(f"{out}/map.png")
    Mj = dict(M)
    Mj["themes"] = R["theme"]
    Mj["floor_heights"] = [float(v) for v in R["h"]]
    json.dump(Mj, open(f"{out}/report.json", "w"), indent=1)
    print(f"  wrote {out}: {m}x{m} raster, {len(trees)} trees, {sum(l.startswith('shot') for l in lines)} shots, "
          f"heights {hz.min():.0f}..{hz.max():.0f} m")


if __name__ == "__main__":
    seed, name = int(sys.argv[1]), sys.argv[2]
    W, PL, M = build(seed, parse([a for a in sys.argv[3:] if "=" in a and not a.startswith("--")]))
    export(W, PL, M, world_dir(name))
