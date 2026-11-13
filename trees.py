"""Where trees grow, and which ones.

Trees are not sprinkled; they follow the land.

1. Site. Every cell gets a few plain measures: how wet it is (drainage area,
   hollows, distance to a lake), how shaded (north-facing), how exposed (ridges),
   how steep, and how close to the treeline.
2. Canopy. Those measures, the room's theme and broad noise give a canopy cover
   0..1. Thresholding it gives stands with real edges and glades instead of an
   even scatter. Open ground gets small copses, and drainage lines and lake
   shores get strips of trees.
3. Species. Each species scores a cell by what it likes (spruce: wet, shaded,
   high; pine: dry, sunny, steep; oak: low open ground; birch: edges and water),
   plus its own broad noise, so neighbours tend to be the same kind and stands
   mix at their borders.
4. Trunks. Dart throwing with a spacing that follows canopy cover: tall trees
   first, then saplings in the gaps. Trees are tallest inside a stand and
   shorter at its edge, on poor ground and near the treeline.

How readily each room theme grows trees, and which species it leans to, comes
from the biome table (`biomes.toml`); bushes are scattered here too, from the
table's `bushes` layer.

`plant` returns the trees and the canopy field (the ground under a stand is
forest floor, so the exporter's cover map reads it too).
"""
import numpy as np
from scipy import ndimage
from dress import _ss
from noise import fbm
import biomes

SPRUCE, PINE, OAK, BIRCH, SNAG, BUSH = range(6)
NAMES = ["spruce", "pine", "oak", "birch", "snag"]
# Height range in metres, and trunk spacing as a share of height.
HEIGHT = [(14, 25), (11, 20), (9, 16), (9, 15), (6, 12)]
CROWN = [0.30, 0.38, 0.55, 0.42, 0.30]
# Closest two full-grown trees stand in a closed stand, metres.
SPACING_CLOSED = 5.5
SPACING_OPEN = 15.0


def site(W, lake, rng):
    """Site measures on the world grid, each 0..1."""
    n, dx = W["n"], W["dx"]
    z, slope = W["z"], W["slope"]
    px = lambda metres: metres / dx
    # Wet: big drainage area, hollows, and the ground beside a lake.
    drain = _ss(np.log10(W["area"]), 3.3, 4.8)
    drain = ndimage.maximum_filter(drain, 3)
    hollow = _ss(ndimage.gaussian_filter(z, px(50)) - z, -1.0, 5.0)
    lake_d = ndimage.distance_transform_edt(~lake) * dx
    shore = np.exp(-lake_d / 45.0)
    wet = np.clip(0.55 * drain + 0.45 * hollow + 0.6 * shore, 0, 1)
    wet = ndimage.gaussian_filter(wet, px(10))
    # Shade: slopes that face north (array rows run north).
    gy, _ = np.gradient(ndimage.gaussian_filter(z, px(12)), dx)
    shade = np.clip(0.5 - 1.6 * gy, 0, 1)
    # Exposure: ground that stands above its surroundings.
    ridge = _ss(z - ndimage.gaussian_filter(z, px(90)), 3.0, 22.0)
    n1 = fbm(n, 10, 4, rng)
    treeline = 400 + 50 * n1
    high = _ss(z, treeline - 150, treeline)
    return dict(wet=wet, shade=shade, ridge=ridge, high=high, lake_d=lake_d, drain=drain,
                above=z > treeline, slope=slope)


def canopy_cover(W, S, PL, road, lake, rng, weights, table):
    """Canopy cover 0..1: closed stands, their edges, copses and water-side strips.
    How readily each room theme grows them comes from the biome table."""
    n, dx = W["n"], W["dx"]
    base = biomes.mix(weights, table, "trees", "cover")
    limit = biomes.mix(weights, table, "trees", "limit")
    copses = biomes.mix(weights, table, "trees", "copses")
    broad = fbm(n, 14, 3, rng)                              # stands, about 600 m
    mid = fbm(n, 70, 3, rng)                                # glades and bays, about 120 m
    want = base + 0.30 * (S["wet"] - 0.35) + 0.14 * (S["shade"] - 0.5) - 0.30 * S["ridge"] \
        + 0.30 * broad + 0.22 * mid
    stand = _ss(want, 0.42, 0.62)
    # Copses: small clumps on open ground, more of them in farmland.
    clump = fbm(n, 160, 2, rng)
    copse = _ss(clump, 0.42, 0.58) * np.clip(copses, 0, 1) \
        * _ss(broad + 0.6 * mid, -0.25, 0.35)
    # Strips along drainage lines and lake shores.
    line = ndimage.gaussian_filter(_ss(np.log10(W["area"]), 4.1, 5.0), 1.5)
    strip = np.clip(1.6 * line, 0, 1) * _ss(mid, -0.55, 0.0)
    bank = _ss(S["lake_d"], 6, 14) * (1 - _ss(S["lake_d"], 30, 60)) * _ss(mid + 0.5 * broad, -0.2, 0.3)
    c = np.maximum.reduce([stand, 0.75 * copse, 0.8 * strip, 0.7 * bank])

    # Where nothing grows, or less.
    c = np.minimum(c, np.clip(limit + 0.6 * strip * _ss(limit, 0.05, 0.2), 0, 1))
    c *= 1 - S["high"]
    c *= 1 - _ss(S["slope"], 0.55, 0.95)
    # Clearings: towns (soft edge), landmarks, roads.
    X, Y = np.meshgrid((np.arange(n) + 0.5) * dx, (np.arange(n) + 0.5) * dx)
    for p in PL["sites"]["town"]:
        d = np.hypot(X - p[0], Y - p[1]) + 40 * mid
        c *= 0.12 + 0.88 * _ss(d, 120, 230)
        c[d < 105] = 0
    for p in PL["sites"]["landmark"]:
        c[np.hypot(X - p[0], Y - p[1]) < 45] = 0
    road_d = ndimage.distance_transform_edt(~road) * dx
    c *= _ss(road_d, 7, 16)
    ok = W["land"] & ~lake & (S["lake_d"] > 5) & (W["z"] > 3) & ~S["above"]
    return np.where(ok, np.clip(c, 0, 1), 0.0)


def species_scores(W, S, canopy, rng, weights, table):
    """One score field per species; higher = more at home. Each theme may lean
    toward a species (`[forest.trees] spruce = 0.45`)."""
    n = W["n"]
    noise = [0.55 * fbm(n, 26, 2, rng) for _ in range(5)]   # stands about 300 m across
    wet, shade, high = S["wet"], S["shade"], S["high"]
    low = 1 - _ss(W["z"], 90, 240)
    edge = 1 - np.abs(canopy - 0.4) / 0.4                    # 1 at a stand's edge
    edge = np.clip(edge, 0, 1)
    steep = _ss(S["slope"], 0.3, 0.7)
    shore = np.exp(-S["lake_d"] / 35.0)
    lean = [biomes.mix(weights, table, "trees", name) for name in NAMES]
    return np.stack([
        0.10 + 0.7 * wet + 0.5 * shade + 0.9 * high + lean[0] + noise[0],
        0.05 + 0.6 * (1 - wet) + 0.4 * (1 - shade) + 0.7 * steep + 0.6 * high + lean[1] + noise[1],
        0.9 * low * (1 - 0.6 * wet) + 0.2 * (1 - canopy) + lean[2] + noise[2],
        0.15 + 0.55 * edge + 0.9 * shore + 0.4 * S["drain"] + lean[3] + noise[3],
        -0.55 + 0.5 * S["ridge"] + lean[4] + 0.3 * noise[4],
    ], 0)


def _throw(cand, radius, n_cells, cell, taken=None):
    """Dart throwing: keep each candidate that no kept tree crowds. Two trees
    crowd when closer than the mean of their radii. Returns kept indices."""
    grid = {} if taken is None else taken
    kept = []
    reach = int(np.ceil(radius.max() / cell))
    for k in range(len(cand)):
        x, y, r = cand[k, 0], cand[k, 1], radius[k]
        gi, gj = int(y / cell), int(x / cell)
        free = True
        for a in range(max(gi - reach, 0), min(gi + reach, n_cells - 1) + 1):
            for b in range(max(gj - reach, 0), min(gj + reach, n_cells - 1) + 1):
                for (ox, oy, orad) in grid.get((a, b), ()):
                    lim = 0.5 * (r + orad)
                    if (ox - x) ** 2 + (oy - y) ** 2 < lim * lim:
                        free = False
                        break
                if not free:
                    break
            if not free:
                break
        if free:
            grid.setdefault((gi, gj), []).append((x, y, r))
            kept.append(k)
    return kept, grid


def plant(W, PL, lake, road, rng, weights, table, log=print):
    """Returns (trees, canopy, site). trees: array of (x, y, height, species) in
    world metres; canopy: cover 0..1 on the world grid; site: the site measures,
    with `canopy` and `edge` added, for the other biome layers."""
    n, dx = W["n"], W["dx"]
    S = site(W, lake, rng)
    canopy = canopy_cover(W, S, PL, road, lake, rng, weights, table)
    scores = species_scores(W, S, canopy, rng, weights, table)
    age = fbm(n, 90, 2, rng)                                 # patches of young and old growth
    cliff_pine = (S["slope"] > 0.75) & (S["slope"] < 1.25) & W["land"] & ~lake & (W["z"] > 6) & ~S["above"] \
        & (biomes.mix(weights, table, "trees", "limit") > 0.5)

    def candidates(count_per_cell, mask):
        ii, jj = np.nonzero(mask)
        reps = rng.poisson(count_per_cell[ii, jj])
        ii, jj = np.repeat(ii, reps), np.repeat(jj, reps)
        order = rng.permutation(len(ii))
        ii, jj = ii[order], jj[order]
        xy = np.stack([(jj + rng.uniform(0, 1, len(jj))) * dx, (ii + rng.uniform(0, 1, len(ii))) * dx], 1)
        return ii, jj, xy

    def dress(ii, jj, small):
        """Species and height for candidates at cells (ii, jj)."""
        s = scores[:, ii, jj] * 5.0
        p = np.exp(s - s.max(0))
        p /= p.sum(0)
        u = rng.uniform(0, 1, len(ii))
        sp = (u[None] > np.cumsum(p, 0)).sum(0).clip(0, 4)
        c = canopy[ii, jj]
        lo = np.array([h[0] for h in HEIGHT])[sp]
        hi = np.array([h[1] for h in HEIGHT])[sp]
        # Tall inside a stand, shorter at its edge, on high ground and in young patches.
        vigour = np.clip(0.25 + 0.75 * _ss(c, 0.15, 0.8) + 0.25 * age[ii, jj], 0, 1) * (1 - 0.6 * S["high"][ii, jj])
        h = lo + (hi - lo) * np.clip(vigour + rng.normal(0, 0.16, len(ii)), 0, 1)
        if small:
            h *= rng.uniform(0.32, 0.6, len(ii))
        return sp, h

    cell_area = dx * dx
    spacing = SPACING_OPEN + (SPACING_CLOSED - SPACING_OPEN) * _ss(canopy, 0.1, 0.85)
    grow = canopy > 0.06
    # Offer several times more darts than can land, so stands fill to their spacing.
    offered = np.where(grow, 3.0 * cell_area / (0.6 * spacing ** 2), 0.0) * np.clip(canopy * 2.2, 0.15, 1)
    ii, jj, xy = candidates(offered, grow)
    sp, h = dress(ii, jj, small=False)
    rad = np.maximum(spacing[ii, jj] * (0.75 + 0.5 * np.array(CROWN)[sp] / 0.4) * (h / 16.0) ** 0.5, 2.5)
    cell = 8.0
    n_cells = int(n * dx / cell) + 1
    keep, grid = _throw(xy, rad, n_cells, cell)
    trees = [np.column_stack([xy[keep], h[keep], sp[keep]])]

    # Saplings and understorey in the gaps, thickest at stand edges and in young patches.
    young = np.clip(canopy * (1.3 - canopy) * 2.0, 0, 1) * (0.6 + 0.5 * _ss(age, -0.2, 0.3))
    ii, jj, xy = candidates(np.where(grow, 0.035 * young * cell_area / 4.0, 0.0), grow)
    sp, h = dress(ii, jj, small=True)
    keep, grid = _throw(xy, np.full(len(xy), 3.2), n_cells, cell, grid)
    trees.append(np.column_stack([xy[keep], h[keep], sp[keep]]))

    # A few pines and snags hang on to ledges of steep ground.
    ii, jj, xy = candidates(np.where(cliff_pine, 0.012, 0.0), cliff_pine)
    h = rng.uniform(5, 11, len(ii))
    sp = np.where(rng.uniform(0, 1, len(ii)) < 0.8, PINE, SNAG)
    keep, grid = _throw(xy, np.full(len(xy), 9.0), n_cells, cell, grid)
    trees.append(np.column_stack([xy[keep], h[keep], sp[keep]]))

    # Bushes grow in thickets, not as a sprinkle. Where the thickets are is
    # built from a few habits, each weighted per room theme by the table:
    #   mantle    a band hugging the outside of every stand and copse
    #   gullies   along the lines water drains down
    #   banks     the strip above a lake shore
    #   hedges    broken rows beside roads, as farmland keeps them
    #   thickets  round clumps out on open ground
    #   singles   the odd bush on its own, anywhere
    S["canopy"] = canopy
    S["edge"] = np.clip(1 - np.abs(ndimage.gaussian_filter(canopy, 1.5) - 0.35) / 0.35, 0, 1)
    habit = lambda key, default=0.0: biomes.mix(weights, table, "bushes", key, default)
    wide = ndimage.gaussian_filter(canopy, 14.0 / dx)
    mantle = _ss(wide, 0.06, 0.3) * (1 - _ss(canopy, 0.3, 0.6)) * _ss(fbm(n, 70, 2, rng), 0.0, 0.3)
    gully = _ss(S["drain"], 0.5, 0.9) * _ss(fbm(n, 90, 2, rng), 0.0, 0.3)
    bank = _ss(S["lake_d"], 5, 10) * (1 - _ss(S["lake_d"], 18, 34)) * _ss(fbm(n, 110, 2, rng), 0.05, 0.3)
    road_d = ndimage.distance_transform_edt(~road) * dx
    hedge = _ss(road_d, 3.5, 6.5) * (1 - _ss(road_d, 8, 12)) * _ss(fbm(n, 160, 2, rng), 0.15, 0.35)
    blobs = _ss(fbm(n, 55, 3, rng) + 0.5 * fbm(n, 400, 2, rng), 0.55, 0.72)
    cover = np.maximum.reduce([
        habit("mantle") * mantle, habit("gullies") * gully, habit("banks") * bank,
        habit("hedges") * hedge, habit("thickets") * blobs,
    ])
    # The table's site responses (wet, ridge and so on) scale the whole layer.
    cover = np.clip(cover * biomes.density(weights, table, "bushes", S, key="amount"), 0, 1)
    can_grow = W["land"] & ~lake & (S["lake_d"] > 3) & (W["z"] > 2.5) & (S["slope"] < 0.7) & ~S["above"] \
        & (road_d > 3.5)
    cover = np.where(can_grow, ndimage.gaussian_filter(cover, 0.8), 0.0)
    # Inside a thicket bushes stand shoulder to shoulder; outside, a rare single.
    per_m2 = 0.12 * _ss(cover, 0.2, 0.6) + np.where(can_grow, habit("singles") / 2500.0, 0.0)
    ii, jj, xy = candidates(per_m2 * cell_area * 2.5, per_m2 > 0)
    # Tall in the heart of a thicket, low at its rim. `h` is nominal; the crown
    # stands half as tall again.
    heart = cover[ii, jj]
    h = (0.6 + 1.5 * heart ** 1.5) * rng.uniform(0.7, 1.25, len(ii))
    keep, grid = _throw(xy, 1.3 + 1.3 * h, n_cells, cell, grid)
    trees.append(np.column_stack([xy[keep], h[keep], np.full(len(keep), BUSH)]))

    trees = np.concatenate(trees)
    counts = np.bincount(trees[:, 3].astype(int), minlength=6)
    log(f"  planted {len(trees)}: spruce {counts[0]}, pine {counts[1]}, oak {counts[2]}, birch {counts[3]}, "
        f"snag {counts[4]}, bushes {counts[5]}; canopy covers {100 * (canopy > 0.5).mean():.1f}% of the grid")
    return trees, canopy, S
