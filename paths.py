"""Paths between the places of a world: roads, footpaths and animal trails.

Three kinds, each a line with a width:

- road: the graded routes `rplace.place` already lays from town to town through
  each door. A cart's width of packed earth. Paved with rough cobbles only
  where people bothered: in and near a town, at a door, and the odd surviving
  stretch in between.
- footpath: from each town to its landmark and to the nearest lake shore.
  Paved only on the last steps up to the landmark.
- trail: narrow tracks worn by animals and wanderers. A handful of quiet spots
  in each room are joined to whatever path or spot is nearest, so trails branch
  off the roads and off each other instead of crossing the land in straight
  lines. All routes take the easiest walking line (`place.find_route`).

`path_map` turns the lines into an image the game reads (`paths.png`). It is a
distance map, not a drawing: every pixel stores how far it is from the nearest
path's centre line, so the game can draw a crisp edge of any width between
pixels and knows how far a tuft or tree stands from the path.

    R  distance to the centre line, 0..MAX_DIST metres
    G  half width of that path, 0..MAX_HALF metres
    B  how paved it is there, 0..1
"""
import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from place import find_route

ROAD, FOOTPATH, TRAIL = 0, 1, 2
HALF = [1.9, 0.8, 0.38]         # half width of each kind, metres
MAX_DIST = 12.0                 # R = 255 at this distance or beyond
MAX_HALF = 4.0                  # G = 255 at this half width
PIXEL = 2.0                     # metres per pixel of the path map


def _smooth(line, spacing=0.5, wobble=0.0, rng=None):
    """A route (points every 20 m or so) as a smooth curve sampled every `spacing` m.
    `wobble` adds a slow sideways wander in metres, so a trail is not a ruled line."""
    line = np.asarray(line, float)
    keep = np.concatenate([[True], np.hypot(*np.diff(line, axis=0).T) > 1e-3])
    line = line[keep]
    if len(line) < 3:
        return None
    d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    t = np.arange(0, d[-1], spacing)
    cs = CubicSpline(d, line)
    p = cs(t)
    if wobble > 0:
        tang = cs(t, 1)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        side = np.stack([-tang[:, 1], tang[:, 0]], 1)
        knots = np.arange(0, d[-1] + 60, 30.0)
        off = np.interp(t, knots, rng.normal(0, wobble, len(knots)))
        off = ndimage.gaussian_filter1d(off, 12 / spacing)
        ends = np.clip(np.minimum(t, d[-1] - t) / 25.0, 0, 1)   # meet the junctions cleanly
        p = p + side * (off * ends)[:, None]
    return p


def network(W, PL, rng, log=print):
    """Returns a list of paths: dict(kind, line (k x 2, world metres, every 0.5 m),
    paved (k,) 0..1)."""
    S, R = PL["sites"], W["rooms"]
    n, dx = W["n"], W["dx"]
    towns, marks, doors = np.array(S["town"]), np.array(S["landmark"]), np.array(S["door"])
    out = []

    def route(a, b):
        try:
            line = find_route(W, np.asarray(a, float), np.asarray(b, float))
        except Exception:
            return None
        return line if line is not None and len(line) >= 3 else None

    def near(p, pts, lo, hi):
        """1 within `lo` metres of any of `pts`, falling to 0 at `hi`."""
        d = np.min(np.hypot(p[:, None, 0] - pts[None, :, 0], p[:, None, 1] - pts[None, :, 1]), 1)
        return np.clip((hi - d) / (hi - lo), 0, 1)

    # Roads: paved in town and at doors, plus a few surviving stretches.
    for line in PL["roads"]:
        if line is None:
            continue
        p = _smooth(line)
        if p is None:
            continue
        t = np.arange(len(p)) * 0.5
        knots = np.arange(0, t[-1] + 120, 60.0)
        relic = np.interp(t, knots, (rng.uniform(0, 1, len(knots)) > 0.8).astype(float))
        paved = np.maximum.reduce([near(p, towns, 150, 260), near(p, doors, 30, 70), 0.75 * relic])
        out.append(dict(kind=ROAD, line=p, paved=paved))

    # Footpaths: town to landmark, town to the nearest shore.
    lake_d = ndimage.distance_transform_edt(~W["lake"]) * dx
    shore = W["land"] & ~W["lake"] & (lake_d > 6) & (lake_d < 16)
    si, sj = np.nonzero(shore)
    shore_xy = np.stack([(sj + 0.5) * dx, (si + 0.5) * dx], 1)
    for i, town in enumerate(towns):
        line = route(town, marks[i])
        p = _smooth(line, wobble=0.8, rng=rng) if line is not None else None
        if p is not None:
            out.append(dict(kind=FOOTPATH, line=p, paved=near(p, marks[i:i + 1], 25, 60)))
        if len(shore_xy):
            d = np.hypot(*(shore_xy - town).T)
            k = int(np.argmin(d))
            if 60 < d[k] < 1500:
                line = route(town, shore_xy[k])
                p = _smooth(line, wobble=0.8, rng=rng) if line is not None else None
                if p is not None:
                    out.append(dict(kind=FOOTPATH, line=p, paved=np.zeros(len(p))))

    # Trails: quiet spots in each room, each joined to the nearest thing already
    # on the network, so they branch.
    walk = W["land"] & ~W["lake"] & (W["slope"] < 0.4) & (W["z"] > 4)
    knots = [p["line"][::40] for p in out]                    # a point every 20 m of path
    for i in range(R["n"]):
        ii, jj = np.nonzero(walk & (W["label"] == i))
        if len(ii) < 50:
            continue
        area_km2 = len(ii) * dx * dx / 1e6
        dry = float(W["desert"][ii, jj].mean() + W["ash"][ii, jj].mean())
        count = int(np.clip(3 + 3.0 * area_km2, 3, 10) * (1 - 0.6 * min(dry, 1)))
        for _ in range(count):
            # A spot well away from everything already linked.
            net = np.concatenate(knots)
            tree = cKDTree(net)
            pick = rng.choice(len(ii), min(60, len(ii)), replace=False)
            spots = np.stack([(jj[pick] + 0.5) * dx, (ii[pick] + 0.5) * dx], 1)
            d, idx = tree.query(spots)
            k = int(np.argmax(np.minimum(d, 900)))
            if d[k] < 120:
                continue
            line = route(net[idx[k]], spots[k])
            p = _smooth(line, wobble=2.0, rng=rng) if line is not None else None
            if p is None:
                continue
            out.append(dict(kind=TRAIL, line=p, paved=np.zeros(len(p))))
            knots.append(p[::40])
    km = [sum(len(p["line"]) for p in out if p["kind"] == k) * 0.5 / 1000 for k in range(3)]
    log(f"  paths: {km[0]:.1f} km of road, {km[1]:.1f} km of footpath, {km[2]:.1f} km of trail")
    return out


def path_map(W, paths):
    """The network as an RGBA image, row 0 at the south edge (as the world grids).
    Returns (image uint8, distance in metres, half width in metres) at PIXEL."""
    size = int(round(W["n"] * W["dx"] / PIXEL))
    dist = np.full((size, size), MAX_DIST, np.float32)
    half = np.zeros((size, size), np.float32)
    paved = np.zeros((size, size), np.float32)
    best = np.full((size, size), 1e9, np.float32)             # distance past the path's edge
    reach = int(np.ceil(MAX_DIST / PIXEL)) + 1
    for kind in range(3):
        mine = [p for p in paths if p["kind"] == kind]
        if not mine:
            continue
        pts = np.concatenate([p["line"] for p in mine])
        pv = np.concatenate([p["paved"] for p in mine])
        # Only pixels near a path of this kind need a distance.
        mask = np.zeros((size, size), bool)
        mask[np.clip((pts[:, 1] / PIXEL).astype(int), 0, size - 1),
             np.clip((pts[:, 0] / PIXEL).astype(int), 0, size - 1)] = True
        mask = ndimage.binary_dilation(mask, iterations=reach)
        ii, jj = np.nonzero(mask)
        d, idx = cKDTree(pts).query(np.stack([(jj + 0.5) * PIXEL, (ii + 0.5) * PIXEL], 1), workers=-1)
        d = d.astype(np.float32)
        win = (d - HALF[kind]) < best[ii, jj]
        ii, jj, d, idx = ii[win], jj[win], d[win], idx[win]
        best[ii, jj] = d - HALF[kind]
        dist[ii, jj] = np.minimum(d, MAX_DIST)
        half[ii, jj] = HALF[kind]
        paved[ii, jj] = pv[idx]
    img = np.zeros((size, size, 4), np.uint8)
    img[..., 0] = np.round(dist / MAX_DIST * 255)
    img[..., 1] = np.round(half / MAX_HALF * 255)
    img[..., 2] = np.round(np.clip(paved, 0, 1) * 255)
    img[..., 3] = 255
    return img, dist, half
