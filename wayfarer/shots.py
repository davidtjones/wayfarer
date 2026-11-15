"""Shots that judge walls: does the high ground between two rooms read as land?

For the longest borders between rooms, three views each:
  face   from the room floor, looking square at the wall
  along  from the top of the wall, looking down its length (a thin wall shows here)
  air    from above and to one side, the wall running across the frame
"""
import numpy as np
from scipy import ndimage


def borders(W, min_len=900.0):
    """Border cells per room pair, ordered along the border. -> [(a, b, xy[k, 2])]"""
    lab, dx, n = W["label"], W["dx"], W["n"]
    cells = {}
    for di, dj in ((0, 1), (1, 0)):
        a, b = lab[:n - di, :n - dj], lab[di:, dj:]
        for i, j in zip(*np.nonzero((a != b) & (a >= 0) & (b >= 0))):
            cells.setdefault((min(a[i, j], b[i, j]), max(a[i, j], b[i, j])), []).append((j * dx, i * dx))
    out = []
    for (a, b), v in cells.items():
        v = np.array(v, float)
        if len(v) * dx < min_len:
            continue
        # order along the border by the main axis of its cells
        c = v - v.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
        out.append((int(a), int(b), v[np.argsort(c @ axis)]))
    return sorted(out, key=lambda t: -len(t[2]))


def wall_shots(W, shot, count=5):
    lab, dx, n = W["label"], W["dx"], W["n"]
    cen = [np.array(ndimage.center_of_mass(lab == i))[::-1] * dx for i in range(W["rooms"]["n"])]
    inside = lambda p: 0 <= p[0] < n * dx and 0 <= p[1] < n * dx and W["land"][int(p[1] / dx), int(p[0] / dx)]
    for k, (a, b, line) in enumerate(borders(W)[:count]):
        m = len(line)
        mid, q1, q3 = line[m // 2], line[m // 4], line[3 * m // 4]
        t = q3 - q1
        t /= np.linalg.norm(t) + 1e-9
        nrm = np.array([-t[1], t[0]])
        if (cen[a] - mid) @ nrm < 0:
            nrm = -nrm                                      # points into room a
        tag = f"wall{k:02d}_{W['rooms']['theme'][a]}{a}_{W['rooms']['theme'][b]}{b}"
        for dist, name in ((900.0, "face_far"), (450.0, "face_near")):
            e = mid + dist * nrm
            if inside(e):
                shot(f"{tag}_{name}", e, mid, up=28.0, pitch=2.0, fov=65.0)
        shot(f"{tag}_along", q1, q3, up=30.0, pitch=-10.0, fov=70.0)
        e = mid + 1100 * (0.75 * nrm + 0.66 * t)
        shot(f"{tag}_air", e, mid, up=380.0, pitch=-20.0, fov=60.0)


def lake_shots(W, shot):
    """Each lake of any size from the air on four sides, to check its shore sits right."""
    dx = W["dx"]
    lab, k = ndimage.label(W["lake"])
    for q in range(1, k + 1):
        m = lab == q
        if m.sum() * dx * dx < 2e4:
            continue
        c = np.array(ndimage.center_of_mass(m))[::-1] * dx
        r = 0.9 * np.sqrt(m.sum()) * dx + 250
        for j, a in enumerate((0.6, 2.2, 3.8, 5.3)):
            e = c + r * np.array([np.cos(a), np.sin(a)])
            shot(f"lake{q:02d}_air{j}", e, c, up=170.0, pitch=-24.0, fov=60.0)


def river_shots(W, shot, count=6):
    """A few spread-out river spots, each from the air and from the bank."""
    ii, jj = np.nonzero(W["river"])
    if len(ii) == 0:
        return
    dx = W["dx"]
    pts = np.stack([(jj + 0.5) * dx, (ii + 0.5) * dx], 1)
    lvl = W["river_z"][ii, jj]
    pick = [int(np.argmax(lvl))]                     # start high, then keep spots far apart
    for _ in range(count - 1):
        d = np.min([np.hypot(*(pts - pts[k]).T) for k in pick], axis=0)
        pick.append(int(np.argmax(d)))
    for q, k in enumerate(pick):
        c = pts[k]
        near = pts[(np.hypot(*(pts - c).T) < 120) & (lvl > lvl[k])]
        v = (near.mean(0) - c) if len(near) else np.array([1.0, 0.0])
        v = v / (np.linalg.norm(v) + 1e-9)           # points upstream
        side = np.array([-v[1], v[0]])
        shot(f"river{q:02d}_air", c + 150 * v + 60 * side, c, up=80.0, pitch=-26.0, fov=60.0)
        shot(f"river{q:02d}_bank", c + 45 * v + 14 * side, c - 40 * v, up=4.0, pitch=-6.0, fov=65.0)
    # the steepest stretch: how a fall or rapid looks
    rz = np.where(W["river_z"] > -999, W["river_z"], np.nan)
    gy, gx = np.gradient(rz, dx)
    g = np.where(W["river"], np.hypot(gx, gy), np.nan)
    if np.isfinite(g).any():
        i, j = np.unravel_index(np.nanargmax(g), g.shape)
        c = np.array([(j + 0.5) * dx, (i + 0.5) * dx])
        v = -np.array([gx[i, j], gy[i, j]]) / (g[i, j] + 1e-9)      # downstream
        side = np.array([-v[1], v[0]])
        shot("river_fall_below", c + 70 * v + 25 * side, c, up=6.0, pitch=4.0, fov=65.0)
        shot("river_fall_air", c + 120 * v + 90 * side, c, up=70.0, pitch=-24.0, fov=60.0)
