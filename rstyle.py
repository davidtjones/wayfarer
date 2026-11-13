"""Regional character for a rooms world: the same deliberate exaggerations as
style.py, driven by each room's theme instead of one fixed brief."""
import numpy as np
from scipy import ndimage
from style import flow, _stream_base
from noise import fbm
from rivers import carve


def finish(lay, z, log=print):
    f, dx, n, R = lay.f, lay.dx, lay.n, lay.rooms
    rng = np.random.default_rng(lay.seed + 5)
    sea = ~f["land"]
    X, Y = lay.grid()
    lab0 = np.maximum(f["label"], 0)
    # Themes fade into each other over a few hundred metres, along a ragged line
    # rather than the ruled room border. Across high ground they blend wider still,
    # so a range is not painted half and half down its crest.
    brng = np.random.default_rng(lay.seed + 31)
    ii, jj = np.indices((n, n))
    wi = np.clip(ii + np.rint(11 * fbm(n, 9, 3, brng) + 4 * fbm(n, 30, 2, brng)).astype(int), 0, n - 1)
    wj = np.clip(jj + np.rint(11 * fbm(n, 9, 3, brng) + 4 * fbm(n, 30, 2, brng)).astype(int), 0, n - 1)
    labw = np.where(f["land"][wi, wj], lab0[wi, wj], lab0)
    landf = f["land"].astype(float)

    def tw(name):
        w = np.array([t == name for t in R["theme"]], float)[labw] * landf
        blur = lambda sg: ndimage.gaussian_filter(w, sg) / np.maximum(ndimage.gaussian_filter(landf, sg), 1e-3)
        near, wide = blur(9.0), blur(30.0)
        return near + (wide - near) * np.clip(f["prof"] * 4.0, 0, 1)

    # 1. valley floors: flatten ground that sits just above a real stream
    zf, rcv, order, area = flow(z, sea, dx)
    base, sa = _stream_base(zf.ravel(), rcv, order, area, 2.5e5)
    base, sa = base.reshape(n, n), sa.reshape(n, n)
    hand = np.maximum(zf - base, 0)
    h0 = np.clip(5 + 9 * np.log10(np.maximum(sa, 2.5e5) / 2.5e5), 5, 26)
    z = base + hand - 0.8 * h0 * (1 - np.exp(-hand / h0))
    z = np.where(sea, 0.0, z)
    lake = f["lake"]
    z[lake] = f["pin_z"][lake]

    # 2. a young cone in the first ash room
    pts = dict(lay.pts)
    for i, t in enumerate(R["theme"]):
        if t != "ash":
            continue
        m = f["label"] == i
        inner = np.where(m, np.minimum(f["wall_d"], f["coast_d"]), 0)
        if np.any(R.get("mount", False)):
            # in mountain country the cone stands deep in the massif, not in the middle of its own room
            mm = np.asarray(R["mount"])[np.maximum(f["label"], 0)] & f["land"]
            deep = ndimage.distance_transform_edt(mm) * lay.dx
            inner = np.where(m, np.minimum(deep, f["coast_d"]), 0)
        k = np.unravel_index(np.argmax(inner), inner.shape)
        v = np.array([X[k], Y[k]])
        Rc = float(np.clip(inner[k] * 0.9, 350, 1300))
        Hc, rc = 0.42 * Rc, 0.09 * Rc
        r = np.hypot(X - v[0], Y - v[1])
        th = np.arctan2(Y - v[1], X - v[0])
        q = np.clip(1 - r / Rc, 0, 1)
        gul = sum(np.cos(kk * th + rng.uniform(0, 6.3)) * a for kk, a in [(7, 1.0), (13, 0.7), (23, 0.5)])
        cone = z[k] * 0 + np.percentile(z[m], 30) + Hc * q ** 1.8 + 9 * gul * q * (1 - q) * 4 * (r > rc * 1.5)
        rim = np.percentile(z[m], 30) + Hc * (1 - rc / Rc) ** 1.8
        cone = np.where(r < rc, rim - 45 * (1 - np.clip(r / rc, 0, 1) ** 2), cone)
        z = np.where(f["land"] & (r < Rc), np.maximum(z, cone), z)
        pts["volcano"] = v
        pts["volcano_r"] = rc
        break

    # 3. refine the grid x2 and add grain where the ground is steep
    N = n * 2
    up = lambda a, o=1: ndimage.zoom(np.asarray(a, np.float64), 2, order=o)
    Z = up(z, 3)
    land2 = up(f["land"]) > 0.5
    lake2 = up(lake) > 0.5
    pin2 = ndimage.zoom(np.where(lake, f["pin_z"], 0), 2, order=0)
    label2 = ndimage.zoom(f["label"], 2, order=0)
    wallw = np.clip(up(f["prof"]), 0, 1)
    des, ash, moor, forest, field = (up(tw(k)) for k in ("desert", "ash", "moor", "forest", "field"))
    gy, gx = np.gradient(Z, dx / 2)
    sl = np.hypot(gx, gy)
    grain = fbm(N, 40, 5, rng, gain=0.6)
    Z = Z + np.clip(sl / 0.7, 0, 1) * 6 * grain * (1 + 1.2 * wallw)
    # desert floors carry flat-topped buttes with cliff edges
    wd = np.clip((des - 0.55) / 0.25, 0, 1) * (1 - wallw) * (sl < 0.5)
    bn = fbm(N, 15, 4, rng)
    edge = np.clip((bn - 0.12) / 0.03, 0, 1)
    edge2 = np.clip((bn - 0.24) / 0.025, 0, 1)
    butte = (38 + 18 * fbm(N, 3, 2, rng)) * edge + 34 * edge2
    Z = Z + wd * butte
    crag = 1 - np.abs(fbm(N, 22, 4, rng, gain=0.55)) * 2.2                  # sharp-crested noise
    Z = Z + wallw * np.clip((Z - 220) / 300, 0, 1) * 40 * (crag - 0.45)
    Z = np.where(land2, np.maximum(Z, 0.3), -6.0)
    # A lake's surface is set by its lowest shore, as real water is: it spills
    # there. Erosion may have cut the ground beside a lake below the level it was
    # designed at, and water left at that level would hang in the air.
    lab_l, k_l = ndimage.label(lake2)
    for q in range(1, k_l + 1):
        m = lab_l == q
        if m.sum() * (dx / 2) ** 2 < 2e4:           # a sliver is not a lake
            lake2[m] = False
            continue
        ring = ndimage.binary_dilation(m, iterations=3) & ~m & land2
        if ring.any():
            pin2[m] = min(float(pin2[m].max()), float(Z[ring].min()) - 0.3)
    Z[lake2] = pin2[lake2]
    # A lake whose level dropped is left in a pit with rock walls. Ease the ground
    # round it back to banks a lake would have: gentle on most sides, steeper on some.
    d_l, (ni, nj) = ndimage.distance_transform_edt(~lake2, return_indices=True)
    d_l = d_l * dx / 2
    grade = 0.22 + 0.5 * np.clip(0.5 + 1.2 * fbm(N, 6, 3, rng), 0, 1) ** 2
    eased = np.minimum(Z, pin2[ni, nj] + 0.5 + grade * d_l)
    Z = np.where(land2 & ~lake2, Z + (eased - Z) * (1 - np.clip((d_l - 90) / 130, 0, 1) ** 2), Z)
    shore = ndimage.binary_dilation(lake2, iterations=2) & ~lake2
    near = ndimage.maximum_filter(pin2, 5)
    Z[shore] = np.maximum(Z[shore], near[shore] + 0.4)

    Z, river, river_z, _ = carve(Z, land2, lake2, pin2, dx / 2, lay.seed, log=log)

    W = dict(z=Z, land=land2, lake=lake2, lake_zmap=pin2, river=river, river_z=river_z, dx=dx / 2, n=N, pts=pts, seed=lay.seed,
             label=label2, wallw=wallw, butte=wd * edge, desert=des, ash=ash, moor=moor, forest=forest,
             field=field, rooms=R, w=lay.w, nn=lay.nn)
    zf, rcv, order, area = flow(Z, ~land2, W["dx"])
    W["area"] = area.reshape(N, N)
    gy, gx = np.gradient(Z, W["dx"])
    W["slope"] = np.hypot(gx, gy)
    log(f"  finished surface {N}x{N} at {W['dx']:.0f} m, peak {Z.max():.0f} m")
    return W
