"""Pictures of a world for people: the diagnostic map, and ground-level views of
a dressed copy of the land. A viewing aid, nothing more.

  diag   top-down map: relief, room borders, floors, doors, roads, sites.
  dress  the land made viewable: ground cover by room theme, trees, water, snow,
         and simple oversized structures at the placed sites.
  view   walks rays over the dressed surface, shades with sun, sky light, cast
         shadows and distance haze, and mirrors the land in still water.
"""
import numpy as np
from numba import njit, prange
from PIL import Image, ImageDraw
from scipy import ndimage
from .noise import fbm, smoothstep

LIGHTS = {
    "afternoon": dict(az=228, el=36, sun=(5.2, 4.7, 4.0), zen=(0.16, 0.34, 0.75), hor=(0.62, 0.76, 0.92),
                      haze=9000.0, exp=0.62),
    "sunset": dict(az=276, el=5.5, sun=(6.5, 3.0, 1.1), zen=(0.13, 0.20, 0.46), hor=(1.0, 0.56, 0.26),
                   haze=6500.0, exp=0.85),
    "morning": dict(az=110, el=24, sun=(5.4, 4.4, 3.4), zen=(0.15, 0.32, 0.72), hor=(0.72, 0.78, 0.88),
                    haze=8000.0, exp=0.68),
}


@njit(cache=True, inline="always")
def bil(H, inv, x, y):
    n = H.shape[0]
    fx = min(max(x * inv - 0.5, 0.0), n - 1.001)
    fy = min(max(y * inv - 0.5, 0.0), n - 1.001)
    i, j = int(fy), int(fx)
    a, b = fx - j, fy - i
    return (H[i, j] * (1 - a) + H[i, j + 1] * a) * (1 - b) + (H[i + 1, j] * (1 - a) + H[i + 1, j + 1] * a) * b


@njit(cache=True)
def march(H, inv, L, zmax, o, d, t0, tmax, rel):
    t, prev = t0, t0
    while t < tmax:
        x, y, z = o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t
        if z > zmax and d[2] >= 0:
            return -1.0
        if x < 0 or x > L or y < 0 or y > L:
            if d[2] < 0:
                return -o[2] / d[2]                       # open sea beyond the map
            return -1.0
        if z < bil(H, inv, x, y):
            lo, hi = prev, t
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                if o[2] + d[2] * mid < bil(H, inv, o[0] + d[0] * mid, o[1] + d[1] * mid):
                    hi = mid
                else:
                    lo = mid
            return hi
        prev = t
        t += max(1.0, rel * t)
    return -1.0


@njit(cache=True)
def sky(d, sun, sunc, zen, hor, tex, clouds):
    t = max(d[2], 0.0)
    k = (1 - t) ** 5
    c = hor * k + zen * (1 - k)
    sd = max(d[0] * sun[0] + d[1] * sun[1] + d[2] * sun[2], 0.0)
    c = c + sunc * (0.05 * sd ** 6 + 0.25 * sd ** 60)
    if sd > 0.99993:
        c = c + sunc * 25
    if clouds and d[2] > 0.02:
        u, v = d[0] / d[2] * 0.11, d[1] / d[2] * 0.11
        m = tex.shape[0]
        q = tex[int(u * m) % m, int(v * m) % m] * 0.65 + tex[int(u * 3.1 * m) % m, int(v * 3.1 * m) % m] * 0.35
        a = min(max((q - 0.52) / 0.2, 0.0), 1.0) * min(d[2] * 6, 1.0) * 0.85
        lit = hor * 0.55 + sunc * (0.10 + 0.25 * sd ** 4)
        c = c * (1 - a) + lit * a
    return c


@njit(cache=True)
def lit(H, alb, emis, inv, L, zmax, p, sun, sunc, amb, tex, shadows):
    """Colour of a land point under sun and sky."""
    n = H.shape[0]
    i = min(max(int(p[1] * inv), 0), n - 1)
    j = min(max(int(p[0] * inv), 0), n - 1)
    e = 2.5
    gx = (bil(H, inv, p[0] + e, p[1]) - bil(H, inv, p[0] - e, p[1])) / (2 * e)
    gy = (bil(H, inv, p[0], p[1] + e) - bil(H, inv, p[0], p[1] - e)) / (2 * e)
    q = np.sqrt(gx * gx + gy * gy + 1)
    nx, ny, nz = -gx / q, -gy / q, 1 / q
    m = tex.shape[0]
    # fine surface grain: tilts the normal a little so near ground is not flat colour
    f1, e1 = 0.045, 0.12
    c0 = tex[int(p[0] * f1 * m) % m, int(p[1] * f1 * m) % m]
    bx = (tex[int((p[0] + e1) * f1 * m) % m, int(p[1] * f1 * m) % m] - c0) / e1
    by = (tex[int(p[0] * f1 * m) % m, int((p[1] + e1) * f1 * m) % m] - c0) / e1
    nx, ny = nx - 0.5 * bx, ny - 0.5 * by
    q = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / q, ny / q, nz / q
    ndl = max(nx * sun[0] + ny * sun[1] + nz * sun[2], 0.0)
    sh = 1.0
    if ndl > 0 and shadows:
        ps = p + np.array([nx, ny, nz + 0.6])
        if march(H, inv, L, zmax, ps, sun, 2.0, 5000.0, 0.03) > 0:
            sh = 0.0
    g = 0.55 + 0.45 * (tex[int(p[0] * 0.006 * m) % m, int(p[1] * 0.006 * m) % m]
                       + tex[int(p[1] * 0.031 * m) % m, int(p[0] * 0.031 * m) % m])
    fx = min(max(p[0] * inv - 0.5, 0.0), n - 1.001)
    fy = min(max(p[1] * inv - 0.5, 0.0), n - 1.001)
    i0, j0 = int(fy), int(fx)
    a, b = fx - j0, fy - i0
    col = (alb[i0, j0] * (1 - a) + alb[i0, j0 + 1] * a) * (1 - b) + (alb[i0 + 1, j0] * (1 - a) + alb[i0 + 1, j0 + 1] * a) * b
    g *= 0.8 + 0.4 * tex[int(p[0] * 0.17 * m) % m, int(p[1] * 0.17 * m) % m]
    return col * g * (sunc * ndl * sh + amb * (0.55 + 0.45 * nz)) + emis[i, j]


@njit(cache=True)
def hazed(c, d, t, sun, sunc, zen, hor, haze, tex):
    f = 1 - np.exp(-t / haze)
    dh = np.array([d[0], d[1], 0.0])
    dh /= np.sqrt(dh[0] ** 2 + dh[1] ** 2) + 1e-9
    return c * (1 - f) + sky(dh, sun, sunc, zen, hor, tex, False) * f


@njit(cache=True)
def shade(H, alb, emis, water, inv, L, zmax, o, d, t, sun, sunc, zen, hor, haze, tex):
    if t < 0:
        return sky(d, sun, sunc, zen, hor, tex, True)
    p = o + d * t
    n = H.shape[0]
    inside = p[0] > 0 and p[0] < L and p[1] > 0 and p[1] < L
    i = min(max(int(p[1] * inv), 0), n - 1)
    j = min(max(int(p[0] * inv), 0), n - 1)
    amb = hor * 0.30 + zen * 0.35
    if (not inside) or water[i, j]:
        m = tex.shape[0]
        s = 0.9 if inside and p[2] > 1.0 else 2.2                # lake calm, sea rougher
        a = tex[int(p[0] * 0.031 * m) % m, int(p[1] * 0.031 * m) % m] - 0.5
        b = tex[int(p[1] * 0.043 * m) % m, int(p[0] * 0.043 * m) % m] - 0.5
        nrm = np.array([a * 0.10 * s, b * 0.10 * s, 1.0])
        nrm /= np.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + 1)
        dn = d[0] * nrm[0] + d[1] * nrm[1] + d[2] * nrm[2]
        r = d - 2 * dn * nrm
        if r[2] < 0.01:
            r[2] = 0.01
        r /= np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        refl = sky(r, sun, sunc, zen, hor, tex, False)
        if inside:                                               # the land, mirrored
            p2 = p + np.array([0.0, 0.0, 0.3])
            t2 = march(H, inv, L, zmax, p2, r, 4.0, 9000.0, 0.012)
            if t2 > 0:
                c2 = lit(H, alb, emis, inv, L, zmax, p2 + r * t2, sun, sunc, amb, tex, False)
                refl = hazed(c2, r, t2, sun, sunc, zen, hor, haze, tex)
        fres = 0.03 + 0.97 * (1 - min(max(-dn, 0.0), 1.0)) ** 5
        sd = max(r[0] * sun[0] + r[1] * sun[1] + r[2] * sun[2], 0.0)
        base = np.array([0.02, 0.07, 0.10]) if not inside else alb[i, j].astype(np.float64)
        c = base * amb * 1.3 * (1 - fres) + refl * fres + sunc * 3.0 * sd ** 350
    else:
        c = lit(H, alb, emis, inv, L, zmax, p, sun, sunc, amb, tex, True)
    return hazed(c, d, t, sun, sunc, zen, hor, haze, tex)


@njit(parallel=True, cache=True)
def _render(H, alb, emis, water, inv, L, zmax, o, fwd, right, upv, tanf, wpx, hpx,
            sun, sunc, zen, hor, haze, tex, ortho):
    img = np.zeros((hpx, wpx, 3), np.float32)
    for y in prange(hpx):
        for x in range(wpx):
            u = (2 * (x + 0.5) / wpx - 1) * tanf * wpx / hpx
            v = (1 - 2 * (y + 0.5) / hpx) * tanf
            if ortho > 0:
                oo = o + right * u * ortho + upv * v * ortho
                d = fwd.copy()
            else:
                oo = o
                d = fwd + right * u + upv * v
                d /= np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            t = march(H, inv, L, zmax, oo, d, 0.5, 30000.0, 0.0035)
            img[y, x] = shade(H, alb, emis, water, inv, L, zmax, oo, d, t, sun, sunc, zen, hor, haze, tex)
    return img


def _tex(seed=3, m=512):
    """Tileable soft noise (filtered in frequency space, so it wraps cleanly)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(m)[:, None] ** 2 + np.fft.fftfreq(m)[None, :] ** 2
    f[0, 0] = 1
    t = np.fft.ifft2(np.fft.fft2(rng.normal(size=(m, m))) / f ** 0.85).real
    t = (t - t.mean()) / t.std()
    return np.clip(0.5 + t / 5, 0, 1).astype(np.float32)


TEX = _tex()


def view(D, pos, look_at=None, yaw=None, pitch=0.0, eye=1.7, fov=72, light="afternoon",
         size=(1280, 720), abs_z=None):
    """Render from pos (x, y). Aim with look_at (x, y[, z]) or yaw in compass degrees."""
    lt = LIGHTS[light]
    inv = 1.0 / D["dx"]
    oz = abs_z if abs_z is not None else bil(D["ground"], inv, pos[0], pos[1]) + eye
    o = np.array([pos[0], pos[1], oz])
    if look_at is not None:
        tz = look_at[2] if len(look_at) > 2 else bil(D["ground"], inv, look_at[0], look_at[1])
        f = np.array([look_at[0] - o[0], look_at[1] - o[1], 0.0])
        f[2] = np.tan(np.radians(pitch)) * np.hypot(f[0], f[1]) + (tz - oz if len(look_at) > 2 else 0)
    else:
        a = np.radians(yaw)
        f = np.array([np.sin(a), np.cos(a), np.tan(np.radians(pitch))])
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    a, e = np.radians(lt["az"]), np.radians(lt["el"])
    sun = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    ss = 2
    img = _render(D["H"], D["alb"], D["emis"], D["water"], inv, D["L"], float(D["H"].max()) + 5, o, f, r, u,
                  np.tan(np.radians(fov / 2)), size[0] * ss // 1, size[1] * ss // 1,
                  sun, np.array(lt["sun"]), np.array(lt["zen"]), np.array(lt["hor"]), lt["haze"], TEX, 0.0)
    x = img * lt["exp"]
    x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    out = (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    return Image.fromarray(out).resize(size, Image.LANCZOS)


TINT = dict(field=(120, 150, 80), lake=(110, 150, 100), forest=(60, 105, 60), desert=(205, 150, 95),
            ash=(95, 88, 86), moor=(150, 140, 110))


def diag(W, PL, px=1100):
    z, dx, n, R = W["z"], W["dx"], W["n"], W["rooms"]
    gy, gx = np.gradient(z, dx)
    sh = np.clip(0.62 + 0.5 * (-gx * 0.6 + gy * 0.6), 0.15, 1.25)
    col = np.array([TINT[t] for t in R["theme"]] + [(40, 70, 110)], float)[W["label"]]
    hi = np.clip((z - 260) / 300, 0, 1)[..., None]
    col = col * (1 - hi) + np.array([235, 232, 228]) * hi
    c = col * sh[..., None]
    c[W["slope"] > 0.75] *= 0.72
    if "floor" in W:
        c[W["floor"]] = c[W["floor"]] * 0.8 + np.array([255, 255, 200]) * 0.2
    c[~W["land"]] = (38, 66, 104)
    c[W["lake"]] = (70, 120, 170)
    c[W["river"]] = (60, 120, 200)
    lab = W["label"]
    edge = (lab != np.roll(lab, 1, 0)) | (lab != np.roll(lab, 1, 1))
    c[edge & W["land"]] = (255, 90, 90)
    im = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)[::-1]).resize((px, px), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    L = n * dx
    P = lambda p: (p[0] / L * px, (1 - p[1] / L) * px)
    for road in PL["roads"]:
        if road is not None:
            dr.line([P(p) for p in road], fill=(255, 235, 120), width=2)
    S = PL["sites"]
    for i, p in enumerate(S["town"]):
        x, y = P(p)
        dr.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255), outline=(0, 0, 0))
        dr.text((x + 8, y - 8), f"{i} {R['theme'][i]} {R['h'][i]:.0f}m", fill=(255, 255, 255), stroke_width=2,
                stroke_fill=(0, 0, 0), font_size=15)
    for p in S["landmark"]:
        x, y = P(p)
        dr.polygon([(x, y - 8), (x - 6, y + 5), (x + 6, y + 5)], fill=(255, 120, 255), outline=(0, 0, 0))
    for d, p in zip(R["doors"], S["door"]):
        x, y = P(p)
        dr.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(0, 255, 255) if d["kind"] == "door" else (255, 160, 0), width=3)
    x, y = P(S["peak"])
    dr.text((x - 6, y - 10), "^", fill=(255, 0, 0), font_size=26, stroke_width=2, stroke_fill=(255, 255, 255))
    return im



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
    ash = smoothstep(ash_w + 0.2 * n1, 0.45, 0.7)
    desert = smoothstep(des_w + 0.15 * n1, 0.45, 0.65) * (1 - ash)
    moor = smoothstep(moor_w + 0.2 * n1, 0.45, 0.7)
    north = -gy / (slope + 1e-6)

    area = W["area"]
    riv = np.zeros_like(area, bool)
    for a_min, grow in ((4e5, 0), (3e6, 1)):
        m = area > a_min
        riv |= ndimage.binary_dilation(m, iterations=grow) if grow else m
    riv = (zo(riv) > 0.55) & land & ~lake

    dry = smoothstep(n1 + 0.5 * fld_w, -0.2, 0.6)
    grass = np.stack([0.06 + 0.08 * dry, 0.11 + 0.035 * dry, 0.03 + 0.02 * dry], -1)
    tundra = np.array([0.17, 0.15, 0.085])
    hi = np.maximum(smoothstep(z, 330, 480), 0.85 * moor)[..., None]
    grass = grass * (1 - hi) + tundra * hi
    rock = np.array([0.30, 0.28, 0.26]) * (0.8 + 0.4 * n3[..., None])
    alb = grass.copy()
    w = smoothstep(slope + 0.25 * n2, 0.85, 1.15)[..., None]
    alb = alb * (1 - w) + rock * w
    band = 0.5 + 0.5 * np.sin((z + 30 * n1) / 7.0)
    strata = np.array([0.36, 0.13, 0.06]) * (1 - band[..., None]) + np.array([0.50, 0.33, 0.19]) * band[..., None]
    sand = np.array([0.36, 0.21, 0.10]) * (0.9 + 0.2 * n2[..., None])
    st = smoothstep(slope, 0.35, 0.7)[..., None]
    dcol = sand * (1 - st) + strata * st
    alb = alb * (1 - desert[..., None]) + dcol * desert[..., None]
    acol = np.array([0.075, 0.07, 0.068]) * (0.7 + 0.6 * n3[..., None])
    alb = alb * (1 - ash[..., None]) + acol * ash[..., None]
    snowline = 520 + 60 * n1 - 80 * north
    snow = smoothstep(z, snowline, snowline + 70) * (1 - smoothstep(slope + 0.2 * n2, 0.9, 1.3))
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
        & (smoothstep(n1 * 0.5 + 0.5, 1 - dens - 0.15, 1 - dens + 0.15) > 0.5) & (n3 < 0.85)
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
    pine = np.clip(smoothstep(z, 120, 300) + 0.5 * moor, 0, 1)
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
