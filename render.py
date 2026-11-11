"""Ground-level and map views of a dressed world. A viewing aid, nothing more.

Walks rays over the height surface, shades with sun, sky light, cast shadows and
distance haze, and mirrors the land in still water.
"""
import numpy as np
from numba import njit, prange
from PIL import Image, ImageDraw

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


def map_view(D, PL, px=1500, light="afternoon"):
    """Top-down map with the routes and sites drawn on."""
    lt = LIGHTS[light]
    L = D["L"]
    a, e = np.radians(lt["az"]), np.radians(50)
    sun = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    o = np.array([L / 2, L / 2, 6000.0])
    img = _render(D["H"], D["alb"], D["emis"], D["water"], 1.0 / D["dx"], L, float(D["H"].max()) + 5, o,
                  np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 1.0, px, px,
                  sun, np.array(lt["sun"]), np.array(lt["zen"]), np.array(lt["hor"]), 1e9, TEX, L / 2)
    x = img * 0.7
    x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    im = Image.fromarray((np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    P = lambda p: (p[0] / L * px, (1 - p[1] / L) * px)
    for line, col in ((PL["route"], (255, 235, 120)), (PL["leg2"], (255, 170, 90)), (PL["leg3"], (230, 230, 230))):
        dr.line([P(p) for p in line], fill=col, width=3)
    for k, p in PL["sites"].items():
        x0, y0 = P(p)
        dr.ellipse([x0 - 6, y0 - 6, x0 + 6, y0 + 6], outline=(255, 255, 255), width=2)
        dr.text((x0 + 9, y0 - 16), k, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font_size=20)
    return im
