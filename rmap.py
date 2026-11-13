"""Diagnostic map of a rooms world: relief, room borders, floors, doors, roads, sites."""
import numpy as np
from PIL import Image, ImageDraw

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
