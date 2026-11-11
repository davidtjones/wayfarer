"""Preview views of a rooms world, one walk through every door.

    pixi run python rshots.py <seed> <out_dir> [key=value ...]
"""
import os
import sys
import numpy as np
from rrun import build, parse
from rdress import dress
from rmap import diag
import render


def door_views(W, PL):
    """For each door road: 150 m before the door, at the door, and 200 m past it."""
    dx, lab, R = W["dx"], W["label"], W["rooms"]
    out = []
    for k, (d, road) in enumerate(zip(R["doors"], PL["roads"])):
        if road is None:
            continue
        li = lab[(road[:, 1] / dx).astype(int), (road[:, 0] / dx).astype(int)]
        cross = np.nonzero(li == d["b"])[0]
        if len(cross) == 0:
            continue
        c0 = int(cross[0])
        for tag, off in (("a_before", -9), ("b_door", 0), ("c_after", 9)):
            i = int(np.clip(c0 + off, 0, len(road) - 14))
            out.append((f"door{k:02d}_{d['a']}to{d['b']}_{tag}", road[i], road[min(i + 12, len(road) - 1)]))
    return out


if __name__ == "__main__":
    seed, out = int(sys.argv[1]), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    W, PL, M = build(seed, parse([a for a in sys.argv[3:] if "=" in a and not a.startswith("--")]))
    diag(W, PL).save(f"{out}/map.png")
    D = dress(W, PL)
    S = PL["sites"]
    jobs = [(f"town{i}_to_landmark", p, S["landmark"][i]) for i, p in enumerate(S["town"])]
    jobs += door_views(W, PL)
    only = [a[7:] for a in sys.argv if a.startswith("--only=")]
    for name, pos, look in jobs:
        if only and not any(o in name for o in only):
            continue
        render.view(D, pos=pos, look_at=look, pitch=4, fov=75).save(f"{out}/{name}.jpg", quality=90)
        print(" ", name)
    L = W["n"] * W["dx"]
    render.view(D, pos=(L / 2, -900), look_at=(L / 2, L / 2, 0), abs_z=3600, pitch=0, fov=62).save(f"{out}/aerial.jpg", quality=90)
