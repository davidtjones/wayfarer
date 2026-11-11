"""Stage 1, end to end: brief -> layout -> land -> placed journey -> report + views.

    pixi run python run.py <seed> [--no-views]
"""
import json
import sys
import time
import numpy as np
from layout import make_layout
from erode import form
from style import finish
from place import place, bearing, ground
from dress import dress
import render


def build(seed, log=print):
    t = time.time()
    lay = make_layout(seed)
    z, _ = form(lay)
    W = finish(lay, z, log)
    PL = place(W, log)
    log(f"  built in {time.time() - t:.0f} s")
    return W, PL


def shots(W, PL, out):
    D = dress(W, PL)
    S, rt, r1 = PL["sites"], PL["route"], PL["reveal_index"]
    ahead = lambda line, k, d=8: line[min(k + d, len(line) - 1)]
    pk = int(np.argmax([ground(W, p) for p in PL["leg2"]]))
    horn3 = (*S["horn"], ground(W, S["horn"]) - 150)
    west = lambda p, d=600: p + W["w"] * d
    r0 = int(np.argmax(PL["lake_profile"] > 0.3))          # the reveal, on the road itself
    k6 = int(len(rt) * 0.62)
    jobs = [
        ("00_map", None),
        ("01_start", dict(pos=rt[1], look_at=ahead(rt, 1, 12), pitch=3)),
        ("02_walk_out", dict(pos=rt[r0 // 2], look_at=ahead(rt, r0 // 2, 12), pitch=3)),
        ("03_road_before_lip", dict(pos=rt[max(r0 - 6, 0)], look_at=west(rt[max(r0 - 6, 0)]), pitch=0)),
        ("04_road_at_lip", dict(pos=rt[min(r0 + 3, len(rt) - 1)], look_at=west(rt[min(r0 + 3, len(rt) - 1)]), pitch=-6)),
        ("05_overlook", dict(pos=S["overlook"], look_at=W["pts"]["lake"], pitch=3, fov=80)),
        ("06_overlook_sunset", dict(pos=S["overlook"], look_at=W["pts"]["lake"], pitch=4, fov=80, light="sunset")),
        ("07_lakeside", dict(pos=rt[k6], look_at=S["horn"], pitch=10, fov=75)),
        ("08_town_arrival", dict(pos=rt[-22], look_at=S["town"], pitch=4, fov=65)),
        ("09_town_keep_horn", dict(pos=PL["leg3"][14], look_at=S["keep"], pitch=9, fov=60)),
        ("10_town_volcano", dict(pos=PL["leg3"][14], look_at=S["volcano"], pitch=5, fov=65, light="morning")),
        ("11_pass_desert", dict(pos=S["pass"] + 45 * (S["oasis"] - S["pass"]) / np.linalg.norm(S["oasis"] - S["pass"]), look_at=S["oasis"], pitch=-1, fov=80)),
        ("12_desert", dict(pos=PL["leg2"][-16], look_at=S["oasis"], pitch=5, fov=75)),
        ("13_aerial", dict(pos=W["pts"]["lake_e"] - W["w"] * 2600, look_at=horn3[:2], abs_z=1500, pitch=-22, fov=70)),
    ]
    for name, kw in jobs:
        t = time.time()
        im = render.map_view(D, PL) if kw is None else render.view(D, **kw)
        im.save(f"{out}/{name}.jpg", quality=92)
        print(f"  {name}  {time.time() - t:.1f} s")


if __name__ == "__main__":
    import os
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = f"out/seed{seed}"
    os.makedirs(out, exist_ok=True)
    W, PL = build(seed)
    R = PL["report"]
    for k, v in R["checks"].items():
        print(f"  [{'pass' if v else 'FAIL'}] {k}")
    print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.items() if k != "checks"})
    json.dump(R, open(f"{out}/report.json", "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    np.savez_compressed(f"{out}/world.npz", z=W["z"].astype(np.float32), land=W["land"], lake=W["lake"],
                        route=PL["route"], lake_profile=PL["lake_profile"])
    if "--no-views" not in sys.argv:
        shots(W, PL, out)
