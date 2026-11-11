"""Build one rooms world and measure it.

    pixi run python rrun.py <seed> [key=value ...]     e.g. rooms=9 wall=160 loops=0.35 hub=1.6
"""
import json
import sys
import time
import numpy as np
from rooms import make_rooms, Cfg
from erode import form
from rstyle import finish
from rplace import place, measure


def build(seed, cfg=Cfg(), log=print):
    t = time.time()
    lay = make_rooms(seed, cfg)
    z, _ = form(lay)
    W = finish(lay, z, log)
    PL = place(W, log)
    M = measure(W, PL, log)
    log(f"  built and measured in {time.time() - t:.0f} s")
    return W, PL, M


def parse(args):
    kw = {}
    for a in args:
        k, v = a.split("=")
        kw[k] = int(v) if k in ("rooms", "n") else float(v)
    return Cfg(**kw)


SHOW = ("see_self", "see_door", "see_wall", "see_far", "land_self", "land_door", "land_wall", "land_far",
        "mark_self", "mark_door", "mark_wall", "mark_far", "rooms_in_view", "marks_in_view", "peak_seen", "doors", "walls", "blocked", "leaks",
        "reveal_before", "reveal_after", "reveal_good", "reveal_landmark", "floor_frac", "room_km2", "peak_m")

if __name__ == "__main__":
    seed = int(sys.argv[1])
    args = [a for a in sys.argv[2:] if "=" in a and not a.startswith("--")]
    W, PL, M = build(seed, parse(args))
    for a in sys.argv:
        if a.startswith("--map="):
            from rmap import diag
            diag(W, PL).save(a[6:])
    for k in SHOW:
        v = M.get(k)
        print(f"  {k:16s} {v if not isinstance(v, float) else round(v, 3)}")
    print("  doors per room  ", M["doors_per_room"])
    print("  themes          ", W["rooms"]["theme"])
