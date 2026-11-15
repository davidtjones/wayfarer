"""Limits of rooms and doors: vary one thing at a time, three seeds each.

    pixi run sweep out/rooms/sweep.json
"""
import json
import sys
from dataclasses import asdict
from wayfarer.rooms import Cfg
from wayfarer.build import build

BASE = dict(rooms=9, wall=160.0, loops=0.35)
ARMS = [("rooms", v) for v in (4, 6, 9, 12, 16, 22)] + [("wall", v) for v in (30.0, 70.0, 260.0)] \
    + [("loops", v) for v in (0.0, 1.0)] + [("hub", v) for v in (1.6, 2.4)]
SEEDS = (0, 1, 2)

if __name__ == "__main__":
    rows = []
    for key, val in ARMS:
        kw = dict(BASE)
        kw[key] = val
        if key == "hub":
            kw["rooms"] = 10
        for seed in SEEDS:
            try:
                W, PL, M = build(seed, Cfg(**kw), log=lambda *a: None)
            except Exception as e:                       # a failed build is a result too
                rows.append(dict(arm=key, value=val, seed=seed, error=repr(e)))
                print(key, val, seed, "FAILED", repr(e), flush=True)
                continue
            M.pop("V")
            rows.append(dict(arm=key, value=val, seed=seed, **M))
            print(key, val, seed, {k: round(M[k], 2) for k in ("see_self", "land_self", "land_door", "land_wall",
                  "land_far", "mark_self", "rooms_in_view", "marks_in_view", "peak_seen", "reveal_good", "doors_seen", "floor_frac")
                  if M.get(k) is not None}, "blocked", M["blocked"], "leaks", M["leaks"],
                  "of", M["doors"], M["walls"], flush=True)
            json.dump(rows, open(sys.argv[1], "w"))
