"""Print the sweep as one row per arm (mean of the seeds).

    pixi run sweep-table out/rooms/sweep.json
"""
import json
import sys
from itertools import groupby
import numpy as np

COLS = [("land_self", "own room seen"), ("land_door", "next room seen"), ("mark_self", "own landmark"),
        ("mark_door", "next landmark"), ("peak_seen", "peak seen"), ("doors_seen", "exits seen"),
        ("reveal_before", "before door"), ("reveal_after", "after door"), ("reveal_good", "clean reveal"),
        ("reveal_landmark", "landmark at door"), ("floor_frac", "flat floor"), ("room_km2", "room km2")]

if __name__ == "__main__":
    rows = [r for r in json.load(open(sys.argv[1])) if "error" not in r]
    print("| arm | " + " | ".join(n for _, n in COLS) + " | doors | blocked | walls | leaks |")
    print("|---" * (len(COLS) + 5) + "|")
    for (arm, val), g in groupby(rows, key=lambda r: (r["arm"], r["value"])):
        g = list(g)
        cell = lambda k: f"{np.mean([r[k] for r in g if r.get(k) is not None]):.2f}"
        tot = lambda k: str(sum(r[k] for r in g))
        print(f"| {arm} {val:g} | " + " | ".join(cell(k) for k, _ in COLS)
              + f" | {tot('doors')} | {tot('blocked')} | {tot('walls')} | {tot('leaks')} |")
    A, Nn, Sl = [], [], []
    for r in rows:
        A += r["room_area_km2"]
        Nn += r["neighbours"]
        Sl += r["door_slots"]
    A, Nn, Sl = np.array(A), np.array(Nn), np.array(Sl)
    print("\n| room size km2 | rooms | land neighbours mean | max | door slots mean | max |\n|---|---|---|---|---|---|")
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 8), (8, 14), (14, 40)):
        m = (A >= lo) & (A < hi)
        if m.any():
            print(f"| {lo}-{hi} | {m.sum()} | {Nn[m].mean():.1f} | {Nn[m].max()} | {Sl[m].mean():.1f} | {Sl[m].max()} |")
