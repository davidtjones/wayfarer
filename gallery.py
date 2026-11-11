"""Collect game3d captures of exported worlds into one local page.

    pixi run python gallery.py <name> [<name> ...]      -> out/rooms/index.html
"""
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
G3D = os.path.join(HERE, "..", "..", "game3d")

if __name__ == "__main__":
    out = os.path.join(HERE, "out", "rooms")
    html = ["<html><body style='background:#15171a;color:#ddd;font-family:sans-serif;max-width:1500px;margin:auto'>",
            "<h1>Rooms and doors - game3d renders</h1>",
            "<p>Each row of a door sheet is one walk through one door: 180 m before, at the door, 180 m past it.</p>"]
    for name in sys.argv[1:]:
        cap = os.path.join(G3D, "capture", name)
        dst = os.path.join(out, name)
        os.makedirs(dst, exist_ok=True)
        subprocess.run([sys.executable, os.path.join(HERE, "gsheet.py"), cap, os.path.join(dst, "sheet")], check=True)
        for f in ("map.png", "report.json"):
            shutil.copy(os.path.join(G3D, "assets", "terrain", name, f), dst)
        if os.path.exists(f"{cap}/00_overview.png"):
            shutil.copy(f"{cap}/00_overview.png", dst)
        r = json.load(open(f"{dst}/report.json"))
        html.append(f"<h2>{name}</h2><p>{len(r['themes'])} rooms, {r['doors']} doors, {r['walls']} closed walls, "
                    f"{r['blocked']} blocked doors, {r['leaks']} leaking walls. Doors per room: {r['doors_per_room']}</p>")
        for f in ["00_overview.png", "map.png"] + sorted(os.path.basename(p) for p in glob.glob(f"{dst}/sheet_*.jpg")):
            if os.path.exists(f"{dst}/{f}"):
                html.append(f"<p><a href='{name}/{f}'><img src='{name}/{f}' style='max-width:100%'></a></p>")
    html.append("</body></html>")
    open(os.path.join(out, "index.html"), "w").write("\n".join(html))
    print("wrote", os.path.join(out, "index.html"))
