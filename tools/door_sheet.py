"""Contact sheets of game3d captures: one row per door (before, at, after).

    pixi run python -m tools.door_sheet <capture_dir> <out_prefix>
"""
import glob
import os
import sys
from PIL import Image, ImageDraw

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    doors = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{src}/door*_1before.png")})
    w, h = 640, 360
    rows = [doors[i:i + 6] for i in range(0, len(doors), 6)]
    for k, grp in enumerate(rows):
        im = Image.new("RGB", (3 * w, len(grp) * h))
        dr = ImageDraw.Draw(im)
        for r, d in enumerate(grp):
            for c, tag in enumerate(("1before", "2door", "3after")):
                f = f"{src}/{d}_{tag}.png"
                if os.path.exists(f):
                    im.paste(Image.open(f).convert("RGB").resize((w, h), Image.LANCZOS), (c * w, r * h))
                dr.text((c * w + 8, r * h + 6), f"{d}  {tag[1:]}", fill=(255, 255, 255), stroke_width=2,
                        stroke_fill=(0, 0, 0), font_size=18)
        im.save(f"{out}_doors{k}.jpg", quality=88)
    towns = sorted(glob.glob(f"{src}/room*_town.png"))
    if towns:
        cols = 3
        im = Image.new("RGB", (cols * w, ((len(towns) + cols - 1) // cols) * h))
        dr = ImageDraw.Draw(im)
        for i, f in enumerate(towns):
            x, y = (i % cols) * w, (i // cols) * h
            im.paste(Image.open(f).convert("RGB").resize((w, h), Image.LANCZOS), (x, y))
            dr.text((x + 8, y + 6), os.path.basename(f)[:-4], fill=(255, 255, 255), stroke_width=2,
                    stroke_fill=(0, 0, 0), font_size=18)
        im.save(f"{out}_rooms.jpg", quality=88)
