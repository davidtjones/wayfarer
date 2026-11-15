"""Contact sheet of the wall shots of a game3d capture: one row per wall.

    pixi run python -m tools.wall_sheet <capture_dir> <out.jpg>
"""
import glob
import os
import sys
from PIL import Image, ImageDraw

TAGS = ("face_far", "face_near", "along", "air")

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    walls = sorted({os.path.basename(f)[:-len("_air.png")] for f in glob.glob(f"{src}/wall*_air.png")})
    w, h = 640, 360
    im = Image.new("RGB", (len(TAGS) * w, len(walls) * h))
    dr = ImageDraw.Draw(im)
    for r, name in enumerate(walls):
        for c, tag in enumerate(TAGS):
            f = f"{src}/{name}_{tag}.png"
            if os.path.exists(f):
                im.paste(Image.open(f).convert("RGB").resize((w, h), Image.LANCZOS), (c * w, r * h))
            dr.text((c * w + 8, r * h + 6), f"{name} {tag}", fill=(255, 255, 255), stroke_width=2,
                    stroke_fill=(0, 0, 0), font_size=18)
    im.save(out, quality=88)
    over = f"{src}/00_overview.png"
    if os.path.exists(over):
        o = Image.open(over).convert("RGB")
        o.thumbnail((1800, 1800))
        o.save(out[:-4] + "_overview.jpg", quality=88)
