"""Contact sheet of one seed's views."""
import sys, glob
from PIL import Image, ImageDraw
d = sys.argv[1]
files = [f for f in sorted(glob.glob(f"{d}/*.jpg")) if "00_map" not in f and "sheet" not in f]
cols, w, h = 3, 640, 360
rows = (len(files) + cols - 1) // cols
im = Image.new("RGB", (cols * w, rows * h))
for k, f in enumerate(files):
    t = Image.open(f).resize((w, h), Image.LANCZOS)
    ImageDraw.Draw(t).text((8, 6), f.split("/")[-1][:-4], fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), font_size=18)
    im.paste(t, ((k % cols) * w, (k // cols) * h))
im.save(f"{d}/sheet.jpg", quality=88)
