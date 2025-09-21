import random, os
from pathlib import Path 
from PIL import Image, ImageDraw 

random.seed(0)

OUT = Path("data")
(OUT/"images"/"train").mkdir(parents=True, exist_ok=True)
(OUT/"images"/"val").mkdir(parents=True, exist_ok=True)
(OUT/"labels"/"train").mkdir(parents=True, exist_ok=True)
(OUT/"labels"/"val").mkdir(parents=True, exist_ok=True)

IMGSZ = 256
N_TRAIN = 800
N_VAL = 200

# cat: a circle with two triangles
def draw_cat(draw, cx, cy, s):
  # head
  r = s//2
  draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(30,30,30))
  # ears (triangles)
  draw.polygon([(cx-r//2,cy-r-2), (cx-r,cy-2), (cx-2,cy-r//2)], fill=(30,30,30))
  draw.polygon([(cx+r//2,cy-r-2), (cx+r,cy-2), (cx+2,cy-r//2)], fill=(30,30,30))

def draw_dog(draw, cx, cy, s):
  # head
  r = s//2
  draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(60,60,60))
  # ears (rounded)
  ear_r = s//4

# 0 = cat, 1 = dog
def make_sample(i, split):
  img = Image.new("RGB", (IMGSZ, IMGSZ), (240, 240, 240))
  draw = ImageDraw.Draw(img)
  cls = 0 if random.random() < 0.5 else 1
  size = random.randint(80, 140)
  # margin: to never draw a cat/dog being cropped by image-borders
  margin = size//2 + 4
  cx = random.randint(margin, IMGSZ-margin)
  cy = random.randint(margin, IMGSZ-margin)

  if cls == 0:
    draw_cat(draw, cx, cy, size)
  else:
    draw_cat(draw, cx, cy, size)
    #draw_dog(draw, cx, cy, size)
  img.save("test.jpg")

for i in range(1):
  make_sample(i, 'train')