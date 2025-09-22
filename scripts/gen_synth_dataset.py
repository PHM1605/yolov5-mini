import random
from pathlib import Path 
from PIL import Image, ImageDraw 

random.seed(0)

OUT = Path("data")
for split in ["train", "val"]:
  (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
  (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

IMGSZ = 256
N_SAMPLES = {"train": 500, "val": 100}

# 0 = cat, 1 = dog
def make_sample(idx, split):
  img = Image.new("RGB", (IMGSZ, IMGSZ), (255, 255, 255))
  draw = ImageDraw.Draw(img)
  cls = random.randint(0, 1) # 0 = cat (triangle); 1 = dog (circle)

  size = random.randint(80, 120)
  # never draw a cat/dog being cropped by image-borders
  cx = random.randint(size//2, IMGSZ-size//2)
  cy = random.randint(size//2, IMGSZ-size//2)

  # Triangle (cat)
  if cls == 0:
    half = size//2
    points = [(cx,cy-half), (cx-half,cy+half), (cx+half, cy+half)]
    draw.polygon(points, fill=(0,0,0))
    x0, y0 = cx-half, cy-half
    x1, y1 = cx+half, cy+half
  # Circle (dog)
  else:
    r = size//2
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(0,0,0))
    x0, y0 = cx-r, cy-r
    x1, y1 = cx+r, cy+r

  # Clip box inside image
  x0, y0 = max(0,x0), max(0,y0)
  x1, y1 = min(IMGSZ-1,x1), min(IMGSZ-1,y1)
  
  # YOLO label: class cx cy w h (normalized)
  w_b, h_b = x1-x0, y1-y0
  cx_n, cy_n = (x0+x1)/2/IMGSZ, (y0+y1)/2/IMGSZ 
  w_n, h_n = w_b/IMGSZ, h_b/IMGSZ

  w = h = size 
  x1, y1 = cx-w//2, cy-h//2
  x2, y2 = cx+w//2, cy+h//2
  img_path = OUT / "images" / split / f"{i:05d}.jpg"
  lbl_path = OUT / "labels" / split / f"{i:05d}.txt"

  img.save(OUT / "images" / split / f"{idx:05d}.jpg")
  with open(OUT / "labels" / split / f"{idx:05d}.txt", 'w') as f:
    f.write(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")
  

for split, n in N_SAMPLES.items():
  for i in range(n):
    make_sample(i, split)
print('Synthetic triangle(cat) vs circle (dog) dataset ready in ./data')