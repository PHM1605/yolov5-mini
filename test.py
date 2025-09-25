import torch
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from training.dataset import YoloToyDataset

def save_debug_image(data_dir="data/", split="train", index=0, img_size=256):
  dataset = YoloToyDataset(data_dir, split, img_size)
  img, target = dataset[index]
  
  img_np = img.permute(1,2,0).cpu().numpy()
  box = target["box"].squeeze().cpu().numpy()
  cl = target["cls"].argmax().item()

  img_size = img_np.shape[0]
  cx, cy, w, h = box 
  x1 = (cx-w/2)*img_size 
  y1 = (cy-h/2)*img_size 
  x2 = (cx+w/2)*img_size
  y2 = (cy+h/2)*img_size
  rect_w = x2 - x1 
  rect_h = y2 - y1

  plt.figure(figsize=(6,6))
  plt.imshow(img_np)
  rect = patches.Rectangle(
    (x1, y1),
    rect_w,
    rect_h,
    linewidth=2,
    edgecolor='r',
    facecolor='none'
  )
  plt.gca().add_patch(rect)
  plt.title(f"Class: {cl}")
  plt.axis('off')
  plt.savefig('test.jpg', bbox_inches='tight')
  plt.close()
  print("Saved to test.jpg")

if __name__ == "__main__":
  save_debug_image()