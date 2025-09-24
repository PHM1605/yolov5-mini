import torch
from torch.utils.data import Dataset 
from pathlib import Path
from PIL import Image

class YoloToyDataset(Dataset):
  def __init__(self, root, split="train", img_size=256, num_anchors=3, cell_size=8):
    self.root = Path(root)
    self.img_path = self.root / "images" / split 
    self.label_path = self.root / "labels" / split 
    self.files = sorted([path for path in self.img_path.glob("*.jpg")])
    self.img_size = img_size 
    self.num_anchors = num_anchors 
    self.cell_size = cell_size 
  
  def __len__(self):
    return len(self.files)
  
  def __getitem(self, i):
    # img: (H,W,3)
    img = Image.open(self.files[i]).convert("RGB"), resize((self.img_size, self.img_size))
    x = torch.from_numpy(np.array(img)).float().permute(2,0,1) / 255.0 # (3,H,W), normalized
    # label
    label_file = (self.label_path / (self.files[i].stem+".txt"))
    if label_file.exists():
      # NOTE: we assume label file has only 1 line (i.e. one image = one box of detection only)
      line = open(label_file).read().strip().split()
      c = int(line[0])
      cx, cy, w, h = map(float, line[1:])
      target = {'cls':torch.zeros(2)}
      target['cls'][c] = 1.0
      # Assign to grid cell + anchor0
      num_rows, num_cols = self.img_size // self.cell_size 
      grid_x = int(cx * now_cols)
      grid_y = int(cy * num_rows)
      grid_anchor = 0
      target["grid_x"] = torch.tensor([grid_x])
      target["grid_y"] = torch.tensor([grid_y])
      target["grid_anchor"] = torch.tensor([grid_anchor])

      target['box'] = torch.tensor([[cx, cy, w, h]])
    else:
      target = None
    # target: {"cls":(num_classes,), "grid_x":(1,), "grid_y":(1,), "grid_anchor":(1,), "box":(1,4)}
    return x, target

