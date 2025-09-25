import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .losses import YoloLoss
from .model import YoloMini 
from .dataset import YoloToyDataset
from tqdm import tqdm 
import os, json, math

# box1, box2: (cx, cy, w, h) (normalized)
def compute_iou(box1, box2):
  def to_xyxy(cx, cy, w, h):
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
  
  x1, y1, x2, y2 = to_xyxy(box1[0], box1[1], box1[2], box1[3])
  x1g, y1g, x2g, y2g = to_xyxy(box2[0], box2[1], box2[2], box2[3])
  
  xi1 = max(x1, x1g)
  yi1 = max(y1, y1g)
  xi2 = min(x2, x2g)
  yi2 = min(y2, y2g)
  inter = max(0, xi2-xi1) * max(0, yi2-yi1)
  area1 = (x2-x1) * (y2-y1)
  area2 = (x2g-x1g) * (y2g-y1g)
  union = area1 + area2 - inter 
  return inter / (union + 1e-6)

class Trainer:
  def __init__(self, cfg):
    self.cfg = cfg 
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = YoloMini(num_classes=cfg['num_classes']).to(self.device)
    self.criterion = YoloLoss(cls=0.5, obj=1.0)
    self.opt = optim.SGD(self.model.parameters(), lr=cfg['lr'], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
  
  # batch: [(image1, target1), (image2, target2)]
  # zip(*batch): [(image1,image2,...), (target1,target2,...)]
  def collate_fn(self, batch): 
    images, targets = zip(*batch) # images:(image1,image2,...), targets:(target1,target2,...)
    images = torch.stack(images, dim=0) # (batch,3,h,w)
    return images, list(targets)

  def loaders(self):
    train_ds = YoloToyDataset(self.cfg['data_dir'], 'train', self.cfg['img_size'])
    val_ds = YoloToyDataset(self.cfg['data_dir'], 'val', self.cfg['img_size'])

    return (
      DataLoader(train_ds, batch_size=self.cfg['batch_size'], shuffle=True, collate_fn=self.collate_fn),
      DataLoader(val_ds, batch_size=self.cfg['batch_size'], shuffle=True, collate_fn=self.collate_fn)
    )

  def train(self):
    cfg = self.cfg 
    train_loader, val_loader = self.loaders()
    best = float('inf')
    hist = []
    
    for epoch in range(cfg['epochs']):
      self.model.train()
      total_loss = 0
      # targets: list of {"cls":(num_classes,), "grid_x":(1,), "grid_y":(1,), "grid_anchor":(1,), "box":(1,4)}
      for imgs, targets in tqdm(train_loader, desc=f"Train {epoch+1}/{cfg['epochs']}"):
        imgs = imgs.to(self.device)
        # preds: (batch, num_anchors, h, w, 5+num_classes)
        preds = self.model(imgs)
        # batch_targets: list of {"cls":(num_classes,), "grid_x":(1,), "grid_y":(1,), "grid_anchor":(1,), "box":(1,4)}
        batch_targets = []
        for t in targets:
          batch_targets.append(t 
            if t is None 
            else {k:(v.to(self.device)) for k,v in t.items()} 
          )
        loss = self.criterion(preds, batch_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        total_loss += loss.item()
      
      # validation
      self.model.eval()
      correct = 0
      total = 0 
      with torch.no_grad():
        for imgs, targets in val_loader:
          imgs = imgs.to(self.device)
          preds = self.model(imgs) # (batch,num_anchors,h_cell,w_cell,5+num_classes)
          batch, num_anchors, h_cell, w_cell, dim = pred.shape # last dim: 5+num_classes
          cell_size = self.cfg["img_size"] // h_cell 

          for img_idx in range(batch):
            if targets[img_idx] is None:
              continue 
            pred = preds[img_idx] # (num_anchors,h_cell,w_cell,5+num_classes)
            obj = pred[...,4] # (num_anchors,h_cell,w_cell)
            # obj.argmax(): flatten, then tell which cell has object
            # unravel_index(): location anchor & grid_x & grid_y of that max cell
            anchor, grid_x, grid_y = torch.unravel_index(obj.argmax(), obj.shape)
            # predicted values
            pr = pred[anchor,grid_x,grid_y] # (5+num_classes,)
            pred_cls = pr[5:].argmax().item() # a number
            # Predicted box - in image-size-percent
            cx_cell = torch.sigmoid(pr[0]).item()
            cy_cell = torch.sigmoid(pr[1]).item()
            cx_abs = (grid_x+cx_cell) / w_cell 
            cy_abs = (grid_y+cy_cell) / h_cell 
            w = torch.exp(pr[2]).item() / self.cfg["img_size"]
            h = torch.exp(pr[3]).item() / self.cfg["img_size"]
            # ground-truth
            tgt = targets[img_idx]
            tgt_cls = tgt["cls"].argmax().item()
            cx_gt, cy_gt, w_gt, h_gt = tgt["box"].squeeze().tolist()
            # check if iou>0.5 and class match
            iou = compute_iou([cx_abs,cy_abs,w,h], [cx_gt,cy_gt,w_gt,h_gt])
            if pred_cls == tgt_cls and iou>0.5:
              correct += 1
            total += 1

      # epoch summary
      val_acc = correct / max(total, 1)
      avg_loss = total_loss / len(train_loader)
      print(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")
      hist.append({'epoch': epoch+1, 'loss': avg_loss, 'val_acc': val_acc})
      if avg_loss < best:
        best = avg_loss
        torch.save(self.model.state_dict(), 'runs/exp1/best.pt')
      os.makedirs('runs/exp1', exist_ok=True)
      for entry in hist:
        for k, v in entry.items():
          if isinstance(v, float) and not math.isfinite(v):
            entry[k] = 0.0
      with open('runs/exp1/metrics.json', 'w') as f:
        json.dump(hist, f, indent=2)

    torch.save(self.model.state_dict(), 'runs/exp1/last.pt')
    return hist 
