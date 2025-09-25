import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .losses import YoloLoss
from .model import YoloMini 
from .dataset import YoloToyDataset
from tqdm import tqdm 
import os, json, math

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
          pred = self.model(imgs)
          batch, num_anchors, h_cell, w_cell, _ = pred.shape # last dim: 5+num_classes
          obj = pred[...,4] # (batch, num_anchors, h_cell, w_cell)
          best_idx = obj.view(batch,-1).argmax(-1) # argmax of (batch, num_anchors*h_cell*w_cell)
          grid_x = best_idx % w_cell
          grid_y = (best_idx // w_cell) % h_cell
          grid_anchor = (best_idx // (h_cell*w_cell)) % num_anchors
          for idx, tar in enumerate(targets):
            if tar is None: continue
            if grid_x[idx].item()==tar['grid_x'].item() and grid_y[idx].item()==tar['grid_y'].item():
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
