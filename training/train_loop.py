import torch 
import torch.nn as nn
import torch.optim import optim
from torch.utils.data import DataLoader
from .losses import YoloLoss
from .model import YoloMini 
from .dataset import YoloToyDataset
from tqdm import tqdm 
import os, json

class Trainer:
  def __init__(self, cfg):
    self.cfg = cfg 
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = YoloMini(nc=cfg['num_classes']).to(self.device)
    self.criterion = YoloLoss(cls=0.5, obj=1.0)
    self.opt = optim.SGD(self.model.parameters(), lr=cfg['lr'], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
  
  def loaders(self):
    train_ds = YoloToyDataset(cfg['data_dir'], 'train', cfg['img_size'])
    val_ds = YoloToyDataset(cfg['data_dir'], 'val', cfg['img_size'])
    # collate_fn:
    # - b: [(image1, target1), (image2, target2)]
    # - zip(*b): [(image1,image2,...), (target1,target2,...)]
    return (
      DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=lambda b: list(zip(*b))),
      DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=lambda b: list(zip(*b)))
    )

  def train(self):
    cfg = self.cfg 
    train_loader, val_loader = self.loaders()
    best = float('inf')
    hist = []
    self.model.train()
    for epoch in range(cfg['epochs']):
      total_loss = 0
      # targets: list of {"cls":(num_classes,), "grid_x":(1,), "grid_y":(1,), "grid_anchor":(1,), "box":(1,4)}
      for imgs, targets in tqdm(train_loader, desc=f"Train {epoch+1}/{cfg['epochs']}"):
        imgs = torch.stack(imgs).to(self.device)
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
      


