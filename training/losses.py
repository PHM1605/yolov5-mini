import torch 
import torch.nn as nn

def bbox_ciou(box1, box2, eps=1e-7):
  # box1: (N,4) or (4,) => (1,4)
  # box2: (M,4) or (4,) => (1,4)
  if box1.ndim == 1:
    box1 = box1.unsqueeze(0)
  if box2.ndim == 1:
    box2 = box2.unsqueeze(0)
  
  b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3] # (N,)
  b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3] # (M,)

  # intersection
  # NOTE: 
  # b1_x1[:,None] => (N,1)
  # b2_x1[None,:] => (1,M)
  inter_x1 = torch.max(b1_x1[:,None], b2_x1[None,:]) # broadcast => (N,M)
  inter_y1 = torch.max(b1_y1[:,None], b2_y1[None,:]) # broadcast => (N,M)
  inter_x2 = torch.min(b1_x2[:,None], b2_x2[None,:]) # (N,M)
  inter_y2 = torch.min(b1_y2[:,None], b2_y2[None,:]) # (N,M)
  inter = (inter_x2-inter_x1).clamp(0) * (inter_y2-inter_y1).clamp(0)

class YoloLoss(nn.Module):
  def __init__(self, cls, obj):
    super().__init__()
    self.cls_loss = nn.BCEWithLogitsLoss()
    self.obj_loss = nn.BCEWithLogitsLoss()
  
  def forward(self, predict, target):
    # predict: (batch, num_anchors, H, W, 5+num_classes)
    # target: list of {"cls":(num_classes,), "grid_x":(1,), "grid_y":(1,), "grid_anchor":(1,), "box":(1,4)}
    for pred, tar in zip(predict, target):
      if tar is None:
        obj_target = torch.zeros_like(pred[...,4]) 
      grid_x, grid_y, grid_anchor = tar["grid_x"], tar["grid_y"], tar["grid_anchor"]
      # pred[...,4]: flag if there is object or not at that cell
      # (num_anchors,H,W)
      # object_loss calculation
      obj_target = torch.zeros_like(pred[...,4]) 
      obj_target[grid_anchor,grid_y,grid_x] = 1.0
      loss_obj += self.obj_loss(pred[...,4], obj_target)
      # box_loss calculation
      pr = pred[grid_anchor,grid_y,grid_x,:4] # (4,)
      if pr.ndim == 1:
        pr = pr.unsqueeze(0) # (1,4)
      ciou = bbox_ciou(pr, pred["box"])
