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
  # area
  area1 = (b1_x2-b1_x1) * (b1_y2-b1_y1) 
  area2 = (b2_x2-b2_x1) * (b2_y2-b2_y1)
  union = area1[:,None] + area2[None,:] - inter 
  # iou
  iou = inter / (union + eps)
  # diagonal**2 of the box <enc> enclosing 2 boxes 
  enc_w = torch.max(b1_x2[:,None], b2_x2[None,:]) - torch.min(b1_x1[:,None], b2_x1[None,:])
  enc_h = torch.max(b1_y2[:,None], b2_y2[None,:]) - torch.min(b1_y1[:,None], b2_y1[None,:])
  diag2 = enc_w**2 + enc_h**2 + eps # (N,M)
  # center-distance**2 of the two boxes
  cen_x_b1, cen_y_b1 = (b1_x1+b1_x2)/2, (b1_y1+b1_y2)/2
  cen_x_b2, cen_y_b2 = (b2_x1+b2_x2)/2, (b2_y1+b2_y2)/2
  dist2 = (cen_x_b1[:,None]-cen_x_b2[None,:])**2 + (cen_y_b1[:,None]-cen_y_b2[None,:])**2
  # aspect ratio term
  angle1 = torch.atan((b1_x2-b1_x1)/(b1_y2-b1_y1+eps))
  angle2 = torch.atan((b2_x2-b2_x1)/(b2_y2-b2_y1+eps))
  v = (4/(3.1416**2)) * (angle1[:,None] - angle2[None,:])**2 # (N,M)
  with torch.no_grad():
    alpha = v / (1-iou+v+eps)
  ciou = iou - dist2/diag2 - alpha*v
  return ciou 

class YoloLoss(nn.Module):
  def __init__(self, cls, obj):
    super().__init__()
    self.cls_loss = nn.BCEWithLogitsLoss()
    self.obj_loss = nn.BCEWithLogitsLoss()
  
  def forward(self, predict, target):
    # predict: (batch, num_anchors, H, W, 5+num_classes)
    # target: list of {"cls":(num_classes,), "grid_x":(1,), "grid_y":(1,), "grid_anchor":(1,), "box":(1,4)}
    loss_obj = loss_box = loss_cls = 0
    total_loss = 0
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
      # Complete IoU
      ciou = bbox_ciou(pr, tar["box"]) 
      loss_box += 1.0 - ciou.mean()
      # class prediction loss
      cls_pred = pred[grid_anchor, grid_y, grid_x, 5:] # (num_classes,)
      cls_true = tar["cls"].float()
      loss_cls += self.cls_loss(cls_pred, cls_true)

    # total loss 
    total_loss = loss_obj + loss_box + loss_cls
      
    return total_loss