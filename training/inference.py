import torch 
from PIL import Image 
import numpy as np 
from .model import YoloMini 
from .utils import xywh2xyxy 

class Predictor:
  def __init__(self, weights='runs/exp1/best.pt', num_classes=2, conf_thres=0.3, iou_thres=0.5):
    self.model = YoloMini(num_classes=num_classes)
    self.model.load_state_dict(torch.load(weights, map_location='cpu'))
    self.model.eval()
    self.conf_thres = conf_thres 
    self.iou_thres = iou_thres 
  
  def __call__(self, img_p):
    img = Image.open(img_p).convert("RGB").resize((256,256))
    x = torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.0 # (3,H,W)
    x = x.unsqueeze(0) # (1,3,H,W)

    with torch.no_grad():
      pred = self.model(x)[0] # (num_anchors, h_cell, w_cell, 5+num_classes)
      num_anchors, h_cell, w_cell, _ = pred.shape 
      pred = pred.reshape(-1, pred.shape[-1]) # (num_anchors*h_cell*w_cell, 5+num_classes)
      xywh = pred[:,:4].sigmoid() # (num_anchors*h_cell*w_cell, 4)
      obj = pred[:,4].sigmoid() # (num_anchors*h_cell*w_cell)
      cl = pred[:,5:].sigmoid() # # (num_anchors*h_cell*w_cell, num_classes)
      confs, cls_ids = (obj[:,None]*cl).max(dim=1) # (position,1)=>(position,class)=> max to (position,)
      locs_flags = confs>self.conf_thres # (position,)
      boxes = xywh2xyxy(xywh[locs_flags]) # input: (num_objs, 4)
      conf = confs[locs_flags] # (num_objs,)
      cls_id = cls_ids[locs_flags] # (num_objs,)
      if boxes.numel() == 0: return []
      keep = nms(boxes, conf, self.iou_thres)
      out = []
      for i in keep:
        box = boxes[i].tolist()
        cl = int(cls_id[i].item())
        con = float(conf[i].item())
        out.append({'xyxy':box, 'cls':cl, 'conf':con})
      return out # list of dicts
