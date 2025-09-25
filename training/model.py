import torch
import torch.nn as nn

def conv_bn_relu(channel_in, channel_out, kernel=3, stride=1, padding=1):
  return nn.Sequential(
    nn.Conv2d(channel_in, channel_out, kernel, stride, padding, bias=False),
    nn.BatchNorm2d(channel_out),
    nn.SiLU()
  )

# Cross-Stage Partial Block
class CSPBlock(nn.Module):
  def __init__(self, channel, n=1):
    super().__init__()
    layers = []
    for _ in range(n):
      layers += [conv_bn_relu(channel, channel), conv_bn_relu(channel, channel)]
    self.block = nn.Sequential(*layers)
  
  def forward(self, x):
    return x + self.block(x)

class YoloMini(nn.Module):
  def __init__(self, num_classes=2, num_anchors=3):
    super().__init__()
    self.stem = nn.Sequential(
      conv_bn_relu(3, 32, 3, 2, 1), # (3,256,256) => (32,128,128)
      conv_bn_relu(32, 64, 3, 2, 1), # (32,128,128) => (64,64,64)
      CSPBlock(64, 1), # (64,64,62)
      conv_bn_relu(64, 128, 3, 2, 1), # (128,32,32)
      CSPBlock(128,1) # (128,32,32)
    )
    ch = 128
    self.head = nn.Conv2d(ch, num_anchors*(5+num_classes), 1) # [x,y,w,h,obj,cls...] => 
    self.num_anchors = num_anchors 
    self.num_classes = num_classes

  def forward(self, x):
    x = self.stem(x) # (batch,3,256,256)=>(batch,128,32,32)
    p = self.head(x) # (batch,num_outputs,32,32)
    bs, _, h, w = p.shape
    # (batch, num_anchors, 5+num_classes, h, w) => (batch, num_anchors, h, w, 5+num_classes)
    return p.view(bs, self.num_anchors, 5+self.num_classes, h, w).permute(0,1,3,4,2).contiguous()