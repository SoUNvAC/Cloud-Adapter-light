"""Shadow signed-distance-guided feature transport before the decoder."""
import torch
from torch import nn
from torch.nn import functional as F

def gradient(x):
 dx=F.pad((x[...,2:]-x[...,:-2])*.5,(1,1,0,0),mode='replicate')
 dy=F.pad((x[...,2:,:]-x[...,:-2,:])*.5,(0,0,1,1),mode='replicate')
 return torch.cat([dx,dy],1)

def distance_target(y,size):
 import numpy as np
 from scipy.ndimage import distance_transform_edt
 valid=y!=255
 fraction=F.adaptive_avg_pool2d(((y==2)&valid)[:,None].float(),size)
 good=F.adaptive_avg_pool2d(valid[:,None].float(),size)>.999
 masks=(fraction[:,0]>=.5).cpu().numpy();vg=good[:,0].cpu().numpy();fields=[];weights=[]
 for mask,v in zip(masks,vg):
  mask=mask&v
  if not mask.any():d=np.full(mask.shape,-1.,dtype=np.float32)
  elif mask.all():d=np.ones(mask.shape,dtype=np.float32)
  else:d=np.clip((distance_transform_edt(mask)-distance_transform_edt(~mask))/8,-1,1).astype(np.float32)
  # Unknown labels must not define a nearby supervised shadow contour.
  reliable=v if v.all() else v&(distance_transform_edt(v)>8)
  fields.append(d);weights.append(reliable)
 return torch.as_tensor(np.stack(fields),device=y.device)[:,None],torch.as_tensor(np.stack(weights),device=y.device)[:,None]

class ContourTransport(nn.Module):
 def __init__(self,normal=False):
  super().__init__()
  self.detail=nn.Conv2d(64,24,1)
  self.semantic=nn.ModuleList([nn.Conv2d(384,8,1) for _ in range(4)])
  self.fuse=nn.Sequential(nn.Conv2d(62,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU())
  self.distance=nn.Conv2d(32,1,1)
  self.motion=nn.Conv2d(32,3,1)
  nn.init.zeros_(self.motion.weight);nn.init.zeros_(self.motion.bias)
  self.register_buffer('normal_mode',torch.tensor(normal,dtype=torch.bool))
  self.enabled=True;self.normal_enabled=True;self.cnn=None;self.rgb=None;self.sdf=None;self.flow=None

 def forward(self,features):
  if not self.enabled:return torch.zeros_like(features[0])
  feature=features[0];size=feature.shape[-2:]
  sem=[F.interpolate(p(F.layer_norm(x.permute(0,2,3,1),(384,)).permute(0,3,1,2)),size=size,mode='bilinear',align_corners=False) for p,x in zip(self.semantic,features)]
  rgb=F.adaptive_avg_pool2d(self.rgb,size)
  z=self.fuse(torch.cat([self.detail(self.cnn),*sem,(rgb+.02).log(),rgb/(rgb.sum(1,keepdim=True)+.02)],1))
  self.sdf=self.distance(z).tanh();motion=self.motion(z)
  vector=2*motion[:,:2].tanh()
  vector=vector/(vector.square().sum(1,keepdim=True)+1e-12).sqrt().clamp_min(2)*2
  if self.normal_mode and self.normal_enabled:
   smooth=F.avg_pool2d(F.pad(self.sdf,(1,1,1,1),mode='replicate'),3,1)
   g=gradient(smooth);normal=g/(g.square().sum(1,keepdim=True)+1e-12).sqrt().clamp_min(1e-4)
   vector=(vector*normal).sum(1,keepdim=True)*normal
  self.flow=vector
  b,_,h,w=feature.shape
  yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=feature.device),torch.linspace(-1,1,w,device=feature.device),indexing='ij')
  grid=torch.stack([xx,yy],-1)[None].expand(b,-1,-1,-1)
  offset=vector.permute(0,2,3,1)*vector.new_tensor([2/(w-1),2/(h-1)])
  # Subtract an identically sampled baseline: exact zero update at zero flow.
  moved=F.grid_sample(feature,grid+offset,padding_mode='border',align_corners=True)
  same=F.grid_sample(feature,grid,padding_mode='border',align_corners=True)
  gate=motion[:,2:3].sigmoid()*torch.exp(-4*self.sdf.abs())
  return (moved-same)*gate

 def auxiliary(self,y,reference=None):
  target,valid=distance_target(y,self.sdf.shape[-2:]);error=F.smooth_l1_loss(self.sdf,target,reduction='none')
  losses=[]
  for side in [target>0,target<0]:
   mask=valid&side
   if mask.any():losses.append((error*mask).sum()/mask.sum())
  distance=sum(losses)/max(len(losses),1) if losses else self.sdf.sum()*0
  # Neighbour differences, excluding invalid neighbours and saturated interior.
  near=valid&(target.abs()<1)
  gx=(self.sdf[...,1:]-self.sdf[...,:-1])-(target[...,1:]-target[...,:-1])
  gy=(self.sdf[...,1:,:]-self.sdf[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])
  vx=near[...,1:]&near[...,:-1];vy=near[...,1:,:]&near[...,:-1,:]
  grad=((gx.abs()*vx).sum()+(gy.abs()*vy).sum())/(vx.sum()+vy.sum()).clamp_min(1)
  return distance+.5*grad

def install(backbone,module):
 def prepare(m,args):
  x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
  module.rgb=((x*std+mean)/255).clamp(0,1)
 backbone.register_forward_pre_hook(prepare)
 backbone.spatial.register_forward_hook(lambda m,a,o:setattr(module,'cnn',o[1]))
 backbone.register_forward_hook(lambda m,a,o:(o[0]+module(o),*o[1:]),prepend=True)
