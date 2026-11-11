"""Source-error-supervised experts in the high-resolution decoder feature path."""
import torch
from torch import nn
from torch.nn import functional as F


class ErrorExperts(nn.Module):
    def __init__(self, routed=False):
        super().__init__()
        self.detail = nn.Conv2d(64,24,1)
        self.semantic = nn.ModuleList([nn.Conv2d(384,8,1) for _ in range(4)])
        self.fuse = nn.Sequential(nn.Conv2d(62,32,1),nn.GELU(),nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU())
        self.route = nn.Conv2d(32,4,1)
        self.experts = nn.ModuleList([nn.Sequential(nn.Conv2d(32,16,1),nn.GELU(),nn.Conv2d(16,384,1)) for _ in range(4)])
        for e in self.experts:
            nn.init.zeros_(e[-1].weight);nn.init.zeros_(e[-1].bias)
        self.register_buffer('routed_mode',torch.tensor(routed,dtype=torch.bool))
        self.enabled=True;self.routing_enabled=True;self.cnn=None;self.rgb=None;self.route_logits=None

    def forward(self,features):
        if not self.enabled:return torch.zeros_like(features[0])
        size=features[0].shape[-2:]
        detail=self.detail(self.cnn)
        sem=[F.interpolate(p(F.layer_norm(x.permute(0,2,3,1),(384,)).permute(0,3,1,2)),size=size,mode='bilinear',align_corners=False) for p,x in zip(self.semantic,features)]
        rgb=F.adaptive_avg_pool2d(self.rgb,size)
        appearance=torch.cat([(rgb+.02).log(),rgb/(rgb.sum(1,keepdim=True)+.02)],1)
        z=self.fuse(torch.cat([detail,*sem,appearance],1))
        self.route_logits=self.route(z)
        weights=self.route_logits.softmax(1) if self.routed_mode and self.routing_enabled else torch.full_like(self.route_logits,.25)
        return sum(e(z)*weights[:,i:i+1] for i,e in enumerate(self.experts))

    @staticmethod
    def targets(y,reference):
        # 0=other, 1=missed shadow, 2=surface false shadow, 3=cloud false shadow.
        old=reference.argmax(1);t=torch.zeros_like(y)
        t[(y==2)&(old!=2)]=1
        t[(y==0)&(old==2)]=2
        t[(y==1)&(old==2)]=3
        t[y==255]=255
        return t

    def auxiliary(self,y,reference):
        t=self.targets(y,reference);valid=t!=255
        one=F.one_hot(t.clamp(0,3),4).permute(0,3,1,2).float()*valid[:,None]
        mass=F.adaptive_avg_pool2d(one,self.route_logits.shape[-2:])
        count=mass.sum((0,2,3));present=count>0
        # Equal contribution of present error types; ignore pixels never contribute.
        loss=-(mass*self.route_logits.log_softmax(1)).sum((0,2,3))/count.clamp_min(1e-8)
        return (loss*present).sum()/present.sum().clamp_min(1)


def install(backbone,module):
    def prepare(m,args):
        x=args[0];mean=x.new_tensor([123.675,116.28,103.53])[None,:,None,None];std=x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
        module.rgb=((x*std+mean)/255).clamp(0,1)
    backbone.register_forward_pre_hook(prepare)
    backbone.spatial.register_forward_hook(lambda m,a,o:setattr(module,'cnn',o[1]))
    backbone.register_forward_hook(lambda m,a,o:(o[0]+module(o),*o[1:]),prepend=True)
