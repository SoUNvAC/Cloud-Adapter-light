"""Color covariance evidence: preserve channel co-variation before pooling."""
import torch
from torch import nn
from torch.nn import functional as F
from .texture_base import ShadowRedistribution as Base

def box(x,k):
    # Separable box filter; replicated image boundaries, no zero padding evidence.
    x=F.avg_pool2d(F.pad(x,(k//2,k//2,0,0),mode='replicate'),(1,k),1)
    return F.avg_pool2d(F.pad(x,(0,0,k//2,k//2),mode='replicate'),(k,1),1)

@torch.no_grad()
def covariance_evidence(rgb):
    # Display RGB statistics, not physical reflectance or recovered illumination.
    pairs=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
    products=torch.cat([rgb[:,i:i+1]*rgb[:,j:j+1] for i,j in pairs],1)
    codes=[]
    for k in (3,9,21):
        mean=box(rgb,k)
        cov=box(products,k)-torch.cat([mean[:,i:i+1]*mean[:,j:j+1] for i,j in pairs],1)
        cov=torch.cat([cov[:,:3].clamp_min(0),cov[:,3:]],1)
        trace=cov[:,:3].sum(1,keepdim=True)
        code=torch.cat([cov/(trace+1e-4),torch.log1p(100*trace),mean/(mean.sum(1,keepdim=True)+.03)],1)
        codes.append(F.avg_pool2d(code,4,4))
    local=torch.cat(codes,1)
    ring=(49*box(local,7)-9*box(local,3))/40
    return torch.cat([local,local-ring],1)

class ShadowRedistribution(Base):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(49901)
            self.covariance=nn.Sequential(nn.Conv2d(60,16,3,padding=1,bias=False),nn.GELU(),
                nn.Conv2d(16,16,3,padding=1,groups=16,bias=False),nn.GELU(),
                nn.Conv2d(16,16,1,bias=False),nn.GELU())
            self.covariance_inject=nn.Conv2d(16,384,1,bias=False)
            nn.init.zeros_(self.covariance_inject.weight)
        self.covariance_enabled=True

    def forward(self,native,detail):
        delta=super().forward(native,detail)
        if self.enabled and self.covariance_enabled:
            code=self.covariance(covariance_evidence(self.rgb))
            delta=delta+self.covariance_inject(code)
        return delta
