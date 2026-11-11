import json
from pathlib import Path
import torch
from .texture import ShadowRedistribution,covariance_evidence,box
from .texture_base import ShadowRedistribution as Base
R=Path('/home/scv/shared/phase499_20261110')
def seeded(cls):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42904);return cls(ordinal=False)
def check_smoke():
    ck=torch.load(R/'preflight_texture/iter_10.pth',map_location='cpu')['state_dict']
    m=seeded(ShadowRedistribution)
    changes={n:not torch.equal(p,ck['backbone.texture_adapter.'+n]) for n,p in m.named_parameters()}
    assert all(changes.values()),changes
    (R/'SMOKE_PARAMETERS.json').write_text(json.dumps(changes,indent=2))
def main():
    torch.set_num_threads(4)
    base=seeded(Base);m=seeded(ShadowRedistribution)
    assert all(torch.equal(p,m.state_dict()[n]) for n,p in base.state_dict().items())
    assert sum(p.numel() for p in m.parameters())-sum(p.numel() for p in base.parameters())==15184
    rgb=torch.rand(2,3,64,64);base.rgb=rgb;m.rgb=rgb
    native=torch.randn(2,384,4,4);detail=torch.randn(2,64,16,16)
    assert torch.equal(base(native,detail),m(native,detail))
    for x in [torch.zeros_like(rgb),torch.ones_like(rgb),rgb]:
        c=covariance_evidence(x);assert c.shape==(2,60,16,16) and torch.isfinite(c).all()
    x=torch.rand(1,2,32,32)
    for k in [3,9,21]:
        brute=torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x,(k//2,)*4,mode='replicate'),k,1)
        assert torch.allclose(box(x,k),brute,atol=2e-6)
    # Equal grayscale, distinct color covariance: genuinely additional input evidence.
    flat=torch.full((1,3,64,64),.5);t=(torch.arange(64)%2).float()[None,None,None,:].expand(1,1,64,64)*.2-.1
    color=flat+torch.cat([.587*t,-.299*t,torch.zeros_like(t)],1)
    w=flat.new_tensor([.299,.587,.114])[None,:,None,None]
    assert torch.allclose((flat*w).sum(1),(color*w).sum(1),atol=1e-7)
    assert (covariance_evidence(flat)-covariance_evidence(color)).abs().max()>.01
    torch.nn.init.normal_(m.covariance_inject.weight,std=.01)
    m(native,detail).square().mean().backward()
    for n,p in m.named_parameters():
        if n.startswith('covariance'):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,n
    m.covariance_enabled=False
    assert torch.equal(base(native,detail),m(native,detail))
    (R/'UNIT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,retained_initialization_equal=True,zero_init_identity=True,finite_flat_and_black=True,separable_box_matches_dense=True,equal_gray_distinct_color=True,new_branch_gradient=True,off_identity=True,extra_parameters=15184),indent=2))
if __name__=='__main__':main()
