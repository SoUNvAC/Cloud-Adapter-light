import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .decoder import PairedBoundary
R=Path('/home/scv/shared/phase500_20261111')
def seeded():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(50001);return PairedBoundary()
def check_smoke():
    ck=torch.load(R/'preflight_texture/iter_10.pth',map_location='cpu')['state_dict']
    m=seeded()
    changes={n:not torch.equal(p,ck['decode_head.pixel_decoder.boundary.'+n]) for n,p in m.named_parameters()}
    assert all(changes.values()),changes
    (R/'SMOKE_PARAMETERS.json').write_text(json.dumps(changes,indent=2))
def main():
    torch.set_num_threads(4);m=seeded()
    coarse=torch.randn(2,128,8,8,requires_grad=True);detail=torch.randn(2,64,16,16,requires_grad=True);rgb=torch.rand(2,3,16,16)
    high=F.interpolate(coarse,scale_factor=2,mode='bilinear',align_corners=False)
    assert torch.equal(m(coarse,detail,rgb),high)
    ramp=torch.arange(16).float()[None,None,None,:].expand(1,1,16,16)
    dx=torch.ones(1,16,16);dy=torch.zeros_like(dx)
    out,valid=m.sample(ramp,dx,dy)
    assert torch.allclose(out[:,:,:,:-1],ramp[:,:,:,1:]) and not valid[:,:,-1].any()
    assert valid[:,:,:-1].all()
    torch.nn.init.normal_(m.inject.weight,std=.01)
    out=m(coarse,detail,rgb);assert out.shape==(2,128,16,16) and torch.isfinite(out).all()
    out.square().mean().backward()
    changes={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0) for n,p in m.named_parameters()}
    assert all(changes.values()),changes
    assert detail.grad.abs().sum()>0 and coarse.grad.abs().sum()>0
    m.enabled=False;assert torch.equal(m(coarse,detail,rgb),high)
    (R/'UNIT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,initial_residual_zero=True,half_resolution=True,sampling_and_validity=True,all_branch_gradients=True,detail_and_semantic_gradients=True,off_equals_bilinear=True,boundary_parameters=sum(p.numel() for p in m.parameters())),indent=2))
if __name__=='__main__':main()
