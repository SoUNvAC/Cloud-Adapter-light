import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .decoder import DetailBoundary
from .init_reference import PairedBoundary
R=Path('/home/scv/shared/phase501_20261111')
def seeded():return DetailBoundary()
def check_smoke():
    ck=torch.load(R/'preflight_texture/iter_10.pth',map_location='cpu')['state_dict']
    m=seeded()
    changes={n:not torch.equal(p,ck['decode_head.pixel_decoder.boundary.'+n]) for n,p in m.named_parameters()}
    assert all(changes.values()),changes
    (R/'SMOKE_PARAMETERS.json').write_text(json.dumps(changes,indent=2))
def main():
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(50001);ref=PairedBoundary()
    m=seeded();ref.pair_enabled=False
    assert sum(p.numel() for p in m.parameters())==6160
    assert not any(n.startswith(('angles','pair','score')) for n,_ in m.named_parameters())
    for name in ['detail','semantic','center','inject']:
        assert all(torch.equal(v,getattr(ref,name).state_dict()[k]) for k,v in getattr(m,name).state_dict().items())
    coarse=torch.randn(2,128,8,8,requires_grad=True);detail=torch.randn(2,64,16,16,requires_grad=True);rgb=torch.rand(2,3,16,16)
    high=F.interpolate(coarse,scale_factor=2,mode='bilinear',align_corners=False)
    assert torch.equal(m(coarse,detail,rgb),high)
    with torch.no_grad():
        torch.nn.init.normal_(m.inject.weight,std=.01);ref.inject.weight.copy_(m.inject.weight)
    # Beyond trivial zero output, retained function equals Phase500 with paired input zero.
    assert torch.allclose(m(coarse,detail,rgb),ref(coarse,detail,rgb),atol=2e-6,rtol=1e-5)
    # The removed sampler cannot execute even accidentally.
    saved=F.grid_sample
    def forbidden(*args,**kwargs):raise AssertionError('paired sampling was not removed')
    F.grid_sample=forbidden
    try:out=m(coarse,detail,rgb)
    finally:F.grid_sample=saved
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())
    assert detail.grad.abs().sum()>0 and coarse.grad.abs().sum()>0
    m.enabled=False;assert torch.equal(m(coarse,detail,rgb),high)
    (R/'UNIT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,extra_parameters=6160,retained_500_initialization_equal=True,folded_zero_channels_equivalent=True,no_runtime_pair_parameters=True,no_runtime_grid_sample=True,half_resolution=True,all_branch_gradients=True),indent=2))
if __name__=='__main__':main()
