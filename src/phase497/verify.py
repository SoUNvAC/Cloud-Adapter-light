"""Check spatial indexing, invalid candidates, null match, and gradient paths."""
import json
from pathlib import Path
import torch
from .texture import ShadowRedistribution,shift
from .texture_base import ShadowRedistribution as Base
R=Path('/home/scv/shared/phase497_20261110')
def check_smoke():
    checkpoint=torch.load(R/'preflight_texture/iter_10.pth',map_location='cpu')['state_dict']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42904);m=ShadowRedistribution(ordinal=False)
    changes={n:not torch.equal(p,checkpoint['backbone.texture_adapter.'+n]) for n,p in m.named_parameters()}
    assert all(changes.values()),changes
    (R/'SMOKE_PARAMETERS.json').write_text(json.dumps(changes,indent=2))
def main():
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42904);base=Base(ordinal=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42904);m=ShadowRedistribution(ordinal=False)
    for n,p in base.named_parameters():
        if n!='evidence.0.weight':assert torch.equal(p,dict(m.named_parameters())[n]),n
    x=torch.arange(20).reshape(1,1,4,5).float()
    assert shift(x,0,1)[0,0,0,-1]==0 and shift(x,0,1)[0,0,0,0]==1
    native=torch.randn(2,384,4,4);detail=torch.randn(2,64,16,16);m.rgb=torch.rand(2,3,64,64)
    w=m.routing(native,(16,16))
    assert torch.allclose(w.sum(1),torch.ones_like(w[:,0]),atol=1e-6)
    for i,(dy,dx) in enumerate(m.offsets):
        valid=shift(torch.ones_like(w[:,:1]),dy*4,dx*4)
        assert (w[:,i+1:i+2]*(1-valid)).count_nonzero()==0
    micro=base.ordinal_evidence(m.rgb)[:,:24]
    assert torch.equal(m.descriptors(m.rgb)[:,10:],micro)
    assert m(native,detail).count_nonzero()==0
    # With nonzero output projection, gradients must reach matcher and confidence.
    for layer in (m.inject,m.evidence_inject,m.ordinal_inject):torch.nn.init.normal_(layer.weight,std=.01)
    out=m(native,detail);out.square().mean().backward()
    for n,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
    for n in ['reference_projection.weight','null_logit','log_temperature']:
        assert dict(m.named_parameters())[n].grad.abs().sum()>0,n
    with torch.no_grad():m.null_logit.fill_(100)
    null=m.routing(native,(16,16))
    assert float(null[:,1:].sum())<1e-20
    (R/'UNIT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,common_retained_init_equal=True,zero_init=True,no_wrap_or_invalid_matches=True,normalized_routing=True,microtexture_unchanged=True,null_match=True,matcher_gradient=True)))
if __name__=='__main__':main()
