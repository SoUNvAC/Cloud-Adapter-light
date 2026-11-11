"""Check retained initialization, shared evidence, stage isolation and gradients."""
import json
from pathlib import Path
import torch
from .texture import ShadowRedistribution
from .texture_base import ShadowRedistribution as Base
R=Path('/home/scv/shared/phase498_20261110')

def seeded(cls):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42904)
        return cls(ordinal=False)

def check_smoke():
    checkpoint=torch.load(R/'preflight_texture/iter_10.pth',map_location='cpu')['state_dict']
    m=seeded(ShadowRedistribution)
    changes={n:not torch.equal(p,checkpoint['backbone.texture_adapter.'+n]) for n,p in m.named_parameters()}
    assert all(changes.values()),changes
    (R/'SMOKE_PARAMETERS.json').write_text(json.dumps(changes,indent=2))

def main():
    torch.set_num_threads(4)
    base=seeded(Base);m=seeded(ShadowRedistribution)
    for n,p in base.state_dict().items():assert torch.equal(p,m.state_dict()[n]),n
    assert sum(p.numel() for p in m.parameters())-sum(p.numel() for p in base.parameters())==8192
    rgb=torch.rand(2,3,64,64)
    native=torch.randn(2,384,4,4);detail=torch.randn(2,64,16,16);context=torch.randn(2,64,4,4)
    base.rgb=rgb;m.rgb=rgb;m.prepare_evidence()
    assert torch.equal(base(native,detail),m(native,detail))
    for s in range(4):assert torch.equal(context,m.early_context(context,s))
    # Nonzero retained output maps test exact evidence reuse, beyond zero-init identity.
    with torch.no_grad():
        for name in ['inject','evidence_inject','ordinal_inject']:
            torch.nn.init.normal_(getattr(base,name).weight,std=.01)
            getattr(m,name).weight.copy_(getattr(base,name).weight)
    assert torch.equal(base(native,detail),m(native,detail))
    torch.nn.init.normal_(m.early_inject[0].weight,std=.01)
    assert not torch.equal(context,m.early_context(context,0))
    assert torch.equal(context,m.early_context(context,1))
    m.early_enabled=False
    assert torch.equal(context,m.early_context(context,0))
    m.early_enabled=True
    for layer in m.early_inject:torch.nn.init.normal_(layer.weight,std=.01)
    # Early path alone must backpropagate to BOTH shared evidence encoders.
    sum(m.early_context(context,s).square().mean() for s in range(4)).backward()
    for n,p in m.named_parameters():
        if n.startswith(('evidence.','ordinal.','early_inject.')):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,n
    m.zero_grad(set_to_none=True);m.prepare_evidence();m(native,detail).square().mean().backward()
    for n,p in m.named_parameters():
        if not n.startswith('early_inject.'):
            assert p.grad is not None and torch.isfinite(p.grad).all(),n
    (R/'UNIT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,retained_initialization_equal=True,initial_early_identity=True,late_evidence_exact=True,stage_isolation=True,early_off_identity=True,early_to_shared_encoders_gradient=True,extra_parameters=8192),indent=2))
if __name__=='__main__':main()
