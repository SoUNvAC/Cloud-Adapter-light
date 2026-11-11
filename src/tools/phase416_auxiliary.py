"""Auxiliary-only diagnostic: identical Phase413 semantic head, no structural delta."""
import torch
from phase416_optimizer import optimizer_self_test
import torch.nn.functional as F
from phase411_msre import LayerwiseMsRE
from phase413_physics import PhysicsMsRE
MODES=['aux_separate_clip']
class AuxiliaryMsRE(PhysicsMsRE):
    def __init__(self, **kwargs):
        super().__init__(mode='dual_operator', **kwargs)
        self.inactive_prefixes=('material.','operator.','assignment.','offset.','unmatched.','output.')
        self.freeze_inactive()
    def forward(self, features, block_index, height, width):
        out=LayerwiseMsRE.forward(self,features,block_index,height,width)
        if self.enabled and block_index==self.injection_indices[-1]:
            x=F.layer_norm(features[:,1:],(self.embed_dims,))
            self.aux_logits=self.semantic(x.detach()).transpose(1,2).reshape(x.shape[0],3,height,width)
        return out

def self_test():
    optimizer_self_test()
    torch.set_num_threads(4)
    torch.manual_seed(4090101);base=LayerwiseMsRE()
    torch.manual_seed(4090101);model=AuxiliaryMsRE()
    torch.manual_seed(4090101);prior=PhysicsMsRE(mode='dual_operator')
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    assert all(torch.equal(v,model.semantic.state_dict()[k]) for k,v in prior.semantic.state_dict().items())
    for index in [2,5,8,11]:
        x=torch.randn(1,17,384,requires_grad=True)
        assert torch.equal(base(x,index,4,4),model(x,index,4,4))
    model.aux_logits.square().mean().backward()
    assert x.grad is None
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith("semantic."))
    assert model.semantic.weight.grad.abs().sum()>0
    assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(model.inactive_prefixes))
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count==355591,count
    return {'aux_separate_clip':{'parameters':count,'baseline_output_exact':True,'semantic_initialization_matches_phase413':True,'aux_gradient_blocked':True}}
