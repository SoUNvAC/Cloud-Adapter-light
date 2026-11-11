"""Spatial scale selection; shared gate starts at original equal averaging."""
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from cloud_adapter.models.backbones.target_msre_dinov2 import SparseMsREResidual

class AdaptiveMsRE(SparseMsREResidual):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.scale_gate = nn.Conv2d(self.embed_dims, 3, 1)
        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)

    def forward(self, features: Tensor, block_index: int, height: int, width: int) -> Tensor:
        if not self.enabled or block_index not in self.layer_slots:
            return features
        cls_token, patches = torch.tensor_split(features, [1], dim=1)
        batch, patch_count, channels = patches.shape
        if patch_count != height * width:
            raise ValueError(
                f"patch grid mismatch: {patch_count} tokens for {height}x{width}"
            )
        tokens = self.learnable_tokens[self.layer_slots[block_index]]
        identity = patches
        spatial = patches.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        multi_scale = torch.stack([conv(spatial) for conv in self.convs], dim=1)
        multi_scale = multi_scale.permute(0, 1, 3, 4, 2).reshape(
            batch, 3, patch_count, channels
        )
        multi_scale = multi_scale * self.level_encoding.weight[None, :, None, :]
        attention = torch.einsum("bsnc,mc->bsnm", multi_scale, tokens)
        attention = F.softmax(attention * (channels ** -0.5), dim=-1)
        # The public MsRE implementation reserves token zero and uses the
        # remaining tokens to form the feature residual.
        delta = torch.einsum(
            "bsnm,mc->bsnc",
            attention[..., 1:],
            self.mlp_token2feat(tokens[1:]),
        )
        weights = self.scale_gate(spatial).softmax(dim=1).reshape(batch, 3, patch_count, 1)
        delta = (delta * weights).sum(dim=1) + identity
        projected = self.projector(
            delta.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        )
        projected = projected.permute(0, 2, 3, 1).reshape(batch, patch_count, channels)
        adapted = patches + self.scale * (projected + delta)
        return torch.cat([cls_token, adapted], dim=1)

def self_test():
    torch.set_num_threads(4)
    torch.manual_seed(41)
    base=SparseMsREResidual(embed_dims=8, token_length=4)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(41)
    adaptive=AdaptiveMsRE(embed_dims=8, token_length=4)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,adaptive.state_dict()[k]) for k,v in base.state_dict().items())
    x=torch.randn(2,10,8)
    errors=[]
    for index in [2,5,8,11]:
        a=base(x,index,3,3);b=adaptive(x,index,3,3)
        errors.append((a-b).abs().max().item())
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    adaptive(x,2,3,3).square().sum().backward()
    assert adaptive.scale_gate.weight.grad.abs().sum().item()>0
    assert adaptive.scale_gate.bias.grad.abs().sum().item()>0
    spatial=x[:,1:].transpose(1,2).reshape(2,8,3,3)
    assert torch.allclose(adaptive.scale_gate(spatial).softmax(1),torch.full((2,3,3,3),1/3))
    with torch.no_grad():adaptive.scale_gate.weight[0,0,0,0]=1
    weights=adaptive.scale_gate(spatial).softmax(1)
    assert weights[:,0].std().item()>0
    adaptive.enabled=False
    assert torch.equal(adaptive(x,2,3,3),x)
    adaptive.enabled=True
    assert torch.equal(adaptive(x,0,3,3),x)
    return {'passed':True,'max_initial_error':max(errors),'common_initial_state_equal':True,'RNG_preserved':True,'gate_gradient_nonzero':True,'spatial_gate_response':True,'disabled_identity':True,'production_added_parameters':3*384+3}
