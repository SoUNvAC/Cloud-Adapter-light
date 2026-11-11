"""One independently trained residual coefficient per MsRE injection layer."""
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from cloud_adapter.models.backbones.target_msre_dinov2 import SparseMsREResidual

class LayerwiseMsRE(SparseMsREResidual):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = nn.Parameter(self.scale.detach().repeat(len(self.injection_indices)))

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
        delta = delta.mean(dim=1) + identity
        projected = self.projector(
            delta.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        )
        projected = projected.permute(0, 2, 3, 1).reshape(batch, patch_count, channels)
        adapted = patches + self.scale[self.layer_slots[block_index]] * (projected + delta)
        return torch.cat([cls_token, adapted], dim=1)

def self_test():
    from phase400_common import state_sha
    torch.set_num_threads(4)
    torch.manual_seed(41);base=SparseMsREResidual(embed_dims=8,token_length=4)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(41);model=LayerwiseMsRE(embed_dims=8,token_length=4)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items() if k!='scale')
    x=torch.randn(2,10,8)
    for slot,index in enumerate([2,5,8,11]):
        torch.testing.assert_close(base(x,index,3,3),model(x,index,3,3),rtol=0,atol=0)
        model.zero_grad();model(x,index,3,3).square().sum().backward()
        assert model.scale.grad[slot].abs()>0 and model.scale.grad.count_nonzero()==1
    with torch.no_grad():model.scale[0]*=2
    assert not torch.equal(base(x,2,3,3),model(x,2,3,3))
    for index in [5,8,11]:assert torch.equal(base(x,index,3,3),model(x,index,3,3))
    model.enabled=False;assert torch.equal(model(x,2,3,3),x)
    model.enabled=True;assert torch.equal(model(x,0,3,3),x)
    torch.manual_seed(4090101);base=SparseMsREResidual()
    torch.manual_seed(4090101);model=LayerwiseMsRE()
    common=lambda m:{k:v for k,v in m.state_dict().items() if k!='scale'}
    assert state_sha(common(base))==state_sha(common(model))
    assert sum(p.numel() for p in base.parameters())==354433
    assert sum(p.numel() for p in model.parameters())==354436
    return {'passed':True,'exact_initial_output_equal':True,'isolated_layer_gradients':True,'isolated_layer_response':True,'disabled_identity':True,'RNG_preserved':True,'baseline_initial_SHA':state_sha(base.state_dict()),'layerwise_initial_SHA':state_sha(model.state_dict()),'common_initial_SHA':state_sha(common(model)),'initial_scales':model.scale.tolist(),'added_parameters':3}
