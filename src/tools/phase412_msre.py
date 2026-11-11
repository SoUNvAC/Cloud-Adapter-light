"""Three dynamic mechanisms, each with a capacity-matched fixed control."""
import os
import torch
from torch import nn
import torch.nn.functional as F
from phase411_msre import LayerwiseMsRE

MODES = ['region_fixed', 'region_dynamic', 'layer_fixed', 'layer_dynamic',
         'site_fixed', 'site_dynamic']

class DynamicMsRE(LayerwiseMsRE):
    def __init__(self, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode or os.environ['PHASE412_MODE']
        assert self.mode in MODES
        c = self.embed_dims
        if self.mode.startswith('region'):
            self.region_query = nn.Linear(c, 16)
            self.region_value = nn.Linear(c, 16)
            self.region_out = nn.Linear(16, c)
            nn.init.zeros_(self.region_out.weight); nn.init.zeros_(self.region_out.bias)
        else:
            self.gates = nn.ModuleList([nn.Linear(c, 1) for _ in self.injection_indices])
            for gate in self.gates:
                nn.init.zeros_(gate.weight); nn.init.zeros_(gate.bias)

    def positions(self, x, h, w):
        y, z = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                              torch.linspace(-1, 1, w, device=x.device), indexing='ij')
        xy = torch.stack([y, z], -1).reshape(1, h*w, 2).to(x.dtype)
        freq = torch.arange(self.embed_dims//2, device=x.device, dtype=x.dtype) + 1
        pos = torch.sin(xy.unsqueeze(-1)*freq).flatten(-2)
        return xy, pos.expand(x.shape[0], -1, -1)

    def forward(self, features, block_index, height, width):
        if not self.enabled or block_index not in self.layer_slots:
            return features
        out = super().forward(features, block_index, height, width)
        x = features[:, 1:]; slot = self.layer_slots[block_index]
        if self.mode.startswith('region'):
            if block_index != self.injection_indices[-1]: return out
            xy, pos = self.positions(x, height, width)
            centres = torch.linspace(-.75, .75, 4, device=x.device, dtype=x.dtype)
            centres = torch.stack(torch.meshgrid(centres, centres, indexing='ij'), -1).reshape(16, 2)
            prior = -8 * (xy.unsqueeze(-2)-centres).square().sum(-1)
            content = F.layer_norm(x if self.mode.endswith('dynamic') else pos, (x.shape[-1],))
            assignment = (prior + self.region_query(content)).softmax(-1)
            value = self.region_value(F.layer_norm(x, (x.shape[-1],)))
            pooled = torch.einsum('bnk,bnd->bkd', assignment, value)
            pooled = pooled / assignment.sum(1).unsqueeze(-1).clamp_min(1e-6)
            regional = torch.einsum('bnk,bkd->bnd', assignment, pooled)
            delta = self.region_out(regional)
            self.last_diagnostics = {'assignment': assignment.detach()}
            return torch.cat([out[:, :1], out[:, 1:] + self.scale[slot]*delta], 1)
        if self.mode.startswith('layer'):
            context = x.mean(1, keepdim=True) if self.mode.endswith('dynamic') else torch.ones_like(x[:, :1])
        else:
            context = x if self.mode.endswith('dynamic') else self.positions(x, height, width)[1]
        context = F.layer_norm(context, (x.shape[-1],)) if self.mode.endswith('dynamic') else context
        gate = 2 * torch.sigmoid(self.gates[slot](context))
        self.last_diagnostics = {'gate': gate.detach()}
        return torch.cat([out[:, :1], out[:, 1:] + (gate-1)*(out[:, 1:]-x)], 1)

def self_test():
    results = {}
    for mode in MODES:
        torch.manual_seed(4090101); base = LayerwiseMsRE(embed_dims=8, token_length=4)
        torch.manual_seed(4090101); model = DynamicMsRE(mode=mode, embed_dims=8, token_length=4)
        assert all(torch.equal(v, model.state_dict()[k]) for k,v in base.state_dict().items())
        x = torch.randn(2, 17, 8)
        for index in model.injection_indices:
            torch.testing.assert_close(model(x,index,4,4),base(x,index,4,4),rtol=0,atol=0)
        model(x,11,4,4).square().sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.enabled=False; assert torch.equal(model(x,11,4,4),x)
        model.enabled=True; assert torch.equal(model(x,0,4,4),x)
        # Fixed routing must ignore content; dynamic routing must respond to it.
        if not mode.startswith('region'):
            with torch.no_grad(): model.gates[-1].weight.copy_(torch.randn_like(model.gates[-1].weight))
        model(x,11,4,4); a={k:v.clone() for k,v in model.last_diagnostics.items()}
        model(torch.randn_like(x),11,4,4); b=model.last_diagnostics
        assert all(torch.equal(a[k],b[k]) for k in a) == mode.endswith('fixed')
        torch.manual_seed(4090101); production=DynamicMsRE(mode=mode)
        results[mode]={'parameters':sum(p.numel() for p in production.parameters()), 'passed':True}
    for prefix in ['region','layer','site']:
        assert results[prefix+'_fixed']['parameters']==results[prefix+'_dynamic']['parameters']
    return results
