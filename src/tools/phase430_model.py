"""Original CloudAdapter DINOv2 plus the existing sparse MsRE, source retraining."""
import json
from pathlib import Path
import torch
from torch import nn
from mmseg.registry import MODELS
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from cloud_adapter.models.backbones.cloud_adapter_dinov2 import CloudAdapterDinoVisionTransformer
from cloud_adapter.models.backbones.target_msre_dinov2 import SparseMsREResidual

@MODELS.register_module()
class Phase430Backbone(CloudAdapterDinoVisionTransformer):
    def __init__(self, **kwargs):
        kwargs['save_backbone'] = True
        super().__init__(**kwargs)
        self.target_msre = SparseMsREResidual(embed_dims=self.embed_dim)
        self.train(True)

    def train(self, mode=True):
        nn.Module.train(self, False)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(('cloud_adapter.', 'target_msre.')))
        self.cloud_adapter.train(mode)
        if hasattr(self, 'target_msre'):
            self.target_msre.train(mode)
        return self

    def forward_features(self, x, masks=None):
        b, _, height, width = x.shape
        h, w = height // self.patch_size, width // self.patch_size
        cache = self.cloud_adapter.cnn(x)
        x = self.prepare_tokens_with_masks(x, masks)
        outputs, slot = [], 0
        for index, block in enumerate(self.blocks):
            if index not in self._active_block_index_set:
                continue
            x = block(x)
            if index in self.adapter_index:
                x = self.cloud_adapter.forward(x, slot, batch_first=True, has_cls_token=True, cache=cache)
                slot += 1
            x = self.target_msre(x, index, h, w)
            if index in self.out_indices:
                outputs.append(x[:, 1:].transpose(1, 2).reshape(b, -1, h, w).contiguous())
        return outputs, cache

@HOOKS.register_module()
class Phase430AuditHook(Hook):
    priority = 'LOW'
    groups = ('backbone.cloud_adapter.cnn.', 'backbone.cloud_adapter.net.',
              'backbone.target_msre.', 'decode_head.')

    def before_train(self, runner):
        model = runner.model
        model.train()
        named = dict(model.named_parameters())
        active = {n: p for n, p in named.items() if p.requires_grad}
        expected = {n for n in named if n.startswith(self.groups)}
        assert set(active) == expected, (set(active) - expected, expected - set(active))
        optimizer_ids = [id(p) for g in runner.optim_wrapper.optimizer.param_groups for p in g['params']]
        assert len(optimizer_ids) == len(set(optimizer_ids))
        assert set(optimizer_ids) == {id(p) for p in active.values()}
        pretrained = torch.load(runner.cfg.model.backbone.init_cfg.checkpoint, map_location='cpu')
        pretrained = pretrained.get('state_dict', pretrained)
        frozen = {n: p for n, p in model.backbone.named_parameters() if not n.startswith(('cloud_adapter.', 'target_msre.'))}
        assert all(n in pretrained and torch.equal(p.detach().cpu(), pretrained[n]) for n, p in frozen.items()), 'DINO pretrained load mismatch'
        self.initial = {n: p.detach().cpu().clone() for n, p in named.items()}
        self.scope = {g: sum(p.numel() for n, p in active.items() if n.startswith(g)) for g in self.groups}
        assert all(self.scope.values())
        self.save(runner, 'ACTIVE_SCOPE.json', {'groups': self.scope, 'trainable_names': list(active),
                  'optimizer_exact': True, 'DINO_frozen': True, 'DINO_pretrained_exact': True,
                  'original_MsRE': True, 'max_iters': runner.max_iters})

    @staticmethod
    def save(runner, name, value):
        Path(runner.work_dir, name).write_text(json.dumps(value, indent=2))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        step = runner.iter + 1
        if step in (10, 100) or step % 1000 == 0:
            named = dict(runner.model.named_parameters())
            changed = {g: sum(not torch.equal(self.initial[n], p.detach().cpu())
                              for n, p in named.items() if n.startswith(g)) for g in self.groups}
            frozen_ok = all(torch.equal(self.initial[n], p.detach().cpu())
                            for n, p in named.items() if not n.startswith(self.groups))
            assert frozen_ok and all(changed.values()), (step, changed, frozen_ok)
            for p in named.values():
                assert torch.isfinite(p).all(), 'nonfinite parameter'
            self.save(runner, 'UPDATE_CHECK.json', {'step': step, 'changed_tensors': changed,
                      'frozen_unchanged': frozen_ok, 'finite_parameters': True})

