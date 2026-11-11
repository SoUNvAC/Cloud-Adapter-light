"""Small mmseg/mmengine bridge for the standalone network."""
import hashlib
import json
from pathlib import Path
import torch
from mmseg.registry import MODELS, DATASETS
from mmseg.datasets import BaseSegDataset
from mmengine.registry import HOOKS, OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.hooks import Hook
from mmengine.optim import DefaultOptimWrapperConstructor
from .network import CloudBackbone

MODELS.register_module(name='DinoCloud431', module=CloudBackbone)

@DATASETS.register_module()
class CloudSEN431(BaseSegDataset):
    METAINFO = dict(classes=('surface_visible','cloud','shadow'),
                    palette=[[80,130,70],[245,245,245],[70,70,70]])

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)

    def load_data_list(self):
        rows = super().load_data_list()
        for row in rows:
            row['label_map'] = {0:0, 1:1, 2:1, 3:2}
        return rows

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TrainableOnly431(DefaultOptimWrapperConstructor):
    def __call__(self, model):
        model.train()
        wrapper = super().__call__(model)
        # Default constructor includes frozen parameters; remove them explicitly.
        wrapper.optimizer.param_groups[:] = [g for g in wrapper.optimizer.param_groups
                                             if any(p.requires_grad for p in g['params'])]
        return wrapper

def digest(items):
    sha = hashlib.sha256()
    for name, tensor in items:
        sha.update(name.encode())
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()

@HOOKS.register_module()
class Audit431(Hook):
    def save(self, runner, name, value):
        Path(runner.work_dir, name).write_text(json.dumps(value, indent=2))

    def before_train(self, runner):
        model = runner.model
        named = dict(model.named_parameters())
        active = {n:p for n,p in named.items() if p.requires_grad}
        expected = {n for n in named if not n.startswith('backbone.encoder.')}
        assert set(active) == expected
        optimizer_ids = [id(p) for g in runner.optim_wrapper.optimizer.param_groups for p in g['params']]
        assert len(optimizer_ids) == len(set(optimizer_ids))
        assert set(optimizer_ids) == {id(p) for p in active.values()}
        self.frozen = digest(model.backbone.encoder.state_dict().items())
        checkpoint = torch.load(runner.cfg.model.backbone.weights, map_location='cpu')
        current = model.backbone.encoder.state_dict()
        assert set(current) == set(checkpoint)
        assert all(torch.equal(v.cpu(), checkpoint[n]) for n,v in current.items()), 'Pretrained encoder overwritten'
        del checkpoint
        self.initial = {n:p.detach().cpu().clone() for n,p in active.items()}
        common = [(n,p) for n,p in named.items() if n.startswith(
            ('backbone.spatial.', 'backbone.interactions.', 'decode_head.'))]
        common_sha = digest(common)
        reference = Path('/home/scv/shared/phase429_20261102/fused/INITIAL.json')
        ref = json.loads(reference.read_text())
        assert ref['common_sha'] == common_sha and ref['frozen_sha'] == self.frozen
        layer = model.backbone.output_residual
        assert sum(p.numel() for p in layer.parameters()) == 26144
        assert all(b[-1].weight.count_nonzero() == 0 and b[-1].bias.count_nonzero() == 0 for b in layer.branches)

        if runner.iter == 0:
            other = 'fused' if model.backbone.variant == 'baseline' else 'baseline'
            reference = Path(runner.work_dir).parent / other / 'INITIAL.json'
            if reference.exists():
                ref = json.loads(reference.read_text())
                assert ref['common_sha'] == common_sha and ref['frozen_sha'] == self.frozen
        self.save(runner, 'INITIAL.json', dict(frozen_sha=self.frozen, common_sha=digest(common),
            trainable_parameters=sum(p.numel() for p in active.values()), optimizer_exact=True,
            names=list(active)))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        step = runner.iter + 1
        if step > 10 and step % 2000:
            return
        tensors = [(f'image{i}',x) for i,x in enumerate(data_batch['inputs'])]
        tensors += [(f'label{i}',x.gt_sem_seg.data) for i,x in enumerate(data_batch['data_samples'])]
        row = dict(step=step, batch_sha=digest(tensors))
        with Path(runner.work_dir, 'INPUT_TRACE.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        step = runner.iter + 1
        if step not in (10,100) and step % 2000:
            return
        named = dict(runner.model.named_parameters())
        assert digest(runner.model.backbone.encoder.state_dict().items()) == self.frozen
        groups = ['backbone.spatial.', 'backbone.interactions.', 'decode_head.', 'backbone.output_residual.']
        if runner.model.backbone.variant == 'fused':
            groups += ['backbone.refinement.', 'backbone.local.']
        changed = {g:sum(not torch.equal(p, named[n].detach().cpu()) for n,p in self.initial.items()
                        if n.startswith(g)) for g in groups}
        assert all(changed.values()), changed
        assert all(torch.isfinite(p).all() for p in named.values())
        self.save(runner, 'UPDATE_CHECK.json', dict(step=step, changed=changed, frozen_unchanged=True))
