"""Matched full-network Catalogue fine-tuning of four source-trained models."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn

ROOT = Path('/home/scv/shared/phase438_20261103')
OLD = Path('/home/scv/shared/phase400_20261031')
SRC = Path('/home/scv/Cloud-Adapter-light/src')
PHASE = 431
BUDGET = sys.argv[1]
assert BUDGET in 'ABCDE' and len(BUDGET)==1
FROZEN = BUDGET in ('C','D')
LOW_LR = BUDGET in ('B','D')
R = ROOT/BUDGET
PARENT = Path(f'/home/scv/shared/phase{PHASE}_20261102')
sys.path[:0] = [str(PARENT/'code'), str(PARENT), str(OLD/'code'), str(SRC)]
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.hooks import Hook
from mmengine.evaluator import BaseMetric
from mmengine.registry import HOOKS, OPTIM_WRAPPER_CONSTRUCTORS
from mmseg.registry import METRICS, TRANSFORMS
from mmseg.utils import register_all_modules
from report_phase400 import ap_exact, describe


def save(name, value):
    path = R/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_train(self, mode=True):
    nn.Module.train(self, mode)
    self.requires_grad_(True)
    if FROZEN:
        self.encoder.requires_grad_(False)
        self.encoder.eval()
    return self


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class Full433(DefaultOptimWrapperConstructor):
    def __call__(self, model):
        model.requires_grad_(True)
        model.train()
        wrapper=super().__call__(model)
        wrapper.optimizer.param_groups[:]=[g for g in wrapper.optimizer.param_groups if any(p.requires_grad for p in g['params'])]
        return wrapper


@METRICS.register_module()
class Metric433(BaseMetric):
    def process(self, data_batch, data_samples):
        for d in data_samples:
            truth = d['gt_sem_seg']['data'][0].cpu().numpy().astype(np.uint8)
            pred = d['pred_sem_seg']['data'][0].cpu().numpy().astype(np.uint8)
            valid = truth != 255
            cm = np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
            scores = d['seg_logits']['data'].float().clamp_min(0)
            scores = scores/scores.sum(0,keepdim=True).clamp_min(1e-12)
            assert torch.isfinite(scores).all()
            product = Path(d['img_path']).name.removesuffix('_rgb.npy')
            row = ROWS[product]
            assert np.bincount(truth[valid],minlength=3).tolist() == row['label_counts']
            self.results.append(dict(product=product,cm=cm,score=scores[2].cpu().numpy()[valid],y=truth[valid]==2))

    def compute_metrics(self, results):
        assert len(results) == len({x['product'] for x in results}) == 32
        cm = sum(x['cm'] for x in results)
        aps = [ap_exact(x['score'],x['y']) for x in results]
        aps = [x for x in aps if x is not None]
        assert len(aps) == 7
        report = dict(**describe(cm),macro_shadow_AP=float(np.mean(aps)*100),
            pooled_shadow_AP=ap_exact(np.concatenate([x['score'] for x in results]),np.concatenate([x['y'] for x in results]))*100,
            per_group=[dict(product=x['product'],**describe(x['cm'])) for x in results])
        save(f'evaluations/step_{CURRENT_STEP}.json',report)
        return {'mIoU':report['mIoU'],'shadow_IoU':report['Shadow_IoU'],
                'macro_shadow_AP':report['macro_shadow_AP'],'pooled_shadow_AP':report['pooled_shadow_AP']}


@HOOKS.register_module()
class Audit433(Hook):
    def before_train(self, runner):
        model = runner.model
        model.train()
        named = dict(model.named_parameters())
        assert all(p.requires_grad == (not (FROZEN and n.startswith('backbone.encoder.'))) for n,p in named.items())
        ids = [id(p) for g in runner.optim_wrapper.optimizer.param_groups for p in g['params']]
        assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in named.values() if p.requires_grad}
        source = torch.load(CHECKPOINT,map_location='cpu')['state_dict']
        actual = model.state_dict()
        assert set(source) == set(actual)
        assert all(torch.equal(v.cpu(),source[n]) for n,v in actual.items()), 'Source checkpoint not loaded exactly'
        self.initial = {n:p.detach().cpu().clone() for n,p in named.items()}
        self.backbone = [n for n in named if (n.startswith('backbone.encoder.') if PHASE!=430 else n.startswith('backbone.blocks.'))]
        self.tasks = [n for n in named if n not in self.backbone]
        assert self.backbone
        save('INITIAL.json',dict(frozen_encoder=FROZEN,optimizer_exact=True,strict_checkpoint_match=True,
             trainable_parameters=sum(p.numel() for p in named.values() if p.requires_grad),checkpoint_sha256=sha(CHECKPOINT)))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        step = runner.iter+1
        allowed=set(json.loads((ROOT/'SUBSETS.json').read_text())[BUDGET])
        assert all(Path(d.img_path).name.removesuffix('_rgb.npy') in allowed for d in data_batch['data_samples'])
        if step<=10 or step%500==0:
            h=hashlib.sha256()
            for image, sample in zip(data_batch['inputs'],data_batch['data_samples']):
                h.update(image.numpy().tobytes());h.update(sample.gt_sem_seg.data.numpy().tobytes())
            with (R/'INPUT_TRACE.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,sha256=h.hexdigest()))+'\n')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        step=runner.iter+1
        if step in (10,100) or step%500==0:
            named=dict(runner.model.named_parameters())
            changed={key:sum(not torch.equal(self.initial[n],named[n].detach().cpu()) for n in names)
                     for key,names in [('DINO',self.backbone),('task',self.tasks)]}
            assert changed['task']>0 and (changed['DINO']==0 if FROZEN else changed['DINO']>0)
            assert all(torch.isfinite(p).all() for p in named.values())
            save('UPDATE_CHECK.json',dict(step=step,changed=changed,finite=True,frozen_encoder=FROZEN))
        if step%500==0:
            import shutil
            assert shutil.disk_usage(ROOT).free>350*1024**2, 'Insufficient disk for safe checkpoint write'
            state={n:v.detach().cpu().clone() for n,v in runner.model.state_dict().items() if not (FROZEN and n.startswith('backbone.encoder.'))}
            path=R/f'iter_{step}.pth'
            torch.save(dict(state_dict=state,meta=dict(step=step,partial=FROZEN,source_checkpoint=str(CHECKPOINT),source_sha256=sha(CHECKPOINT))),str(path)+'.tmp')
            os.replace(str(path)+'.tmp',path)

    def before_val(self, runner):
        global CURRENT_STEP
        CURRENT_STEP=runner.iter


def main():
    global ROWS, CHECKPOINT, CURRENT_STEP
    CURRENT_STEP=0
    R.mkdir(exist_ok=False)
    os.chdir(SRC)
    torch.set_num_threads(4)
    register_all_modules()
    if PHASE == 430:
        import cloud_adapter.models
        import cloud_adapter.datasets
        from phase430_model import Phase430Backbone
        Phase430Backbone.train=full_train
        cfg=Config.fromfile(str(PARENT/'phase430_original_msre_40k.py'))
        work=PARENT/'work_dirs/original_msre'
        custom={f'backbone.{name}':dict(lr_mult=0.1) for name in ('blocks','patch_embed','cls_token','pos_embed','mask_token','norm')}
    else:
        module=__import__(f'phase{PHASE}.network',fromlist=['CloudBackbone'])
        module.CloudBackbone.train=full_train
        cfg=Config.fromfile(str(PARENT/f'code/phase{PHASE}/config.py'))
        cfg.model.backbone.variant='fused'
        source_train_cfg=cfg.train_dataloader.copy()
        work=PARENT/'fused'
        custom={'backbone.encoder':dict(lr_mult=0.1)}
    candidates=list(work.glob('best_mIoU_iter_*.pth'))
    assert len(candidates)==1
    CHECKPOINT=candidates[0]
    assert sha(CHECKPOINT)=='ffc0a9d6de9a25ae29fb0eb7b158c6d6d483d84fff6704309816336b34852205'
    spec=importlib.util.spec_from_file_location('catalogue433_loader',SRC/'cloud_adapter/datasets/phase65_catalogue.py')
    if TRANSFORMS.get('LoadPhase65RGB') is None:spec.loader.exec_module(importlib.util.module_from_spec(spec))
    rows=json.loads((OLD/'INPUT_LOCK.json').read_text())['target_rows']
    fit=[x for x in rows if x['split']=='fit']; val=[x for x in rows if x['split']=='development_val']
    assert len(fit)==153 and len(val)==32
    subset=json.loads((ROOT/'SUBSETS.json').read_text())[BUDGET]
    fit=[x for x in fit if x['product'] in subset]
    assert len(fit)==len(subset)
    rows=fit+val
    assert not {x['group_id'] for x in fit}&{x['group_id'] for x in val}
    ROWS={x['product']:x for x in val}
    for row in rows:
        for key,suffix in [('rgb_path','_rgb.npy'),('mask_path','_mask.npy')]:assert sha(row[key])==row['prepared_sha256'][suffix]
    manifest=R/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(scope='exploratory_fit_development_only',rows=rows)))
    trainpipe=[dict(type='LoadPhase65RGB'),dict(type='LoadPhase65Mask'),dict(type='RandomCrop',crop_size=(512,512)),dict(type='RandomFlip',prob=0.5),dict(type='PhotoMetricDistortion'),dict(type='PackSegInputs')]
    valpipe=[dict(type='LoadPhase65RGB'),dict(type='Resize',scale=(512,512)),dict(type='LoadPhase65Mask'),dict(type='PackSegInputs')]
    cfg.train_dataloader=dict(batch_size=1,num_workers=2,persistent_workers=True,sampler=dict(type='InfiniteSampler',shuffle=True,seed=43301),dataset=dict(type='Phase65CatalogueDataset',prepared_manifest=str(manifest),cohort='fit_representatives',pipeline=trainpipe))
    cfg.val_dataloader=dict(batch_size=1,num_workers=2,persistent_workers=True,sampler=dict(type='DefaultSampler',shuffle=False),dataset=dict(type='Phase65CatalogueDataset',prepared_manifest=str(manifest),cohort='development_val',pipeline=valpipe))
    cfg.val_evaluator=dict(type='Metric433')
    cfg.test_dataloader=None;cfg.test_evaluator=None;cfg.test_cfg=None
    cfg.optim_wrapper=dict(type='OptimWrapper',constructor='Full433',optimizer=dict(type='AdamW',lr=(1e-5 if LOW_LR else 1e-4),weight_decay=.05,eps=1e-8,betas=(.9,.999)),clip_grad=dict(max_norm=1.,norm_type=2),paramwise_cfg=dict(norm_decay_mult=0.,custom_keys=custom))
    cfg.param_scheduler=[dict(type='LinearLR',start_factor=.001,begin=0,end=100,by_epoch=False),dict(type='PolyLR',power=.9,eta_min=0.,begin=100,end=4000,by_epoch=False)]
    cfg.train_cfg=dict(type='IterBasedTrainLoop',max_iters=4000,val_interval=500)
    cfg.custom_hooks=[dict(type='Audit433')]
    cfg.default_hooks.checkpoint=None
    cfg.load_from=str(CHECKPOINT);cfg.resume=False;cfg.work_dir=str(R)
    cfg.randomness=dict(seed=43301,deterministic=False)
    cfg.env_cfg=dict(cudnn_benchmark=False,mp_cfg=dict(mp_start_method='fork',opencv_num_threads=0),dist_cfg=dict(backend='nccl'))
    save('INPUT_LOCK.json',dict(parent_phase=PHASE,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),input_sha256=sha(OLD/'INPUT_LOCK.json'),subset_sha256=sha(ROOT/'SUBSETS.json'),arm=BUDGET,supervision_percent=10,frozen_encoder=FROZEN,low_lr=LOW_LR,fit_groups=len(fit),worker_sha256=sha(__file__),steps=4000,all_trainable=not FROZEN))
    runner=Runner.from_cfg(cfg)
    if BUDGET=='E':
        from replay_phase438 import install_replay
        install_replay(runner,source_train_cfg,R)
    # Fix data/augmentation RNG after architecture-dependent model construction.
    import random
    random.seed(43301);np.random.seed(43301);torch.manual_seed(43301)
    start=time.monotonic();runner.train();assert runner.iter==4000
    reports=[(int(p.stem.split('_')[-1]),json.loads(p.read_text())) for p in (R/'evaluations').glob('step_*.json')]
    best=max(reports,key=lambda x:x[1]['mIoU']);final=dict(reports)[4000]
    save('COMPLETE.json',dict(passed=True,steps=4000,best_step=best[0],best=best[1],final=final,seconds=time.monotonic()-start))


if __name__=='__main__':
    try:main()
    except BaseException:
        R.mkdir(exist_ok=True);save('FAILURE.json',dict(error=traceback.format_exc()));raise
