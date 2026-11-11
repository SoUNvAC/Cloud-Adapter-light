"""Separate trained weights/BN drift; label-free fit-image BN calibration."""
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

R=Path('/home/scv/shared/phase435_20261103')
P=Path('/home/scv/shared/phase431_20261102')
T=Path('/home/scv/shared/phase433_20261103/431')
OLD=Path('/home/scv/shared/phase400_20261031')
SRC=Path('/home/scv/Cloud-Adapter-light/src')
sys.path[:0]=[str(P/'code'),str(OLD/'code')]


def save(name,value):
    (R/name).write_text(json.dumps(value,indent=2,allow_nan=False))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    from report_phase400 import ap_exact,describe,writecsv
    register_all_modules()
    import phase431.integration
    spec=importlib.util.spec_from_file_location('catalogue435',SRC/'cloud_adapter/datasets/phase65_catalogue.py')
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    torch.set_num_threads(4);torch.manual_seed(43501)
    torch.backends.cudnn.benchmark=False
    cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused'
    model=MODELS.build(cfg.model).cuda().eval()
    paths=dict(source=P/'fused/best_mIoU_iter_32000.pth',best=T/'best_mIoU_iter_3500.pth',final=T/'iter_4000.pth')
    states={k:torch.load(v,map_location='cpu')['state_dict'] for k,v in paths.items()}
    buffers_path=Path('/home/scv/shared/phase434_20261103/final_fit_bn_buffers.pth')
    buffers=torch.load(buffers_path,map_location='cpu')
    stable={**states['final'],**buffers}
    head_prefix='backbone.output_residual.'
    encoder_prefix='backbone.encoder.'
    def head_off(state):
        result=dict(state)
        for n,v in state.items():
            if n.startswith(head_prefix) and (n.endswith('.2.weight') or n.endswith('.2.bias')):
                result[n]=torch.zeros_like(v)
        return result
    def restore(state,prefix):
        return {n:(states['source'][n] if n.startswith(prefix) else v) for n,v in state.items()}
    arms={
        'source_on':states['source'], 'source_off':head_off(states['source']),
        'final_on':stable, 'final_off':head_off(stable),
        'final_source_head':restore(stable,head_prefix),
        'final_source_encoder':restore(stable,encoder_prefix),
        'final_source_encoder_off':head_off(restore(stable,encoder_prefix)),
    }
    rows=json.loads((OLD/'INPUT_LOCK.json').read_text())['target_rows']
    fit=[x for x in rows if x['split']=='fit'];val=[x for x in rows if x['split']=='development_val']
    assert len(fit)==153 and len(val)==32
    assert not {x['group_id'] for x in fit}&{x['group_id'] for x in val}
    for row in rows:assert sha(row['rgb_path'])==row['prepared_sha256']['_rgb.npy']
    save('PLAN_LOCK.json',dict(checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()},
        arms=list(arms),buffers_sha256=sha(buffers_path),
        scope='post-training surgical diagnostics; no gradients; fixed calibrated BN for all final arms',
        caveat='rollback changes coadapted representations, not equivalent to training ablations',
        input_lock_sha256=sha(OLD/'INPUT_LOCK.json'),script_sha256=sha(__file__)))
    evalpipe=Compose([dict(type='LoadPhase65RGB'),dict(type='Resize',scale=(512,512)),dict(type='LoadPhase65Mask'),dict(type='PackSegInputs')])
    # This separate pipeline never invokes a mask loader.
    imagepipe=Compose([dict(type='LoadPhase65RGB'),dict(type='Resize',scale=(512,512)),dict(type='PackSegInputs')])
    summary=[];groups=[]

    def evaluate(name,state):
        model.load_state_dict(state,strict=True);model.eval()
        cm=np.zeros((3,3),np.int64);scores=[];labels=[];aps=[]
        for i,row in enumerate(val):
            sample=evalpipe(dict(img_path=row['rgb_path'],seg_map_path=row['mask_path'],seg_fields=[],label_map=None,reduce_zero_label=False))
            with torch.inference_mode():d=model.test_step(pseudo_collate([sample]))[0]
            truth=sample['data_samples'].gt_sem_seg.data[0].numpy();pred=d.pred_sem_seg.data[0].cpu().numpy();valid=truth!=255
            assert np.bincount(truth[valid],minlength=3).tolist()==row['label_counts']
            c=np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
            semantic=d.seg_logits.data.float().clamp_min(0);prob=semantic/semantic.sum(0,keepdim=True).clamp_min(1e-12)
            s=prob[2].cpu().numpy()[valid];y=truth[valid]==2
            assert np.isfinite(s).all()
            ap=ap_exact(s,y)
            if ap is not None:aps.append(ap)
            scores.append(s);labels.append(y);cm+=c
            groups.append(dict(arm=name,product=row['product'],**describe(c),AP=None if ap is None else ap*100))
        assert len(aps)==7
        report=dict(arm=name,**describe(cm),macro_AP=float(np.mean(aps)*100),pooled_AP=ap_exact(np.concatenate(scores),np.concatenate(labels))*100)
        save(name+'.json',report);summary.append(report)
        save('STATUS.json',dict(stage='evaluation',arm=name,completed=len(summary)))
        return report

    for name,state in arms.items():
        report=evaluate(name,state)
        if name in ('source_on','final_on'):
            expected_path=Path('/home/scv/shared/phase434_20261103')/('source.json' if name=='source_on' else 'final_fit_bn.json')
            assert report['confusion']==json.loads(expected_path.read_text())['confusion']
    writecsv(R/'SUMMARY.csv',summary);writecsv(R/'PER_GROUP.csv',groups)
    from svd_phase435 import analyze
    spectral=analyze(states['source'],states['final'],R)
    save('COMPLETE.json',dict(passed=True,summary=summary,evaluation_calls=32*len(summary),training_steps=0,
        endpoint_confusions_exact=True,spectral=spectral))


if __name__=='__main__':
    try:main()
    except BaseException:
        save('FAILURE.json',dict(error=traceback.format_exc()));raise
