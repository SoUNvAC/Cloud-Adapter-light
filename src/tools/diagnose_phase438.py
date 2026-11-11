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

ARM=sys.argv[1]
STEP=int(sys.argv[2])
ROOT=Path('/home/scv/shared/phase438_20261103')
R=ROOT/ARM/'diagnostics'/str(STEP)
R.mkdir(parents=True,exist_ok=True)
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
    from mmseg.registry import MODELS,DATASETS
    from report_phase400 import ap_exact,describe,writecsv
    register_all_modules()
    import phase431.integration
    spec=importlib.util.spec_from_file_location('catalogue434',SRC/'cloud_adapter/datasets/phase65_catalogue.py')
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    torch.set_num_threads(4);torch.manual_seed(43401)
    torch.backends.cudnn.benchmark=False
    cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused'
    model=MODELS.build(cfg.model).cuda().eval()
    source_path=P/'fused/best_mIoU_iter_32000.pth'
    paths=dict(final=(source_path if ARM=='SOURCE' else ROOT/ARM/f'iter_{STEP}.pth'))
    base=torch.load(source_path,map_location='cpu')['state_dict']
    payload=torch.load(paths['final'],map_location='cpu')
    loaded=payload['state_dict']
    if payload.get('meta',{}).get('partial'):
        assert payload['meta']['source_sha256']==sha(source_path)
        assert set(base)-set(loaded)=={n for n in base if n.startswith('backbone.encoder.')}
        loaded={**base,**loaded}
    states=dict(final=loaded)
    bn_names=[n for n,m in model.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    bn_keys={n+'.'+b for n in bn_names for b in ('running_mean','running_var','num_batches_tracked')}
    assert bn_names and all(n.startswith('backbone.spatial.') for n in bn_names),bn_names
    rows=json.loads((OLD/'INPUT_LOCK.json').read_text())['target_rows']
    fit=[x for x in rows if x['split']=='fit'];val=[x for x in rows if x['split']=='development_val']
    assert len(fit)==153 and len(val)==32
    assert not {x['group_id'] for x in fit}&{x['group_id'] for x in val}
    for row in rows:assert sha(row['rgb_path'])==row['prepared_sha256']['_rgb.npy']
    save('PLAN_LOCK.json',dict(checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()},
        bn_modules=bn_names,bn_buffers=sorted(bn_keys),calibration='153 fit RGB only, fixed layerwise population moments, no masks or gradients',
        supervised_diagnostic_arms=['final_fit_bn'],supervision_budget=10,experiment_arm=ARM,step=STEP,
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
        if ARM=='SOURCE':
            reference=Path('/home/scv/shared/phase434_20261103')/('source.json' if name=='raw_target' else 'source_fit_bn.json')
            assert report['confusion']==json.loads(reference.read_text())['confusion'], 'Source endpoint mismatch'
        save(name+'.json',report);summary.append(report)
        save('STATUS.json',dict(stage='evaluation',arm=name,completed=len(summary)))
        return report

    def swapped(weights,buffers):
        return {n:(states[buffers][n] if n in bn_keys else v) for n,v in states[weights].items()}

    def calibrate(name,state):
        model.load_state_dict(state,strict=True);model.eval()
        modules=dict(model.named_modules())
        parameter_hash={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
        # Sequential topology: upstream BN is already calibrated when measuring the next layer.
        for index,bn_name in enumerate(bn_names):
            bn=modules[bn_name];total=None;squares=None;count=0
            def capture(module,args):
                nonlocal total,squares,count
                x=args[0].detach().double()
                a=x.sum((0,2,3));b=x.square().sum((0,2,3))
                total=a if total is None else total+a;squares=b if squares is None else squares+b
                count+=x.shape[0]*x.shape[2]*x.shape[3]
            handle=bn.register_forward_pre_hook(capture)
            with torch.inference_mode():
                for row in fit:
                    sample=imagepipe(dict(img_path=row['rgb_path'],seg_fields=[]))
                    data=model.data_preprocessor(pseudo_collate([sample]),training=False)
                    model.backbone.spatial(data['inputs'])
            handle.remove()
            mean=total/count;variance=(squares/count-mean.square()).clamp_min(0)
            assert torch.isfinite(mean).all() and torch.isfinite(variance).all()
            bn.running_mean.copy_(mean.to(bn.running_mean));bn.running_var.copy_(variance.to(bn.running_var))
            bn.num_batches_tracked.fill_(len(fit))
            save('STATUS.json',dict(stage='label_free_calibration',arm=name,bn_layer=index+1,total_layers=len(bn_names)))
        assert parameter_hash=={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
        result={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        changed=[n for n,v in result.items() if not torch.equal(v,state[n])]
        assert changed and set(changed)<=bn_keys
        torch.save({n:result[n] for n in bn_keys},R/(name+'_buffers.pth'))
        save(name+'_CALIBRATION.json',dict(images=153,label_reads=0,gradient_steps=0,parameters_unchanged=True,changed_buffers=changed,spatial_forward_calls=153*len(bn_names)))
        return result

    def eval_samples(name,samples,state):
        model.load_state_dict(state,strict=True);model.eval()
        cm=np.zeros((3,3),np.int64);details=[]
        with torch.inference_mode():
            for sample in samples:
                d=model.test_step(pseudo_collate([sample]))[0]
                truth=sample['data_samples'].gt_sem_seg.data[0].numpy()
                pred=d.pred_sem_seg.data[0].cpu().numpy();valid=truth!=255
                assert np.isin(truth[valid],[0,1,2]).all()
                c=np.bincount(3*truth[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
                cm+=c;details.append(dict(image=sample['data_samples'].img_path,**describe(c)))
        report=dict(**describe(cm),images=len(details),per_image=details)
        save(name+'.json',report)
        return report

    start=time.monotonic()
    evaluate('raw_target',states['final'])
    calibrated_final=calibrate('final_fit_bn',states['final'])
    evaluate('final_fit_bn',calibrated_final)
    selected=set(json.loads((ROOT/'SUBSETS.json').read_text())['A'])
    target_train=[x for x in fit if x['product'] in selected]
    assert len(target_train)==15
    eval_samples('target_train', (evalpipe(dict(img_path=x['rgb_path'],seg_map_path=x['mask_path'],seg_fields=[],label_map=None,reduce_zero_label=False)) for x in target_train),calibrated_final)
    # Evaluate source retention under domain-matched, label-free BN calibration.
    # The calibration set is fixed source TRAIN RGB, never source validation RGB.
    source_train_cfg=dict(cfg.train_dataloader.dataset)
    source_train_cfg['pipeline']=[dict(type='LoadImageFromFile'),dict(type='Resize',scale=(512,512)),dict(type='PackSegInputs')]
    source_train=DATASETS.build(source_train_cfg)
    source_val=DATASETS.build(cfg.val_dataloader.dataset)
    source_paths=sorted(source_train.get_data_info(i)['img_path'] for i in range(len(source_train)))[:153]
    assert len(source_paths)==153 and all('/train/' in x for x in source_paths)
    fit=[dict(rgb_path=x) for x in source_paths]
    imagepipe=Compose(source_train_cfg['pipeline'])
    save('SOURCE_CALIBRATION_INPUT.json',dict(paths=source_paths,labels_read=0))
    calibrated_source=calibrate('source_domain_bn',states['final'])
    eval_samples('source_val', (source_val[i] for i in range(len(source_val))),calibrated_source)
    if (R/'FAILURE.json').exists():
        (R/'FAILURE.json').rename(R/'FAILURE_resolved.json')
    writecsv(R/'SUMMARY.csv',summary);writecsv(R/'PER_GROUP.csv',groups)
    save('COMPLETE.json',dict(passed=True,summary=summary,evaluation_calls=32*len(summary)+15+len(source_val),training_steps=0,seconds=time.monotonic()-start,
        target_supervision_budget=10,experiment_arm=ARM,step=STEP,calibration_scope='fit images only, test distribution not fitted',parameters_unchanged_during_calibration=True))


if __name__=='__main__':
    try:main()
    except BaseException:
        save('FAILURE.json',dict(error=traceback.format_exc()));raise
