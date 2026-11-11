"""Paired appearance interventions, fixed weights/BN, feature and error tracing."""
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
from appearance_phase439 import conditions,transform,self_test

ROOT=Path('/home/scv/shared/phase439_20261103')
OLD=Path('/home/scv/shared/phase400_20261031')
P=Path('/home/scv/shared/phase431_20261102')
SRC=Path('/home/scv/Cloud-Adapter-light/src')
ARM=sys.argv[1]
assert ARM in ('source','adapted')
R=ROOT/ARM
sys.path[:0]=[str(P/'code'),str(OLD/'code')]


def save(name,value):
    path=R/name;path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Trace:
    def __init__(self,backbone):
        self.values={};self.enabled=True
        self.handles=[backbone.spatial.register_forward_hook(self.spatial)]
        for i,module in enumerate(backbone.interactions):
            self.handles.append(module.register_forward_hook(self.interaction(i)))
        self.handles.append(backbone.refinement.register_forward_hook(self.refinement))
        self.handles.append(backbone.output_residual.register_forward_hook(self.output))

    def put(self,n,x):
        if self.enabled:self.values[n]=x.detach().float().clone()

    def spatial(self,module,args,out):
        self.put('spatial.context',out[0]);self.put('spatial.detail',out[1])

    def interaction(self,i):
        def hook(module,args,out):
            self.put(f'stage{i+1}.pre',args[0]);self.put(f'stage{i+1}.context',args[1])
            self.put(f'stage{i+1}.post_attention',out)
            self.put(f'stage{i+1}.attention_delta',out-args[0])
        return hook

    def refinement(self,module,args,out):
        i=int(args[2])+1
        self.put(f'stage{i}.local_read',args[1])
        self.put(f'stage{i}.post_refinement',out)
        self.put(f'stage{i}.refinement_delta',out-args[0])

    def output(self,module,args,out):
        i=int(args[1])+1
        self.put(f'stage{i}.output_pre',args[0]);self.put(f'stage{i}.output_post',out)


def feature_changes(original,current):
    assert set(original)==set(current)
    keys=list(original);values=[]
    for name in keys:
        a=original[name].flatten();b=current[name].flatten()
        an=torch.linalg.vector_norm(a);bn=torch.linalg.vector_norm(b)
        diff=torch.linalg.vector_norm(b-a)
        cosine=(torch.dot(a,b)/(an*bn).clamp_min(1e-20)).clamp(-1,1)
        values.append(torch.stack([an,bn,diff,diff/an.clamp_min(1e-12),cosine]))
    values=torch.stack(values).cpu().tolist()
    result={}
    for name,(an,bn,diff,relative,cosine) in zip(keys,values):
        result[name]=dict(reference_norm=an,changed_norm=bn,delta_norm=diff,
                          relative_change=relative if an>1e-8 else None,
                          cosine=cosine if an*bn>1e-16 else None)
    for i in range(1,5):
        pre=result[f'stage{i}.pre']['relative_change']
        for suffix in ('post_attention','post_refinement'):
            out=result[f'stage{i}.{suffix}']
            out['relative_amplification_vs_pre']=out['relative_change']/pre if pre is not None and pre>1e-8 else None
    return result


def main():
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmseg.registry import MODELS,TRANSFORMS
    from mmseg.utils import register_all_modules
    from report_phase400 import describe
    R.mkdir(exist_ok=False)
    os.chdir(SRC);torch.set_num_threads(4);torch.manual_seed(43901)
    torch.backends.cudnn.benchmark=False
    register_all_modules()
    import phase431.integration
    spec=importlib.util.spec_from_file_location('catalogue439',SRC/'cloud_adapter/datasets/phase65_catalogue.py')
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    save('TRANSFORM_TESTS.json',self_test())
    checkpoint=(P/'fused/best_mIoU_iter_32000.pth' if ARM=='source' else Path('/home/scv/shared/phase433_20261103/431/iter_4000.pth'))
    bn_path=Path('/home/scv/shared/phase434_20261103')/('source_fit_bn_buffers.pth' if ARM=='source' else 'final_fit_bn_buffers.pth')
    cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused'
    model=MODELS.build(cfg.model).cuda().eval();model.requires_grad_(False)
    state=torch.load(checkpoint,map_location='cpu')['state_dict'];bn=torch.load(bn_path,map_location='cpu')
    assert len(bn)==30 and all(n.endswith(('running_mean','running_var','num_batches_tracked')) for n in bn)
    model.load_state_dict({**state,**bn},strict=True)
    before=state_hash(model)
    rows=[x for x in json.loads((OLD/'INPUT_LOCK.json').read_text())['target_rows'] if x['split']=='development_val']
    assert len(rows)==len({x['group_id'] for x in rows})==32
    for row in rows:
        for key,suffix in [('rgb_path','_rgb.npy'),('mask_path','_mask.npy'),('image_valid_path','_image_valid.npy')]:
            assert sha(row[key])==row['prepared_sha256'][suffix]
    save('LOCK.json',dict(weight=str(checkpoint),weight_sha=sha(checkpoint),bn=str(bn_path),bn_sha=sha(bn_path),
        model_state_sha=before,conditions=conditions(),input_lock_sha=sha(OLD/'INPUT_LOCK.json'),
        worker_sha=sha(__file__),transform_sha=sha(Path(__file__).with_name('appearance_phase439.py')),
        training_steps=0,bn_updates=0,threshold_tuning=False,
        feature_locations='spatial outputs; each attention pre/post and delta; local read; refinement post/delta; output residual pre/post'))
    trace=Trace(model.backbone)
    pipe=Compose([dict(type='Resize',scale=(512,512)),dict(type='LoadPhase65Mask'),dict(type='PackSegInputs')])
    standard=Compose([dict(type='LoadPhase65RGB'),dict(type='Resize',scale=(512,512)),dict(type='LoadPhase65Mask'),dict(type='PackSegInputs')])
    totals={c['id']:np.zeros((3,3),np.int64) for c in conditions()}
    transfers={c['id']:np.zeros((3,3,3),np.int64) for c in conditions()}
    calls=0;start=time.monotonic();audit=[];hook_checks=[]

    def infer(rgb,row):
        nonlocal calls
        sample=pipe(dict(img=rgb[...,::-1].copy(),img_path=row['rgb_path'],seg_map_path=row['mask_path'],
            ori_shape=rgb.shape[:2],img_shape=rgb.shape[:2],seg_fields=[],label_map=None,reduce_zero_label=False))
        trace.values={}
        with torch.inference_mode():prediction=model.test_step(pseudo_collate([sample]))[0]
        calls+=1
        return prediction.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8),trace.values,sample

    for index,row in enumerate(rows):
        rgb=np.load(row['rgb_path']);truth=np.load(row['mask_path']);iv=np.load(row['image_valid_path'])
        valid=truth!=255
        assert np.array_equal(np.bincount(truth[valid],minlength=3),row['label_counts'])
        assert np.isfinite(rgb).all() and (rgb[iv]>=0).all()
        audit.append(dict(product=row['product'],dtype=str(rgb.dtype),min=float(rgb[iv].min()),max=float(rgb[iv].max()),
                          above255_fraction=float((rgb[iv]>255).any(1).mean())))
        baseline,reference,sample=infer(rgb,row)
        if True:
            canonical=standard(dict(img_path=row['rgb_path'],seg_map_path=row['mask_path'],seg_fields=[],label_map=None,reduce_zero_label=False))
            assert torch.equal(canonical['inputs'],sample['inputs'])
            trace.enabled=False
            with torch.inference_mode():plain=model.test_step(pseudo_collate([canonical]))[0]
            calls+=1;trace.enabled=True
            flipped=int((baseline!=plain.pred_sem_seg.data[0].cpu().numpy()).sum())
            assert flipped<=32,'Hook parity exceeds numerical tolerance'
            hook_checks.append(dict(product=row['product'],different_pixels=flipped,total_pixels=int(baseline.size)))
            save('HOOK_PARITY.json',dict(passed=True,checked_images=len(hook_checks),different_pixels=sum(x['different_pixels'] for x in hook_checks),input_pipeline_exact=True,per_image=hook_checks))
        y=truth[valid].astype(np.int64);base=baseline[valid].astype(np.int64)
        for condition in conditions():
            cid=condition['id'];image,meta,clipped=transform(rgb,iv,condition)
            if cid=='original':pred=baseline;features=reference
            else:pred,features,_=infer(image,row)
            pv=pred[valid].astype(np.int64)
            cm=np.bincount(3*y+pv,minlength=9).reshape(3,3)
            transition=np.bincount(9*y+3*base+pv,minlength=27).reshape(3,3,3)
            assert np.array_equal(transition.sum(1),cm)
            clean=valid&~clipped
            clean_base=np.bincount(3*truth[clean].astype(np.int64)+baseline[clean],minlength=9).reshape(3,3)
            clean_new=np.bincount(3*truth[clean].astype(np.int64)+pred[clean],minlength=9).reshape(3,3)
            meta.update(evaluation_pixels=int(valid.sum()),newly_clipped_evaluation_pixels=int((clipped&valid).sum()),
                        transformed_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            data=dict(product=row['product'],group_id=row['group_id'],weight=ARM,condition=condition,
                metrics=describe(cm),transition=transition.tolist(),appearance=meta,
                clean_base=describe(clean_base),clean_changed=describe(clean_new),
                features=feature_changes(reference,features))
            save(f'paired/{index:02d}_{cid}.json',data)
            totals[cid]+=cm;transfers[cid]+=transition
            save('STATUS.json',dict(image=index+1,images=32,condition=cid,forward_calls=calls,seconds=time.monotonic()-start))
        del reference,features
    endpoint=json.loads((Path('/home/scv/shared/phase434_20261103')/('source_fit_bn.json' if ARM=='source' else 'final_fit_bn.json')).read_text())
    actual=describe(totals['original'])
    differences=[abs(x-y) for x,y in zip(actual['IoU'],endpoint['IoU'])]
    save('ENDPOINT_CHECK.json',dict(reference_mIoU=endpoint['mIoU'],actual_mIoU=actual['mIoU'],delta_mIoU=actual['mIoU']-endpoint['mIoU'],per_class_abs_delta=differences,confusion_delta=(totals['original']-np.asarray(endpoint['confusion'])).tolist(),tolerance_IoU_pp=.001,exact_confusion=totals['original'].tolist()==endpoint['confusion']))
    assert max(differences)<.001,'Original endpoint exceeds numerical tolerance'
    assert state_hash(model)==before and all(not m.training for m in model.modules())
    assert all(p.grad is None for p in model.parameters())
    save('INPUT_AUDIT.json',audit)
    save('SUMMARY.json',{c['id']:dict(condition=c,**describe(totals[c['id']]),transition=transfers[c['id']].tolist()) for c in conditions()})
    save('COMPLETE.json',dict(passed=True,forward_calls=calls,expected_calls=544,training_steps=0,bn_updates=0,
        state_unchanged=True,original_endpoint_within_0_001pp=True,seconds=time.monotonic()-start))
    assert calls==544


if __name__=='__main__':
    try:main()
    except BaseException:
        R.mkdir(exist_ok=True);save('FAILURE.json',dict(error=traceback.format_exc()));raise
