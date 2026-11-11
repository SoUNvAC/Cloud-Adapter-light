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

ROOT=Path('/home/scv/shared/phase440_20261103')
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
    os.chdir(SRC);torch.set_num_threads(4);torch.manual_seed(44001)
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
        model_state_sha=before,conditions=[1.,.70,.68,.66,.64,.62,.60,.58,.56,.54,.52,.50],input_lock_sha=sha(OLD/'INPUT_LOCK.json'),
        worker_sha=sha(__file__),transform_sha=sha(Path(__file__).with_name('appearance_phase439.py')),
        training_steps=0,bn_updates=0,threshold_tuning=False,
        intervention_location='backbone.interactions[0] only'))

    ratios=[1.,.70,.68,.66,.64,.62,.60,.58,.56,.54,.52,.50]
    modes=['baseline','attention_off','clean_context']
    save('PROTOCOL.json',dict(ratios=ratios,modes=modes,
        intervention='Only backbone.interactions[0]: identity residual bypass or replace context with same-image original context; all other operations unchanged',
        cohorts='Baseline original correct surface: endpoint-cloud flippers versus surface at every ratio; descriptive outcome-selected cohorts',
        scores='Mask2Former nonnegative semantic scores normalized by class sum, no extra softmax',
        original_reference_required=True,clean_context_is_oracle=True))
    control=dict(mode='baseline',capture=False,context=None)
    def before_attention(module,args):
        if control['capture']:control['context']=args[1].detach().clone()
        if control['mode']=='clean_context':return (args[0],control['context'])
    def after_attention(module,args,out):
        if control['mode']=='attention_off':return args[0]
    module=model.backbone.interactions[0]
    handles=[module.register_forward_pre_hook(before_attention),module.register_forward_hook(after_attention)]
    pipe=Compose([dict(type='Resize',scale=(512,512)),dict(type='LoadPhase65Mask'),dict(type='PackSegInputs')])
    totals={m:[np.zeros((3,3),np.int64) for _ in ratios] for m in modes}
    calls=0;start=time.monotonic();parity=[]
    def infer(image,row):
        nonlocal calls
        sample=pipe(dict(img=image[...,::-1].copy(),img_path=row['rgb_path'],seg_map_path=row['mask_path'],
            ori_shape=image.shape[:2],img_shape=image.shape[:2],seg_fields=[],label_map=None,reduce_zero_label=False))
        with torch.inference_mode():out=model.test_step(pseudo_collate([sample]))[0]
        calls+=1
        pred=out.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        score=out.seg_logits.data.float().cpu().numpy()
        assert np.isfinite(score).all() and score.min()>=0
        assert np.array_equal(score.argmax(0),pred)
        raw=score[0]-score[1]
        score/=np.maximum(score.sum(0,keepdims=True),1e-12)
        return pred,(score[0]-score[1]),raw,score[0],score[1]
    def stats(v,mask):
        a=v[mask].astype(np.float64)
        return dict(n=int(a.size),sum=float(a.sum()),sum_sq=float(np.square(a).sum()),
            mean=float(a.mean()) if a.size else None,
            quantiles=np.quantile(a,[.05,.25,.5,.75,.95]).tolist() if a.size else None)
    for index,row in enumerate(rows):
        rgb=np.load(row['rgb_path']);truth=np.load(row['mask_path']);iv=np.load(row['image_valid_path'])
        valid=truth!=255;y=truth[valid].astype(np.int64)
        control.update(mode='baseline',capture=True)
        original=infer(rgb,row);control['capture']=False
        base=original[0];eligible=(truth==0)&(base==0)
        records={};allpred={};input_hashes=[]
        for mode in modes:
            control['mode']=mode
            predictions=[];margins=[];raws=[];surfaces=[];clouds=[];entries=[]
            for j,ratio in enumerate(ratios):
                image,meta,clip=transform(rgb,iv,dict(family='red_blue',value=ratio))
                assert not clip.any() and meta['lower_clipped_pixels']==0
                digest=hashlib.sha256(image.tobytes()).hexdigest()
                if mode=='baseline':input_hashes.append(digest)
                else:assert input_hashes[j]==digest
                result=original if mode=='baseline' and j==0 else infer(image,row)
                pred,margin,raw,surface,cloud=result
                if mode=='clean_context' and j==0:
                    assert np.array_equal(pred,base) and np.array_equal(margin,original[1])
                    parity.append(dict(product=row['product'],prediction_exact=True,score_margin_exact=True))
                cm=np.bincount(3*y+pred[valid],minlength=9).reshape(3,3)
                transition=np.bincount(9*y+3*base[valid]+pred[valid],minlength=27).reshape(3,3,3)
                totals[mode][j]+=cm
                entries.append(dict(ratio=ratio,metrics=describe(cm),transition=transition.tolist(),appearance=meta,input_sha=digest))
                # Keep only surface cohort score arrays, at full float32 precision.
                predictions.append(pred[eligible]);margins.append(margin[eligible]);raws.append(raw[eligible])
                surfaces.append(surface[eligible]);clouds.append(cloud[eligible])
                save('STATUS.json',dict(image=index+1,images=32,mode=mode,ratio=ratio,forward_calls=calls,seconds=time.monotonic()-start))
            pp=np.stack(predictions);mm=np.stack(margins);rr=np.stack(raws)
            ss=np.stack(surfaces);cc=np.stack(clouds)
            if mode=='baseline':
                cohorts=dict(flippers=pp[-1]==1,stable=np.all(pp==0,axis=0),all_original_correct_surface=np.ones(pp.shape[1],bool))
                baseline_predictions=pp.copy()
                # Sample actual fixed pixel identities uniformly; cohorts remain descriptive.
                rng=np.random.default_rng(44001+index);sample={}
                flat=np.flatnonzero(eligible)
                for name,mask in cohorts.items():
                    ids=np.flatnonzero(mask);ids=rng.choice(ids,min(128,len(ids)),replace=False)
                    sample[name]=ids
            summaries=[];trajectories={}
            for name,mask in cohorts.items():
                for j,ratio in enumerate(ratios):
                    summaries.append(dict(cohort=name,ratio=ratio,margin=stats(mm[j],mask),raw_margin=stats(rr[j],mask),
                        surface_score=stats(ss[j],mask),cloud_score=stats(cc[j],mask),
                        cloud_pixels=int((pp[j,mask]==1).sum()),surface_pixels=int((pp[j,mask]==0).sum()),
                        negative_margin=int((mm[j,mask]<0).sum())))
                ids=sample[name]
                trajectories[name]=dict(pixel_flat_index=flat[ids].tolist(),margin=mm[:,ids].tolist(),prediction=pp[:,ids].tolist())
            clouds_bool=pp==1;ever=clouds_bool.any(0);first=clouds_bool.argmax(0)
            crossing=dict(first_cloud=[int(((first==j)&ever).sum()) for j in range(len(ratios))],
                never_cloud=int((~ever).sum()),cloud_to_noncloud_reversals=int((clouds_bool[:-1]&~clouds_bool[1:]).sum()),
                adjacent_new_cloud=[int((~clouds_bool[j-1]&clouds_bool[j]).sum()) for j in range(1,len(ratios))],
                adjacent_leave_cloud=[int((clouds_bool[j-1]&~clouds_bool[j]).sum()) for j in range(1,len(ratios))])
            records[mode]=dict(entries=entries,cohorts=summaries,crossing=crossing,trajectories=trajectories,
                baseline_cloud_recovered=[int(((baseline_predictions[j]==1)&(pp[j]==0)).sum()) for j in range(len(ratios))],
                baseline_stable_harmed=[int((cohorts['stable']&(pp[j]!=0)).sum()) for j in range(len(ratios))])
        save(f'paired/{index:02d}.json',dict(weight=ARM,product=row['product'],group_id=row['group_id'],
            cohort_counts={k:int(v.sum()) for k,v in cohorts.items()},records=records))
        save('IDENTITY_CHECK.json',dict(passed=True,images=len(parity),checks=parity))
    prior=json.loads((Path('/home/scv/shared/phase439_20261103')/ARM/'SUMMARY.json').read_text())
    checks=[]
    for ratio,key in [(1.,'original'),(.7,'red_blue_0.7'),(.5,'red_blue_0.5')]:
        actual=describe(totals['baseline'][ratios.index(ratio)])
        diff=max(abs(a-b) for a,b in zip(actual['IoU'],prior[key]['IoU']))
        checks.append(dict(ratio=ratio,max_class_delta_pp=diff));assert diff<.001
    save('ENDPOINT_CHECK.json',dict(passed=True,checks=checks,tolerance_pp=.001))
    assert state_hash(model)==before and all(not m.training for m in model.modules())
    assert all(p.grad is None for p in model.parameters())
    save('SUMMARY.json',{m:[dict(ratio=r,**describe(cm)) for r,cm in zip(ratios,cms)] for m,cms in totals.items()})
    assert calls==32*len(ratios)*len(modes)
    save('COMPLETE.json',dict(passed=True,forward_calls=calls,training_steps=0,bn_updates=0,state_unchanged=True,seconds=time.monotonic()-start))

if __name__=='__main__':
    try:main()
    except BaseException:
        R.mkdir(exist_ok=True);save('FAILURE.json',dict(error=traceback.format_exc()));raise
