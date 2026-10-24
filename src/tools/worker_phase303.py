"""CPU cached support, strictly fixed Source recovery, and CPU score diagnostics."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from phase303_common import *


def cached():
    rows,schedule,locked=inputs();guard(rows)
    assert 'torch' not in sys.modules
    (ROOT/'scores').mkdir()
    files={n:sha(OLD/n) for n in ('FIT_IMAGES.json','CROP_SCHEDULE.json','PSEUDO_LOCK.json','TRAIN_CURVE.jsonl')}
    records=[];steps=[];full=[];totals=np.zeros(3,np.int64);valid_total=0
    curve={r['attempt']:r for r in map(json.loads,(OLD/'TRAIN_CURVE.jsonl').read_text().splitlines())}
    updates=[]
    for i,row in enumerate(rows):
        pid=row['product'];p=OLD/'teacher'/(pid+'.npz');assert sha(p)==locked[pid]['sha256'];files['teacher/'+pid+'.npz']=sha(p)
        with np.load(p) as z:
            valid=z['valid'];maximum=z['maximum'];pseudo=z['pseudo']
            assert np.isfinite(maximum).all() and np.array_equal(pseudo!=255,valid&(maximum>=.90))
            cls=count_classes(pseudo,pseudo!=255);totals+=cls;valid_total+=int(valid.sum())
            assert cls==locked[pid]['counts'] and int((pseudo!=255).sum())==locked[pid]['accepted']
            if 'scores' in z:
                s=z['scores'];a,b,total,pred=confidence(s)
                error=float(np.abs(a-maximum).max());assert error<=1e-6
                assert np.array_equal(pred,z['pred']) and np.array_equal(np.where(valid&(a>=.90),pred,255),pseudo)
                np.save(ROOT/'scores'/(pid+'.npy'),s)
                full.append(dict(product=pid,array_sha256=array_sha(s),maximum_error=error,finite_nonnegative=True))
            previous=None
            for threshold in THRESHOLDS:
                mask=valid&(maximum>=threshold)
                if previous is not None:assert not (mask&~previous).any()
                previous=mask
                records.append(dict(product=pid,definition='A_cached',threshold=threshold,valid=int(valid.sum()),accepted=int(mask.sum()),
                    class_distribution='known at0.90 only; ignore255 is not a low-threshold class',
                    Surface=cls[0] if threshold==.90 else None,Cloud=cls[1] if threshold==.90 else None,Shadow=cls[2] if threshold==.90 else None))
                for crop in (v for v in schedule if v['fit_index']==i):
                    y,x=crop['top'],crop['left'];n=int(mask[y:y+512,x:x+512].sum())
                    steps.append(dict(attempt=crop['attempt'],product=pid,threshold=threshold,accepted=n))
                    if threshold==.90:
                        assert n==curve[crop['attempt']]['accepted'] and bool(n)==curve[crop['attempt']]['updated']
                        if n:
                            counts=count_classes(pseudo[y:y+512,x:x+512],mask[y:y+512,x:x+512])
                            updates.append(dict(attempt=crop['attempt'],product=pid,top=y,left=x,horizontal_flip=crop['horizontal_flip'],
                                Surface=counts[0],Cloud=counts[1],Shadow=counts[2],accepted=n,original_updated=True,reproduced=True))
    assert valid_total==104448397 and totals.tolist()==[101857,0,0] and len(updates)==6 and len(full)==4
    assert sum(locked[r['product']]['accepted']>0 for r in rows)==5
    table('CACHED_PRODUCT_THRESHOLDS.csv',records);table('CACHED_CROP_COUNTS.csv',sorted(steps,key=lambda r:(r['threshold'],r['attempt'])))
    table('ORIGINAL_SIX_UPDATES.csv',sorted(updates,key=lambda r:r['attempt']))
    write(ROOT/'CACHED_SCORE_SAMPLES.json',full)
    write(ROOT/'INPUT_SHA256.json',files)
    write(ROOT/'CACHED_ACCEPTANCE.json',dict(status='passed',valid=valid_total,accepted=int(totals.sum()),classes=totals.tolist(),
        accepted_products=5,updates=6,skipped=994,gradient_products=len({r['product'] for r in updates}),
        full_score_products=[r['product'] for r in full],numeric_tests=numerical_tests(),neural_calls=0))
    print('cached admission passed',flush=True)


def recover():
    import torch
    rows,schedule,locked=inputs();guard(rows)
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert torch.__version__=='2.1.0+cu121' and torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    sys.path.insert(0,str(R/'src'))
    from mmseg.utils import register_all_modules
    register_all_modules();import cloud_adapter.models
    import execute_phase302 as original
    original.ROOT=ROOT
    def count(kind):
        path=ROOT/'CALLS.jsonl';n=sum(1 for _ in path.open()) if path.exists() else 0
        assert n<900
        with path.open('a') as f:f.write(json.dumps(dict(number=n+1,kind=kind,product=current[0]))+'\n')
    original.count=count
    original.seeded();current=['constructor'];model,base,_,cfg=original.construct(False)
    cfg.dump(str(ROOT/'RESOLVED_SOURCE_CONFIG.py'))
    from phase302_images import rgb
    known=set(load(ROOT/'CACHED_ACCEPTANCE.json')['full_score_products']);first=sorted(known)[0]
    targets=[next(r for r in rows if r['product']==first)]+[r for r in rows if r['product'] not in known]
    assert len(targets)==97
    results=[]
    for row in targets:
        pid=row['product'];current[0]=pid
        image,valid=rgb(row,ROOT/'FIT_IMAGE_ACCESS.jsonl')
        s=original.stitched(model,image,'source_reconstruction_window')
        with np.load(OLD/'teacher'/(pid+'.npz')) as z:
            assert np.array_equal(valid,z['valid'])
            maximum=torch.from_numpy(s).softmax(0).max(0).values.numpy()
            error=float(np.abs(maximum-z['maximum']).max());assert error<=1e-6,(pid,error)
            accepted=valid&(maximum>=.90)
            assert np.array_equal(accepted,z['pseudo']!=255)
            pred=s.argmax(0).astype(np.uint8);assert np.array_equal(pred[accepted],z['pseudo'][accepted])
            score_error=None
            if pid==first:
                score_error=float(np.abs(s-z['scores']).max());assert score_error<=1e-6
                assert np.array_equal(pred,z['pred'])
            else:np.save(ROOT/'scores'/(pid+'.npy'),s)
        results.append(dict(product=pid,kind='cached_reconstruction_check' if pid==first else 'missing_scores_recovered',
            maximum_error=error,scores_error=score_error,mask_identical=True,full_scores_sha256=array_sha(s)))
        print('recovered',len(results),pid,flush=True)
    assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in base.items())
    table('RECONSTRUCTION_RECEIPTS.csv',results)
    write(ROOT/'RECOVERY_ACCEPTANCE.json',dict(status='passed',new_products=96,reconstruction_products=1,
        calls=sum(1 for _ in (ROOT/'CALLS.jsonl').open()),frozen_source_unchanged=True,maximum_tolerance=1e-6))


def diagnose():
    rows,schedule,locked=inputs();guard(rows);assert 'torch' not in sys.modules
    assert load(ROOT/'RECOVERY_ACCEPTANCE.json')['status']=='passed'
    products=[];crops=[];intersections=[];zero_rows=[];scales=[];index=[]
    by_product={i:[v for v in schedule if v['fit_index']==i] for i in range(100)}
    for i,row in enumerate(rows):
        pid=row['product'];p=ROOT/'scores'/(pid+'.npy');s=np.load(p,mmap_mode='r')
        with np.load(OLD/'teacher'/(pid+'.npz')) as z:valid=z['valid'];cached_a=z['maximum'];pseudo=z['pseudo']
        a,b,total,pred=confidence(s);assert float(np.abs(a-cached_a).max())<=1e-6
        # A uses the original stored maximum, avoiding threshold-boundary drift from CPU exp rounding.
        a=cached_a;allows={'A':valid&(total>0),'B':valid&(total>1e-12)};conf={'A':a,'B':b}
        zero_rows.append(dict(product=pid,valid=int(valid.sum()),zero_sum=int((valid&(total==0)).sum()),
            tiny_sum=int((valid&(total<=1e-12)).sum()),ties=int((valid&((s==s.max(0)).sum(0)>1)).sum())))
        index.append(dict(product=pid,file=p.name,sha256=sha(p),array_sha256=array_sha(s),retained=pid in load(ROOT/'CACHED_ACCEPTANCE.json')['full_score_products']))
        for definition in ('A','B'):
            previous=None
            for threshold in THRESHOLDS:
                mask=allows[definition]&(conf[definition]>=threshold)
                if previous is not None:assert not (mask&~previous).any()
                previous=mask;counts=count_classes(pred,mask);n=int(mask.sum());assert sum(counts)==n
                products.append(dict(product=pid,definition=definition,threshold=threshold,valid=int(valid.sum()),accepted=n,
                    coverage_pct=100*n/int(valid.sum()),Surface=counts[0],Cloud=counts[1],Shadow=counts[2]))
                if definition=='A' and threshold==.90:
                    assert np.array_equal(mask,pseudo!=255) and counts==locked[pid]['counts']
                for crop in by_product[i]:
                    y,x=crop['top'],crop['left'];cut=mask[y:y+512,x:x+512];pp=pred[y:y+512,x:x+512]
                    if crop['horizontal_flip']:cut=cut[:,::-1];pp=pp[:,::-1]
                    counts=count_classes(pp,cut);n=int(cut.sum())
                    # Independent whole-map indicator slicing, separate from class-mask counting.
                    second=[int(((pred==c)&mask)[y:y+512,x:x+512].sum()) for c in range(3)]
                    assert counts==second and n==sum(counts)
                    crops.append(dict(definition=definition,threshold=threshold,attempt=crop['attempt'],product=pid,
                        top=y,left=x,horizontal_flip=crop['horizontal_flip'],Surface=counts[0],Cloud=counts[1],Shadow=counts[2],accepted=n,would_update=bool(n)))
        for threshold in THRESHOLDS:
            ma=allows['A']&(a>=threshold);mb=allows['B']&(b>=threshold)
            intersections.append(dict(product=pid,threshold=threshold,A=int(ma.sum()),B=int(mb.sum()),intersection=int((ma&mb).sum()),
                union=int((ma|mb).sum()),B_added=count_classes(pred,mb&~ma),B_removed=count_classes(pred,ma&~mb)))
        if index[-1]['retained']:
            base_b=b.copy()
            for scale in (.5,1,2):
                aa,bb,ss,pp=confidence(np.asarray(s)*np.float32(scale));error=float(np.abs(bb-base_b).max())
                assert np.array_equal(pp,pred) and error<=1e-6
                scales.append(dict(product=pid,scale=scale,argmax_changes=int(((pp!=pred)&valid).sum()),B_max_difference=error,
                    A_counts=[int((valid&(ss>0)&(aa>=t)).sum()) for t in THRESHOLDS],
                    B_counts=[int((valid&(ss>1e-12)&(bb>=t)).sum()) for t in THRESHOLDS]))
        del s;gc.collect()
    table('PRODUCT_THRESHOLDS.csv',products);table('CROP_REPLAY_8000.csv',sorted(crops,key=lambda r:(r['definition'],r['threshold'],r['attempt'])))
    table('DEFINITION_INTERSECTIONS.csv',intersections);table('ZERO_SUM_TIES.csv',zero_rows);table('SCALE_CHECKS.csv',scales)
    write(ROOT/'SCORE_INDEX.json',index)
    summary=[]
    for definition in ('A','B'):
        for threshold in THRESHOLDS:
            selected=[r for r in products if r['definition']==definition and r['threshold']==threshold]
            steps=[r for r in crops if r['definition']==definition and r['threshold']==threshold]
            n=sum(r['accepted'] for r in selected);v=sum(r['valid'] for r in selected);rank=sorted((r['accepted'] for r in selected),reverse=True)
            summary.append(dict(definition=definition,threshold=threshold,valid=v,accepted=n,pooled_coverage_pct=100*n/v,
                product_mean_coverage_pct=sum(r['coverage_pct'] for r in selected)/100,
                Surface=sum(r['Surface'] for r in selected),Cloud=sum(r['Cloud'] for r in selected),Shadow=sum(r['Shadow'] for r in selected),
                nonempty_products=sum(r['accepted']>0 for r in selected),nonempty_steps=sum(r['would_update'] for r in steps),
                gradient_products=len({r['product'] for r in steps if r['would_update']}),
                cumulative_pixel_uses=sum(r['accepted'] for r in steps),
                class_present_steps=[sum(r[c]>0 for r in steps) for c in CLASSES],
                top1_fraction=rank[0]/n if n else None,top5_fraction=sum(rank[:5])/n if n else None))
    assert next(r for r in summary if r['definition']=='A' and r['threshold']==.90)['nonempty_steps']==6
    table('THRESHOLD_SUMMARY.csv',summary)
    write(ROOT/'DIAGNOSTIC_ACCEPTANCE.json',dict(status='passed',products=100,combinations=8,replayed_rows=len(crops),
        independent_class_crop_checks=8000,all_original090_reproduced=True,monotonic_thresholds=True,scale_samples=4,
        A_uses='original cached maximum; recomputed softmax(s) bounded within1e-6',neural_calls=0))


def quantiles():
    rows,schedule,locked=inputs();guard(rows);assert 'torch' not in sys.modules
    labels=[]
    for row in rows:
        with np.load(OLD/'teacher'/(row['product']+'.npz')) as z:valid=z['valid']
        s=np.load(ROOT/'scores'/(row['product']+'.npy'),mmap_mode='r');labels.append(s.argmax(0).astype(np.uint8)[valid]);del s
    labels=np.concatenate(labels);output=[]
    names=('s_Surface','s_Cloud','s_Shadow','sum_s','max_s','top1_minus_top2','confidence_A','confidence_B')
    for name in names:
        values=[]
        for row in rows:
            s=np.load(ROOT/'scores'/(row['product']+'.npy'),mmap_mode='r')
            with np.load(OLD/'teacher'/(row['product']+'.npz')) as z:valid=z['valid'];maximum=z['maximum']
            if name.startswith('s_'):v=s[CLASSES.index(name[2:])]
            elif name=='sum_s':v=s.sum(0)
            elif name=='max_s':v=s.max(0)
            elif name=='top1_minus_top2':v=np.sort(s,axis=0)[2]-np.sort(s,axis=0)[1]
            elif name=='confidence_A':v=maximum
            else:v=np.divide(s.max(0),s.sum(0),out=np.zeros(s.shape[1:],np.float32),where=s.sum(0)>1e-12)
            values.append(v[valid].copy());del s,v
        values=np.concatenate(values)
        for cls in ('all',0,1,2):
            chosen=values if cls=='all' else values[labels==cls]
            q=np.quantile(chosen,[0,.01,.10,.50,.90,.99,1]).tolist() if len(chosen) else [None]*7
            output.append(dict(predicted_class='all' if cls=='all' else CLASSES[cls],support=len(chosen),quantity=name,
                **dict(zip(('min','p1','p10','p50','p90','p99','max'),q))))
        print('quantiles',name,flush=True);del values;gc.collect()
    table('SCORE_QUANTILES.csv',output)
    # Remove only this phase's owned transient reconstruction files after exact quantiles.
    deleted=[]
    for row in load(ROOT/'SCORE_INDEX.json'):
        p=ROOT/'scores'/row['file'];assert p.resolve().parent==(ROOT/'scores').resolve()
        assert sha(p)==row['sha256']
        if not row['retained']:deleted.append(dict(file=row['file'],sha256=row['sha256'],bytes=p.stat().st_size));p.unlink()
    write(ROOT/'TRANSIENT_SCORE_CLEANUP.json',dict(deleted=deleted,retained=4,
        reason='96 temporary full-score arrays consumed by complete diagnostics and exact quantiles; SHA retained'))
    write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',exact_quantile_fields=8,prediction_groups=4,quantiles=[0,1,10,50,90,99,100],
        zero_training=True,zero_optimizer_updates=True,no_truth_or_development_access=True,neural_calls=0))


if __name__=='__main__':
    try:{'cached':cached,'recover':recover,'diagnose':diagnose,'quantiles':quantiles}[sys.argv[1]]()
    except BaseException:
        with (ROOT/'FAILURES.jsonl').open('a') as f:f.write(json.dumps(dict(stage=sys.argv[1],traceback=traceback.format_exc()))+'\n')
        raise
