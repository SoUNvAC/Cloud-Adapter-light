"""Phase304 independent host CPU analysis; inherits Phase302 metric and support rules."""
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion
from transfer_phase302 import sha,load,write

R=Path(__file__).resolve().parents[2]
REMOTE=True
ROOT=Path('/home/scv/shared/phase304_20261024')
D=ROOT if REMOTE else ROOT/'remote_backup'
OLD=Path('/home/scv/shared/phase300_20261020') if REMOTE else R/'outputs/phase300/phase300_20261020/remote_backup/phase300_20261020'
SUPPORT=Path('/home/scv/shared/phase205_20261019') if REMOTE else R/'outputs/phase205/phase205_20261019/delivery'
ANALYSIS_SUBDIR=os.environ.get('PHASE302_ANALYSIS_SUBDIR','analysis')
assert ANALYSIS_SUBDIR.startswith('analysis') and Path(ANALYSIS_SUBDIR).name==ANALYSIS_SUBDIR
O=ROOT/ANALYSIS_SUBDIR
S=R/'src/research_results/phase304_20261024'
C=('Surface','Cloud','Shadow')
INPUTS={}
CHECKS=0


def pin(p): INPUTS[str(p)]=sha(p);return load(p)
def confusion_matrix(y,p,labels):
    assert labels==[0,1,2]
    return np.histogram2d(y,p,bins=(np.arange(4)-.5,np.arange(4)-.5))[0].astype(np.int64)
def table(name,rows):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:'NA' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def tensor(y,a,b):
    global CHECKS
    t=np.bincount((9*y+3*a+b).astype(np.int64),minlength=27).reshape(3,3,3)
    for c in range(3):
        for i in range(3):
            for j in range(3):
                assert t[c,i,j]==int(((y==c)&(a==i)&(b==j)).sum());CHECKS+=1
    assert np.array_equal(t.sum(2),confusion_matrix(y,a,labels=[0,1,2]))
    assert np.array_equal(t.sum(1),confusion_matrix(y,b,labels=[0,1,2]))
    return t


def cells(t,prefix):
    return [dict(**prefix,truth=C[c],Source_pred=C[i],student_pred=C[j],
        truth_support=int(t[c].sum()),count=int(t[c,i,j])) for c in range(3) for i in range(3) for j in range(3)]


def events(t,prefix):
    rows=[]
    for c in range(3):
        for j in range(3):
            if j==c:continue
            old=int(t[c,j].sum());new=int(t[c,:,j].sum());same=int(t[c,j,j]);n=int(t[c].sum())
            removed=old-same;added=new-same;corrected=int(t[c,j,c])
            assert removed-added==old-new
            rows.append(dict(**prefix,error=C[c]+'->'+C[j],truth_support=n,Source_errors=old,student_errors=new,
                removed=removed,added=added,net_reduction=old-new,corrected=corrected,
                changed_to_other_error=removed-corrected,from_correct=int(t[c,c,j]),
                rate_change_pp=100*(new-old)/n if n else None))
    return rows


def ap(y,s):
    global CHECKS
    if not y.any() or y.all():return None
    order=np.argsort(-s,kind='stable');values=s[order];yy=y[order]
    ends=np.r_[np.flatnonzero(values[:-1]!=values[1:]),len(y)-1]
    tp=np.cumsum(yy,dtype=np.int64)[ends]
    result=float(np.dot(np.diff(np.r_[0,tp]),tp/(ends+1))/tp[-1])
    # Independent threshold-group path: no sorting positive labels by sample rank.
    unique,inverse=np.unique(s,return_inverse=True)
    totals=np.bincount(inverse,minlength=len(unique))[::-1]
    positives=np.bincount(inverse,weights=y.astype(np.int64),minlength=len(unique))[::-1]
    grouped_tp=np.cumsum(positives);grouped_n=np.cumsum(totals)
    other=float(np.sum((positives/grouped_tp[-1])*(grouped_tp/grouped_n)))
    assert abs(result-other)<1e-12;CHECKS+=1
    return result


def ious(mat):
    diag=np.diag(mat);union=mat.sum(0)+mat.sum(1)-diag
    return np.divide(diag,union,out=np.full(3,np.nan),where=union>0)


def metric(y,p,s):
    mat=np.bincount((3*y+p).astype(np.int64),minlength=9).reshape(3,3)
    assert np.array_equal(mat,confusion_matrix(y,p,labels=[0,1,2]))
    iou=ious(mat)
    for c in range(3):
        union=int(((y==c)|(p==c)).sum())
        if union:assert abs(iou[c]-int(((y==c)&(p==c)).sum())/union)<1e-12
    positive=int((y==2).sum());negative=len(y)-positive
    return dict(support=len(y),positive=positive,negative=negative,confusion=mat.tolist(),
        Surface_IoU_pct=100*float(iou[0]) if np.isfinite(iou[0]) else None,
        Cloud_IoU_pct=100*float(iou[1]) if np.isfinite(iou[1]) else None,
        Shadow_IoU_pct=100*float(iou[2]) if np.isfinite(iou[2]) else None,
        mIoU_pct=100*float(np.nanmean(iou)),AP_pct=None if (a:=ap(y==2,s)) is None else 100*a,
        recall_pct=100*int(mat[2,2])/positive if positive else None,
        FPR_pct=100*int(mat[:2,2].sum())/negative if negative else None)


def regions(y,valid):
    cloud=y==1;edge=np.zeros(y.shape,bool)
    v=valid[:-1]&valid[1:]&(cloud[:-1]!=cloud[1:]);h=valid[:,:-1]&valid[:,1:]&(cloud[:,:-1]!=cloud[:,1:])
    edge[:-1]|=v;edge[1:]|=v;edge[:,:-1]|=h;edge[:,1:]|=h
    a=np.pad(edge,2);b=np.pad(valid,2);expanded=np.zeros_like(valid);inside=np.ones_like(valid)
    for dy in range(5):
        for dx in range(5):
            expanded|=a[dy:dy+len(y),dx:dx+y.shape[1]]
            inside&=b[dy:dy+len(y),dx:dx+y.shape[1]]
    assert np.array_equal(expanded,binary_dilation(edge,structure=np.ones((5,5),bool)))
    assert np.array_equal(inside,binary_erosion(valid,structure=np.ones((5,5),bool),border_value=0))
    return dict(boundary=expanded&inside,nonboundary=~expanded&inside,excluded=valid&~inside)


def average(v):
    v=[x for x in v if x is not None];return sum(v)/len(v) if v else None


def main():
    started=time.monotonic();assert 'torch' not in sys.modules
    assert not O.exists();O.mkdir()
    if not REMOTE:assert pin(ROOT/'BACKUP_RECEIPT.json')['all_archive_members_verified']
    assert pin(D/'EVALUATION_LOCK.json')['status']=='completed'
    endpoint=pin(D/'ENDPOINT_LOCK.json');assert endpoint['all_base_tensors_unchanged']
    assert sha(D/'final_lora.pth')==endpoint['final_sha256']
    assert sha(D/'TRAIN_CURVE.jsonl')==endpoint['curve_sha256']
    assert sha(D/'PSEUDO_LOCK.json')==endpoint['pseudo_lock_sha256']
    assert sha(D/'EXECUTION_LOCK.json')==endpoint['execution_lock_sha256']
    assert time.time()>=endpoint['sealed_epoch']
    x,a,b=np.indices((3,3,3)).reshape(3,-1).astype(np.uint8)
    assert np.array_equal(tensor(x,a,b),np.ones((3,3,3),np.int64))
    assert all(r['removed']==r['added']==2 for r in events(tensor(x,a,b),{}))
    plan=[r for r in pin(OLD/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue']
    expected={r['product_id']:r for r in pin(OLD/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}
    lock=pin(D/'EVALUATION_LOCK.json');score_sha={(r['arm'],r['product']):r['sha256'] for r in lock['rows']}
    arrays={'Source':[],'LoRA':[]};truths=[];per=[];group_cells=[];group_events=[];boundary_events=[];boundary_cells=[];supports=[]
    pooled=np.zeros((3,3,3),np.int64);region_total={k:pooled.copy() for k in ('boundary','nonboundary','excluded')}
    masks_path=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip';INPUTS[str(masks_path)]=sha(masks_path)
    with zipfile.ZipFile(masks_path) as masks:
        for row in plan:
            pid=row['product_id'];gid=row['group']['group_id']
            old=SUPPORT/'infer/R4/scores/Catalogue'/(pid+'.npz');INPUTS[str(old)]=sha(old)
            with np.load(old) as z: ids=z['flat_indices'];common=z['valid']
            ex=expected[pid]
            assert hashlib.sha256(ids.tobytes()).hexdigest()==ex['flat_indices_sha256']
            assert hashlib.sha256(common.tobytes()).hexdigest()==ex['common_valid_sha256']
            raw=masks.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==row['group']['mask_member_sha256']
            mask=np.load(io.BytesIO(raw));truth=mask.argmax(-1).astype(np.uint8)
            assert (mask.reshape(-1,3)[ids[common]].sum(1)==1).all()
            support=np.zeros((1022,1022),bool);support.ravel()[ids[common]]=True
            assert int(support.sum())==ex['count']
            y=truth[support];preds={}
            for arm in ('Source','LoRA'):
                folder=Path('/home/scv/shared/phase302_20261020/evaluation') if arm=='Source' else D/'evaluation'
                p=folder/(arm+'_'+pid+'.npz');assert sha(p)==score_sha[arm,pid];INPUTS[str(p)]=sha(p)
                with np.load(p) as z:
                    loss=int((support&~z['valid']).sum())
                    if loss:
                        write(O/'SUPPORT_FAILURE.json',dict(product=pid,lost=loss,reason='RGB finite/nonzero/fill support differs from Phase300; no support silently removed'))
                    assert loss==0, 'RGB lost common support: '+pid
                    scores=z['scores'];prediction=z['pred'];shadow=z['shadow']
                    assert np.array_equal(prediction,scores.argmax(0))
                    shifted=scores-scores.max(0,keepdims=True);prob=np.exp(shifted);prob/=prob.sum(0,keepdims=True)
                    assert np.allclose(shadow,prob[2],rtol=1e-6,atol=1e-7)
                    p1=prediction[support].copy();s=shadow[support].copy();preds[arm]=prediction.copy()
                arrays[arm].append((p1,s));per.append(dict(strategy=arm,product=pid,group_id=gid,**metric(y,p1,s)))
            t=tensor(y,preds['Source'][support],preds['LoRA'][support]);pooled+=t
            group_cells+=cells(t,dict(product=pid,group_id=gid));group_events+=events(t,dict(product=pid,group_id=gid))
            region=regions(truth,support)
            assert np.array_equal(sum(v.astype(np.uint8) for v in region.values()),support.astype(np.uint8))
            for name,choose in region.items():
                rt=tensor(truth[choose],preds['Source'][choose],preds['LoRA'][choose]);region_total[name]+=rt
                prefix=dict(product=pid,group_id=gid,region=name)
                supports.append(dict(**prefix,support=int(choose.sum()),Surface=int((truth[choose]==0).sum()),Cloud=int((truth[choose]==1).sum()),Shadow=int((truth[choose]==2).sum())))
                boundary_events+=events(rt,prefix);boundary_cells+=cells(rt,prefix)
            truths.append(y);gc.collect()
    assert np.array_equal(sum(region_total.values()),pooled)
    table('TRANSITIONS27.csv',cells(pooled,{}));table('PER_GROUP_TRANSITIONS.csv',group_cells)
    event_rows=events(pooled,{});table('ERROR_EVENTS.csv',event_rows);table('PER_GROUP_ERRORS.csv',group_events)
    table('BOUNDARY_EVENTS.csv',boundary_events);table('BOUNDARY_TRANSITIONS.csv',boundary_cells);table('BOUNDARY_SUPPORT.csv',supports)
    table('BOUNDARY_POOLED_EVENTS.csv',[r for name,t in region_total.items() for r in events(t,dict(region=name))])
    table('PER_GROUP_METRICS.csv',per)
    y=np.concatenate(truths);summaries=[]
    for arm in ('Source','LoRA'):
        p=np.concatenate([a for a,s in arrays[arm]]);s=np.concatenate([s for a,s in arrays[arm]])
        pooled_metric=metric(y,p,s);parts=[r for r in per if r['strategy']==arm]
        expected_confusion=pooled.sum(2 if arm=='Source' else 1)
        assert np.array_equal(np.array(pooled_metric['confusion']),expected_confusion)
        summaries.append(dict(strategy=arm,groups=len(parts),AP_groups=sum(r['AP_pct'] is not None for r in parts),
            group_mean_mIoU_pct=average(r['mIoU_pct'] for r in parts),pooled_mIoU_pct=pooled_metric['mIoU_pct'],
            **{k:pooled_metric[k] for k in ('Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct','AP_pct')},
            macro_AP_pct=average(r['AP_pct'] for r in parts),macro_recall_pct=average(r['recall_pct'] for r in parts),
            macro_FPR_pct=average(r['FPR_pct'] for r in parts),max_group_FPR_pct=max(r['FPR_pct'] for r in parts if r['FPR_pct'] is not None),
            confusion=pooled_metric['confusion']))
        del p,s;gc.collect()
    table('FIXED_RESULTS.csv',summaries)
    delta={k[:-4]+'_pp':summaries[1][k]-summaries[0][k] for k in summaries[0] if k.endswith('_pct') and summaries[0][k] is not None and summaries[1][k] is not None}
    write(O/'RESULT_DIFFERENCES_PP.json',delta)
    table('RESULT_DIFFERENCES_PP.csv',[dict(strategy='LoRA minus Source',**delta)])
    probes=[];probe_cells=[];full_cells=[]
    fit_source=Path('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json') if REMOTE else R/'outputs/phase208/phase208_20261020/12201_FIT_INPUT_LOCK.json'
    fit_masks={r['product']:r for r in pin(fit_source)['bindings']}
    with zipfile.ZipFile(masks_path) as masks:
        for pid in sorted(r['product'] for r in pin(D/'FIT_IMAGES.json')['rows'])[:4]:
            raw=masks.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==fit_masks[pid]['mask_member_sha256']
            m=np.load(io.BytesIO(raw))[255:767,255:767];truth=m.argmax(-1).astype(np.uint8)
            with np.load(D/'acceptance'/(pid+'.npz')) as z:source=z['scores'];initial=source.argmax(0)
            with np.load(D/'teacher'/(pid+'.npz')) as z:full=z['pred'][255:767,255:767]
            common=None
            for attempt in (0,100,500,1000):
                path=D/'probes'/(pid+'_'+str(attempt)+'.npz');assert sha(path)==endpoint['probes'][path.name]
                with np.load(path) as z:
                    current=z['pred'];valid=z['valid']&(m.sum(-1)==1)
                    if attempt==0:assert float(np.abs(z['scores']-source).max())<=1e-6
                if common is None:common=valid
                assert np.array_equal(common,valid)
                t=tensor(truth[valid],initial[valid],current[valid]);probe_cells+=cells(t,dict(product=pid,attempt=attempt))
                for c in range(3):
                    correct=int(t[c,c,c]);damaged=int(t[c,c].sum())-correct;repaired=int(t[c,:,c].sum())-correct
                    probes.append(dict(product=pid,attempt=attempt,truth=C[c],support=int(t[c].sum()),
                        Source_correct_student_wrong=damaged,Source_wrong_student_correct=repaired,net_correct=repaired-damaged,
                        Shadow_prediction_pct=100*int((current[valid]==2).sum())/int(valid.sum()) if valid.any() else None))
                if attempt in (0,1000):full_cells+=cells(tensor(truth[valid],full[valid],current[valid]),dict(product=pid,attempt=attempt))
    table('SAME512_TRAINING.csv',probes);table('SAME512_TRANSITIONS.csv',probe_cells);table('FULL_TEACHER_TO_512.csv',full_cells)
    curve=[json.loads(s) for s in (D/'TRAIN_CURVE.jsonl').read_text().splitlines()]
    assert len(curve)==1000 and sum(r['updated'] for r in curve)==endpoint['updates']
    table('TRAINING_CURVE.csv',curve)
    pseudo=pin(D/'PSEUDO_LOCK.json')['rows']
    table('PSEUDO_COVERAGE.csv',[dict(product=r['product'],valid=r['valid'],accepted=r['accepted'],
        Surface=r['counts'][0],Cloud=r['counts'][1],Shadow=r['counts'][2],sha256=r['sha256']) for r in pseudo])
    extra_audits(per,group_events,endpoint)
    write(O/'INPUT_SHA256.json',INPUTS)
    write(O/'CPU_ACCEPTANCE.json',dict(status='passed',seconds=time.monotonic()-started,independent_checks=CHECKS,
        groups=32,source_student_marginals_exact=True,boundary_partition_exact=True,all_probe_zero_lora_matches_source=True,
        all_common_reference_support_retained=True,AP_independent_threshold_group_checked=True,neural_calls=0))
    report(summaries,delta,event_rows,boundary_events,group_events,probes,endpoint)
    if not REMOTE:
        S.mkdir(parents=True,exist_ok=True)
        import shutil
        for p in O.iterdir():shutil.copyfile(p,S/p.name)
        for name in ('SOURCE_ADMISSION.json','MODEL_ACCEPTANCE.json','PREPROCESS_ACCEPTANCE.json','ENDPOINT_LOCK.json','EXECUTION_LOCK.json','EVALUATION_LOCK.json','train_RECEIPT.json','evaluate_RECEIPT.json'):
            shutil.copyfile(D/name,S/name)
        shutil.copyfile(ROOT/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    print(json.dumps(load(O/'CPU_ACCEPTANCE.json')))


def extra_audits(per,group_events,endpoint):
    """Reference access is isolated here, after sealed training and evaluation."""
    from phase304_common import array_sha,counts
    assert 'torch' not in sys.modules
    started=time.monotonic()
    lock=pin(D/'PSEUDO_LOCK.json');rows=pin(D/'FIT_IMAGES.json')['rows']
    coverage={r['product']:r for r in lock['rows']}
    bindings={r['product']:r for r in pin(Path('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json'))['bindings']}
    totals={k:np.zeros((3,3),np.int64) for k in ('accepted','rejected','all')}
    product=[];support_rows=[];quality=[];probe_strata=[];probe_cells=[];bias=[]
    predicted_all=np.zeros(3,np.int64);predicted_accepted=np.zeros(3,np.int64)
    with zipfile.ZipFile(R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as masks:
        for row in rows:
            pid=row['product'];entry=coverage[pid];p=D/'teacher'/(pid+'.npz')
            assert sha(p)==entry['sha256'];INPUTS[str(p)]=sha(p)
            scores_path=Path(entry['scores_path']);assert sha(scores_path)==entry['scores_sha256'];INPUTS[str(scores_path)]=sha(scores_path)
            scores=np.load(scores_path);assert array_sha(scores)==entry['scores_array_sha256']
            raw=masks.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==bindings[pid]['mask_member_sha256']
            m=np.load(io.BytesIO(raw));y=m.argmax(-1).astype(np.uint8);one=m.sum(-1)==1
            with np.load(p) as z:pred=z['pred'];pseudo=z['pseudo'];valid=z['valid'];maximum=z['maximum']
            assert np.array_equal(pred,scores.argmax(0))
            total=scores.sum(0);q=np.divide(scores,total[None],out=np.zeros_like(scores),where=total[None]>1e-12)
            assert np.array_equal(q.max(0),maximum)
            accepted=pseudo!=255
            assert np.array_equal(accepted,valid&(total>1e-12)&(maximum>=.9))
            assert np.array_equal(pseudo[accepted],pred[accepted])
            support=valid&one
            masks_regions=dict(accepted=support&accepted,rejected=support&~accepted,all=support)
            groupmat={}
            for region,v in masks_regions.items():
                mat=confusion_matrix(y[v],pred[v],labels=[0,1,2])
                independent=np.bincount((3*y[v]+pred[v]).astype(np.int64),minlength=9).reshape(3,3)
                assert np.array_equal(mat,independent)
                groupmat[region]=mat;totals[region]+=mat
                for c in range(3):
                    for j in range(3):product.append(dict(product=pid,region=region,truth=C[c],prediction=C[j],count=int(mat[c,j])))
            assert np.array_equal(groupmat['accepted']+groupmat['rejected'],groupmat['all'])
            support_rows.append(dict(product=pid,image_valid=int(valid.sum()),reference_valid=int(support.sum()),
                reference_ignore_excluded=int((valid&~one).sum()),training_accepted=int(accepted.sum()),
                reference_accepted=int((support&accepted).sum()),reference_rejected=int((support&~accepted).sum())))
            predicted_all += np.array([int(((pred==c)&valid).sum()) for c in range(3)])
            predicted_accepted += np.array(counts(pseudo))
            if pid in sorted(coverage)[:4]:
                yy=y[255:767,255:767];v=support[255:767,255:767];acc=accepted[255:767,255:767]
                with np.load(D/'acceptance'/(pid+'.npz')) as z:a=z['scores'].argmax(0).astype(np.uint8)
                for attempt in (0,100,500,1000):
                    path=D/'probes'/(pid+'_'+str(attempt)+'.npz');assert sha(path)==endpoint['probes'][path.name]
                    with np.load(path) as z:b=z['pred'];assert np.array_equal(z['valid']&one[255:767,255:767],v)
                    for region,selected in dict(accepted=v&acc,rejected=v&~acc,all=v).items():
                        t=tensor(yy[selected],a[selected],b[selected])
                        probe_cells+=cells(t,dict(product=pid,attempt=attempt,region=region))
                        for c in range(3):
                            correct=int(t[c,c,c]);damaged=int(t[c,c].sum())-correct;repaired=int(t[c,:,c].sum())-correct
                            probe_strata.append(dict(product=pid,attempt=attempt,region=region,truth=C[c],support=int(t[c].sum()),
                                Source_correct_student_wrong=damaged,Source_wrong_student_correct=repaired,net_correct=repaired-damaged))
            del scores,m,raw;gc.collect()
    assert len(support_rows)==100 and sum(r['image_valid'] for r in support_rows)==104448397
    assert predicted_accepted.tolist()==[40622307,33554491,1292360]
    assert np.array_equal(totals['accepted']+totals['rejected'],totals['all'])
    for region,mat in totals.items():
        for c in range(3):
            denom=int(mat[:,c].sum());truth_all=int(totals['all'][c].sum());truth_accepted=int(totals['accepted'][c].sum())
            quality.append(dict(rule='Q90 fixed; all100fit teacher only',region=region,category=C[c],
                reference_region_support=int(mat.sum()),predicted_class_count=denom,correct_class_count=int(mat[c,c]),
                precision_pct=100*int(mat[c,c])/denom if denom else None,
                truth_class_total=truth_all,truth_class_accepted=truth_accepted,
                truth_class_acceptance_pct=100*truth_accepted/truth_all if truth_all else None))
    for c in range(3):bias.append(dict(category=C[c],image_valid_predicted=int(predicted_all[c]),
        image_only_accepted=int(predicted_accepted[c]),predicted_class_acceptance_pct=100*int(predicted_accepted[c])/int(predicted_all[c]),
        denominator='all image-valid pixels predicted as this class; NOT precision'))
    table('PSEUDO_QUALITY.csv',quality);table('PSEUDO_CONFUSIONS_PER_PRODUCT.csv',product)
    table('PSEUDO_CONFUSIONS_POOLED.csv',[dict(region=region,truth=C[c],prediction=C[j],count=int(mat[c,j]))
        for region,mat in totals.items() for c in range(3) for j in range(3)])
    table('PSEUDO_REFERENCE_SUPPORT.csv',support_rows);table('PREDICTED_CLASS_ACCEPTANCE.csv',bias)
    table('SAME512_Q90_STRATA.csv',probe_strata);table('SAME512_Q90_TRANSITIONS.csv',probe_cells)
    damage=[]
    for source in (r for r in per if r['strategy']=='Source'):
        student=next(r for r in per if r['strategy']=='LoRA' and r['product']==source['product'])
        def unions(r):
            mat=np.array(r['confusion']);return int(np.count_nonzero(mat.sum(0)+mat.sum(1)-np.diag(mat)))
        damage.append(dict(product=source['product'],group_id=source['group_id'],Source_mIoU_pct=source['mIoU_pct'],
            Q90_mIoU_pct=student['mIoU_pct'],delta_pp=student['mIoU_pct']-source['mIoU_pct'],
            Source_union_classes=unions(source),Q90_union_classes=unions(student)))
    table('GROUP_DAMAGE_AND_ABSENCE.csv',sorted(damage,key=lambda r:r['delta_pp']))
    concentration=[]
    for error in ('Cloud->Shadow','Surface->Cloud'):
        selected=[r for r in group_events if r['error']==error];added=sum(r['added'] for r in selected);net=sum(r['net_reduction'] for r in selected)
        for r in selected:concentration.append(dict(r,added_pool_fraction=r['added']/added if added else None,
            net_reduction_pool_fraction=r['net_reduction']/net if net else None))
    table('GROUP_ERROR_CONCENTRATION.csv',concentration)
    curve=[json.loads(s) for s in (D/'TRAIN_CURVE.jsonl').read_text().splitlines()]
    assert sum(r['updated'] for r in curve)==955 and [sum(r[c]>0 for r in curve) for c in C]==[524,644,458]
    table('TRAINING_SUPPORT.csv',[
        dict(strategy='A90 historical, unequal effective budget',valid=104448397,accepted=101857,Surface=101857,Cloud=0,Shadow=0,
            nonempty_products=5,gradient_products=2,attempts=1000,updates=6,skipped=994,class_present_steps=[6,0,0],cumulative_crop_pixel_uses=152948,base_unchanged=True),
        dict(strategy='Q90-PL current',valid=104448397,accepted=75469158,Surface=40622307,Cloud=33554491,Shadow=1292360,
            nonempty_products=97,gradient_products=len({r['product'] for r in curve if r['updated']}),attempts=1000,updates=955,skipped=45,
            class_present_steps=[524,644,458],cumulative_crop_pixel_uses=sum(r['accepted'] for r in curve),base_unchanged=True)])
    historical=R/'src/research_results/phase302_20261020/FIXED_RESULTS.csv'
    if historical.exists():
        import shutil
        INPUTS[str(historical)]=sha(historical);shutil.copyfile(historical,O/'A90_UNEQUAL_BUDGET_HISTORICAL.csv')
    write(O/'PSEUDO_AUDIT_ACCEPTANCE.json',dict(status='passed',PID=os.getpid(),after_seal=True,seal_epoch=endpoint['sealed_epoch'],
        completed_epoch=time.time(),seconds=time.monotonic()-started,fit_products=100,student_audit_probes_only=4,
        image_valid=104448397,reference_valid=sum(r['reference_valid'] for r in support_rows),
        reference_ignore_excluded=sum(r['reference_ignore_excluded'] for r in support_rows),
        training_mask_unmodified=True,accepted_rejected_partition_exact=True,confusions_histogram_bincount_equal=True,
        target_labels_not_used_for_training=True,extra_forward_calls=0))


def report(summaries,delta,events,boundaries,groups,probes,endpoint):
    from phase304_common import read
    def fmt(v):return 'NA' if v is None else f'{v:.3f}'
    quality=read(O/'PSEUDO_QUALITY.csv');bias=read(O/'PREDICTED_CLASS_ACCEPTANCE.csv')
    audit=load(O/'PSEUDO_AUDIT_ACCEPTANCE.json');worst=read(O/'GROUP_DAMAGE_AND_ABSENCE.csv')[0]
    lines=['# Phase304：固定Q90-PL三类适配','','## 数据结果','',
        'gzs cloud-lite-pt210，固定Source64dd…59dc9；全新LoRA rank4/alpha4、294912参数、seed30201。初始化与Phase302逐张量相同；原Source参数/buffer冻结eval且初末完全一致。',
        'Q90接受Surface/Cloud/Shadow=40,622,307/33,554,491/1,292,360，占104,448,397纯影像有效像元72.255%。全部100产品逐项及1000裁剪复现Phase303；955更新/45跳过，三类分别出现524/644/458步，97产品有梯度。唯一像元与累计裁剪次数见TRAINING_SUPPORT.csv。',
        '教师平均九窗s后算q=s/sum(s)，q仅用于接受；损失CE(s,pseudo)，评价Shadow评分softmax(s)[2]。末步固定封存后才读取标签和development。', '',
        '|策略|组均/池化mIoU %|Surface/Cloud/Shadow IoU %|Shadow宏/池化AP %|宏召回 %|宏/最大组误报 %|',
        '|---|---|---|---|---|---|']
    for r in summaries:
        lines.append('|'+ '|'.join(['Source-only' if r['strategy']=='Source' else 'Q90-PL',
            f"{fmt(r['group_mean_mIoU_pct'])}/{fmt(r['pooled_mIoU_pct'])}",
            '/'.join(fmt(r[k]) for k in ('Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct')),
            f"{fmt(r['macro_AP_pct'])}/{fmt(r['AP_pct'])}",fmt(r['macro_recall_pct']),
            f"{fmt(r['macro_FPR_pct'])}/{fmt(r['max_group_FPR_pct'])}"])+'|')
    lines.append('|Δ pp|'+ '|'.join([f"{fmt(delta['group_mean_mIoU_pp'])}/{fmt(delta['pooled_mIoU_pp'])}",
        '/'.join(fmt(delta[k]) for k in ('Surface_IoU_pp','Cloud_IoU_pp','Shadow_IoU_pp')),
        f"{fmt(delta['macro_AP_pp'])}/{fmt(delta['AP_pp'])}",fmt(delta['macro_recall_pp']),
        f"{fmt(delta['macro_FPR_pp'])}/{fmt(delta['max_group_FPR_pp'])}"])+'|')
    lines+=['',f"全部32组保留，AP有效组{summaries[0]['AP_groups']}/{summaries[1]['AP_groups']}；union=0的IoU为NA，组mIoU只平均有union类；AP须正负均有，宏召回/误报只用相应有支持组。A90六更新仅历史附表，非等有效训练预算消融。",'',
        '|错误|Source/Q90|消除/新增|净减少|消除中改正/转另一错|错误率Δpp|','|---|---|---|---:|---|---:|']
    for e in events:lines.append(f"|{e['error']}|{e['Source_errors']}/{e['student_errors']}|{e['removed']}/{e['added']}|{e['net_reduction']}|{e['corrected']}/{e['changed_to_other_error']}|{fmt(e['rate_change_pp'])}|")
    for error in ('Cloud->Shadow','Surface->Cloud'):
        e=next(r for r in events if r['error']==error);top=max((r for r in groups if r['error']==error),key=lambda r:r['added'])
        away=sum(r['added'] for r in boundaries if r['error']==error and r['region']=='nonboundary')
        lines.append(f"- {error}新增{e['added']:,}，参考非边界占{100*away/e['added'] if e['added'] else 0:.1f}%；最大新增组{top['product']}占{100*top['added']/e['added'] if e['added'] else 0:.1f}%。全部组与边界/非边界/排除支持另表。")
    lines+=['',f"最大组均损伤：{worst['product']}，{float(worst['Source_mIoU_pct']):.3f}→{float(worst['Q90_mIoU_pct']):.3f}%，Δ{float(worst['delta_pp']):+.3f}pp，union类数{worst['Source_union_classes']}→{worst['Q90_union_classes']}。未删除该组或更换缺类计分。",'',
        '|教师预测类|接受/拒绝/全部precision %|预测类接受率 %（非正确率）|真值类接受率 %|','|---|---|---|---|']
    for c in C:
        def q(region):return next(r for r in quality if r['category']==c and r['region']==region)
        b=next(r for r in bias if r['category']==c)
        lines.append('|'+ '|'.join([c,'/'.join('NA' if q(reg)['precision_pct']=='NA' else f"{float(q(reg)['precision_pct']):.3f}" for reg in ('accepted','rejected','all')),
            f"{float(b['predicted_class_acceptance_pct']):.3f}",f"{float(q('all')['truth_class_acceptance_pct']):.3f}"])+'|')
    lines+=['',f"全100fit教师质量审计：参考交集{audit['reference_valid']:,}，参考ignore另排除{audit['reference_ignore_excluded']:,}；训练覆盖和伪标签未按真值改动。precision分母是该区域预测类数，真值类接受率分母是该真值类全支持。",'']
    final=[r for r in probes if r['attempt']==1000];repair=sum(r['Source_wrong_student_correct'] for r in final);damage=sum(r['Source_correct_student_wrong'] for r in final)
    lines+=[f'4探针严格同512末步纠错{repair:,}、损伤{damage:,}、净正确{repair-damage:+,}。4时点×3真值类、Q90接受/拒绝分层另表；全教师→裁剪对照单列。学生fit审计仅4探针，未追加100整图学生预测。', '', '## 结论分析','']
    cs=next(e for e in events if e['error']=='Cloud->Shadow');sc=next(e for e in events if e['error']=='Surface->Cloud')
    lines.append('本轮'+('复现Cloud→Shadow净减少与Surface→Cloud净增加并存。' if cs['net_reduction']>0 and sc['net_reduction']<0 else '未同时复现Cloud→Shadow净减少与Surface→Cloud净增加。')+
        'Q90提供三类训练支持，但高得分占比并不等于标签可靠；类别质量、支持偏置和组集中度均须结合上表解释。')
    lines+=['过滤更换同时把有效更新从6改为955，不能将与Phase302的差异单独归因于置信度公式。此为固定单seed和历史development支持上的描述性结果，不外推总体机制。',
        'Source使用CloudSEN12官方train/val与源验证选模，已有空间审计未发现132产品源footprint重叠；DINO预训练交集未知。目标开发标签历史已用，不是新盲测。参考边界是半径2标签邻接邻域，不称薄云或物理边缘。全部计算在gzs；不改环境、不增seed/约束/阈值或启动下一phase。']
    (O/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        from phase304_common import append
        append('FAILURES.jsonl',dict(stage='independent_CPU_analysis',traceback=traceback.format_exc(),epoch=time.time()))
        raise

