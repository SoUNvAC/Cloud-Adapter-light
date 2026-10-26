"""Post-both-seal original32 reference readout, fixed event sets, and seed-only summaries."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import gc
import io
import resource
import sys
import time
import zipfile
from collections import defaultdict
import numpy as np
import analyze_phase304 as legacy
from phase315_common import *
C=('Surface','Cloud','Shadow');SEEDS=(30201,31501,31502)
def ah(a):return hashlib.sha256(a.tobytes()).hexdigest()
def mean(v):
    a=[x for x in v if x is not None];return sum(a)/len(a) if a else None
def main():
    assert sys.platform=='linux' and 'torch' not in sys.modules;start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3));seal=load(ROOT/'BOTH_SEALED.json');assert seal['status']=='sealed' and time.time()>seal['epoch']
    for seed in [31501,31502]:assert sha(ROOT/str(seed)/'EVALUATION_LOCK.json')==seal['evaluation_locks'][str(seed)]
    assert not (ROOT/'CPU_ACCEPTANCE.json').exists();inputs={}
    def pin(p):inputs[str(p)]=sha(p);return load(p)
    old=Path('/home/scv/shared/phase300_20261020');support={r['product_id']:r for r in pin(old/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'};plan={r['product_id']:r for r in pin(old/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue'}
    rows=load(ROOT/'EVAL_IMAGES.json')['rows'];assert len(rows)==32
    oldlock=pin(PL/'EVALUATION_LOCK.json');oldsha={(r['product'],r['arm']):r['sha256'] for r in oldlock['rows']};newlocks={s:{r['product']:r for r in pin(ROOT/str(s)/'EVALUATION_LOCK.json')['rows']} for s in [31501,31502]}
    masked=BASE/'masks.zip';inputs[str(masked)]=sha(masked);per=[];flows=[];cells=[];bcells=[];bflows=[];bsupport=[];events=[];overlap=[];truths=[];parts={s:[] for s in ['Source']+list(SEEDS)};pooled={s:np.zeros((3,3,3),np.int64) for s in SEEDS};Dtotal=defaultdict(int);Btotal=defaultdict(int);initials=[]
    with zipfile.ZipFile(masked) as masks:
        for row in rows:
            pid=row['product'];gid=row['group_id'];sfile=Path('/home/scv/shared/phase205_20261019/infer/R4/scores/Catalogue')/(pid+'.npz');inputs[str(sfile)]=sha(sfile)
            with np.load(sfile) as z:ids=z['flat_indices'];v=z['valid']
            ex=support[pid];assert ah(ids)==ex['flat_indices_sha256'] and ah(v)==ex['common_valid_sha256'];valid=np.zeros((1022,1022),bool);valid.ravel()[ids[v]]=True;assert int(valid.sum())==ex['count']
            raw=masks.read('masks/'+pid+'.npy');assert hashlib.sha256(raw).hexdigest()==plan[pid]['group']['mask_member_sha256'];mask=np.load(io.BytesIO(raw));assert mask.dtype==bool and mask.shape==(1022,1022,3) and (mask.sum(-1)[valid]==1).all();truth=mask.argmax(-1).astype(np.uint8);y=truth[valid];truths.append(y);preds={};scores={}
            for seed in ['Source']+list(SEEDS):
                path=Path('/home/scv/shared/phase302_20261020/evaluation')/f'Source_{pid}.npz' if seed=='Source' else PL/'evaluation'/f'LoRA_{pid}.npz' if seed==30201 else Path(newlocks[seed][pid]['file']);expected=oldsha[pid,'Source' if seed=='Source' else 'LoRA'] if seed in ['Source',30201] else newlocks[seed][pid]['sha256'];assert sha(path)==expected;inputs[str(path)]=expected
                with np.load(path) as z:p=z['pred'];sc=z['shadow'];vv=z['valid'];assert not (valid&~vv).any() and np.array_equal(p,z['scores'].argmax(0))
                preds[seed]=p;scores[seed]=sc;measure=legacy.metric(y,p[valid],sc[valid]);mat=np.array(measure['confusion']);measure['union_classes']=int(np.count_nonzero(mat.sum(0)+mat.sum(1)-np.diag(mat)));per.append(dict(seed=seed,product=pid,group_id=gid,**measure));parts[seed].append((p[valid],sc[valid]))
            riskD=int((valid&(truth==0)&(preds['Source']==0)).sum());riskB=int((valid&(truth==1)&(preds['Source']==2)).sum());ds={s:valid&(truth==0)&(preds['Source']==0)&(preds[s]==1) for s in SEEDS};bs={s:valid&(truth==1)&(preds['Source']==2)&(preds[s]==1) for s in SEEDS}
            regions=legacy.regions(truth,valid)
            for name,r in regions.items():bsupport.append(dict(product=pid,group_id=gid,region=name,support=int(r.sum())))
            for seed in SEEDS:
                t=legacy.tensor(y,preds['Source'][valid],preds[seed][valid]);pooled[seed]+=t;prefix=dict(seed=seed,product=pid,group_id=gid);cells+=legacy.cells(t,prefix);flows+=legacy.events(t,prefix)
                Dn=int(ds[seed].sum());Bn=int(bs[seed].sum());Dtotal[seed]+=Dn;Btotal[seed]+=Bn
                events.append(dict(**prefix,D=Dn,B=Bn,truth_Surface=int((y==0).sum()),Source_correct_Surface_risk=riskD,truth_Cloud=int((y==1).sum()),Source_Cloud_as_Shadow_risk=riskB,D_fraction_Surface=Dn/int((y==0).sum()) if (y==0).any() else None,D_fraction_Source_correct_Surface=Dn/riskD if riskD else None,B_fraction_Cloud=Bn/int((y==1).sum()) if (y==1).any() else None,B_fraction_Source_Cloud_as_Shadow=Bn/riskB if riskB else None))
                for region,r in regions.items():
                    bt=legacy.tensor(truth[r],preds['Source'][r],preds[seed][r]);bcells+=legacy.cells(bt,dict(**prefix,region=region));bflows+=legacy.events(bt,dict(**prefix,region=region))
            for seed in [31501,31502]:
                for kind,sets in [('D',ds),('B',bs)]:
                    a,b=sets[30201],sets[seed];inter=int((a&b).sum());union=int((a|b).sum());overlap.append(dict(seed=seed,reference_seed=30201,product=pid,group_id=gid,event=kind,old=int(a.sum()),new=int(b.sum()),intersection=inter,union=union,old_only=int((a&~b).sum()),new_only=int((b&~a).sum()),Jaccard=inter/union if union else None,old_to_Surface=int((a&(preds[seed]==0)).sum()),old_to_Cloud=int((a&(preds[seed]==1)).sum()),old_to_Shadow=int((a&(preds[seed]==2)).sum()),interpretation='initialization agreement only; not intervention repair/retention'))
            gc.collect()
    assert Dtotal[30201]==506798 and Btotal[30201]==784478
    y=np.concatenate(truths);metrics=[]
    for seed in ['Source']+list(SEEDS):
        p=np.concatenate([t[0] for t in parts[seed]]);s=np.concatenate([t[1] for t in parts[seed]]);m=legacy.metric(y,p,s);rr=[r for r in per if r['seed']==seed]
        metrics.append(dict(seed=seed,groups=32,group_mean_mIoU_pct=mean([r['mIoU_pct'] for r in rr]),pooled_mIoU_pct=m['mIoU_pct'],Surface_IoU_pct=m['Surface_IoU_pct'],Cloud_IoU_pct=m['Cloud_IoU_pct'],Shadow_IoU_pct=m['Shadow_IoU_pct'],macro_Shadow_AP_pct=mean([r['AP_pct'] for r in rr]),pooled_Shadow_AP_pct=m['AP_pct'],macro_Shadow_recall_pct=mean([r['recall_pct'] for r in rr]),macro_Shadow_FPR_pct=mean([r['FPR_pct'] for r in rr]),max_group_Shadow_FPR_pct=max(r['FPR_pct'] for r in rr if r['FPR_pct'] is not None),AP_groups=sum(r['AP_pct'] is not None for r in rr),recall_groups=sum(r['recall_pct'] is not None for r in rr),FPR_groups=sum(r['FPR_pct'] is not None for r in rr),confusion=m['confusion']))
        if seed!='Source':assert np.array_equal(np.array(m['confusion']),pooled[seed].sum(1))
        del p,s;gc.collect()
    for seed,t in pooled.items():
        assert np.array_equal(t.sum(2),metrics[0]['confusion']);flows+=legacy.events(t,dict(seed=seed,product='POOLED32',group_id='32relatedgroups'));cells+=legacy.cells(t,dict(seed=seed,product='POOLED32',group_id='32relatedgroups'))
    base_metric=metrics[0];numeric=[k for k in base_metric if k.endswith('_pct')];summary=[]
    for metric in numeric:
        vals=[m[metric] for m in metrics[1:] if m[metric] is not None];v=np.array(vals)
        summary.append(dict(metric=metric,Source=base_metric[metric],seeds=len(v),mean=float(v.mean()) if len(v) else None,sample_SD_ddof1=float(v.std(ddof=1)) if len(v)>1 else None,min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None,mean_minus_Source=float(v.mean())-base_metric[metric] if len(v) and base_metric[metric] is not None else None,Source_SD=None,estimand='initialization seeds only; fixed32 relatedgroups not96independentcases'))
    replication=[]
    for seed,m in zip(SEEDS,metrics[1:]):
        cs=next(r for r in flows if r['seed']==seed and r['product']=='POOLED32' and r['error']=='Cloud->Shadow');sc=next(r for r in flows if r['seed']==seed and r['product']=='POOLED32' and r['error']=='Surface->Cloud')
        replication.append(dict(seed=seed,group_mIoU_delta_pp=m['group_mean_mIoU_pct']-base_metric['group_mean_mIoU_pct'],pooled_mIoU_delta_pp=m['pooled_mIoU_pct']-base_metric['pooled_mIoU_pct'],Cloud_Shadow_net_reduction=cs['net_reduction'],Surface_Cloud_net_increase=-sc['net_reduction'],B_cloud_corrections=Btotal[seed],D_surface_damage=Dtotal[seed],macro_Shadow_recall_delta_pp=m['macro_Shadow_recall_pct']-base_metric['macro_Shadow_recall_pct'],group_mIoU_improves=m['group_mean_mIoU_pct']>base_metric['group_mean_mIoU_pct'],Cloud_Shadow_decreases=cs['net_reduction']>0,Surface_Cloud_increases=sc['net_reduction']<0))
    repstats=[dict(metric=k,mean=float(np.mean([r[k] for r in replication])),sample_SD_ddof1=float(np.std([r[k] for r in replication],ddof=1)),min=min(r[k] for r in replication),max=max(r[k] for r in replication),seeds=3,Source_SD=None) for k in ['group_mIoU_delta_pp','pooled_mIoU_delta_pp','Cloud_Shadow_net_reduction','Surface_Cloud_net_increase','B_cloud_corrections','D_surface_damage','macro_Shadow_recall_delta_pp']]
    table(ROOT/'REPLICATION_STATISTICS.csv',repstats);table(ROOT/'SEED_SOURCE_DELTAS.csv',[dict(seed=m['seed'],metric=k,Source=base_metric[k],seed_value=m[k],delta_pp=None if m[k] is None or base_metric[k] is None else m[k]-base_metric[k]) for m in metrics[1:] for k in numeric])
    directions=[dict(observation=k,direction_count=sum(r[k] for r in replication),seeds=3,effect_min=min(r[metric] for r in replication),effect_max=max(r[metric] for r in replication),no_p_value=True,scope='fixed schedule/Source/LoRA initialization repeat only') for k,metric in [('group_mIoU_improves','group_mIoU_delta_pp'),('Cloud_Shadow_decreases','Cloud_Shadow_net_reduction'),('Surface_Cloud_increases','Surface_Cloud_net_increase')]]
    for seed in [31501,31502]:
        for kind in ['D','B']:
            rr=[r for r in overlap if r['seed']==seed and r['event']==kind];total={k:sum(r[k] for r in rr) for k in ['old','new','intersection','union','old_only','new_only','old_to_Surface','old_to_Cloud','old_to_Shadow']};overlap.append(dict(seed=seed,reference_seed=30201,product='POOLED32',group_id='32relatedgroups',event=kind,**total,Jaccard=total['intersection']/total['union'] if total['union'] else None,interpretation='initialization agreement only; not intervention repair/retention'))
    group_summary=[]
    for row in rows:
        pid=row['product'];src=next(r for r in per if r['seed']=='Source' and r['product']==pid)
        for key in ['mIoU_pct','Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct','AP_pct','recall_pct','FPR_pct']:
            vals=[r[key]-src[key] for r in per if r['product']==pid and r['seed']!='Source' and r[key] is not None and src[key] is not None];group_summary.append(dict(product=pid,group_id=row['group_id'],metric=key,valid_seeds=len(vals),mean=mean(vals),min=min(vals) if vals else None,max=max(vals) if vals else None,positive=sum(v>0 for v in vals),negative=sum(v<0 for v in vals),zero=sum(v==0 for v in vals)))
    hist=load(ROOT/'HISTORICAL_INITIAL_IDENTITY.json');ident=[dict(seed=30201,evidence='historical sealed endpoint, no retraining',initial_A_sha256=hist['A_sha256'],B_zero=hist['B_all_zero'],parameters=hist['parameters'],attempts=1000,updates=955,skipped=45,Source_SHA=SOURCE,schedule_SHA=sha(ROOT/'CROP_SCHEDULE.json'),pseudo_SHA=sha(PL/'PSEUDO_LOCK.json'),crop_identity_SHA=sha(ROOT/'CROP_IDENTITY.csv'),accepted_cumulative=181307540,class_present_steps=[524,644,458],base_unchanged=True,actual_new_formal_forwards=0)]
    for seed in [31501,31502]:
        d=ROOT/str(seed);gate=load(d/'MODEL_ACCEPTANCE.json');ep=load(d/'ENDPOINT_LOCK.json');curves=[__import__('json').loads(s) for s in (d/'TRAIN_CURVE.jsonl').read_text().splitlines()];assert len(curves)==1000 and sum(r['updated'] for r in curves)==955 and sum(r['accepted'] for r in curves)==181307540
        assert all(r['mask_sha256']==e['mask_sha256'] for r,e in zip(curves,read(ROOT/'CROP_IDENTITY.csv')))
        calls=[__import__('json').loads(line) for line in (ROOT/'CALLS.jsonl').read_text().splitlines()];selected=[r for r in calls if r['seed']==seed];formal=[sum(r['context']=='train' and r['kind']=='train' for r in selected),sum(r['context']=='evaluate' and r['kind']=='evaluation_window' for r in selected),sum(r['context'].startswith('probe_') and r['kind']=='probe' for r in selected)];assert formal==[955,288,16]
        ident.append(dict(seed=seed,evidence='new independently initialized sealed repeat',initial_A_sha256=gate['A_sha256'],B_zero=gate['B_all_zero'],parameters=gate['parameters'],attempts=1000,updates=ep['updates'],skipped=ep['skipped'],Source_SHA=SOURCE,schedule_SHA=sha(ROOT/'CROP_SCHEDULE.json'),pseudo_SHA=sha(PL/'PSEUDO_LOCK.json'),crop_identity_SHA=sha(ROOT/'CROP_IDENTITY.csv'),accepted_cumulative=181307540,class_present_steps=[sum(r[c]>0 for r in curves) for c in C],base_unchanged=ep['all_base_tensors_unchanged'],actual_new_formal_forwards=sum(formal)))
    assert len({r['initial_A_sha256'] for r in ident})==3
    top3=[r['product'] for r in sorted([r for r in events if r['seed']==30201],key=lambda r:-r['D'])[:3]]
    for n,v in {'SEED_IDENTITY.csv':ident,'SEED_METRICS.csv':metrics,'SEED_METRIC_SUMMARY.csv':summary,'PER_GROUP_METRICS.csv':per,'ERROR_FLOW.csv':flows,'TRANSITIONS27.csv':cells,'BOUNDARY_TRANSITIONS.csv':bcells,'BOUNDARY_ERROR_FLOW.csv':bflows,'BOUNDARY_SUPPORT.csv':bsupport,'D_B_PER_GROUP.csv':events,'EVENT_OVERLAP.csv':overlap,'REPLICATION_SUMMARY.csv':replication,'DIRECTION_COUNTS.csv':directions,'GROUP_SEED_DELTA_SUMMARY.csv':group_summary,'HISTORICAL_TOP3.csv':[dict(rank=i+1,product=p,evidence='fixed historical30201 D0 ranking, posthoc diagnostic only') for i,p in enumerate(top3)],'TOP3_SEED_DIAGNOSTIC.csv':[r for r in events if r['product'] in top3]}.items():table(ROOT/n,v)
    historical={ (r['strategy'],r['product']):r for r in read(PL/'analysis/PER_GROUP_METRICS.csv')}
    for r in per:
        if r['seed'] in ['Source',30201]:assert __import__('json').loads(historical['Source' if r['seed']=='Source' else 'LoRA',r['product']]['confusion'])==r['confusion']
    write(ROOT/'CPU_INPUT_SHA256.json',inputs);available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024;assert available>=4000000000
    write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',both_endpoints_sealed_before_references=True,groups=32,seeds=list(SEEDS),historical_D0=Dtotal[30201],historical_B0=Btotal[30201],historical64_confusions_exact=True,tensor_margins_exact=True,all32groups_retained=True,no_torch=True,seconds=time.monotonic()-start,peak_cpu_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,host_available_end=available))
    assert 'torch' not in sys.modules;write(ROOT/'STATUS.json',dict(status='CPU_complete_pending_audit_and_admission',neural_forwards=seal['forwards']));print(__import__('json').dumps(replication));print(__import__('json').dumps(directions))
if __name__=='__main__':main()
