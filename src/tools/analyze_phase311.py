"""Sealed independent CPU comparison on original reference support; no neural imports."""
import gc
import io
import sys
import time
import zipfile
import numpy as np
from phase311_common import *
from support_phase311 import partitions
import analyze_phase304 as legacy
ARMS=('Source','Q90-PL','Q90-SP','Q90-CM')
COMPARE=(('Source','Q90-CM'),('Q90-PL','Q90-CM'),('Q90-SP','Q90-CM'))
INPUTS={}
def pin(p,expected=None):
    h=sha(p);assert expected is None or h==expected,str(p)
    INPUTS[str(p)]=h;return load(p) if Path(p).suffix=='.json' else h
def damage(y,source,pl,sp,choose,prefix):
    d=choose&(y==0)&(source==0)&(pl==1);b=choose&(y==1)&(source==2)&(pl==1)
    ds=[int((d&(sp==i)).sum()) for i in range(3)];bs=[int((b&(sp==i)).sum()) for i in range(3)]
    assert sum(ds)==int(d.sum()) and sum(bs)==int(b.sum())
    outsideD=choose&(y==0)&(source==0)&(pl!=1)&(sp==1)
    newB=choose&(y==1)&(source==2)&(pl!=1)&(sp==1)
    newAny=choose&(y==1)&(source!=1)&(pl!=1)&(sp==1)
    return dict(**prefix,original_D=int(d.sum()),D_to_Surface=ds[0],D_still_Cloud=ds[1],D_to_Shadow=ds[2],D_repair_fraction=ds[0]/int(d.sum()) if d.any() else None,outside_D_new_Surface_to_Cloud=int(outsideD.sum()),outside_D_new_Surface_to_Shadow=int((choose&(y==0)&(source==0)&~d&(sp==2)).sum()),original_Cloud_correction=int(b.sum()),benefit_to_Surface=bs[0],benefit_retained_Cloud=bs[1],benefit_back_Shadow=bs[2],Cloud_correction_retention=bs[1]/int(b.sum()) if b.any() else None,outside_B_new_SourceShadow_to_Cloud=int(newB.sum()),outside_B_new_SourceWrong_to_Cloud=int(newAny.sum()))
def main():
    assert 'torch' not in sys.modules;endpoint=pin(ROOT/'ENDPOINT_LOCK.json');assert endpoint['status']=='sealed' and time.time()>=endpoint['sealed_epoch']
    pin(ROOT/'final_lora.pth',endpoint['final_sha256']);pin(ROOT/'TRAIN_CURVE.jsonl',endpoint['curve_sha256']);pin(ROOT/'MASK_LOCK.json',endpoint['mask_lock_sha256'])
    assert endpoint['all_base_tensors_unchanged'] and endpoint['teacher_all_tensors_unchanged']
    old=Path('/home/scv/shared/phase300_20261020');plan=[r for r in pin(old/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue'];expected={r['product_id']:r for r in pin(old/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}
    locked=pin(ROOT/'EVALUATION_LOCK.json');scoreidx={(r['arm'],r['product']):r for r in locked['rows']}
    masks=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip';pin(masks)
    metricrows=[];truths=[];parts={a:[] for a in ARMS};transitions=[];flows=[];damages=[];cost=[];bounds=[]
    pooled={(cmp,reg):np.zeros((3,3,3),np.int64) for cmp in tuple(a+'->'+b for a,b in COMPARE) for reg in ('all','P','A','other')}
    pooled_damage={(arm,reg):[] for arm in ARMS[1:] for reg in ('all','P','A','other')};nonG_D=0
    with zipfile.ZipFile(masks) as archive:
        for row in plan:
            pid=row['product_id'];gid=row['group']['group_id'];supportpath=Path('/home/scv/shared/phase205_20261019/infer/R4/scores/Catalogue')/(pid+'.npz');pin(supportpath)
            with np.load(supportpath) as z:ids=z['flat_indices'];valid=z['valid']
            assert array_sha(ids)==expected[pid]['flat_indices_sha256'] and array_sha(valid)==expected[pid]['common_valid_sha256']
            common=np.zeros((1022,1022),bool);common.ravel()[ids[valid]]=True;assert int(common.sum())==expected[pid]['count']
            raw=archive.read('masks/'+pid+'.npy');assert __import__('hashlib').sha256(raw).hexdigest()==row['group']['mask_member_sha256']
            m=np.load(io.BytesIO(raw));assert (m.sum(-1)[common]==1).all();y=m.argmax(-1).astype(np.uint8);pred={};source_s=None;imagevalid=None
            for arm in ARMS:
                entry=scoreidx[arm,pid];p=Path(entry['path']);pin(p,entry['sha256'])
                with np.load(p) as z:s=z['scores'];pr=z['pred'];sv=z['shadow'];v=z['valid']
                assert np.array_equal(pr,s.argmax(0)) and not (common&~v).any()
                exp=np.exp(s-s.max(0,keepdims=True));exp/=exp.sum(0,keepdims=True);assert np.allclose(sv,exp[2],rtol=1e-6,atol=1e-7)
                if arm=='Source':source_s=s;imagevalid=v
                pred[arm]=pr;parts[arm].append((pr[common].copy(),sv[common].copy()))
                measurements=legacy.metric(y[common],pr[common],sv[common]);mat=np.asarray(measurements['confusion']);unions=mat.sum(0)+mat.sum(1)-np.diag(mat)
                metricrows.append(dict(strategy=arm,product=pid,group_id=gid,union_classes=int((unions>0).sum()),**measurements))
            A,P,_=partitions(source_s,imagevalid);regions=dict(all=common,P=common&P,A=common&A,other=common&~(P|A))
            assert np.array_equal(regions['P'].astype(np.uint8)+regions['A']+regions['other'],common.astype(np.uint8))
            nonG_D+=int((regions['P']&(y==0)&(pred['Source']==0)&(pred['Q90-PL']==1)).sum())
            for reg,choose in regions.items():
                for arm in ARMS[1:]:
                    dr=damage(y,pred['Source'],pred['Q90-PL'],pred[arm],choose,dict(strategy=arm,product=pid,group_id=gid,region=reg));damages.append(dr);pooled_damage[arm,reg].append(dr)
                for c in range(3):
                    cc=choose&(y==c);n=int(cc.sum());pr=pred['Q90-CM']
                    for oldarm in ARMS[:-1]:
                        oldpr=pred[oldarm]
                        cost.append(dict(product=pid,region=reg,comparison=oldarm+'->Q90-CM',truth=C[c],support=n,CM_Surface=int((cc&(pr==0)).sum()),CM_Cloud=int((cc&(pr==1)).sum()),CM_Shadow=int((cc&(pr==2)).sum()),old_correct_CM_wrong=int((cc&(oldpr==c)&(pr!=c)).sum()),old_wrong_CM_correct=int((cc&(oldpr!=c)&(pr==c)).sum())))
                for a,b in COMPARE:
                    comparison=a+'->'+b;t=legacy.tensor(y[choose],pred[a][choose],pred[b][choose]);pooled[comparison,reg]+=t;pre=dict(product=pid,group_id=gid,region=reg,comparison=comparison)
                    transitions+=legacy.cells(t,pre);flows+=legacy.events(t,pre)
            # Boundary tables are secondary checks, with the original radius2 valid-interior convention.
            for reg,choose in legacy.regions(y,common).items():
                for a,b in COMPARE:bounds+=legacy.events(legacy.tensor(y[choose],pred[a][choose],pred[b][choose]),dict(product=pid,group_id=gid,region=reg,comparison=a+'->'+b))
            truths.append(y[common]);gc.collect();print('CPU group',pid,flush=True)
    summaries=[];y=np.concatenate(truths)
    for arm in ARMS:
        p=np.concatenate([p for p,s in parts[arm]]);s=np.concatenate([s for p,s in parts[arm]]);mt=legacy.metric(y,p,s);per=[r for r in metricrows if r['strategy']==arm]
        summaries.append(dict(strategy=arm,groups=len(per),AP_groups=sum(r['AP_pct'] is not None for r in per),group_mean_mIoU_pct=legacy.average(r['mIoU_pct'] for r in per),pooled_mIoU_pct=mt['mIoU_pct'],Surface_IoU_pct=mt['Surface_IoU_pct'],Cloud_IoU_pct=mt['Cloud_IoU_pct'],Shadow_IoU_pct=mt['Shadow_IoU_pct'],macro_AP_pct=legacy.average(r['AP_pct'] for r in per),pooled_AP_pct=mt['AP_pct'],macro_recall_pct=legacy.average(r['recall_pct'] for r in per),macro_FPR_pct=legacy.average(r['FPR_pct'] for r in per),max_group_FPR_pct=max(r['FPR_pct'] for r in per if r['FPR_pct'] is not None),confusion=mt['confusion']))
        del p,s;gc.collect()
    for comparison,reg in pooled:
        pre=dict(product='ALL',group_id='ALL',region=reg,comparison=comparison);t=pooled[comparison,reg];transitions+=legacy.cells(t,pre);flows+=legacy.events(t,pre)
    for (arm,reg),records in pooled_damage.items():
        numeric=[k for k in records[0] if k not in ('strategy','product','group_id','region','D_repair_fraction','Cloud_correction_retention')]
        merged={k:sum(r[k] for r in records) for k in numeric};merged.update(D_repair_fraction=merged['D_to_Surface']/merged['original_D'] if merged['original_D'] else None,Cloud_correction_retention=merged['benefit_retained_Cloud']/merged['original_Cloud_correction'] if merged['original_Cloud_correction'] else None)
        damages.append(dict(strategy=arm,product='ALL',group_id='ALL',region=reg,**merged))
    total=next(r for r in damages if r['strategy']=='Q90-CM' and r['product']=='ALL' and r['region']=='all');assert total['original_D']==506798 and total['original_Cloud_correction']==784478 and nonG_D==450058
    oldsp=next(r for r in damages if r['strategy']=='Q90-SP' and r['product']=='ALL' and r['region']=='all')
    assert [oldsp[k] for k in ('benefit_retained_Cloud','benefit_back_Shadow','benefit_to_Surface')]==[362132,417682,4664]
    for comparison in tuple(a+'->'+b for a,b in COMPARE):assert np.array_equal(sum(pooled[comparison,reg] for reg in ('P','A','other')),pooled[comparison,'all'])
    # Baseline metric reproduction includes all32 groups and original missing-class treatment.
    baseline=read(PL/'analysis/FIXED_RESULTS.csv')
    for name,oldname in [('Source','Source'),('Q90-PL','LoRA')]:
        observed=next(r for r in summaries if r['strategy']==name);ref=next(r for r in baseline if r['strategy']==oldname)
        for key in ('group_mean_mIoU_pct','pooled_mIoU_pct','Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct','macro_AP_pct','macro_recall_pct','macro_FPR_pct','max_group_FPR_pct'):assert abs(observed[key]-float(ref[key]))<1e-6,key
    oldmetrics=read(SP/'METRICS.csv');pin(SP/'METRICS.csv')
    for oldrow in oldmetrics:
        observed=next(r for r in summaries if r['strategy']==oldrow['strategy'])
        for key in observed:
            if key.endswith('_pct'):assert abs(observed[key]-float(oldrow[key]))<1e-6,key
    pin(PL/'analysis/FIXED_RESULTS.csv');table(ROOT/'METRICS.csv',summaries);table(ROOT/'PER_GROUP_METRICS.csv',metricrows)
    delta=[]
    for lhs,rhs in [('Q90-CM','Q90-PL'),('Q90-CM','Q90-SP'),('Q90-CM','Source')]:
        a=next(r for r in summaries if r['strategy']==lhs);b=next(r for r in summaries if r['strategy']==rhs)
        delta.append(dict(comparison=lhs+' minus '+rhs,**{k.replace('_pct','_pp'):a[k]-b[k] for k in a if k.endswith('_pct') and a[k] is not None and b[k] is not None}))
    table(ROOT/'METRIC_DIFFERENCES.csv',delta);table(ROOT/'TRANSITIONS27.csv',transitions);table(ROOT/'ERROR_FLOW.csv',flows);table(ROOT/'DAMAGE_BENEFIT.csv',damages);table(ROOT/'PROTECTION_COST.csv',cost);table(ROOT/'BOUNDARY_ERROR_FLOW.csv',bounds)
    probes=[];bindings={r['product']:r for r in pin(Path('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json'))['bindings']};fit=load(ROOT/'FIT_IMAGES.json')['rows']
    with zipfile.ZipFile(masks) as archive:
        for pid in sorted(r['product'] for r in fit)[:4]:
            raw=archive.read('masks/'+pid+'.npy');assert __import__('hashlib').sha256(raw).hexdigest()==bindings[pid]['mask_member_sha256'];m=np.load(io.BytesIO(raw))[255:767,255:767];truth=m.argmax(-1)
            sourcepath=PL/'acceptance'/(pid+'.npz');pin(sourcepath)
            with np.load(sourcepath) as z:src=z['scores'].argmax(0)
            with np.load(SP/'masks'/(pid+'.npz')) as z:P=z['P'][255:767,255:767];A=z['A'][255:767,255:767]
            for attempt in (0,100,500,1000):
                path=ROOT/'probes'/(pid+'_'+str(attempt)+'.npz');pin(path,endpoint['probes'][path.name])
                with np.load(path) as z:cm=z['pred'];valid=z['valid']&(m.sum(-1)==1)
                oldpath=PL/'probes'/path.name;pin(oldpath,load(PL/'ENDPOINT_LOCK.json')['probes'][path.name])
                with np.load(oldpath) as z:pl=z['pred'];assert np.array_equal(z['valid']&(m.sum(-1)==1),valid)
                sppath=SP/'probes'/path.name;pin(sppath,load(SP/'ENDPOINT_LOCK.json')['probes'][path.name])
                with np.load(sppath) as z:sp=z['pred'];assert np.array_equal(z['valid']&(m.sum(-1)==1),valid)
                for region,choose in dict(all=valid,P=valid&P,A=valid&A,other=valid&~(P|A)).items():
                    for c in range(3):
                        v=choose&(truth==c)
                        probes.append(dict(product=pid,attempt=attempt,region=region,truth=C[c],support=int(v.sum()),Source_correct_CM_wrong=int((v&(src==c)&(cm!=c)).sum()),Source_wrong_CM_correct=int((v&(src!=c)&(cm==c)).sum()),PL_correct_CM_wrong=int((v&(pl==c)&(cm!=c)).sum()),PL_wrong_CM_correct=int((v&(pl!=c)&(cm==c)).sum()),SP_correct_CM_wrong=int((v&(sp==c)&(cm!=c)).sum()),SP_wrong_CM_correct=int((v&(sp!=c)&(cm==c)).sum()),same512=True))
    table(ROOT/'PROBES.csv',probes)
    for comparison in (a+'->Q90-CM' for a in ARMS[:-1]):
        for region in ('all','P','A','other'):
            for truth in C:
                records=[r for r in cost if r['comparison']==comparison and r['region']==region and r['truth']==truth and r['product']!='ALL']
                cost.append(dict(product='ALL',region=region,comparison=comparison,truth=truth,**{k:sum(r[k] for r in records) for k in records[0] if k not in ('product','region','comparison','truth')}))
    table(ROOT/'PROTECTION_COST.csv',cost)
    # Post-seal scene tags remain descriptive and never enter training admission.
    tags=read(R/'data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv');pin(R/'data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv')
    table(ROOT/'SCENE_TAGS_POST_SEAL.csv',[dict(product=r['product'],snow_ice=next(t['snow/ice'] for t in tags if t['scene']==r['product']),metadata_only_not_pixel_truth=True) for r in plan_to_products(plan)])
    curve=[json.loads(s) for s in (ROOT/'TRAIN_CURVE.jsonl').read_text().splitlines()];assert len(curve)==1000 and sum(r['updated'] for r in curve)==955 and sum(not r['updated'] for r in curve)==45
    assert all((r['teacher_calls']==r['student_calls']==int(r['updated'])) for r in curve)
    table(ROOT/'TRAINING_CURVE.csv',curve);masklock=load(ROOT/'MASK_LOCK.json')
    active=[r for r in curve if r['updated']]
    support=dict(strategy='Q90-CM',initial_SHA=endpoint['initial_sha256'],schedule_SHA=sha(ROOT/'CROP_SCHEDULE.json'),attempts=1000,updates=955,skipped=45,A=masklock['A'],A_classes=masklock['classes'],A_class_present_steps=[sum(r[c]>0 for r in curve) for c in C],P_unique_product_pixels=masklock['P_unique_product_pixels'],P_cumulative_all_crops=masklock['P_cumulative_all_crops'],P_cumulative_updated_crops=masklock['P_cumulative_updated_crops'],P_nonempty_all_steps=masklock['P_nonempty_all_steps'],P_nonempty_updated_steps=masklock['P_nonempty_updated_steps'],mean_PL=float(np.mean([r['PL'] for r in active])),mean_CM=float(np.mean([r['CM'] for r in active])),mean_total=float(np.mean([r['total'] for r in active])),base_unchanged=True,teacher_unchanged=True)
    table(ROOT/'TRAINING_SUPPORT.csv',[support]);table(ROOT/'GRADIENT_AUDITS.csv',[r for r in active if r['CM_grad_norm'] is not None])
    assert masklock['P_unique_product_pixels']==8201741 and masklock['P_cumulative_updated_crops']==19937970
    assert all(r['active_count']<=r['P'] and (r['updated'] or r['CM'] is None) for r in curve)
    table(ROOT/'CONSTRAINT_ACTIVITY.csv',[dict(attempt=r['attempt'],updated=r['updated'],P=r['P'],**{k:r[k] for k in ('active_count','active_fraction','mT_mean','mS_mean','d_mean','d_max','CM')}) for r in curve])
    write(ROOT/'ACTIVITY_SUMMARY.json',dict(updated_steps=955,P_nonempty_updated_steps=sum(r['P']>0 for r in active),active_steps=sum(r['active_count']>0 for r in active),P_updated_cumulative=sum(r['P'] for r in active),active_pixel_exposures=sum(r['active_count'] for r in active),active_fraction_of_all_updated_P=sum(r['active_count'] for r in active)/sum(r['P'] for r in active),mean_CM=float(np.mean([r['CM'] for r in active])),skip_rule='A empty: no forwards, losses/margins NA; active_count recorded 0 means unevaluated, not measured inactive. Updated P empty: differentiable CM0, margins NA.'))
    # Fixed historical top3, no selection from the new result.
    top3=('T25XEG','T40XDR','T35RPM')
    table(ROOT/'FIXED_TOP3_METRICS.csv',[r for r in metricrows if any(k in r['product'] for k in top3)])
    table(ROOT/'FIXED_TOP3_DAMAGE.csv',[r for r in damages if any(k in r['product'] for k in top3)])
    worst=[dict(r) for r in metricrows if 'T25XEG' in r['product']]
    for r in worst:
        mat=np.asarray(r['confusion']);assert mat[0].sum()==1044484 and mat[1:].sum()==0
        r['Surface_support']=int(mat[0].sum());r['Cloud_support']=0;r['Shadow_support']=0
        r['Surface_to_Cloud']=int(mat[0,1]);r['Surface_to_Shadow']=int(mat[0,2])
    table(ROOT/'LARGEST_DAMAGE_GROUP.csv',worst)
    diag=next(r for r in damages if 'T25XEG' in r['product'] and r['strategy']=='Q90-CM' and r['region']=='all');assert diag['original_D']==206076
    diagP=next(r for r in damages if 'T25XEG' in r['product'] and r['strategy']=='Q90-CM' and r['region']=='P');assert diagP['original_D']==197082
    assert next(r for r in worst if r['strategy']=='Q90-SP')['Surface_to_Shadow']==33087
    # Clarify legacy field names: Source_pred/Source_errors mean the comparison's left arm.
    (ROOT/'TABLE_README.md').write_text('27格: truth/Source_pred/LoRA_pred 是 truth/比较左端/比较右端；错误流 Source_errors/LoRA_errors 同理，非恒指Source。P/A/other固定由影像Source定义。D/B均固定于Source与PL，不按CM重定义。AP/IoU零分母NA，group mIoU仅平均union非空类。全部32组均保留。\n',encoding='utf-8')
    write(ROOT/'ANALYSIS_INPUT_SHA256.json',INPUTS);write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',groups=32,strategies=4,AP_independent_checks=legacy.CHECKS,original_D=506798,original_Cloud_correction=784478,nonG_D=nonG_D,all_common_support_retained=True,all27_cells_and_marginals_checked=True,P_A_other_partition_checked=True,baselines_metrics_reproduced=True,probe_rows=len(probes),teacher_calls_formal=955,student_calls_formal=955,empty_A_steps_no_forwards=True,torch_imported=False,neural_calls=0))
    from report_phase311 import report
    report()
    from plot_phase311 import plot
    plot()
def plan_to_products(plan):return [dict(product=r['product_id']) for r in plan]
if __name__=='__main__':
    try:main()
    except BaseException:append('FAILURES.jsonl',dict(stage='independent_CPU_analysis',traceback=__import__('traceback').format_exc()));raise
