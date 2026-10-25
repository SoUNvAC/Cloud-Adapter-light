"""Sealed independent CPU comparison on original reference support; no neural imports."""
import gc
import io
import sys
import time
import zipfile
import numpy as np
from phase310_common import *
from support_phase310 import partitions
import analyze_phase304 as legacy
ARMS=('Source','Q90-PL','Q90-SP')
COMPARE=(('Source','Q90-SP'),('Q90-PL','Q90-SP'))
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
    return dict(**prefix,original_D=int(d.sum()),D_to_Surface=ds[0],D_still_Cloud=ds[1],D_to_Shadow=ds[2],D_repair_fraction=ds[0]/int(d.sum()) if d.any() else None,outside_D_new_Surface_to_Cloud=int(outsideD.sum()),original_Cloud_correction=int(b.sum()),benefit_to_Surface=bs[0],benefit_retained_Cloud=bs[1],benefit_back_Shadow=bs[2],Cloud_correction_retention=bs[1]/int(b.sum()) if b.any() else None,outside_B_new_SourceShadow_to_Cloud=int(newB.sum()),outside_B_new_SourceWrong_to_Cloud=int(newAny.sum()))
def main():
    assert 'torch' not in sys.modules;endpoint=pin(ROOT/'ENDPOINT_LOCK.json');assert endpoint['status']=='sealed' and time.time()>=endpoint['sealed_epoch']
    pin(ROOT/'final_lora.pth',endpoint['final_sha256']);pin(ROOT/'TRAIN_CURVE.jsonl',endpoint['curve_sha256']);pin(ROOT/'MASK_LOCK.json',endpoint['mask_lock_sha256'])
    assert endpoint['all_base_tensors_unchanged'] and endpoint['teacher_all_tensors_unchanged']
    old=Path('/home/scv/shared/phase300_20261020');plan=[r for r in pin(old/'EVALUATION_INPUT_PLAN.json')['rows'] if r['pool']=='Catalogue'];expected={r['product_id']:r for r in pin(old/'EXPECTED_SUPPORT.json')['rows'] if r['pool']=='Catalogue'}
    locked=pin(ROOT/'EVALUATION_LOCK.json');scoreidx={(r['arm'],r['product']):r for r in locked['rows']}
    masks=R/'data/sentinel2_cloud_mask_catalogue_4172871/masks.zip';pin(masks)
    metricrows=[];truths=[];parts={a:[] for a in ARMS};transitions=[];flows=[];damages=[];cost=[];bounds=[]
    pooled={(cmp,reg):np.zeros((3,3,3),np.int64) for cmp in ('Source->Q90-SP','Q90-PL->Q90-SP') for reg in ('all','P','A','other')}
    pooled_damage={reg:[] for reg in ('all','P','A','other')};nonG_D=0
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
                dr=damage(y,pred['Source'],pred['Q90-PL'],pred['Q90-SP'],choose,dict(product=pid,group_id=gid,region=reg));damages.append(dr);pooled_damage[reg].append(dr)
                for c in range(3):
                    cc=choose&(y==c);n=int(cc.sum());pr=pred['Q90-SP'];oldpr=pred['Q90-PL']
                    cost.append(dict(product=pid,region=reg,truth=C[c],support=n,SP_Surface=int((cc&(pr==0)).sum()),SP_Cloud=int((cc&(pr==1)).sum()),SP_Shadow=int((cc&(pr==2)).sum()),PL_correct_SP_wrong=int((cc&(oldpr==c)&(pr!=c)).sum()),PL_wrong_SP_correct=int((cc&(oldpr!=c)&(pr==c)).sum())))
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
    for reg,records in pooled_damage.items():
        numeric=[k for k in records[0] if k not in ('product','group_id','region','D_repair_fraction','Cloud_correction_retention')]
        merged={k:sum(r[k] for r in records) for k in numeric};merged.update(D_repair_fraction=merged['D_to_Surface']/merged['original_D'] if merged['original_D'] else None,Cloud_correction_retention=merged['benefit_retained_Cloud']/merged['original_Cloud_correction'] if merged['original_Cloud_correction'] else None)
        damages.append(dict(product='ALL',group_id='ALL',region=reg,**merged))
    total=next(r for r in damages if r['product']=='ALL' and r['region']=='all');assert total['original_D']==506798 and total['original_Cloud_correction']==784478 and nonG_D==450058
    for comparison in ('Source->Q90-SP','Q90-PL->Q90-SP'):assert np.array_equal(sum(pooled[comparison,reg] for reg in ('P','A','other')),pooled[comparison,'all'])
    # Baseline metric reproduction includes all32 groups and original missing-class treatment.
    baseline=read(PL/'analysis/FIXED_RESULTS.csv')
    for name,oldname in [('Source','Source'),('Q90-PL','LoRA')]:
        observed=next(r for r in summaries if r['strategy']==name);ref=next(r for r in baseline if r['strategy']==oldname)
        for key in ('group_mean_mIoU_pct','pooled_mIoU_pct','Surface_IoU_pct','Cloud_IoU_pct','Shadow_IoU_pct','macro_AP_pct','macro_recall_pct','macro_FPR_pct','max_group_FPR_pct'):assert abs(observed[key]-float(ref[key]))<1e-6,key
    pin(PL/'analysis/FIXED_RESULTS.csv');table(ROOT/'METRICS.csv',summaries);table(ROOT/'PER_GROUP_METRICS.csv',metricrows)
    delta=[]
    for lhs,rhs in [('Q90-SP','Q90-PL'),('Q90-SP','Source'),('Q90-PL','Source')]:
        a=next(r for r in summaries if r['strategy']==lhs);b=next(r for r in summaries if r['strategy']==rhs)
        delta.append(dict(comparison=lhs+' minus '+rhs,**{k.replace('_pct','_pp'):a[k]-b[k] for k in a if k.endswith('_pct') and a[k] is not None and b[k] is not None}))
    table(ROOT/'METRIC_DIFFERENCES.csv',delta);table(ROOT/'TRANSITIONS27.csv',transitions);table(ROOT/'ERROR_FLOW.csv',flows);table(ROOT/'DAMAGE_BENEFIT.csv',damages);table(ROOT/'PROTECTION_COST.csv',cost);table(ROOT/'BOUNDARY_ERROR_FLOW.csv',bounds)
    probes=[];bindings={r['product']:r for r in pin(Path('/home/scv/shared/phase208_20261020/12201_FIT_INPUT_LOCK.json'))['bindings']};fit=load(ROOT/'FIT_IMAGES.json')['rows']
    with zipfile.ZipFile(masks) as archive:
        for pid in sorted(r['product'] for r in fit)[:4]:
            raw=archive.read('masks/'+pid+'.npy');assert __import__('hashlib').sha256(raw).hexdigest()==bindings[pid]['mask_member_sha256'];m=np.load(io.BytesIO(raw))[255:767,255:767];truth=m.argmax(-1)
            sourcepath=PL/'acceptance'/(pid+'.npz');pin(sourcepath)
            with np.load(sourcepath) as z:src=z['scores'].argmax(0)
            with np.load(ROOT/'masks'/(pid+'.npz')) as z:P=z['P'][255:767,255:767];A=z['A'][255:767,255:767]
            for attempt in (0,100,500,1000):
                path=ROOT/'probes'/(pid+'_'+str(attempt)+'.npz');pin(path,endpoint['probes'][path.name])
                with np.load(path) as z:sp=z['pred'];valid=z['valid']&(m.sum(-1)==1)
                oldpath=PL/'probes'/path.name;pin(oldpath)
                with np.load(oldpath) as z:pl=z['pred'];assert np.array_equal(z['valid']&(m.sum(-1)==1),valid)
                for region,choose in dict(all=valid,P=valid&P,A=valid&A,other=valid&~(P|A)).items():
                    for c in range(3):
                        v=choose&(truth==c)
                        probes.append(dict(product=pid,attempt=attempt,region=region,truth=C[c],support=int(v.sum()),Source_correct_SP_wrong=int((v&(src==c)&(sp!=c)).sum()),Source_wrong_SP_correct=int((v&(src!=c)&(sp==c)).sum()),Q90_matched_attempt_correct_SP_wrong=int((v&(pl==c)&(sp!=c)).sum()),Q90_matched_attempt_wrong_SP_correct=int((v&(pl!=c)&(sp==c)).sum()),same512=True))
    table(ROOT/'PROBES.csv',probes)
    # Post-seal scene tags remain descriptive and never enter training admission.
    tags=read(R/'data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv');pin(R/'data/sentinel2_cloud_mask_catalogue_4172871/classification_tags.csv')
    table(ROOT/'SCENE_TAGS_POST_SEAL.csv',[dict(product=r['product'],snow_ice=next(t['snow/ice'] for t in tags if t['scene']==r['product']),metadata_only_not_pixel_truth=True) for r in plan_to_products(plan)])
    curve=[json.loads(s) for s in (ROOT/'TRAIN_CURVE.jsonl').read_text().splitlines()];assert len(curve)==1000 and sum(r['updated'] for r in curve)==955 and sum(not r['updated'] for r in curve)==45
    assert all((r['teacher_calls']==r['student_calls']==int(r['updated'])) for r in curve)
    table(ROOT/'TRAINING_CURVE.csv',curve);masklock=load(ROOT/'MASK_LOCK.json')
    active=[r for r in curve if r['updated']]
    support=dict(strategy='Q90-SP',initial_SHA=endpoint['initial_sha256'],schedule_SHA=sha(ROOT/'CROP_SCHEDULE.json'),attempts=1000,updates=955,skipped=45,A=masklock['A'],A_classes=masklock['classes'],A_class_present_steps=[sum(r[c]>0 for r in curve) for c in C],P_unique_product_pixels=masklock['P_unique_product_pixels'],P_cumulative_all_crops=masklock['P_cumulative_all_crops'],P_cumulative_updated_crops=masklock['P_cumulative_updated_crops'],P_nonempty_all_steps=masklock['P_nonempty_all_steps'],P_nonempty_updated_steps=masklock['P_nonempty_updated_steps'],mean_PL=float(np.mean([r['PL'] for r in active])),mean_SP=float(np.mean([r['SP'] for r in active])),mean_total=float(np.mean([r['total'] for r in active])),base_unchanged=True,teacher_unchanged=True)
    table(ROOT/'TRAINING_SUPPORT.csv',[support]);table(ROOT/'GRADIENT_AUDITS.csv',[r for r in active if r['SP_grad_norm'] is not None])
    write(ROOT/'ANALYSIS_INPUT_SHA256.json',INPUTS);write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',groups=32,strategies=3,AP_independent_checks=legacy.CHECKS,original_D=506798,original_Cloud_correction=784478,nonG_D=nonG_D,all_common_support_retained=True,all27_cells_and_marginals_checked=True,P_A_other_partition_checked=True,baselines_metrics_reproduced=True,probe_rows=len(probes),teacher_calls_formal=955,student_calls_formal=955,empty_A_steps_no_forwards=True,torch_imported=False,neural_calls=0))
    from report_phase310 import report
    report()
    from plot_phase310 import plot
    plot()
def plan_to_products(plan):return [dict(product=r['product_id']) for r in plan]
if __name__=='__main__':
    try:main()
    except BaseException:append('FAILURES.jsonl',dict(stage='independent_CPU_analysis',traceback=__import__('traceback').format_exc()));raise
