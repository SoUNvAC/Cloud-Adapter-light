"""Phase214 local confusion-only audit; no models, pixels or remote access."""
from pathlib import Path
import collections,csv,hashlib,json,math,sys,time,tracemalloc
import numpy as np
ROOT=Path(__file__).resolve().parents[2];IN=ROOT/'outputs/phase213/phase213_20261021';OUT=ROOT/'outputs/phase214/phase214_20261023'
ALPHAS=[0,.25,.5,.75,1];CLASSES=['Surface','Cloud','Shadow'];DIRECTIONS=[(i,j) for i in range(3) for j in range(3) if i!=j]
START=time.monotonic();tracemalloc.start()
def native_peak_bytes():
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    x=Counters();x.cb=ctypes.sizeof(x);k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=wintypes.HANDLE
    f=ctypes.windll.psapi.GetProcessMemoryInfo;f.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    assert f(k.GetCurrentProcess(),ctypes.byref(x),x.cb)
    return int(x.PeakWorkingSetSize)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(n,rs):
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def ratio(a,b):return float(a/b) if b else None
def metrics(cm):
    c=np.asarray(cm,dtype=np.int64);tp=np.diag(c);union=c.sum(0)+c.sum(1)-tp
    iou=np.divide(tp,union,out=np.full(3,np.nan),where=union>0);m={'mIoU':float(100*np.nanmean(iou)),'effective_IoU_classes':int((union>0).sum()),'reference_classes':int((c.sum(1)>0).sum()),'valid_pixels':int(c.sum()),'reference_Shadow_pixels':int(c[2].sum()),'Shadow_recall':ratio(c[2,2],c[2].sum()),'nominal_FPR':ratio(c[:2,2].sum(),c[:2].sum()),'misclassified_pixels':int(c.sum()-tp.sum()),'misclassification_rate':ratio(c.sum()-tp.sum(),c.sum())}
    for i,name in enumerate(CLASSES):m[name+'_IoU']=float(iou[i]*100) if union[i] else None
    return m
def miou_many(c):
    tp=np.diagonal(c,axis1=-2,axis2=-1);u=c.sum(-2)+c.sum(-1)-tp
    return np.nanmean(np.divide(tp,u,out=np.full(tp.shape,np.nan),where=u>0),axis=-1)*100
def net(c,b):
    n=int(c.sum());d=c-b;r={}
    for i,j in DIRECTIONS:
        key=CLASSES[i]+'_to_'+CLASSES[j];r[key+'_net_count']=int(d[i,j]);r[key+'_net_pct_valid']=100*float(d[i,j])/n if n else None
    v=int(sum(d[i,j] for i,j in DIRECTIONS));r.update(total_misclassified_net_count=v,total_misclassified_net_pct_valid=100*v/n if n else None)
    return r
def main():
    refresh='--refresh-incomplete' in sys.argv
    if refresh:
        assert OUT.is_dir() and not (OUT/'DELIVERY_COMPLETE.json').exists()
        assert read(OUT/'RUN_STATUS.json')['status']=='analysis_complete_pending_independent_validation'
    else:OUT.mkdir(parents=True,exist_ok=False)
    complete=read(IN/'DELIVERY_COMPLETE.json');manifest=read(IN/'DELIVERY_MANIFEST.json');assert sha(IN/'DELIVERY_MANIFEST.json')==complete['manifest_sha256']
    for name,r in manifest.items():
        p=IN/name;assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],name
    required=['PER_GROUP.csv','PER_GROUP_CHANGES.csv','TEST_LABEL_METADATA_AUDIT.json','ALPHA_SELECTION_LOCK.json','ALPHA_SELECTION_TABLE.csv','PARETO_TEST_TABLE.csv','ALL_POOL_METRICS.csv','PAIRED_DIFFERENCES_CI.csv','DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json']
    save('INPUT_HASHES.json',{'root':str(IN),'files':{n:sha(IN/n) for n in required},'phase213_manifest_all_files_verified':len(manifest),'class_order':CLASSES,'protocol_sha256':sha(ROOT/'src/research_plans/PHASE214_L8_SUPPORT_SOURCE_TAIL_AUDIT_20261023.md')})
    cases=collections.defaultdict(dict);records=rows(IN/'PER_GROUP.csv')
    for r in records:
        key=(r['target'],r['pool'],float(r['alpha']));g=r['group'];raw=json.loads(r['confusion']);c=np.asarray(raw)
        assert c.shape==(3,3) and np.issubdtype(c.dtype,np.integer) and (c>=0).all() and g not in cases[key]
        c=c.astype(np.int64);m=metrics(c);assert int(r['valid_pixels'])==m['valid_pixels'] and int(r['shadow_reference_pixels'])==m['reference_Shadow_pixels'] and int(r['nonshadow_reference_pixels'])==int(c[:2].sum())
        for name in ['mIoU','Surface_IoU','Cloud_IoU','Shadow_IoU']:
            assert (r[name]=='' and m[name] is None) or abs(float(r[name])-m[name])<1e-10
        for name,metric_key in [('Shadow_recall','Shadow_recall'),('nonShadow_to_Shadow_FPR','nominal_FPR')]:
            assert (r[name]=='' and m[metric_key] is None) or abs(float(r[name])-m[metric_key])<1e-10
        cases[key][g]=c
    assert len(cases)==20
    for (t,p,a),gs in cases.items():
        assert len(gs)==(107 if p=='source' else 16 if t=='l8' else 10)
        base=cases[t,p,0];assert set(base)==set(gs)
        assert all(np.array_equal(c.sum(1),base[g].sum(1)) for g,c in gs.items())
    for a in ALPHAS:assert all(np.array_equal(cases['l8','source',0][g],cases['sparcs','source',0][g]) for g in cases['l8','source',0])
    for r in rows(IN/'ALL_POOL_METRICS.csv'):
        gs=cases[r['target'],r['pool'],float(r['alpha'])];m=metrics(sum(gs.values(),np.zeros((3,3),np.int64)))
        assert abs(float(r['mIoU'])-m['mIoU'])<1e-10 and abs(float(r['group_mean_mIoU'])-np.mean([metrics(c)['mIoU'] for c in gs.values()]))<1e-10
        assert int(r['valid_pixels'])==m['valid_pixels']
    for r in rows(IN/'PARETO_TEST_TABLE.csv'):
        if not r['alpha']:continue
        t=r['target'];a=float(r['alpha'])
        for p,col in [('source','source_pooled_mIoU'),('test','test_pooled_mIoU')]:assert abs(metrics(sum(cases[t,p,a].values(),np.zeros((3,3),np.int64)))['mIoU']-float(r[col]))<1e-10
    for r in rows(IN/'PER_GROUP_CHANGES.csv'):
        t,p,a,g=r['target'],r['pool'],float(r['alpha']),r['group'];assert abs(metrics(cases[t,p,a][g])['mIoU']-metrics(cases[t,p,0][g])['mIoU']-float(r['delta_mIoU_pp']))<1e-10
    lock=read(IN/'ALPHA_SELECTION_LOCK.json');budget_labels=collections.defaultdict(list)
    for r in lock['rows']:
        t=r['target'];base=metrics(sum(cases[t,'source',0].values(),np.zeros((3,3),np.int64)))['mIoU']
        feasible=[a for a in ALPHAS if base-metrics(sum(cases[t,'source',a].values(),np.zeros((3,3),np.int64)))['mIoU']<=r['budget_pp']]
        assert max(feasible)==r['chosen_alpha'];budget_labels[t,r['chosen_alpha']].append('B'+str(r['budget_pp']))
    for r in rows(IN/'ALPHA_SELECTION_TABLE.csv'):
        match=next(x for x in lock['rows'] if x['target']==r['target'] and x['budget_pp']==int(r['budget_pp']));assert float(r['chosen_alpha'])==match['chosen_alpha']
    save('PRIOR_REPRODUCTION.json',{'status':'passed','phase213_original_SHA_preserved':True,'all20_case_pooled_and_group_metrics_reproduced':True,'all_group_integer_confusions_reference_row_sums_exact':True,'original_budget_choices_unchanged':lock['rows'],'no_model_or_pixel_access':True})
    metadata=read(IN/'TEST_LABEL_METADATA_AUDIT.json');status={x['scene']:x['usgs_shadows'] for x in metadata['L8_scene_shadow_status']};assert set(status)==set(cases['l8','test',0])
    supports={'full':sorted(status),'yes':sorted(g for g in status if status[g]=='yes'),'no':sorted(g for g in status if status[g]=='no')};assert [len(v) for v in supports.values()]==[16,6,10]
    strat=[];scene=[];error=[];loo=[];looranges=[];matrices={}
    for support,ids in supports.items():
        for a in ALPHAS:
            cur=np.stack([cases['l8','test',a][g] for g in ids]);base=np.stack([cases['l8','test',0][g] for g in ids]);c=cur.sum(0);b=base.sum(0);m=metrics(c);delta=m['mIoU']-metrics(b)['mIoU'];matrices[support,a]=c
            strat.append(dict(support=support,alpha=a,scenes=len(ids),**m,group_mean_mIoU=float(np.mean([metrics(x)['mIoU'] for x in cur])),delta_mIoU_pp=delta,reference_qualification='provided-mask nominal counts; no is not true-negative ground truth',confusion=json.dumps(c.tolist())))
            error.append(dict(support=support,alpha=a,valid_pixels=m['valid_pixels'],**net(c,b)))
            ds=[]
            for g,cc,bb in zip(ids,cur,base):
                v=metrics(c-cc)['mIoU']-metrics(b-bb)['mIoU'];ds.append((g,v));loo.append(dict(target='l8',pool='test',support=support,alpha=a,left_out=g,remaining_groups=len(ids)-1,pooled_delta_mIoU_pp=v))
                scene.append(dict(support=support,scene=g,usgs_shadows=status[g],alpha=a,**metrics(cc),delta_mIoU_pp=metrics(cc)['mIoU']-metrics(bb)['mIoU'],**net(cc,bb),confusion=json.dumps(cc.tolist())))
            mn=min(ds,key=lambda x:(x[1],x[0]));mx=max(ds,key=lambda x:(x[1],x[0]));looranges.append(dict(target='l8',pool='test',support=support,alpha=a,groups=len(ids),full_delta_mIoU_pp=delta,LOO_min_delta_pp=mn[1],min_left_out=mn[0],LOO_max_delta_pp=mx[1],max_left_out=mx[0]))
    for a in ALPHAS:assert np.array_equal(matrices['yes',a]+matrices['no',a],matrices['full',a])
    table('L8_STRATIFIED_RESULTS.csv',strat);table('L8_SCENE_RESULTS.csv',scene);table('L8_ERROR_NET_CHANGES.csv',error)
    rng=np.random.default_rng(214023);cis=[];bootrows=[];bootrule={}
    for support in ['yes','no']:
        ids=supports[support];n=len(ids);idx=rng.integers(0,n,size=(10000,n));h=np.stack([np.bincount(v,minlength=n) for v in idx]);b=np.stack([cases['l8','test',0][g] for g in ids]);bboot=miou_many((h@b.reshape(n,9)).reshape(10000,3,3))
        bootrule[support]={'scene_ids':ids,'groups':n,'index_sha256':hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()}
        for a in [.25,.5,.75]:
            cc=np.stack([cases['l8','test',a][g] for g in ids]);d=miou_many((h@cc.reshape(n,9)).reshape(10000,3,3))-bboot;lo,hi=np.quantile(d,[.025,.975]);v=metrics(cc.sum(0))['mIoU']-metrics(b.sum(0))['mIoU']
            cis.append(dict(support=support,alpha=a,scenes=n,delta_mIoU_pp=v,CI95_low=float(lo),CI95_high=float(hi),draws=10000,seed=214023,interpretation='posthoc descriptive; conditional on support and fixed endpoints; no new decision gate'))
            bootrows.extend(dict(support=support,alpha=a,draw=i,delta_mIoU_pp=float(x)) for i,x in enumerate(d))
    table('L8_STRATIFIED_DIFFERENCES_CI.csv',cis);table('L8_BOOTSTRAP_DRAWS.csv',bootrows);save('BOOTSTRAP_RULE.json',{'seed':214023,'draws':10000,'stream':'PCG64 sequential yes6 then no10, same paired draws for all internal alphas','quantile':'linear .025/.975','supports':bootrule,'source_intervals':'Phase213 only; no new source bootstrap'})
    tail=[];roirows=[];worst=[];concentration=[]
    for t in ['l8','sparcs']:
        for a in ALPHAS:
            gs=cases[t,'source',a];bs=cases[t,'source',0];c=sum(gs.values(),np.zeros((3,3),np.int64));b=sum(bs.values(),np.zeros((3,3),np.int64));rr=[]
            for g in sorted(gs):
                m=metrics(gs[g]);sm=metrics(bs[g]);l=sm['mIoU']-m['mIoU'];ref=gs[g].sum(1);n=int(ref.sum())
                r=dict(target=t,alpha=a,budget_labels='/'.join(budget_labels[t,a]),ROI=g,loss_pp=l,Source_mIoU=sm['mIoU'],current_mIoU=m['mIoU'],Source_effective_classes=sm['effective_IoU_classes'],current_effective_classes=m['effective_IoU_classes'],effective_class_count_change=m['effective_IoU_classes']-sm['effective_IoU_classes'],reference_classes=m['reference_classes'],valid_pixels=n,Source_error_rate=sm['misclassification_rate'],current_error_rate=m['misclassification_rate'],error_rate_delta_pp=100*(m['misclassification_rate']-sm['misclassification_rate']),**net(gs[g],bs[g]))
                for k,name in enumerate(CLASSES):r.update({f'Source_{name}_IoU':sm[name+'_IoU'],f'current_{name}_IoU':m[name+'_IoU'],f'reference_{name}_pixels':int(ref[k]),f'reference_{name}_fraction':float(ref[k]/n)})
                rr.append(r)
            ordered=sorted(rr,key=lambda r:(-r['loss_pp'],r['ROI']));loss=np.array([r['loss_pp'] for r in rr]);assert len(rr)==107
            tail.append(dict(target=t,alpha=a,budget_labels='/'.join(budget_labels[t,a]),source_pooled_loss_pp=metrics(b)['mIoU']-metrics(c)['mIoU'],ROI_mean_loss_pp=float(loss.mean()),median_loss_pp=float(np.median(loss)),P90_loss_pp=float(np.quantile(loss,.9)),worst_ROI=ordered[0]['ROI'],worst_ROI_loss_pp=ordered[0]['loss_pp'],worst11_mean_loss_pp=float(np.mean([r['loss_pp'] for r in ordered[:11]])),worst11_ROI_ids=';'.join(r['ROI'] for r in ordered[:11]),**{f'ROIs_loss_gt_{k}pp':int((loss>k).sum()) for k in [1,3,5,10]}))
            roirows.extend(rr);worst.extend(dict(rank=i+1,**r) for i,r in enumerate(ordered[:10]))
            positive=sorted([r for r in rr if r['total_misclassified_net_count']>0],key=lambda r:(-r['total_misclassified_net_count'],r['ROI']));den=sum(r['total_misclassified_net_count'] for r in positive);neg=[r for r in rr if r['total_misclassified_net_count']<0]
            cr=dict(target=t,alpha=a,budget_labels='/'.join(budget_labels[t,a]),positive_net_error_sum=den,positive_ROIs=len(positive),negative_net_error_sum_signed=sum(r['total_misclassified_net_count'] for r in neg),error_reduction_magnitude=-sum(r['total_misclassified_net_count'] for r in neg),negative_ROIs=len(neg),all_ROI_net_error_sum=sum(r['total_misclassified_net_count'] for r in rr))
            for k in [1,5,10]:cr.update({f'top{k}_positive_net_errors':sum(r['total_misclassified_net_count'] for r in positive[:k]),f'top{k}_share':ratio(sum(r['total_misclassified_net_count'] for r in positive[:k]),den),f'top{k}_ROI_ids':';'.join(r['ROI'] for r in positive[:k])})
            concentration.append(cr)
            if a in [.25,.5,.75]:
                ds=[]
                for g in sorted(gs):
                    v=metrics(c-gs[g])['mIoU']-metrics(b-bs[g])['mIoU'];ds.append((g,v));loo.append(dict(target=t,pool='source',support='107_ROI',alpha=a,left_out=g,remaining_groups=106,pooled_delta_mIoU_pp=v))
                mn=min(ds,key=lambda x:(x[1],x[0]));mx=max(ds,key=lambda x:(x[1],x[0]));looranges.append(dict(target=t,pool='source',support='107_ROI',alpha=a,groups=107,full_delta_mIoU_pp=metrics(c)['mIoU']-metrics(b)['mIoU'],LOO_min_delta_pp=mn[1],min_left_out=mn[0],LOO_max_delta_pp=mx[1],max_left_out=mx[0]))
    table('SOURCE_TAIL_SUMMARY.csv',tail);table('SOURCE_ROI_NET_CHANGES.csv',roirows);table('SOURCE_WORST10_ROIS.csv',worst);table('SOURCE_ERROR_CONCENTRATION.csv',concentration)
    denominator=[]
    for t in ['l8','sparcs']:
        for a in ALPHAS:
            rr=[r for r in roirows if r['target']==t and r['alpha']==a];damaged=[r for r in rr if r['loss_pp']>5]
            denominator.append(dict(target=t,alpha=a,budget_labels='/'.join(budget_labels[t,a]),ROIs=len(rr),effective_class_count_changed=sum(r['effective_class_count_change']!=0 for r in rr),ROIs_loss_gt5=len(damaged),loss_gt5_with_effective_count_change=sum(r['effective_class_count_change']!=0 for r in damaged),source_effective_class_counts=json.dumps(dict(collections.Counter(r['Source_effective_classes'] for r in rr))),current_effective_class_counts=json.dumps(dict(collections.Counter(r['current_effective_classes'] for r in rr)))))
    table('SOURCE_DENOMINATOR_AUDIT.csv',denominator)
    overlap=[]
    for budget in [1,3,5]:
        alphas={t:next(r['chosen_alpha'] for r in lock['rows'] if r['target']==t and r['budget_pp']==budget) for t in ['l8','sparcs']}
        sets={t:{r['ROI'] for r in roirows if r['target']==t and r['alpha']==alphas[t] and r['loss_pp']>5} for t in alphas};l,s=sets['l8'],sets['sparcs']
        overlap.append(dict(budget_pp=budget,L8_alpha=alphas['l8'],SPARCS_alpha=alphas['sparcs'],L8_ROI_count=len(l),SPARCS_ROI_count=len(s),intersection_count=len(l&s),union_count=len(l|s),intersection_ids=';'.join(sorted(l&s)),union_ids=';'.join(sorted(l|s)),L8_only_ids=';'.join(sorted(l-s)),SPARCS_only_ids=';'.join(sorted(s-l))))
    table('SOURCE_BUDGET_ROI_OVERLAP.csv',overlap);table('LEAVE_ONE_GROUP_OUT.csv',loo);table('LEAVE_ONE_GROUP_OUT_RANGES.csv',looranges)
    source_ci=[r for r in rows(IN/'PAIRED_DIFFERENCES_CI.csv') if r['pool']=='source' and float(r['reference_alpha'])==0];table('SOURCE_EXISTING_213_CI.csv',source_ci)
    save('METHODS.json',{'classes':CLASSES,'alpha_candidates':ALPHAS,'unchanged_budget_selections':lock['rows'],'loss_sign':'Source ROI mIoU minus current ROI mIoU; positive means damage','tail':'11/107 worst ROI arithmetic mean, fixed ceil(10%*107), not population guarantee','percentiles':'linear interpolation','ties':'descending loss then lexicographic ROI ID','concentration':'positive per-ROI net misclassified count increments only; error reductions separately signed','flows':'six net confusion count changes only; not individual pixel transition events','no_Shadows_status_true_negative_claim':True,'missing_class_rule':'union=0 -> NA IoU; mIoU averages nonzero unions; no GT Shadow -> NA recall','no_source_bootstrap_repetition':True})
    elapsed=time.monotonic()-START;peak=tracemalloc.get_traced_memory()[1];native=native_peak_bytes()
    assert elapsed<=1800 and native<=4*1024**3 and sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=200*1024**2
    save('RUN_STATUS.json',{'status':'analysis_complete_pending_independent_validation','elapsed_seconds':elapsed,'Python_tracemalloc_peak_bytes':peak,'native_peak_working_set_bytes':native,'new_model_calls':0,'remote_connections':0,'new_fits':0})
    print(json.dumps({'elapsed_seconds':elapsed,'stratified_internal':cis,'selected_tails':[r for r in tail if r['budget_labels']],'overlap':overlap},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
