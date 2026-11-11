"""Local CPU tables and fixed paired-group bootstrap; no model/pixel access."""
from pathlib import Path
import collections,csv,hashlib,json,time,ctypes
from ctypes import wintypes
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase216/phase216_20261023';RAW=OUT/'delivery';AS=[0,.25,.5,.75,1]
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def write(n,rows):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def miou(c):
    c=np.asarray(c,dtype=np.float64);d=np.diagonal(c,axis1=-2,axis2=-1);u=c.sum(-2)+c.sum(-1)-d
    return 100*np.nanmean(np.divide(d,u,out=np.full_like(d,np.nan),where=u>0),axis=-1)
def metrics(c):
    c=np.asarray(c,dtype=np.int64);d=np.diag(c);u=c.sum(0)+c.sum(1)-d;cs=['Surface','Cloud','Shadow']
    r={'mIoU':float(miou(c)),'valid_pixels':int(c.sum()),'shadow_pixels':int(c[2].sum())}
    for i,n in enumerate(cs):r[n+'_IoU']=100*float(d[i]/u[i]) if u[i] else None
    for name,num,den in [('Shadow_recall',c[2,2],c[2].sum()),('nonShadow_to_Shadow_FPR',c[:2,2].sum(),c[:2].sum())]:r[name]=float(num/den) if den else None
    for i,x in enumerate(cs):
        for j,y in enumerate(cs):
            if i!=j:r[x+'_to_'+y+'_pixels']=int(c[i,j]);r[x+'_to_'+y+'_rate']=float(c[i,j]/c[i].sum()) if c[i].sum() else None
    return r
def data(arm,a,pool):return js(RAW/'evaluation'/('Source__'+pool+'.json' if a==0 else f'{arm}_a{a}__{pool}.json'))
def groups(r):return {g:np.asarray(c,np.int64) for g,c in r.get('per_group',r.get('per_scene')).items()}
def peak():
    class P(ctypes.Structure):_fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    p=P();p.cb=ctypes.sizeof(p);k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=wintypes.HANDLE;f=ctypes.windll.psapi.GetProcessMemoryInfo;f.argtypes=[wintypes.HANDLE,ctypes.POINTER(P),wintypes.DWORD];assert f(k.GetCurrentProcess(),ctypes.byref(p),p.cb);return p.peak
def main():
    clock=time.monotonic();assert (RAW/'EXIT_CODE').read_text().strip()=='0'
    complete=js(RAW/'COMPLETE.json');lock=js(RAW/'SOURCE_SELECTION_LOCK.json');assert lock['target_metrics_loaded_before_lock'] is False
    caches={(arm,a,p):data(arm,a,p) for arm in ['C','D'] for a in AS for p in ['source','val','test']}
    baseline={p:groups(data('C',0,p)) for p in ['source','val','test']};assert [len(baseline[p]) for p in ['source','val','test']]==[107,10,10]
    pooled=[];pergroup=[];flow=[];alltable=[];summary={};tails=[]
    for arm in ['C','D']:
        for a in AS:
            sm=lock['source_metrics'][arm][str(a)]
            sg=groups(caches[arm,a,'source']);ordered=sorted(sg,key=lambda g:(float(miou(sg[g])-miou(baseline['source'][g])),g))
            for rank,g in enumerate(ordered,1):tails.append({'arm':arm,'alpha':a,'rank_by_source_loss':rank,'ROI':g,'loss_pp':float(miou(baseline['source'][g])-miou(sg[g])),'in_T11':rank<=11,'Source_mIoU':float(miou(baseline['source'][g])),'candidate_mIoU':float(miou(sg[g]))})
            for p in ['source','val','test']:
                raw=caches[arm,a,p];gs=groups(raw);assert set(gs)==set(baseline[p])
                total=np.asarray(raw['confusion'],np.int64);assert np.array_equal(total,sum(gs.values()))
                assert all(np.array_equal(c.sum(1),baseline[p][g].sum(1)) for g,c in gs.items())
                ms=[metrics(c) for c in gs.values()];m=metrics(total)
                agg={'arm':arm,'alpha':a,'pool':p,'groups':len(gs),'group_mean_mIoU':float(np.mean([q['mIoU'] for q in ms])),'max_group_FPR':max(q['nonShadow_to_Shadow_FPR'] for q in ms if q['nonShadow_to_Shadow_FPR'] is not None),'groups_without_shadow':sum(q['shadow_pixels']==0 for q in ms),'zero_shadow_recall_groups':sum(q['shadow_pixels']>0 and q['Shadow_recall']==0 for q in ms),**m}
                for n in ['Surface_IoU','Cloud_IoU','Shadow_IoU','Shadow_recall','nonShadow_to_Shadow_FPR']:
                    vals=[q[n] for q in ms if q[n] is not None];agg['group_mean_'+n]=float(np.mean(vals)) if vals else None
                pooled.append(agg);summary[arm,a,p]=agg
                bm=metrics(sum(baseline[p].values()))
                for src in ['Surface','Cloud','Shadow']:
                    for dst in ['Surface','Cloud','Shadow']:
                        if src==dst:continue
                        k=src+'_to_'+dst;flow.append({'arm':arm,'alpha':a,'pool':p,'group':'__pooled__','flow':k,'pixels':m[k+'_pixels'],'net_pixels_vs_Source':m[k+'_pixels']-bm[k+'_pixels'],'rate':m[k+'_rate'],'rate_delta':None if m[k+'_rate'] is None else m[k+'_rate']-bm[k+'_rate']})
                for g,c in sorted(gs.items()):
                    met=metrics(c);base=metrics(baseline[p][g]);pergroup.append({'arm':arm,'alpha':a,'pool':p,'group':g,'confusion':json.dumps(c.tolist()),'delta_mIoU_pp':met['mIoU']-base['mIoU'],**met})
                    for src in ['Surface','Cloud','Shadow']:
                        for dst in ['Surface','Cloud','Shadow']:
                            if src==dst:continue
                            k=src+'_to_'+dst
                            flow.append({'arm':arm,'alpha':a,'pool':p,'group':g,'flow':k,'pixels':met[k+'_pixels'],'net_pixels_vs_Source':met[k+'_pixels']-base[k+'_pixels'],'rate':met[k+'_rate'],'rate_delta':None if met[k+'_rate'] is None else met[k+'_rate']-base[k+'_rate']})
            test=summary[arm,a,'test'];source=summary[arm,a,'source'];val=summary[arm,a,'val']
            train=js(RAW/f'{arm}_TRAIN_COMPLETE.json')
            alltable.append({'arm':arm,'alpha':a,**sm,'source_mIoU':source['mIoU'],'source_group_mean_mIoU':source['group_mean_mIoU'],'val_mIoU':val['mIoU'],'test_mIoU':test['mIoU'],'test_Shadow_IoU':test['Shadow_IoU'],'test_Shadow_recall':test['Shadow_recall'],'test_FPR':test['nonShadow_to_Shadow_FPR'],'training_seconds':train['train_seconds'],'training_units':sum(train['counts_delta'].values()),'new_eval_units':0 if a==0 else 555,'common_Source_eval_units':555 if a==0 else 0})
    write('ALL_CANDIDATES.csv',alltable);write('ALL_POOL_METRICS.csv',pooled);write('PER_GROUP.csv',pergroup);write('ERROR_FLOWS.csv',flow);write('SOURCE_TAIL_GROUPS.csv',tails)
    chosen={(r['arm'],r['budget_pp']):r['alpha'] for r in lock['rows']}
    rng=np.random.default_rng(216023);ix_test=rng.integers(0,10,size=(10000,10));rng=np.random.default_rng(216023);ix_source=rng.integers(0,107,size=(10000,107))
    test_ids=sorted(baseline['test']);source_ids=sorted(baseline['source']);sbase=np.stack([baseline['source'][g] for g in source_ids]);base_group_iou=miou(sbase)
    joint=[];intervals=[];bootstrap={}
    for budget in [1,3,5]:
        ca,da=chosen['C',budget],chosen['D',budget];ct=summary['C',ca,'test'];dt=summary['D',da,'test'];cs=lock['source_metrics']['C'][str(ca)];ds=lock['source_metrics']['D'][str(da)]
        joint.append({'budget_pp':budget,'C_alpha':ca,'D_alpha':da,'C_source_pool_loss_pp':cs['pool_loss_pp'],'D_source_pool_loss_pp':ds['pool_loss_pp'],'C_T11_pp':cs['T11_pp'],'D_T11_pp':ds['T11_pp'],'C_test_mIoU':ct['mIoU'],'D_test_mIoU':dt['mIoU'],'D_minus_C_test_mIoU_pp':dt['mIoU']-ct['mIoU'],'C_Source_fallback':ca==0,'D_Source_fallback':da==0})
        cg=groups(caches['C',ca,'test']);dg=groups(caches['D',da,'test']);c=np.stack([cg[g] for g in test_ids]);d=np.stack([dg[g] for g in test_ids]);tdiff=miou(d[ix_test].sum(1))-miou(c[ix_test].sum(1))
        cg=groups(caches['C',ca,'source']);dg=groups(caches['D',da,'source']);c=np.stack([cg[g] for g in source_ids]);d=np.stack([dg[g] for g in source_ids]);pooldiff=miou(c[ix_source].sum(1))-miou(d[ix_source].sum(1))
        closs=base_group_iou-miou(c);dloss=base_group_iou-miou(d)
        taildiff=np.sort(dloss[ix_source],axis=1)[:,-11:].mean(1)-np.sort(closs[ix_source],axis=1)[:,-11:].mean(1)
        for name,point,draws in [('test_mIoU_D_minus_C_pp',dt['mIoU']-ct['mIoU'],tdiff),('source_pool_loss_D_minus_C_pp',ds['pool_loss_pp']-cs['pool_loss_pp'],pooldiff),('source_T11_D_minus_C_pp',ds['T11_pp']-cs['T11_pp'],taildiff)]:
            lo,hi=np.quantile(draws,[.025,.975]);intervals.append({'budget_pp':budget,'C_alpha':ca,'D_alpha':da,'metric':name,'point':point,'CI95_low':float(lo),'CI95_high':float(hi),'groups':10 if name.startswith('test') else 107,'draws':10000,'seed':216023,'conditional':'fixed trained model and selected alpha; excludes train/replay/selection uncertainty; source already used for selection'})
            bootstrap[str(budget)+'/'+name]={'first100':draws[:100].tolist(),'SHA':hashlib.sha256(draws.tobytes()).hexdigest()}
    write('JOINT_RULE.csv',joint);write('PAIRED_CI.csv',intervals);save('BOOTSTRAP_PROOF.json',bootstrap)
    dominance=[]
    for ca in AS:
        for da in AS:
            c=summary['C',ca,'test'];d=summary['D',da,'test'];cs=lock['source_metrics']['C'][str(ca)];ds=lock['source_metrics']['D'][str(da)]
            dominance.append({'C_alpha':ca,'D_alpha':da,'D_dominates_C':d['mIoU']>=c['mIoU'] and ds['T11_pp']<=cs['T11_pp'] and (d['mIoU']>c['mIoU'] or ds['T11_pp']<cs['T11_pp']),'D_minus_C_test_mIoU_pp':d['mIoU']-c['mIoU'],'D_minus_C_T11_pp':ds['T11_pp']-cs['T11_pp'],'D_minus_C_pool_loss_pp':ds['pool_loss_pp']-cs['pool_loss_pp'],'C_worst_ROI_loss_pp':cs['worst_ROI_loss_pp'],'D_worst_ROI_loss_pp':ds['worst_ROI_loss_pp'],'interpretation':'finite two-dimensional point estimate only; no selection by target test'})
    write('FINITE_SET_DOMINANCE.csv',dominance)
    memory=peak();assert memory<=16*1024**3
    save('ANALYSIS_RECEIPT.json',{'seconds':time.monotonic()-clock,'native_peak_working_set_bytes':memory,'new_model_calls':0,'bootstrap_draws_per_metric':10000,'seed':216023,'source_selection_SHA':sha(RAW/'SOURCE_SELECTION_LOCK.json'),'source_parent_classes':3,'source_label_access_from_training':0,'code_SHA':sha(Path(__file__)),'raw_complete':complete})
    print(json.dumps(joint,indent=2))
if __name__=='__main__':main()
