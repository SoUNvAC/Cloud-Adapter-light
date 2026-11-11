"""CPU-only Phase213 paired ROI/scene bootstrap and complete tables."""
from pathlib import Path
import collections,csv,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/phase213/phase213_20261021'; RAW=OUT/'delivery'
OLD=ROOT/'outputs/phase212/phase212_retrain_20261020/evaluation/delivery'
ALPHAS=[0,.25,.5,.75,1]
sys.path.insert(0,str(ROOT/'src/tools'))
from analyze_phase212_retrained import metrics,miou_many
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(n,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def source_file(t,a):return OLD/('Source__source.json' if a==0 else f'{t}_a{str(a).replace(".","p")}__source.json')
def group_array(r):return {g:np.asarray(c,dtype=np.int64) for g,c in r['per_group'].items()}
def summarize(r):
    gs=group_array(r);cm=np.asarray(r['confusion'],np.int64)
    assert np.array_equal(sum(gs.values(),np.zeros((3,3),np.int64)),cm)
    by=collections.defaultdict(lambda:np.zeros((3,3),np.int64))
    for x in r['per_input']:by[x['group']]+=np.asarray(x['confusion'],np.int64)
    assert set(by)==set(gs) and all(np.array_equal(by[g],gs[g]) for g in gs)
    assert len(r['per_input'])==r['inputs'] and len(gs)==r['groups'] and (cm>=0).all()
    scalar=[]
    for k in range(3):
        den=sum(int(cm[k,j]) for j in range(3))+sum(int(cm[j,k]) for j in range(3))-int(cm[k,k])
        if den:scalar.append(int(cm[k,k])/den*100)
    m=metrics(cm);assert abs(sum(scalar)/len(scalar)-m['mIoU'])<1e-10
    ms=[metrics(c) for c in gs.values()]
    return dict(m,group_mean_mIoU=float(np.mean([x['mIoU'] for x in ms])),max_group_FPR=max((x['nonShadow_to_Shadow_FPR'] for x in ms if x['nonShadow_to_Shadow_FPR'] is not None),default=None),groups=len(gs),inputs=len(r['per_input']),ignore_pixels=sum(x.get('ignore_pixels',0) for x in r['per_input']),no_shadow_groups=sum(x['shadow_reference_pixels']==0 for x in ms),zero_shadow_recall_groups=sum(x['Shadow_recall']==0 for x in ms if x['Shadow_recall'] is not None))
def main():
    assert (RAW/'EXIT_CODE').read_text().strip()=='0'
    complete=read(RAW/'RUN_COMPLETE.json');assert complete['calls']==11420<=12000 and complete['elapsed_seconds']<=10800
    adm=read(RAW/'FINAL_ADMISSION_LOCK.json');assert adm['source_recheck_exact'] and adm['calls_before_target']==535 and adm['selection_unchanged']
    source_recheck=read(RAW/'source_a0.json');source_old=read(OLD/'Source__source.json')
    assert source_recheck['confusion']==source_old['confusion'] and source_recheck['prediction_sha256']==source_old['prediction_sha256']
    lock=read(OUT/'ALPHA_SELECTION_LOCK.json');cases={};summaries={};pergroup=[];errors=[]
    for t in ['l8','sparcs']:
        for a in ALPHAS:
            for pool,r in [('source',read(source_file(t,a))),('test',read(RAW/f'{t}_a{str(a).replace(".","p")}.json'))]:
                cases[t,pool,a]=r;summaries[t,pool,a]=summarize(r)
                assert r['groups']==(107 if pool=='source' else 16 if t=='l8' else 10)
                for g,c in group_array(r).items():
                    m=metrics(c);ref=c.sum(1);pergroup.append(dict(target=t,pool=pool,alpha=a,group=g,**m,missing_reference_classes=','.join(str(i) for i in range(3) if ref[i]==0),zero_shadow_recall=m['Shadow_recall']==0 if ref[2] else None,confusion=json.dumps(c.tolist())))
                    errors.append(dict(target=t,pool=pool,alpha=a,group=g,Shadow_to_Surface=int(c[2,0]),Shadow_to_Cloud=int(c[2,1]),Surface_to_Shadow=int(c[0,2]),Cloud_to_Shadow=int(c[1,2]),shadow_reference=int(ref[2]),nonshadow_reference=int(ref[:2].sum())))
    # One reproducible stream: source ROI first, then L8 test, then SPARCS test.
    rng=np.random.default_rng(213020);weights={};index={};boots={};cirows=[]
    for poolkey,n in [('source',107),('l8',16),('sparcs',10)]:
        ix=rng.integers(0,n,size=(10000,n));h=np.stack([np.bincount(row,minlength=n) for row in ix]);weights[poolkey]=h;index[poolkey]=ix
    for t in ['l8','sparcs']:
        for pool in ['source','test']:
            base=cases[t,pool,0];gs=sorted(base['per_group']);truth=np.stack([np.asarray(base['per_group'][g]).sum(1) for g in gs]);h=weights['source' if pool=='source' else t]
            for a in ALPHAS:
                r=cases[t,pool,a];assert sorted(r['per_group'])==gs
                cm=np.stack([r['per_group'][g] for g in gs]);assert np.array_equal(cm.sum(2),truth)
                sampled=(h@cm.reshape(len(gs),9)).reshape(10000,3,3);v=miou_many(sampled);boots[t,pool,a]=v
                independent_sum=np.zeros(10000);independent_n=np.zeros(10000,dtype=int)
                for k in range(3):
                    num=sampled[:,k,k].astype(float);den=sampled[:,k,:].sum(1)+sampled[:,:,k].sum(1)-num;ok=den>0
                    independent_sum[ok]+=num[ok]/den[ok];independent_n+=ok
                assert np.max(np.abs(independent_sum/independent_n*100-v))<1e-10
                for j in [0,4999,9999]:assert abs(metrics(cm[index['source' if pool=='source' else t][j]].sum(0))['mIoU']-v[j])<1e-10
            for a in ALPHAS:
                for ref in [0,1]:
                    delta=boots[t,pool,a]-boots[t,pool,ref];lo,hi=np.quantile(delta,[.025,.975])
                    ordered=sorted(float(v) for v in delta)
                    for q,value in [(.025,lo),(.975,hi)]:
                        rank=9999*q;j=int(rank);manual=ordered[j]+(rank-j)*(ordered[j+1]-ordered[j]);assert abs(manual-value)<1e-10
                    cirows.append(dict(target=t,pool=pool,alpha=a,reference_alpha=ref,delta_mIoU_pp=summaries[t,pool,a]['mIoU']-summaries[t,pool,ref]['mIoU'],CI95_low=float(lo),CI95_high=float(hi),groups=len(gs),draws=10000,seed=213020))
    meta_path=ROOT/'src/research_plans/protocol_data/l8_biome_usgs_shadow_status.csv'
    meta={r['scene']:r['usgs_shadows'] for r in csv.DictReader(meta_path.open(encoding='utf-8-sig'))}
    quality=[{'scene':g,'usgs_shadows':meta.get(g,'unknown')} for g in sorted(cases['l8','test',0]['per_group'])]
    write('TEST_LABEL_METADATA_AUDIT.json',{'scope':'metadata only; no additional image/label evaluation, no selection changes','L8_full_original_manifest_kept':True,'L8_scene_shadow_status':quality,'status_counts':dict(collections.Counter(r['usgs_shadows'] for r in quality)),'metadata_sha256':hashlib.sha256(meta_path.read_bytes()).hexdigest(),'interpretation':'USGS Shadows=no/unknown does not establish reliable negative shadow truth; raw full-manifest metrics must retain this annotation limitation; no post-test filtering'})
    allrows=[];budget=[]
    for t in ['l8','sparcs']:
        for a in ALPHAS:
            s=summaries[t,'source',a];m=summaries[t,'test',a];ci=next(x for x in cirows if x['target']==t and x['pool']=='test' and x['alpha']==a and x['reference_alpha']==0)
            allrows.append(dict(target=t,alpha=a,source_pooled_mIoU=s['mIoU'],source_group_mean_mIoU=s['group_mean_mIoU'],source_Shadow_IoU=s['Shadow_IoU'],source_loss_pp=summaries[t,'source',0]['mIoU']-s['mIoU'],test_pooled_mIoU=m['mIoU'],test_group_mean_mIoU=m['group_mean_mIoU'],test_Surface_IoU=m['Surface_IoU'],test_Cloud_IoU=m['Cloud_IoU'],test_Shadow_IoU=m['Shadow_IoU'],test_Shadow_recall=m['Shadow_recall'],test_FPR=m['nonShadow_to_Shadow_FPR'],test_max_group_FPR=m['max_group_FPR'],test_zero_recall_groups=m['zero_shadow_recall_groups'],test_no_shadow_groups=m['no_shadow_groups'],test_gain_vs_Source_pp=ci['delta_mIoU_pp'],CI95_low=ci['CI95_low'],CI95_high=ci['CI95_high'],comparison_support='target_test FP32',test_ignore_pixels=m['ignore_pixels']))
        for r in lock['rows']:
            if r['target']!=t:continue
            feasible=[a for a in ALPHAS if summaries[t,'source',0]['mIoU']-summaries[t,'source',a]['mIoU']<=r['budget_pp']]
            assert max(feasible)==r['chosen_alpha'];v=next(x for x in allrows if x['target']==t and x['alpha']==r['chosen_alpha']);budget.append(dict(budget_pp=r['budget_pp'],**v))
    history=read(OLD/'HISTORICAL_REFERENCES.json');historical=[]
    for t in ['l8','sparcs']:
        source_history=read(OLD/f'Source_historical_fp16_diagnostic__{t}.json')
        historical.append(dict(target=t,method='Source historical',source_pooled_mIoU=77.81788921828681,historical_target_val_mIoU=metrics(source_history['confusion'])['mIoU'],comparison_support='historical target_val FP16; not matched to target_test'))
        historical.append(dict(target=t,method='shared-parent historical',source_pooled_mIoU={'l8':63.21,'sparcs':69.92}[t],historical_target_val_mIoU={'l8':64.10,'sparcs':69.65}[t],comparison_support='Phase64 final audit rounded target_val FP16; old head unavailable; not matched to target_test'))
        for method in ['MsRE','LoRA']:
            r=history[method][t];tv=r.get('target_metrics_exact',{}).get('mIoU',r.get('target_mIoU_rounded_only'));sv=r.get('source_metrics_exact',{}).get('mIoU')
            historical.append(dict(target=t,method=method,source_pooled_mIoU=sv,historical_target_val_mIoU=tv,comparison_support='historical target_val FP16; not matched to current target_test; no dominance claim'))
    changes=[]
    for t in ['l8','sparcs']:
        for pool in ['source','test']:
            for a in ALPHAS:
                for g,c in group_array(cases[t,pool,a]).items():
                    m=metrics(c);b=metrics(cases[t,pool,0]['per_group'][g])
                    changes.append(dict(target=t,pool=pool,alpha=a,group=g,delta_mIoU_pp=m['mIoU']-b['mIoU'],delta_Shadow_recall_pp=None if m['Shadow_recall'] is None else 100*(m['Shadow_recall']-b['Shadow_recall']),delta_FPR_pp=None if m['nonShadow_to_Shadow_FPR'] is None else 100*(m['nonShadow_to_Shadow_FPR']-b['nonShadow_to_Shadow_FPR'])))
    table('PER_GROUP_CHANGES.csv',changes)
    table('PARETO_TEST_TABLE.csv',allrows+historical);table('BUDGET_TEST_TABLE.csv',budget);table('PAIRED_DIFFERENCES_CI.csv',cirows);table('PER_GROUP.csv',pergroup);table('ERROR_FLOWS.csv',errors)
    table('ALL_POOL_METRICS.csv',[dict(target=t,pool=p,alpha=a,**m) for (t,p,a),m in summaries.items()])
    write('BOOTSTRAP_RULE.json',{'seed':213020,'draws':10000,'stream':'PCG64, source107,L8test16,SPARCStest10 in order; whole-group confusion summed before IoU; paired all alphas','no_pixel_independence':True})
    write('ANALYSIS_ACCEPTANCE.json',{'status':'passed','cases_checked':len(cases),'per_input_group_pool_confusion_exact':True,'identical_reference_support_all_alphas':True,'independent_scalar_mIoU':True,'independent_selected_bootstrap_draws':True,'all10000_draws_classwise_IoU_verified':True,'manual_quantile_interpolation_verified':True,'selection_max_feasible_unrounded':True,'source_recheck_prediction_exact':True,'new_model_calls':0,'small_scene_samples':[16,10]})
    write('ANALYSIS_SUMMARY.json',{'budgets':budget,'candidates':allrows,'history':historical})
    print(json.dumps({'budgets':budget},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
