"""Local paired whole-group confusion bootstrap; no pixels, weights or ML calls."""
from pathlib import Path
import collections
import csv
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/phase212/phase212_retrain_20261020/evaluation'
RAW=OUT/'delivery'

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(n,rs):
    with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def ratio(n,d):return float(n/d) if d else None
def metrics(cm):
    c=np.asarray(cm,np.int64);tp=np.diag(c);union=c.sum(0)+c.sum(1)-tp
    i=np.divide(tp,union,out=np.full(3,np.nan),where=union>0)
    neg=c[:2].sum()
    return {'mIoU':100*float(np.nanmean(i)),'Surface_IoU':None if np.isnan(i[0]) else 100*float(i[0]),'Cloud_IoU':None if np.isnan(i[1]) else 100*float(i[1]),'Shadow_IoU':None if np.isnan(i[2]) else 100*float(i[2]),'Shadow_recall':ratio(c[2,2],c[2].sum()),'nonShadow_to_Shadow_FPR':ratio(c[:2,2].sum(),neg),'Shadow_to_Surface':ratio(c[2,0],c[2].sum()),'Shadow_to_Cloud':ratio(c[2,1],c[2].sum()),'valid_pixels':int(c.sum()),'shadow_reference_pixels':int(c[2].sum()),'nonshadow_reference_pixels':int(neg)}
def miou_many(c):
    tp=np.diagonal(c,axis1=-2,axis2=-1);u=c.sum(-2)+c.sum(-1)-tp
    return np.nanmean(np.divide(tp,u,out=np.full(tp.shape,np.nan),where=u>0),axis=-1)*100

def main():
    assert (RAW/'EXIT_CODE').read_text().strip()=='0'
    complete=load(RAW/'COMPLETED.json');assert complete['calls']<=16000
    cases={};pergroups=[];errors=[];checks=0
    for p in sorted(RAW.glob('*__*.json')):
        r=load(p);pool=r['pool'];name=r['model'];cm=np.asarray(r['confusion'],np.int64)
        gs={k:np.asarray(v,np.int64) for k,v in r['per_group'].items()}
        assert np.array_equal(cm,sum(gs.values(),np.zeros((3,3),np.int64)))
        ig=collections.defaultdict(lambda:np.zeros((3,3),np.int64))
        for x in r['per_input']:ig[x['group']]+=np.asarray(x['confusion'],np.int64)
        assert set(ig)==set(gs) and all(np.array_equal(ig[g],gs[g]) for g in gs)
        assert int(cm.sum())>0 and np.all(cm>=0)
        # Independent scalar set-union implementation of the pooled mIoU.
        independent=[]
        for k in range(3):
            den=int(sum(int(cm[k,j]) for j in range(3))+sum(int(cm[j,k]) for j in range(3))-cm[k,k])
            if den:independent.append(int(cm[k,k])/den*100)
        assert abs(sum(independent)/len(independent)-metrics(cm)['mIoU'])<1e-12
        checks+=1
        gm=[]
        for g,c in gs.items():
            m=metrics(c);gm.append(m);pergroups.append(dict(model=name,pool=pool,group=g,**m,confusion=json.dumps(c.tolist())))
            errors.append({'model':name,'pool':pool,'group':g,'Shadow_to_Surface_count':int(c[2,0]),'Shadow_to_Cloud_count':int(c[2,1]),'Surface_to_Shadow_count':int(c[0,2]),'Cloud_to_Shadow_count':int(c[1,2]),'Shadow_to_Surface':m['Shadow_to_Surface'],'Shadow_to_Cloud':m['Shadow_to_Cloud'],'FPR':m['nonShadow_to_Shadow_FPR']})
        summary=metrics(cm)
        for label,key in [('group_mean_mIoU','mIoU'),('group_mean_Shadow_recall','Shadow_recall'),('group_mean_FPR','nonShadow_to_Shadow_FPR')]:
            vv=[x[key] for x in gm if x[key] is not None];summary[label]=float(np.mean(vv)) if vv else None
        ff=[x['nonShadow_to_Shadow_FPR'] for x in gm if x['nonShadow_to_Shadow_FPR'] is not None];summary['max_group_FPR']=max(ff) if ff else None
        summary['groups']=len(gs);summary['no_shadow_groups']=sum(x['shadow_reference_pixels']==0 for x in gm);summary['zero_shadow_recall_groups']=sum(x['Shadow_recall']==0 for x in gm if x['Shadow_recall'] is not None)
        cases[(name,pool)]={'raw':r,'groups':gs,'summary':summary}
    csvout('PER_GROUP.csv',pergroups);csvout('ERROR_FLOWS.csv',errors)
    pooled=[dict(model=n,pool=p,**v['summary'],confusion=json.dumps(v['raw']['confusion'])) for (n,p),v in cases.items()]
    csvout('ALL_POOL_METRICS.csv',pooled)
    rng=np.random.default_rng(212020);boots={};resampling={}
    for pool in ['source','l8','sparcs']:
        names=[n for n,p in cases if p==pool and 'diagnostic' not in n];groups=sorted(cases[('Source',pool)]['groups']);ng=len(groups)
        assert ng=={'source':107,'l8':8,'sparcs':10}[pool]
        indices=rng.integers(0,ng,size=(10000,ng));hist=np.zeros((10000,ng),np.int64)
        for i,ix in enumerate(indices):hist[i]=np.bincount(ix,minlength=ng)
        resampling[pool]={'groups':groups,'draws':10000,'seed':212020,'stream':'PCG64 one stream, pools source/l8/sparcs in this order','unit':'official ROI' if pool=='source' else 'original scene'}
        for name in names:
            gs=cases[(name,pool)]['groups'];assert sorted(gs)==groups
            assert np.array_equal(np.stack([gs[g].sum(1) for g in groups]),np.stack([cases[('Source',pool)]['groups'][g].sum(1) for g in groups]))
            cm=np.stack([gs[g] for g in groups]);sampled=np.einsum('bg,gij->bij',hist,cm);bm=miou_many(sampled);boots[(name,pool)]=bm
            independent_sampled=(hist @ cm.reshape(ng,9)).reshape(10000,3,3)
            assert np.array_equal(independent_sampled,sampled)
            independent_sum=np.zeros(10000,np.float64);independent_n=np.zeros(10000,np.int64)
            for k in range(3):
                numerator=independent_sampled[:,k,k].astype(np.float64)
                denominator=independent_sampled[:,k,:].sum(1)+independent_sampled[:,:,k].sum(1)-numerator
                ok=denominator>0;independent_sum[ok]+=numerator[ok]/denominator[ok];independent_n+=ok
            assert np.max(np.abs(independent_sum/independent_n*100-bm))<1e-10
            # Independent direct selected-draw reconstruction, not a second RNG or unpaired interval.
            for j in [0,4999,9999]:assert abs(metrics(cm[indices[j]].sum(0))['mIoU']-bm[j])<1e-10
    cis=[]
    for t in ['l8','sparcs']:
        for alpha in [.25,.5,.75]:
            name=t+'_a'+str(alpha).replace('.','p')
            for pool in ['source',t]:
                for ref in ['Source',t+'_a1']:
                    delta=boots[(name,pool)]-boots[(ref,pool)];ci=np.quantile(delta,[.025,.975])
                    ordered=sorted(float(v) for v in delta)
                    for q,v in zip([.025,.975],ci):
                        rank=(len(ordered)-1)*q;lo=math.floor(rank);hi=math.ceil(rank)
                        manual=ordered[lo]+(rank-lo)*(ordered[hi]-ordered[lo])
                        assert abs(manual-v)<1e-12
                    cis.append({'target':t,'alpha':alpha,'pool':pool,'reference':ref,'pooled_mIoU_delta_pp':cases[(name,pool)]['summary']['mIoU']-cases[(ref,pool)]['summary']['mIoU'],'CI95_low':float(ci[0]),'CI95_high':float(ci[1]),'groups':len(resampling[pool]['groups']),'bootstrap_draws':10000,'seed':212020})
    csvout('PAIRED_DIFFERENCES_CI.csv',cis);save('BOOTSTRAP_RULE.json',resampling)
    history=load(RAW/'HISTORICAL_REFERENCES.json');table=[];hist_comparison=[]
    for t in ['l8','sparcs']:
        baseline=cases[('Source',t)]['summary']['mIoU'];source=cases[('Source','source')]['summary']['mIoU']
        for alpha in [0,.25,.5,.75,1]:
            name='Source' if alpha==0 else t+'_a1' if alpha==1 else t+'_a'+str(alpha).replace('.','p')
            target=cases[(name,t)]['summary'];src=cases[(name,'source')]['summary']
            table.append({'target':t,'alpha_or_method':str(alpha),'target_pooled_mIoU':target['mIoU'],'source_pooled_mIoU':src['mIoU'],'target_gain_vs_Source_pp':target['mIoU']-baseline,'source_loss_vs_Source_pp':source-src['mIoU'],'target_Shadow_IoU':target['Shadow_IoU'],'source_Shadow_IoU':src['Shadow_IoU'],'coordinate_precision':'new FP32 exact confusion','finite_set_status':'pending'})
        for method in ['MsRE','LoRA']:
            h=history.get(method,{}).get(t,{})
            tm=h.get('target_metrics_exact');sm=h.get('source_metrics_exact')
            tv=tm['mIoU'] if tm else h.get('target_mIoU_rounded_only');sv=sm['mIoU'] if sm else None
            # Recover protocol rounded reference only as rounded, never as exact legacy point.
            if tv is None:tv={'l8':61.0182,'sparcs':67.1705}[t]
            if sv is None:sv={'MsRE':{'l8':73.3432,'sparcs':73.7240},'LoRA':{'l8':66.6042,'sparcs':68.1511}}[method][t]
            table.append({'target':t,'alpha_or_method':method+'_seed64_history','target_pooled_mIoU':tv,'source_pooled_mIoU':sv,'target_gain_vs_Source_pp':None,'source_loss_vs_Source_pp':None,'target_Shadow_IoU':tm['per_class_iou']['shadow'] if tm else None,'source_Shadow_IoU':sm['per_class_iou']['shadow'] if sm else None,'coordinate_precision':'historical FP16 exact confusion metrics' if tm and sm else 'historical rounded target/source reference; exact dominance pending','finite_set_status':'pending'})
        tr=[x for x in table if x['target']==t];exact=[x for x in tr if 'rounded' not in x['coordinate_precision']]
        for x in tr:
            if x not in exact:x['finite_set_status']='pending exact historical coordinate';continue
            dominating=[y['alpha_or_method'] for y in exact if y is not x and y['target_pooled_mIoU']>=x['target_pooled_mIoU'] and y['source_pooled_mIoU']>=x['source_pooled_mIoU'] and (y['target_pooled_mIoU']>x['target_pooled_mIoU'] or y['source_pooled_mIoU']>x['source_pooled_mIoU'])]
            x['finite_set_status']='dominated by '+','.join(dominating) if dominating else 'non-dominated in exact-coordinate subset; full set pending rounded history'
        for x in tr:
            if x['alpha_or_method'] not in ['0.25','0.5','0.75']:continue
            for method in ['MsRE','LoRA']:
                y=next(v for v in tr if v['alpha_or_method']==method+'_seed64_history')
                if 'rounded' in y['coordinate_precision']:relation='pending exact historical values'
                elif x['target_pooled_mIoU']>=y['target_pooled_mIoU'] and x['source_pooled_mIoU']>=y['source_pooled_mIoU'] and (x['target_pooled_mIoU']>y['target_pooled_mIoU'] or x['source_pooled_mIoU']>y['source_pooled_mIoU']):relation='interpolation point-estimate dominates historical coordinate; cross-precision exploratory'
                elif y['target_pooled_mIoU']>=x['target_pooled_mIoU'] and y['source_pooled_mIoU']>=x['source_pooled_mIoU'] and (y['target_pooled_mIoU']>x['target_pooled_mIoU'] or y['source_pooled_mIoU']>x['source_pooled_mIoU']):relation='historical coordinate point-estimate dominates interpolation; cross-precision exploratory'
                else:relation='trade-off; no point-estimate dominance'
                hist_comparison.append({'target':t,'alpha':x['alpha_or_method'],'historical_method':method,'relation':relation})
    csvout('MAIN_TABLE.csv',table);csvout('HISTORICAL_POINT_COMPARISONS.csv',hist_comparison)
    save('ANALYSIS_ACCEPTANCE.json',{'status':'passed','independent_scalar_IoU_checks':checks,'per_input_to_group_to_pool_all_exact':True,'all_model_group_reference_support_exact':True,'paired_direct_draw_reconstructions_verified':True,'all10000_draws_independent_matmul_and_classwise_IoU_verified':True,'CI_quantiles_independent_manual_rank_interpolation_verified':True,'draws':10000,'seed':212020,'source_unit':'107 official ROI, not535 patches or independent global scenes','finite_frontier_scope':'finite evaluated exact-coordinate points only; rounded historical target remains pending','zero_new_model_calls':True})
    print(json.dumps({'status':'analyzed','table':table,'history_comparisons':hist_comparison},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
