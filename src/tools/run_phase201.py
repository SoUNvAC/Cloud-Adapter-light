"""Historical selection replay: immutable small tables only, no models or thresholds."""
import csv,datetime,hashlib,json,math,time,tracemalloc
from pathlib import Path
import numpy as np
from prepare_phase201 import R,O,sha,write
def csvout(name,rows,fields=None):
    if fields is None:fields=list(dict.fromkeys(k for row in rows for k in row))
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def avg(x):return sum(x)/len(x) if x else None
def cmmetrics(cm):
    a=np.array(cm,dtype=np.int64);t=a.sum(axis=1);p=a.sum(axis=0);d=np.diag(a);u=t+p-d;valid=u>0
    return dict(miou=float(np.mean(d[valid]/u[valid])),errors=int(a.sum()-d.sum()),error_rate=float((a.sum()-d.sum())/a.sum()),valid=int(a.sum()),
        surface_to_cloud=int(a[0,1]),cloud_to_surface=int(a[1,0]),shadow_to_surface=int(a[2,0]),shadow_to_cloud=int(a[2,1]),nonshadow_to_shadow=int(a[0,2]+a[1,2]))
def summarize(v):
    rr=v['rows'];pos=sum(x['positive'] for x in rr);neg=sum(x['negative'] for x in rr);tp=sum(x['TP'] for x in rr);fp=sum(x['FP'] for x in rr)
    s=dict(groups=len(rr),AP_groups=sum(x['positive']>0 and x['negative']>0 for x in rr),positive_groups=sum(x['positive']>0 for x in rr),negative_groups=sum(x['negative']>0 for x in rr),positive=pos,negative=neg,TP=tp,FP=fp,
        macro_AP=avg([x['AP'] for x in rr if x['positive'] and x['negative']]),pooled_AP=v['pooled_AP'],macro_recall=avg([x['TP']/x['positive'] for x in rr if x['positive']]),macro_FPR=avg([x['FP']/x['negative'] for x in rr if x['negative']]),max_FPR=max(x['FP']/x['negative'] for x in rr if x['negative']),
        pooled_recall=tp/pos if pos else None,pooled_FPR=fp/neg if neg else None,pooled_precision=tp/(tp+fp) if tp+fp else None,
        macro_miou=avg([x['miou'] for x in rr if x['miou'] is not None]),pooled_miou=None,errors=None,mean_error_rate=None)
    if v['pooled_cm'] is not None:
        cm=np.sum([x['cm'] for x in rr],axis=0);assert np.array_equal(cm,np.array(v['pooled_cm']))
        m=cmmetrics(cm);s.update(pooled_miou=m['miou'],errors=m['errors'],mean_error_rate=avg([cmmetrics(x['cm'])['error_rate'] for x in rr]))
    return s
def dominance(a,b,tol=1e-10):
    delta=[b['macro_recall']-a['macro_recall'],a['macro_FPR']-b['macro_FPR'],a['max_FPR']-b['max_FPR']]
    return all(d>=-tol for d in delta) and any(d>tol for d in delta),delta
def budget(summaries,order,chosen,tail=False):
    ok=[m for m in order if summaries[m]['macro_FPR']<=.01 and (not tail or summaries[m]['max_FPR']<=.1)]
    if not ok:return dict(state='no_feasible_candidate',oracle=None,loss_pp=None,excess_mean_pp=max(0,summaries[chosen]['macro_FPR']-.01)*100)
    oracle=max(ok,key=lambda m:summaries[m]['macro_recall'])
    if chosen not in ok:return dict(state='budget_violation',oracle=oracle,loss_pp=None,excess_mean_pp=max(0,summaries[chosen]['macro_FPR']-.01)*100,excess_tail_pp=max(0,summaries[chosen]['max_FPR']-.1)*100 if tail else None)
    return dict(state='feasible',oracle=oracle,loss_pp=(summaries[oracle]['macro_recall']-summaries[chosen]['macro_recall'])*100,excess_mean_pp=0)
def main():
    started=time.monotonic();tracemalloc.start();lock=json.loads((O/('INPUT_CODE_LOCK_REPAIR.json' if (O/'INPUT_CODE_LOCK_REPAIR.json').exists() else 'INPUT_CODE_LOCK.json')).read_text(encoding='utf8'))
    for v in lock['source_files']:assert sha(R/v['path'])==v['sha256'],v['path']
    for p,h in lock['code_sha256'].items():assert sha(R/p)==h,p
    assert sha(O/'NORMALIZED_INPUT.json')==lock['normalized_sha256'];a=json.loads((O/'NORMALIZED_INPUT.json').read_text(encoding='utf8'))
    candidates=[];selections=[];checks=[];domrows=[];cis=[];allstats={};rng=np.random.default_rng(201017)
    for block,b in a['blocks'].items():
        allstats[block]={};pending=[]
        for side in ['selection','evaluation']:
            allstats[block][side]={}
            for model in b['order']:
                p=b[side][model];s=summarize(p);allstats[block][side][model]=s
                for key,want in p['historical_expected'].items():
                    error=abs(s[key]-want);checks.append(dict(block=block,side=side,model=model,metric=key,computed=s[key],historical=want,error=error,status='passed' if error<=1e-6 else 'pending'))
                    if error>1e-6:pending.append(model+':'+key)
                if p.get('published_pooled_counts'):
                    for key,want in p['published_pooled_counts'].items():assert s[key]==want
                candidates.append(dict(block=block,side=side,level='aggregate',model=model,group_id='ALL',pool=b['pool'] if side=='evaluation' else 'original32_development',**s))
                for x in p['rows']:
                    row=dict(block=block,side=side,level='group',model=model,pool=b['pool'] if side=='evaluation' else 'original32_development',**{k:v for k,v in x.items() if k!='cm'})
                    if x['cm'] is not None:
                        row.update(cmmetrics(x['cm']));row.update({f'cm_{i}{j}':x['cm'][i][j] for i in range(3) for j in range(3)})
                    candidates.append(row)
        es=allstats[block]['evaluation'];ss=allstats[block]['selection']
        for chosen in b['order']:
            for other in b['order']:
                if chosen==other:continue
                yes,ds=dominance(es[chosen],es[other]);domrows.append(dict(block=block,selected=chosen,alternative=other,dominates=yes,recall_alt_minus_selected_pp=ds[0]*100,FPR_alt_minus_selected_pp=-ds[1]*100,maxFPR_alt_minus_selected_pp=-ds[2]*100,selected_pooled_miou=es[chosen]['pooled_miou'],alternative_pooled_miou=es[other]['pooled_miou']))
        for rule in ['pooled_miou','pooled_AP','macro_AP']:
            if (block=='D' and rule=='pooled_miou') or any(ss[m][rule] is None for m in b['order']):
                selections.append(dict(block=block,rule=rule,status='not_executable' if block!='D' else 'not_applicable',reason='no existing common three-class output; no reconstruction' if block!='D' else 'D is AP-only'));continue
            maximum=max(ss[m][rule] for m in b['order']);ties=[m for m in b['order'] if ss[m][rule]==maximum];chosen=ties[0]
            dom=[d['alternative'] for d in domrows if d['block']==block and d['selected']==chosen and d['dominates']]
            q=dict(block=block,rule=rule,status='pending' if pending else 'executed',selected=chosen,selection_score=maximum,ties=';'.join(ties),**{'eval_'+k:v for k,v in es[chosen].items()},dominators=';'.join(dom) or 'none')
            if block!='D':
                for tail,prefix in [(False,'mean'),(True,'tail')]:
                    result=budget(es,b['order'],chosen,tail)
                    q.update({prefix+'_'+k:v for k,v in result.items()})
                    oracle=result['oracle']
                    if oracle:
                        lhs={x['group_id']:x for x in b['evaluation'][chosen]['rows']};rhs={x['group_id']:x for x in b['evaluation'][oracle]['rows']}
                        for metric,den,num in [('AP',None,None),('recall','positive','TP'),('FPR','negative','FP')]:
                            gids=[g for g in sorted(lhs) if (lhs[g]['positive'] and lhs[g]['negative'])] if metric=='AP' else [g for g in sorted(lhs) if lhs[g][den]>0]
                            diffs=np.array([rhs[g]['AP']-lhs[g]['AP'] if metric=='AP' else rhs[g][num]/rhs[g][den]-lhs[g][num]/lhs[g][den] for g in gids],dtype=float)
                            samples=diffs[rng.integers(0,len(diffs),size=(10000,len(diffs)))].mean(axis=1)*100
                            cis.append(dict(block=block,rule=rule,budget=prefix,selected=chosen,reference=oracle,metric=metric,groups=len(diffs),delta_reference_minus_selected_pp=float(diffs.mean()*100),CI95_low_pp=float(np.quantile(samples,.025)),CI95_high_pp=float(np.quantile(samples,.975)),seed=201017,resamples=10000,scope='conditional on fixed selected/reference objects; descriptive; not selection uncertainty or joint guarantee'))
            else:q.update(mean_state='not_applicable',tail_state='not_applicable')
            selections.append(q)
    metrows=[];metall=[]
    for phase,order in [('81',['LL','LM','MM','ML']),('84',['LL','R4','F_real','F_shuffle'])]:
        values={}
        for name in order:
            rows=[r for r in a['met'] if r['phase']==phase and r['policy']==name];assert len(rows)==32
            pooled=a['met_pooled'][phase+'_'+name]
            v=dict(old_macro_miou=avg([r['old_mIoU'] for r in rows]),reference_fixed_macro_miou=avg([r['reference_classes_mIoU'] for r in rows]),pooled_miou=pooled['old_mIoU'],errors=sum(r['errors'] for r in rows),mean_error_rate=avg([r['error_rate'] for r in rows]))
            for flow in ['surface_to_cloud','cloud_to_surface','shadow_to_surface','shadow_to_cloud','nonshadow_to_shadow']:v[flow]=sum(r[flow] for r in rows);assert v[flow]==pooled[flow]
            assert v['errors']==pooled['errors'];values[name]=v;metall.append(dict(phase=phase,model=name,**v))
            for x in rows:metall.append(dict(x,model=name,level='group'))
        for rule,key in [('old_mIoU','old_macro_miou'),('reference_classes_mIoU','reference_fixed_macro_miou')]:
            chosen=max(order,key=lambda m:values[m][key]);assert chosen==a['met_rankings'][phase][rule][0]['policy']
            metrows.append(dict(phase=phase,rule=rule,selected=chosen,**values[chosen]))
    csvout('candidate_metrics.csv',candidates);csvout('selection_replay.csv',selections);csvout('dominance_pairs.csv',domrows);csvout('conditional_intervals.csv',cis,['block','rule','budget','selected','reference','metric','groups','delta_reference_minus_selected_pp','CI95_low_pp','CI95_high_pp','seed','resamples','scope'])
    csvout('metric_reproduction.csv',checks);csvout('missing_inputs.csv',a['missing'],['block','side','item','reason']);csvout('definition_sensitivity.csv',metrows);csvout('met_all_candidates_and_groups.csv',metall)
    write(O/'ANALYSIS.json',dict(statistics=allstats,selection=selections,definition=metrows,resource=dict(wall_seconds=time.monotonic()-started,tracemalloc_peak_bytes=tracemalloc.get_traced_memory()[1]),new_training=0,fitting=0,model_calls=0,downloads=0,threshold_search=0,selection_stability='not computed; optional',bootstrap='10000 seed201017 conditional paired groups only'))
    print(json.dumps({'rules_executed':sum(x['status']=='executed' for x in selections),'pending_reproduction':sum(c['status']=='pending' for c in checks),'conditional_intervals':len(cis),'seconds':time.monotonic()-started}))
if __name__=='__main__':main()
