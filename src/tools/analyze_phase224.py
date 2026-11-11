"""Paired H-D analysis of both targets, all finite points and source direct counts."""
from pathlib import Path
import json,csv,collections,hashlib,time,functools
import numpy as np
from analyze_phase216 import metrics,miou,peak
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase224/phase224_20261024';D=O/'delivery';E=D/'evaluation';AA=[0,.25,.5,.75,1];SS=[21701,21702];CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def write(n,rows):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rows for k in v)) or ['no_rows']);w.writeheader();w.writerows(rows)
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def denominators(c):
 c=np.asarray(c);u=c.sum(0)+c.sum(1)-np.diag(c);v={'effective_IoU_classes':int((u>0).sum()),'FPR_nonShadow_reference_pixels':int(c[:2].sum())}
 for i,k in enumerate(CS):v.update({k+'_reference_pixels':int(c[i].sum()),k+'_prediction_pixels':int(c[:,i].sum()),k+'_IoU_union_pixels':int(u[i])})
 return v
@functools.lru_cache(None)
def raw(target,seed,arm,alpha,pool):
 if alpha==0:return js(D/f'historical/{target}/evaluation/Source__{pool}.json')
 return js(E/f'{target}_{seed}_H_a{alpha}__{pool}.json') if arm=='H' else js(D/f'historical/{target}/evaluation/{seed}_{arm}_a{alpha}__{pool}.json')
def main():
 clock=time.monotonic();assert js(D/'COMPLETE.json')['units']==100460 and js(D/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed';lock=js(D/'EXPERIMENT_LOCK.json');selection=js(E/'SOURCE_SELECTION_LOCK.json');assert selection['target_metrics_opened_before_lock'] is False;strata=js(D/'L8_SUPPORT_ACCEPTED.json')['test_strata'];cases={};stats={};poolm={};allc=[];per=[];pools=[];flow=[];tails=[];choices={}
 for target in ['SPARCS','L8']:
  base={p:raw(target,21701,'D',0,p)['per_group'] for p in ['source','val','test']};oldsel=js(D/f'historical/{target}/evaluation/SOURCE_SELECTION_LOCK.json')
  for s in SS:
   for arm in ['C','D','H']:
    for a in AA:
     for pool in ['source','val','test']:
      q=raw(target,s,arm,a,pool);gg={g:np.asarray(v,np.int64) for g,v in q['per_group'].items()};assert set(gg)==set(base[pool]) and all(np.array_equal(c.sum(1),np.asarray(base[pool][g]).sum(1)) for g,c in gg.items());cases[target,s,arm,a,pool]=gg;assert sum(gg.values()).tolist()==q['confusion']
      supports=['full','yes','no'] if target=='L8' and pool=='test' else ['full']
      for support in supports:
       keys=sorted(gg) if support=='full' else strata[support];c=sum(gg[g] for g in keys);row={'target':target,'seed':s,'arm':arm,'alpha':a,'pool':pool,'support':support,'groups':len(keys),'confusion':json.dumps(c.tolist()),**metrics(c),**denominators(c)};pools.append(row);poolm[target,s,arm,a,pool,support]=row
      for g,c in sorted(gg.items()):
       b=np.asarray(base[pool][g]);m=metrics(c);per.append({'target':target,'seed':s,'arm':arm,'alpha':a,'pool':pool,'group':g,'loss_pp':float(miou(b)-miou(c)),'confusion':json.dumps(c.tolist()),**m,**denominators(c)})
       for i,cl in enumerate(CS):
        for j,ds in enumerate(CS):
         if i!=j:flow.append({'target':target,'seed':s,'arm':arm,'alpha':a,'pool':pool,'group':g,'flow':cl+'->'+ds,'pixels':int(c[i,j]),'net_vs_Source_pixels':int(c[i,j]-b[i,j])})
     source=cases[target,s,arm,a,'source'];loss={g:float(miou(np.asarray(base['source'][g]))-miou(c)) for g,c in source.items()};ordered=sorted(source,key=lambda g:(-loss[g],g));st={'pool_loss_pp':float(miou(np.asarray(raw(target,s,arm,0,'source')['confusion']))-miou(sum(source.values()))),'T11_pp':float(np.mean([loss[g] for g in ordered[:11]])),'worst_ROI':ordered[0],'worst_ROI_loss_pp':loss[ordered[0]],'ROI_loss_gt5':sum(v>5 for v in loss.values())};stats[target,s,arm,a]=st
     for rank,g in enumerate(ordered,1):tails.append({'target':target,'seed':s,'arm':arm,'alpha':a,'ROI':g,'rank':rank,'loss_pp':loss[g],'in_T11':rank<=11})
     row={'target':target,'seed':s,'arm':arm,'alpha':a,**st,'test_mIoU':poolm[target,s,arm,a,'test','full']['mIoU'],'val_mIoU':poolm[target,s,arm,a,'val','full']['mIoU']}
     if target=='L8':row.update({f'test_{k}_mIoU':poolm[target,s,arm,a,'test',k]['mIoU'] for k in ['yes','no']})
     allc.append(row)
    for b in [1,3,5]:
     chosen=max(a for a in AA if stats[target,s,arm,a]['pool_loss_pp']<=b and stats[target,s,arm,a]['T11_pp']<=5);choices[target,s,arm,b]=chosen;old=selection if arm=='H' else oldsel;assert chosen==next(v['alpha'] for v in old['rows'] if v['seed']==s and v['arm']==arm and v['budget_pp']==b and (arm!='H' or v['target']==target))
 write('TABLE_C_ALL_CANDIDATES.csv',allc);write('ALL_POOL_METRICS.csv',pools);write('PER_GROUP.csv',per);write('ERROR_FLOW_NET_CHANGES.csv',flow);write('SOURCE_TAIL_GROUPS.csv',tails);write('FOCUS_ROI0124_ALPHA025_AND1.csv',[v for v in per if v['pool']=='source' and v['group']=='ROI_0124' and v['alpha'] in [.25,1]])
 endpoints=[v for v in allc if v['alpha']==1];means=[]
 for t in ['SPARCS','L8']:
  for a in ['C','D','H']:
   rr=[v for v in endpoints if v['target']==t and v['arm']==a];z={'target':t,'seed':'mean_of_two','arm':a,'alpha':1,'worst_ROI':'see individual seeds'}
   for k in rr[0]:
    if k not in z:z[k]=float(np.mean([v[k] for v in rr]))
   means.append(z)
 write('TABLE_A_ENDPOINTS.csv',endpoints+means)
 contexts=['endpoint_alpha1','B1','B3','B5'];comparison=[];joint=[];ci=[];proof={};first={}
 def aa(t,s,context):return (1,1) if context=='endpoint_alpha1' else (choices[t,s,'H',int(context[1:])],choices[t,s,'D',int(context[1:])])
 def point(t,s,ha,da,context):
  h=stats[t,s,'H',ha];d=stats[t,s,'D',da];r={'target':t,'seed':s,'context':context,'H_alpha':ha,'D_alpha':da,'test_mIoU_diff_pp':poolm[t,s,'H',ha,'test','full']['mIoU']-poolm[t,s,'D',da,'test','full']['mIoU'],'source_pool_loss_diff_pp':h['pool_loss_pp']-d['pool_loss_pp'],'T11_diff_pp':h['T11_pp']-d['T11_pp']}
  if t=='L8':r.update({f'test_{k}_mIoU_diff_pp':poolm[t,s,'H',ha,'test',k]['mIoU']-poolm[t,s,'D',da,'test',k]['mIoU'] for k in ['yes','no']})
  return r
 for t in ['SPARCS','L8']:
  ids=sorted(cases[t,21701,'D',0,'source']);bm=np.array([miou(cases[t,21701,'D',0,'source'][g]) for g in ids]);support={'source':ids,'full':sorted(cases[t,21701,'D',0,'test'])};
  if t=='L8':support.update(strata)
  rng=np.random.default_rng(217023);ix={k:rng.integers(0,len(gs),(10000,len(gs))) for k,gs in support.items()};proof[t]={'stream_order':list(support),'index_SHA':{k:hashlib.sha256(v.astype('<i8').tobytes()).hexdigest() for k,v in ix.items()},'groups':{k:len(v) for k,v in support.items()}}
  for context in contexts:
   pts={};draws={};rr=[]
   for s in SS:
    ha,da=aa(t,s,context);p=point(t,s,ha,da,context);rr.append(p);pts[str(s)]={k:v for k,v in p.items() if k.endswith('_diff_pp')};h=np.stack([cases[t,s,'H',ha,'source'][g] for g in ids]);d=np.stack([cases[t,s,'D',da,'source'][g] for g in ids]);hl=bm-miou(h);dl=bm-miou(d);ds={'source_pool_loss_diff_pp':miou(d[ix['source']].sum(1))-miou(h[ix['source']].sum(1)),'T11_diff_pp':np.sort(hl[ix['source']],axis=1)[:,-11:].mean(1)-np.sort(dl[ix['source']],axis=1)[:,-11:].mean(1)}
    for k,gs in support.items():
     if k=='source':continue
     h=np.stack([cases[t,s,'H',ha,'test'][g] for g in gs]);d=np.stack([cases[t,s,'D',da,'test'][g] for g in gs]);name='test_mIoU_diff_pp' if k=='full' else f'test_{k}_mIoU_diff_pp';ds[name]=miou(h[ix[k]].sum(1))-miou(d[ix[k]].sum(1))
    draws[str(s)]=ds
    if context!='endpoint_alpha1':
     for arm,f in [('H',ha),('D',da)]:joint.append(next({**v,'budget_pp':int(context[1:])} for v in allc if v['target']==t and v['seed']==s and v['arm']==arm and v['alpha']==f))
   mean={'target':t,'seed':'mean_of_two','context':context,'H_alpha':'/'.join(str(v['H_alpha']) for v in rr),'D_alpha':'/'.join(str(v['D_alpha']) for v in rr)};mean.update({k:float(np.mean([v[k] for v in rr])) for k in pts[str(SS[0])]});comparison.extend(rr+[mean]);draws['mean_of_two']={k:(draws[str(SS[0])][k]+draws[str(SS[1])][k])/2 for k in draws[str(SS[0])]};pts['mean_of_two']={k:mean[k] for k in pts[str(SS[0])]}
   for scope,dd in draws.items():
    for k,v in dd.items():
     lo,hi=np.quantile(v,[.025,.975]);ci.append({'target':t,'context':context,'scope':scope,'metric':k,'point':pts[scope][k],'CI95_low':float(lo),'CI95_high':float(hi),'draws':10000,'seed':217023});first[t+'/'+context+'/'+scope+'/'+k]=v[:50].tolist()
 write('TABLE_B_H_MINUS_D.csv',comparison);write('JOINT_RULE.csv',joint);write('PAIRED_CI.csv',ci);save('BOOTSTRAP_RULE.json',{'seed':217023,'draws':10000,'targets':proof,'interpretation':'paired groups within target across two fixed seeds; nonlinear metrics per-seed then mean difference; fixed selected alpha; no cross-target average'});save('BOOTSTRAP_FIRST50.json',first)
 finite=[]
 for t in ['SPARCS','L8']:
  for s in SS:
   for ha in AA:
    for da in AA:
     v=point(t,s,ha,da,'finite25');v['H_dominates_D']=v['test_mIoU_diff_pp']>=0 and v['source_pool_loss_diff_pp']<=0 and v['T11_diff_pp']<=0 and (v['test_mIoU_diff_pp']>0 or v['source_pool_loss_diff_pp']<0 or v['T11_diff_pp']<0);finite.append(v)
 write('FINITE_25_PAIRS_ALL.csv',finite)
 agg=collections.defaultdict(lambda:np.zeros((3,3,3,3),np.int64))
 for v in lines(E/'JOINT_COUNTS.jsonl'):agg[v['target'],v['seed'],v['ROI']]+=np.asarray(v['J'],np.int64)
 direct=[];correctness=[];di,hi=np.indices((3,3))
 for t in ['SPARCS','L8']:
  for s in SS:
   ids=sorted(cases[t,s,'D',0,'source'])
   for g in ids+['__pooled__']:
    j=sum(agg[t,s,x] for x in ids) if g=='__pooled__' else agg[t,s,g];lesswrong=lesscorrect=net=0
    for r in range(3):
     for teacher in range(3):
      x=j[r,teacher];net+=int(x[hi!=r].sum()-x[di!=r].sum())
      if teacher==r:lesswrong+=int(x[di!=r].sum()-x[hi!=r].sum())
      else:lesscorrect+=int(x[di==r].sum()-x[hi==r].sum())
     for tier,keep in [('all',[0,1,2]),('correct',[r]),('wrong',[k for k in range(3) if k!=r])]:
      x=j[r,keep].sum(0);direct.append({'target':t,'seed':s,'ROI':g,'reference_class':CS[r],'teacher_correctness':tier,'pixels':int(x.sum()),'D_wrong_H_right':int(x[(di!=r)&(hi==r)].sum()),'D_right_H_wrong':int(x[(di==r)&(hi!=r)].sum()),'H_minus_D_errors':int(x[hi!=r].sum()-x[di!=r].sum())})
    assert net==lesscorrect-lesswrong;correctness.append({'target':t,'seed':s,'ROI':g,'teacher_correct_H_fewer_errors':lesswrong,'teacher_wrong_H_fewer_corrections':lesscorrect,'H_minus_D_errors':net})
 write('DIRECT_D_TO_H_ALPHA1.csv',direct);write('TEACHER_CORRECTNESS_ALPHA1.csv',correctness)
 losscomp=[]
 for job,meta in lock['jobs'].items():
  date='20261023' if meta['target']=='SPARCS' else '20261024';phase=meta['historical_phase'];old=R/f'outputs/phase{phase}/phase{phase}_{date}/delivery/seed{meta["seed"]}/D_LOSSES.jsonl'
  for arm,rr,lam in [('D',lines(old),1),('H',lines(D/job/'H_LOSSES.jsonl'),4)]:
   for segment,vs in [('first100',rr[:100]),('last100',rr[-100:]),('all4000',rr)]:losscomp.append({'target':meta['target'],'seed':meta['seed'],'arm':arm,'lambda':lam,'segment':segment,'target_loss_mean':float(np.mean([v['target_loss'] for v in vs])),'unscaled_KL_mean':float(np.mean([v['source_KL'] for v in vs])),'scaled_KL_mean':float(np.mean([lam*v['source_KL'] for v in vs])),'total_loss_mean':float(np.mean([v['total_loss'] for v in vs]))})
 write('LOSS_COMPARISON.csv',losscomp);assert peak()<16*1024**3;save('ANALYSIS_RECEIPT.json',{'new_model_calls':0,'seconds':time.monotonic()-clock,'CPU_peak_bytes':peak(),'per_group_rows':len(per),'pool_rows':len(pools),'all60_candidates':len(allc),'all100_finite_pairs':len(finite),'no_cross_target_average':True,'code_SHA':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()});print(json.dumps([v for v in comparison if v['seed']=='mean_of_two' and v['context']=='endpoint_alpha1']))
if __name__=='__main__':main()
