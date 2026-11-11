"""217 CPU analysis: paired seeds, shared group draws, mean of per-seed metric differences."""
from pathlib import Path
import csv,json,hashlib,time
import numpy as np
from analyze_phase216 import metrics,miou,peak
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase217/phase217_20261023';RAW=OUT/'delivery';EV=RAW/'evaluation';SEEDS=[21701,21702];AS=[0,.25,.5,.75,1]
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,rows):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  names=list(rows[0]) if rows else ['context','arm','ROI','seed21701_alpha','seed21702_alpha','seed21701_loss_pp','seed21702_loss_pp','both_loss_gt5']
  w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def groups(r):return {g:np.asarray(c,np.int64) for g,c in r['per_group'].items()}
def main():
 clock=time.monotonic();assert js(RAW/'COMPLETE.json')['units']==48893
 lock=js(EV/'SOURCE_SELECTION_LOCK.json');assert lock['target_metrics_opened_before_lock'] is False
 cases={};pooled=[];per=[];flows=[];candidates=[];tail=[];summary={};stat={}
 baseline={p:groups(js(EV/f'Source__{p}.json')) for p in ['source','val','test']}
 assert [len(baseline[p]) for p in ['source','val','test']]==[107,10,10]
 for s in SEEDS:
  for a in ['C','D']:
   for f in AS:
    st=lock['source_metrics'][f'{s}/{a}/{f}'];stat[s,a,f]=st
    for pool in ['source','val','test']:
     raw=js(EV/(f'Source__{pool}.json' if f==0 else f'{s}_{a}_a{f}__{pool}.json'));g=groups(raw);cases[s,a,f,pool]=g
     assert set(g)==set(baseline[pool]) and all(np.array_equal(c.sum(1),baseline[pool][k].sum(1)) for k,c in g.items())
     total=sum(g.values());assert total.tolist()==raw['confusion'];m=metrics(total);gm=[metrics(c) for c in g.values()]
     r={'seed':s,'arm':a,'alpha':f,'pool':pool,'groups':len(g),'group_mean_mIoU':float(np.mean([q['mIoU'] for q in gm])),'max_group_FPR':max(q['nonShadow_to_Shadow_FPR'] for q in gm if q['nonShadow_to_Shadow_FPR'] is not None),'zero_shadow_recall_groups':sum(q['shadow_pixels']>0 and q['Shadow_recall']==0 for q in gm),'no_shadow_groups':sum(q['shadow_pixels']==0 for q in gm),**m}
     for k in ['Surface_IoU','Cloud_IoU','Shadow_IoU','Shadow_recall','nonShadow_to_Shadow_FPR']:
      v=[q[k] for q in gm if q[k] is not None];r['group_mean_'+k]=float(np.mean(v)) if v else None
     pooled.append(r);summary[s,a,f,pool]=r
     for k,c in sorted(g.items()):
      met=metrics(c);b=metrics(baseline[pool][k]);per.append({'seed':s,'arm':a,'alpha':f,'pool':pool,'group':k,'confusion':json.dumps(c.tolist()),'delta_mIoU_pp':met['mIoU']-b['mIoU'],**met})
     for k,c in list(sorted(g.items()))+[('__pooled__',total)]:
      met=metrics(c);b=metrics(sum(baseline[pool].values()) if k=='__pooled__' else baseline[pool][k])
      for src in ['Surface','Cloud','Shadow']:
       for dst in ['Surface','Cloud','Shadow']:
        if src==dst:continue
        name=src+'_to_'+dst;flows.append({'seed':s,'arm':a,'alpha':f,'pool':pool,'group':k,'flow':name,'pixels':met[name+'_pixels'],'net_pixels':met[name+'_pixels']-b[name+'_pixels'],'rate':met[name+'_rate'],'delta_rate':None if met[name+'_rate'] is None else met[name+'_rate']-b[name+'_rate']})
    t=summary[s,a,f,'test'];v=summary[s,a,f,'val'];candidates.append({'seed':s,'arm':a,'alpha':f,**st,'val_mIoU':v['mIoU'],'test_mIoU':t['mIoU'],'test_Shadow_IoU':t['Shadow_IoU'],'test_Shadow_recall':t['Shadow_recall'],'test_FPR':t['nonShadow_to_Shadow_FPR'],'test_group_mean_mIoU':t['group_mean_mIoU']})
    g=cases[s,a,f,'source'];ordered=sorted(g,key=lambda k:(float(miou(g[k])-miou(baseline['source'][k])),k))
    for rank,k in enumerate(ordered,1):tail.append({'seed':s,'arm':a,'alpha':f,'rank':rank,'ROI':k,'loss_pp':float(miou(baseline['source'][k])-miou(g[k])),'in_T11':rank<=11})
 write('ALL_CANDIDATES.csv',candidates);write('ALL_POOL_METRICS.csv',pooled);write('PER_GROUP.csv',per);write('ERROR_FLOWS.csv',flows);write('SOURCE_TAIL_GROUPS.csv',tail)
 selected={(r['seed'],r['arm'],r['budget_pp']):r['alpha'] for r in lock['rows']}
 def comparison(s,ca,da,context):
  c=stat[s,'C',ca];d=stat[s,'D',da];ct=summary[s,'C',ca,'test']['mIoU'];dt=summary[s,'D',da,'test']['mIoU']
  return {'seed':s,'context':context,'C_alpha':ca,'D_alpha':da,'C_test_mIoU':ct,'D_test_mIoU':dt,'D_minus_C_test_mIoU_pp':dt-ct,'C_pool_loss_pp':c['pool_loss_pp'],'D_pool_loss_pp':d['pool_loss_pp'],'D_minus_C_pool_loss_pp':d['pool_loss_pp']-c['pool_loss_pp'],'C_T11_pp':c['T11_pp'],'D_T11_pp':d['T11_pp'],'D_minus_C_T11_pp':d['T11_pp']-c['T11_pp'],'C_worst_ROI_loss_pp':c['worst_ROI_loss_pp'],'D_worst_ROI_loss_pp':d['worst_ROI_loss_pp']}
 endpoints=[comparison(s,1,1,'endpoint_alpha1') for s in SEEDS];joint=[comparison(s,selected[s,'C',b],selected[s,'D',b],f'B{b}') for s in SEEDS for b in [1,3,5]]
 def meanrow(rs):
  r={'seed':'mean_of_21701_21702','context':rs[0]['context'],'C_alpha':';'.join(str(x['C_alpha']) for x in rs),'D_alpha':';'.join(str(x['D_alpha']) for x in rs)}
  for k in rs[0]:
   if k not in r:r[k]=float(np.mean([x[k] for x in rs]))
  return r
 endpoints.append(meanrow(endpoints));joint.extend(meanrow([r for r in joint if r['context']==f'B{b}']) for b in [1,3,5])
 write('ENDPOINTS.csv',endpoints);write('JOINT_RULE.csv',joint)
 # Shared group indices, but nonlinear metrics computed separately within each seed.
 it=np.random.default_rng(217023).integers(0,10,(10000,10));ir=np.random.default_rng(217023).integers(0,107,(10000,107));sourceids=sorted(baseline['source']);testids=sorted(baseline['test']);base_m=np.array([miou(baseline['source'][g]) for g in sourceids]);intervals=[];proof={}
 for context in ['endpoint_alpha1','B1','B3','B5']:
  draws={};points={}
  for s in SEEDS:
   ca,da=(1,1) if context=='endpoint_alpha1' else (selected[s,'C',int(context[1:])],selected[s,'D',int(context[1:])])
   c=np.stack([cases[s,'C',ca,'test'][g] for g in testids]);d=np.stack([cases[s,'D',da,'test'][g] for g in testids]);td=miou(d[it].sum(1))-miou(c[it].sum(1))
   c=np.stack([cases[s,'C',ca,'source'][g] for g in sourceids]);d=np.stack([cases[s,'D',da,'source'][g] for g in sourceids]);sd=miou(c[ir].sum(1))-miou(d[ir].sum(1));cl=base_m-miou(c);dl=base_m-miou(d);taild=np.sort(dl[ir],axis=1)[:,-11:].mean(1)-np.sort(cl[ir],axis=1)[:,-11:].mean(1)
   draws[str(s)]={'test_mIoU_diff_pp':td,'source_pool_loss_diff_pp':sd,'T11_diff_pp':taild};row=comparison(s,ca,da,context);points[str(s)]={'test_mIoU_diff_pp':row['D_minus_C_test_mIoU_pp'],'source_pool_loss_diff_pp':row['D_minus_C_pool_loss_pp'],'T11_diff_pp':row['D_minus_C_T11_pp']}
  draws['mean_of_two']={k:(draws[str(SEEDS[0])][k]+draws[str(SEEDS[1])][k])/2 for k in draws[str(SEEDS[0])]};points['mean_of_two']={k:(points[str(SEEDS[0])][k]+points[str(SEEDS[1])][k])/2 for k in points[str(SEEDS[0])]}
  for scope,ds in draws.items():
   for k,v in ds.items():
    lo,hi=np.quantile(v,[.025,.975]);intervals.append({'context':context,'scope':scope,'metric':k,'point':points[scope][k],'CI95_low':float(lo),'CI95_high':float(hi),'groups':10 if k.startswith('test') else 107,'draws':10000,'seed':217023,'meaning':'fixed trained seeds/Source/replay pool and frozen alphas; no training-seed-population or selection uncertainty'})
    proof[context+'/'+scope+'/'+k]={'first50':v[:50].tolist(),'draws_SHA':hashlib.sha256(v.tobytes()).hexdigest()}
 write('PAIRED_CI.csv',intervals);save('BOOTSTRAP_PROOF.json',proof)
 domination=[]
 for s in SEEDS:
  for c in AS:
   for d in AS:
    r=comparison(s,c,d,'finite_2D');r['D_dominates_C']=r['D_minus_C_test_mIoU_pp']>=0 and r['D_minus_C_T11_pp']<=0 and (r['D_minus_C_test_mIoU_pp']>0 or r['D_minus_C_T11_pp']<0);domination.append(r)
 write('FINITE_SET_DOMINANCE.csv',domination)
 damage=[];worst=[]
 for s in SEEDS:
  for a in ['C','D']:
   for f in AS:
    g=cases[s,a,f,'source'];k=max(sorted(g),key=lambda k:float(miou(baseline['source'][k])-miou(g[k])));worst.append({'seed':s,'arm':a,'alpha':f,'worst_ROI':k,'loss_pp':float(miou(baseline['source'][k])-miou(g[k]))})
 for context in [f'alpha{f}' for f in AS]+['B1','B3','B5']:
  for a in ['C','D']:
   aa=[float(context[5:]) if context.startswith('alpha') else selected[s,a,int(context[1:])] for s in SEEDS]
   for g in sourceids:
    v=[float(miou(baseline['source'][g])-miou(cases[s,a,f,'source'][g])) for s,f in zip(SEEDS,aa)]
    damage.append({'context':context,'arm':a,'ROI':g,'seed21701_alpha':aa[0],'seed21702_alpha':aa[1],'seed21701_loss_pp':v[0],'seed21702_loss_pp':v[1],'both_loss_gt5':all(x>5 for x in v)})
 write('ALL_ROI_DAMAGE_MEMBERSHIP.csv',damage);write('COMMON_DAMAGE_ROIS.csv',[r for r in damage if r['both_loss_gt5']]);write('WORST_ROIS.csv',worst)
 old=ROOT/'outputs/phase216/phase216_20261023';history=list(csv.DictReader((old/'ALL_CANDIDATES.csv').open(encoding='utf-8-sig')));write('HISTORICAL216_ENDPOINTS.csv',[r for r in history if float(r['alpha'])==1])
 oldtail=list(csv.DictReader((old/'SOURCE_TAIL_GROUPS.csv').open(encoding='utf-8-sig')));oldworst=[r for r in oldtail if float(r['alpha'])==1 and int(r['rank_by_source_loss'])==1];track=[]
 for r in oldworst:
  g=r['ROI']
  for s in SEEDS:
   for a in ['C','D']:
    for f in AS:track.append({'historical_seed':21601,'historical_arm':r['arm'],'historical_worst_ROI':g,'historical_alpha1_loss_pp':float(r['loss_pp']),'new_seed':s,'new_arm':a,'new_alpha':f,'new_loss_pp':float(miou(baseline['source'][g])-miou(cases[s,a,f,'source'][g]))})
 write('HISTORICAL216_WORST_ROI_TRACK.csv',track)
 memory=peak();assert memory<16*1024**3;save('ANALYSIS_RECEIPT.json',{'seconds':time.monotonic()-clock,'peak_working_set_bytes':memory,'new_model_calls':0,'new_seed_groups_not_inflated':True,'bootstrap':'same scene/ROI indices across seeds; nonlinear within-seed metrics then average differences; never average confusion first','source_selection_SHA':sha(EV/'SOURCE_SELECTION_LOCK.json'),'code_SHA':sha(Path(__file__))});print(json.dumps(endpoints,indent=2))
if __name__=='__main__':main()
