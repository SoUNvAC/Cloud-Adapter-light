"""Source+B versus frozen234 references; one Source, paired scenes, no selection."""
from pathlib import Path
import csv,hashlib,json,time,collections
import numpy as np
from phase231_metrics import metrics
from analyze_phase229 import miou_batch
from analyze_phase230 import cpu_peak
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase235/phase235_20261027';D=R/'delivery';OLD=REPO/'outputs/phase234/phase234_20261027';START=time.monotonic();KEYS=['mIoU','Cloud_recall','Surface_to_Cloud_FPR','Shadow_recall','nonShadow_to_Shadow_FPR']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def output(n,rows):
 save(n+'.json',rows);keys=list(dict.fromkeys(k for v in rows for k in v))
 with (R/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for v in rows:w.writerow({k:json.dumps(x) if isinstance(x,(dict,list)) else x for k,x in v.items()})
def summary(gs):
 m=metrics(sum(gs.values()));v={'pooled':m,'confusion':m['confusion']}
 for k in KEYS:
  vals=[metrics(c)[k] for c in gs.values() if metrics(c)[k] is not None];v['scene_mean_'+k]=float(np.mean(vals)) if vals else None;v[k+'_defined_scenes']=len(vals)
 return v
def main():
 assert not (R/'CPU_ANALYSIS_COMPLETE.json').exists();assert (D/'EXIT_CODE').read_text().strip()=='0';C=js(D/'COMPLETE.json');assert C['total_calls']<=4800 and C['GPU_stage_seconds']<3600;F=js(D/'BIAS_FROZEN.json');assert sha(D/'BIAS_FROZEN.json')==js(D/'BIAS_SEAL.json')['SHA'];assert js(D/'INDEPENDENT_BIAS_ACCEPTANCE.json')['status']=='passed';assert js(D/'EVALUATION_ACCESS_STARTED.json')['epoch']>F['epoch'];assert js(D/'EVALUATION_ACCESS_STARTED.json')['bias_SHA']==sha(D/'BIAS_FROZEN.json')
 for rec in js(REPO/'src/research_plans/phase235_frozen_20261027/PRIOR_IDENTITIES.json'):assert sha(REPO/rec['path'])==rec['sha256'],rec['path']
 data={};inputs=[];per=[];table1=[];table2=[];table3=[];points=[];scenediff=[];strata=[];E=js(R/'EVAL_INPUTS.json');meta={(d,s,v['name']):v for d,pools in E.items() for s,vs in pools.items() for v in vs};seen=collections.Counter()
 with (D/'EVALUATION.jsonl').open() as f:
  for line in f:
   v=json.loads(line);cm=np.array(v['confusion'],np.int64);assert cm.shape==(3,3) and (cm>=0).all();key=(v['domain'],v['split'],'Source' if v['decision']=='Raw' else 'Source+B','NA');data.setdefault(key,{});data[key].setdefault(v['scene'],np.zeros((3,3),np.int64));data[key][v['scene']]+=cm;assert v['scene']==meta[v['domain'],v['split'],v['name']]['scene'];seen[v['domain'],v['split'],v['decision']]+=1;inputs.append({**v,**metrics(cm)})
 for d,pools in E.items():
  for s,vs in pools.items():assert seen[d,s,'Raw']==seen[d,s,'Source+B']==len(vs)
 prior=js(OLD/'PER_SCENE.json');selection=js(OLD/'delivery/VAL_SELECTION_LOCK.json');assert sha(OLD/'delivery/VAL_SELECTION_LOCK.json')==js(OLD/'delivery/VAL_SELECTION_SEAL.json')['SHA']
 for domain,pools in E.items():
  support='VALID' if domain=='L8' else 'ORIGINAL'
  for split in pools:
   for seed in [21701,21702]:
    for strategy in ['W','P','C']:
     actual='W' if strategy=='C' else strategy;alpha=1. if strategy=='C' else next(v['selected_alpha'] for v in selection['selections'] if v['domain']==domain and v['seed']==seed and v['strategy']==strategy);rows=[v for v in prior if v['domain']==domain and v['split']==split and v['support']==support and v['seed']==seed and v['strategy']==actual and v['alpha']==alpha];assert len(rows)==len({v['scene'] for v in E[domain][split]});data[domain,split,strategy,seed]={v['scene']:np.array(v['confusion'],np.int64) for v in rows}
 for key,gs in data.items():
  domain,split,strategy,seed=key
  for g,c in sorted(gs.items()):per.append({'domain':domain,'split':split,'strategy':strategy,'seed':seed,'scene':g,'confusion':c.tolist(),**metrics(c)})
  m=summary(gs);row={'domain':domain,'split':split,'strategy':strategy,'seed':seed,**m};row.update(m['pooled']);row['alpha']='NA' if strategy in ['Source','Source+B'] else 1. if strategy=='C' else next(v['selected_alpha'] for v in selection['selections'] if v['domain']==domain and v['seed']==seed and v['strategy']==strategy);table2.append(row)
 train=js(D/'TRAIN_SCENE_SUPPORT.json');fit=F['biases']['L8']['fit'];support=np.sum([v['support'] for v in train],axis=0).tolist();S=js(R/'SPARCS_SOURCE_BIAS.json')['bias_receipt'];membership=js(R/'TRAIN_MEMBERSHIP_ACCEPTED.json')
 for d,nscene,ninput,supp,fill,b,origin in [('L8',18,1560,support,sum(v['Fill_pixels'] for v in train),F['biases']['L8']['b'],'new_one_scalar_fit'),('SPARCS',60,60,S['support_by_class'],0,F['biases']['SPARCS']['b'],'reuse232_No_new_fit')]:
  vc=metrics(sum(data[d,'val','Source','NA'].values()));table1.append({'domain':d,'train_scenes':nscene,'train_inputs':ninput,'fit_support_Surface_Cloud_Shadow':supp,'fit_valid_pixels':sum(supp),'Fill_pixels':fill,'b':b,'BCE0':fit['L0'] if d=='L8' else S['BCE0'],'BCEb':fit['Lb'] if d=='L8' else S['BCEb'],'g_b':fit['g_b'] if d=='L8' else S['g_b'],'boundary':fit['boundary'] if d=='L8' else S['boundary'],'origin':origin,'SourceB_val_selection_labels':0,'W_P_val_selection_scenes':8 if d=='L8' else 10,'W_P_val_selection_pixels':vc['Surface_pixels']+vc['Cloud_pixels']+vc['Shadow_pixels'],'SourceB_has_training_labels':True,'C_training_support_note':'original C historical support; L8 Fill evaluation fix did not retroactively fix its training masks','overlap_policy':membership if d=='L8' else 'original60whole512scenes reused232, no new training pixel reads'})
 for d in E:
  gs=sorted(data[d,'test','Source+B','NA']);idx=np.random.default_rng(23501).integers(0,len(gs),(10000,len(gs)));np.savez_compressed(R/f'BOOTSTRAP_INDICES_{d}.npz',indices=idx);b=np.array([data[d,'test','Source+B','NA'][g] for g in gs]);bp=miou_batch(b[idx].sum(1));bpoint=metrics(b.sum(0))['mIoU'];allboot={};allpoints={}
  for strategy,seeds in [('Source',['NA']),('W',[21701,21702]),('C',[21701,21702]),('P',[21701,21702])]:
   ds=[];ps=[]
   for seed in seeds:
    a=np.array([data[d,'test',strategy,seed][g] for g in gs]);point=bpoint-metrics(a.sum(0))['mIoU'];ps.append(point);row={'domain':d,'comparison':'Source+B_minus_'+strategy,'reference_seed':seed,'delta_pooled_mIoU_pp':point,'SourceB_replicates':1}
    if strategy!='P':dist=bp-miou_batch(a[idx].sum(1));ds.append(dist);row['descriptive_95_pp']=np.quantile(dist,[.025,.975]).tolist();row['conditional_on_fixed_bias_and_alpha']=True;table3.append(row)
    else:points.append(row)
    for g in gs:
     x=metrics(data[d,'test','Source+B','NA'][g]);y=metrics(data[d,'test',strategy,seed][g]);scenediff.append({'domain':d,'scene':g,'comparison':'Source+B_minus_'+strategy,'reference_seed':seed,**{k:x[k]-y[k] if x[k] is not None and y[k] is not None else None for k in KEYS}})
   if len(seeds)==2:
    row={'domain':d,'comparison':'Source+B_minus_'+strategy,'reference_seed':'two_seed_difference_mean','delta_pooled_mIoU_pp':float(np.mean(ps)),'SourceB_replicates':1}
    if strategy!='P':row['descriptive_95_pp']=np.quantile(np.mean(ds,axis=0),[.025,.975]).tolist();row['conditional_on_fixed_bias_and_alpha']=True;table3.append(row)
    else:points.append(row)
 statuses={s:{v['scene']:v['usgs_shadows'] for v in vs} for s,vs in E['L8'].items()};assert list(statuses['test'].values()).count('yes')==6 and list(statuses['test'].values()).count('no')==10
 for key,groups in data.items():
  d,split,st,seed=key
  if d!='L8':continue
  for level in ['FULL','yes','no']:
   subset={g:c for g,c in groups.items() if level=='FULL' or statuses[split][g]==level}
   if subset:strata.append({'split':split,'strategy':st,'seed':seed,'stratum':level,'scenes':len(subset),**summary(subset)})
 strata_diffs=[]
 for split in E['L8']:
  for level in ['FULL','yes','no']:
   b=next((v for v in strata if v['split']==split and v['stratum']==level and v['strategy']=='Source+B'),None)
   if not b:continue
   for r in [v for v in strata if v['split']==split and v['stratum']==level and v['strategy']!='Source+B']:strata_diffs.append({'split':split,'stratum':level,'reference_strategy':r['strategy'],'reference_seed':r['seed'],**{k:b['pooled'][k]-r['pooled'][k] if b['pooled'][k] is not None and r['pooled'][k] is not None else None for k in KEYS}})
 for name,rows in [('TABLE1',table1),('TABLE2',table2),('TABLE3',table3),('P_COMPARISON_POINTS',points),('PER_SCENE',per),('PER_INPUT',inputs),('TEST_SCENE_DIFFERENCES',scenediff),('L8_STRATA',strata),('L8_STRATA_DIFFERENCES',strata_diffs)]:output(name,rows)
 save('CPU_ANALYSIS_COMPLETE.json',{'status':'passed','CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'SourceB_models':1,'reference234_selection_unchanged':True,'bootstrap_draws':10000,'bootstrap_seed':23501,'two_reference_seeds_share_scene_indices':True,'no_new_prediction_or_fit':True});assert cpu_peak()<16*1024**3;print(json.dumps({'TABLE1':table1,'TABLE3':table3}))
if __name__=='__main__':main()
