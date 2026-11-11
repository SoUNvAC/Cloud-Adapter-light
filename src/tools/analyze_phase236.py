"""Fixed cached counts only: selection replay, paired scenes, no model access."""
from pathlib import Path
import collections,csv,hashlib,json,time
import numpy as np
from phase231_metrics import metrics
from analyze_phase229 import miou_batch
from analyze_phase230 import cpu_peak
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase236/phase236_20261027';D=R/'delivery'
KEYS=['mIoU','Cloud_recall','Surface_to_Cloud_FPR','Shadow_recall','nonShadow_to_Shadow_FPR']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def output(n,rows):
 save(n+'.json',rows);fields=list(dict.fromkeys(k for v in rows for k in v))
 with (R/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def cid(c):return json.dumps({k:c[k] for k in ['seed','strategy','lr','step','alpha','source']},sort_keys=True)
def summary(groups):
 c=sum(groups.values());pooled=metrics(c);result={**pooled,'pooled':pooled,'scene_count':len(groups)}
 for k in KEYS:
  values=[metrics(c)[k] for c in groups.values() if metrics(c)[k] is not None]
  result['scene_mean_'+k]=float(np.mean(values)) if values else None;result[k+'_defined_scenes']=len(values)
 return result
def source_candidate(seed,st):return dict(seed=seed,strategy=st,lr=None,step=None,alpha=None,source=True)
def selection_order(v):
 c=np.asarray(v['confusion'],np.float64);t=c.diagonal();u=c.sum(0)+c.sum(1)-t;score=float(np.mean(t[u>0]/u[u>0])*100)
 return (-score,0,0,0) if v['source'] else (-score,1,v['step'] if v['strategy']=='E' else v['alpha'],v['lr'])
def main():
 start=time.monotonic();assert not (R/'CPU_ANALYSIS_COMPLETE.json').exists();assert (D/'EXIT_CODE').read_text().strip()=='0'
 complete=js(D/'COMPLETE.json');assert complete['updates']==complete['backwards']==32063 and complete['calls']<=85000 and complete['GPU_stage_seconds']<21600
 lock=js(D/'VAL_SELECTION_LOCK.json');assert sha(D/'VAL_SELECTION_LOCK.json')==js(D/'VAL_SELECTION_SEAL.json')['SHA'];assert js(D/'TEST_ACCESS_STARTED.json')['epoch']>js(D/'VAL_SELECTION_SEAL.json')['epoch']
 E=js(R/'EVAL_INPUTS.json');chosen=lock['selections'];assert len(chosen)==12
 data={};descriptors={};allval=[];replay=[];per=[];t1=[];t2=[];t3=[];signs=[];strata=[];diffs=[];seen=collections.Counter();inputcounts={}
 for domain in E:
  for split in ['val','test']:
   p=D/f'{domain}_{split}_PER_INPUT.jsonl';stage=js(D/f'{split.upper()}_COMPLETE.json');assert sha(p)==stage['pools'][domain]['per_input_SHA']
   meta={v['name']:v for v in E[domain][split]};n=0
   with p.open() as f:
    for line in f:
     v=json.loads(line);assert v['scene']==meta[v['name']]['scene'];c=np.asarray(v['confusion'],np.int64);assert c.shape==(3,3) and (c>=0).all();key=(domain,split,cid(v));descriptors[key]={k:v[k] for k in ['seed','strategy','lr','step','alpha','source']};data.setdefault(key,{});data[key].setdefault(v['scene'],np.zeros((3,3),np.int64));data[key][v['scene']]+=c;seen[key]+=1;n+=1
   inputcounts[domain,split]=n
   table=js(D/f'{domain}_{split}_SUMMARY.json')
   for row in table:
    key=(domain,split,cid(row));assert seen[key]==len(meta) and np.array_equal(sum(data[key].values()),row['confusion']);assert {g:c.tolist() for g,c in data[key].items()}==row['scenes']
    if split=='val':allval.append({k:v for k,v in row.items() if k!='scenes'})
  table=[v for v in allval if v['domain']==domain]
  for seed in [21701,21702]:
   row={'domain':domain,'seed':seed,'Source_val_mIoU':summary(data[domain,'val',cid(source_candidate(seed,'E'))])['mIoU']}
   for st in ['E','W','P']:
    pool=[v for v in table if v['seed']==seed and v['strategy']==st];assert len(pool)==9;winner=min(pool,key=selection_order);frozen=next(v for v in chosen if (v['domain'],v['seed'],v['strategy'])==(domain,seed,st));assert cid(winner)==cid(frozen) and winner['confusion']==frozen['confusion'];replay.append({'domain':domain,'seed':seed,'strategy':st,'candidate_count':9,'winner':cid(winner),'frozen_exact':True})
    row[st]={k:winner[k] for k in ['lr','step','alpha','source','pooled_mIoU']}
   t1.append(row)
 # Keep all candidate per-scene counts; Source aliases are kept in raw candidate
 # files but fixed test summary contains a single Source model only.
 for (domain,split,identity),groups in data.items():
  for g,c in groups.items():per.append({'domain':domain,'split':split,**descriptors[domain,split,identity],'scene':g,**metrics(c)})
 # Historical fixed Source+B counts only, never read fit pixels or refit.
 prior=js(ROOT/'outputs/phase235/phase235_20261027/PER_SCENE.json')
 for domain in E:
  for split in ['val','test']:
   rows=[v for v in prior if (v['domain'],v['split'],v['strategy'])==(domain,split,'Source+B')];desc=dict(seed=None,strategy='Source+B',lr=None,step=None,alpha=None,source=False);key=(domain,split,cid(desc));data[key]={v['scene']:np.asarray(v['confusion'],np.int64) for v in rows};descriptors[key]=desc
   for v in rows:per.append({'domain':domain,'split':split,**desc,'scene':v['scene'],**metrics(v['confusion']),'historical235':True})
 def test_key(domain,st,seed,lr=None):
  return next(k for k in data if k[:2]==(domain,'test') and descriptors[k]['strategy']==st and descriptors[k]['seed']==seed and (lr is None or descriptors[k]['lr']==lr))
 for key,groups in list(data.items()):
  domain,split,identity=key;desc=descriptors[key]
  if split!='test':continue
  row={'domain':domain,**desc,**summary(groups),'historical235':desc['strategy']=='Source+B','val_selection_budget_comparable':desc['strategy']!='Source+B'};t2.append(row)
  if domain=='L8':
   status={v['scene']:v['usgs_shadows'] for v in E[domain][split]}
   for level in ['FULL','yes','no']:
    subset={g:c for g,c in groups.items() if level=='FULL' or status[g]==level};assert len(subset)=={'FULL':16,'yes':6,'no':10}[level];strata.append({'domain':domain,**desc,'stratum':level,**summary(subset)})
 for domain in E:
  for st,lr in [('E',None),('W',None),('P',None),('endpoint',1e-4),('endpoint',3e-5)]:
   rows=[v for v in t2 if v['domain']==domain and v['strategy']==st and isinstance(v['seed'],int) and (lr is None or v['lr']==lr)];assert len(rows)==2
   mean={'domain':domain,'strategy':st,'seed':'two_seed_metric_mean','lr':lr if st=='endpoint' else None,'step':4000 if st=='endpoint' else None,'alpha':None,'source':False,'confusion':None,'aggregation':'arithmetic mean of separately evaluated seed metrics, not score ensemble','scene_count':rows[0]['scene_count'],'class_IoU':[float(np.mean([r['class_IoU'][j] for r in rows])) if all(r['class_IoU'][j] is not None for r in rows) else None for j in range(3)]}
   for k in KEYS:
    for field in [k,'scene_mean_'+k]:mean[field]=float(np.mean([v[field] for v in rows])) if all(v[field] is not None for v in rows) else None
   for k in ['Surface_pixels','Cloud_pixels','Shadow_pixels','nonShadow_pixels',*[x+'_defined_scenes' for x in KEYS]]:assert rows[0][k]==rows[1][k];mean[k]=rows[0][k]
   t2.append(mean)
  source=test_key(domain,'Source',None);sourceval=data[domain,'val',cid(source_candidate(21701,'E'))];source_test=summary(data[source])['mIoU'];source_val=summary(sourceval)['mIoU']
  scenes=sorted(data[source]);idx=np.random.default_rng(23601).integers(0,len(scenes),(10000,len(scenes)));np.savez_compressed(R/f'BOOTSTRAP_INDICES_{domain}.npz',indices=idx)
  for left,right in [('W','E'),('W','P'),('E','Source')]:
   seedpoints=[];seedboot=[]
   for seed in [21701,21702]:
    ka=test_key(domain,left,seed);kb=source if right=='Source' else test_key(domain,right,seed);aa=np.array([data[ka][g] for g in scenes]);bb=np.array([data[kb][g] for g in scenes]);point=metrics(aa.sum(0))['mIoU']-metrics(bb.sum(0))['mIoU'];dist=miou_batch(aa[idx].sum(1))-miou_batch(bb[idx].sum(1));seedpoints.append(point);seedboot.append(dist);t3.append({'domain':domain,'comparison':left+'−'+right,'seed':seed,'delta_mIoU_pp':point,'conditional_95_pp':np.quantile(dist,[.025,.975]).tolist(),'scenes':len(scenes),'bootstrap_reselects':False})
    for g in scenes:
     a=metrics(data[ka][g]);b=metrics(data[kb][g]);diffs.append({'domain':domain,'comparison':left+'−'+right,'seed':seed,'scene':g,**{k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in KEYS}})
   t3.append({'domain':domain,'comparison':left+'−'+right,'seed':'two_seed_difference_mean','delta_mIoU_pp':float(np.mean(seedpoints)),'conditional_95_pp':np.quantile(np.mean(seedboot,axis=0),[.025,.975]).tolist(),'scenes':len(scenes),'bootstrap_reselects':False})
  for seed in [21701,21702]:
   for st in ['E','W','P']:
    selection=next(v for v in chosen if (v['domain'],v['seed'],v['strategy'])==(domain,seed,st));v=summary(data[domain,'val',cid(selection)])['mIoU']-source_val;t=summary(data[test_key(domain,st,seed)])['mIoU']-source_test;signs.append({'domain':domain,'seed':seed,'strategy':st,'val_gain_over_Source_pp':v,'test_gain_over_Source_pp':t,'strict_sign_reversal':bool(v*t<0)})
 labels=[];heads=[]
 for p in sorted((D/'fits').glob('*/COMPLETE.json')):
  fit=js(p);pool=js(p.parent/'TRAIN_POOL.json');sup=pool['valid_support'];scene_count=pool['scene_count'];valid=sum(int(sup[str(k)]) for k in range(3));valsource=summary(data[fit['domain'],'val',cid(source_candidate(fit['seed'],'E'))])
  labels.append({'domain':fit['domain'],'seed':fit['seed'],'lr':fit['lr'],'train_original_scenes':scene_count,'train_listed_inputs':len(pool['census']),'train_active_inputs':len(pool['active_indices']),'train_all_invalid_excluded':pool['excluded_all_invalid'],'unique_listed_crop_pixel_support':sup,'listed_valid_pixels':valid,'selection_scenes':8 if fit['domain']=='L8' else 10,'selection_valid_pixels':sum(valsource[k] for k in ['Surface_pixels','Cloud_pixels','Shadow_pixels']),'scene_exposure_4000':fit['scene_exposure'],'same_scene_and_pixel_budget_across_LR':True,'train_resolution_note':'L8 original512 crops; SP original scene random512 crops, full-scene reference census; not equal training information budgets across targets'})
  for step,rec in fit['heads'].items():heads.append({'fit':p.parent.name,'step':int(step),'head_path':str(p.parent.relative_to(D)/f'head_{step}.pth').replace('\\','/'),'resume_path':str(p.parent.relative_to(D)/f'resume_{step}.pth').replace('\\','/'),**rec,'files_retained_shared_not_transferred':True})
 for n,rows in [('TABLE1',t1),('TABLE2',t2),('TABLE3',t3),('VAL_ALL_CANDIDATES',allval),('SELECTION_REPLAY',replay),('PER_SCENE',per),('L8_TEST_STRATA',strata),('TEST_SCENE_DIFFERENCES',diffs),('VAL_TO_TEST_SOURCE_GAINS',signs),('LABEL_BUDGET',labels),('HEAD_RESUME_INVENTORY',heads)]:output(n,rows)
 assert time.monotonic()-start<14400 and cpu_peak()<16*1024**3
 save('CPU_ANALYSIS_COMPLETE.json',{'status':'passed','seconds':time.monotonic()-start,'CPU_peak_bytes':cpu_peak(),'per_input_records':{d+'_'+s:n for (d,s),n in inputcounts.items()},'selections_replayed':12,'bootstrap_intervals':len(t3),'no_new_model_or_fit_or_reference_pixels':True,'bootstrap_seed':23601,'draws':10000});print(json.dumps({'TABLE1':t1,'TABLE3':t3}))
if __name__=='__main__':main()
