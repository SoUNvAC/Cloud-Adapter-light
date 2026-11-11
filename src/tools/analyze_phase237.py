"""Three fixed subset sensitivity comparisons, cached counts only."""
from pathlib import Path
import collections,csv,hashlib,json,time
import numpy as np
from phase231_metrics import metrics
from analyze_phase229 import miou_batch
from analyze_phase230 import cpu_peak
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase237/phase237_20261028';D=R/'delivery';OLD=ROOT/'outputs/phase236/phase236_20261027'
KEYS=['mIoU','Cloud_recall','Surface_to_Cloud_FPR','Shadow_recall','nonShadow_to_Shadow_FPR']
SUBSETS=['SUBSET_1','SUBSET_2','SUBSET_3'];SEEDS=[21701,21702]
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def output(n,rows):
 save(n+'.json',rows);fields=list(dict.fromkeys(k for v in rows for k in v))
 with (R/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def ident(v):return tuple(v[k] for k in ['subset','seed','strategy','step','alpha','source'])
def summary(gs):
 c=sum(gs.values());pooled=metrics(c);out={**pooled,'pooled':pooled,'scene_count':len(gs)}
 for k in KEYS:
  vals=[metrics(c)[k] for c in gs.values() if metrics(c)[k] is not None];out['scene_mean_'+k]=float(np.mean(vals)) if vals else None;out[k+'_defined_scenes']=len(vals)
 return out
def order(v):return (-v['pooled_mIoU'],0,0) if v['source'] else (-v['pooled_mIoU'],1,v['step'] if v['strategy']=='E' else v['alpha'])
def main():
 start=time.monotonic();assert not (R/'CPU_ANALYSIS_COMPLETE.json').exists();assert (D/'EXIT_CODE').read_text().strip()=='0';C=js(D/'COMPLETE.json');assert C['updates']==C['training_backwards']==24000 and C['calls']<=26000 and C['GPU_stage_seconds']<10800
 L=js(D/'VAL_SELECTION_LOCK.json');assert sha(D/'VAL_SELECTION_LOCK.json')==js(D/'VAL_SELECTION_SEAL.json')['SHA'];assert js(D/'TEST_ACCESS_STARTED.json')['epoch']>js(D/'VAL_SELECTION_SEAL.json')['epoch'];E=js(R/'EVAL_INPUTS.json');T=js(R/'SUBSET_INPUTS.json');full=js(R/'FULL60_REFERENCES.json');fullsel=js(R/'FULL60_VAL_SELECTION.json');groups={};meta={};allcandidates=[];allscene=[];seen=collections.Counter();labelrows=[];headrows=[];t1=[];t2=[];t3=[];subsetdeltas=[];scenedeltas=[];signcounts=[];allinput=[]
 for split in ['val','test']:
  assert sha(D/f'{split}_PER_INPUT.jsonl')==js(D/f'{split.upper()}_COMPLETE.json')['per_input_SHA'];pool=js(D/f'{split.upper()}_SUMMARY.json');raw={};n=0
  with (D/f'{split}_PER_INPUT.jsonl').open() as f:
   for line in f:
    v=json.loads(line);cm=np.asarray(v['confusion'],np.int64);key=ident(v);assert cm.shape==(3,3) and (cm>=0).all();raw.setdefault(key,{});raw[key].setdefault(v['scene'],np.zeros((3,3),np.int64));raw[key][v['scene']]+=cm;seen[split,key]+=1;allinput.append(v);n+=1
  for row in pool:
   key=ident(row);assert seen[split,key]==10 and {g:c.tolist() for g,c in raw[key].items()}==row['scenes'];assert sum(raw[key].values()).tolist()==row['confusion'];m=summary(raw[key]);assert abs(m['mIoU']-row['pooled_mIoU'])<1e-10
   for g,c in raw[key].items():allscene.append({'subset':row['subset'],'seed':row['seed'],'strategy':row['strategy'],'step':row['step'],'alpha':row['alpha'],'source':row['source'],'split':split,'scene':g,**metrics(c)})
   if split=='val' and row['strategy']!='Source':allcandidates.append({k:v for k,v in row.items() if k!='scenes'})
   if row['strategy']=='Source':groups[split,'Source',None,'Source']=raw[key];meta[split,'Source',None,'Source']=row
  if split=='val':
   for subset in SUBSETS:
    for seed in SEEDS:
     for st in ['E','W','P']:
      candidates=[v for v in pool if (v['subset'],v['seed'],v['strategy'])==(subset,seed,st)];assert len(candidates)==5;chosen=min(candidates,key=order);lock=next(v for v in L['subset_selections'] if (v['subset'],v['seed'],v['strategy'])==(subset,seed,st));assert ident(chosen)==ident(lock) and chosen['confusion']==lock['confusion'];groups[split,subset,seed,st]=raw[ident(chosen)];meta[split,subset,seed,st]=chosen
     endpoint=next(v for v in pool if (v['subset'],v['seed'],v['strategy'],v['step'])==(subset,seed,'E',4000));groups[split,subset,seed,'endpoint']=raw[ident(endpoint)];meta[split,subset,seed,'endpoint']=endpoint
  else:
   for row in pool:
    if row['strategy']!='Source':groups[split,row['subset'],row['seed'],row['strategy']]=raw[ident(row)];meta[split,row['subset'],row['seed'],row['strategy']]=row
 for v in full:
  if v['strategy']=='endpoint':items=[(v['split'],v['data'])]
  else:items=[(s,v[s]) for s in ['val','test']]
  for split,row in items:
   key=split,'FULL60',v['seed'],v['strategy'];groups[key]={g:np.asarray(c,np.int64) for g,c in row['scenes'].items()};meta[key]={**row,'subset':'FULL60','strategy':v['strategy']}
   for g,c in groups[key].items():allscene.append({'subset':'FULL60','seed':v['seed'],'strategy':v['strategy'],'step':row['step'],'alpha':row['alpha'],'source':row['source'],'split':split,'scene':g,**metrics(c),'reused236_no_new_forward':True})
 for v in fullsel:
  for row in v['candidates']:allcandidates.append({'subset':'FULL60',**row,'reused236_no_new_forward':True})
 # Training labels/exposures: raw original-scene annotation counts differ from
 # repeated random512 training crops; never add repetitions to label budgets.
 for subset in ['FULL60',*SUBSETS]:
  for seed in SEEDS:
   p=(OLD/'delivery/fits'/f'SPARCS_{seed}_1e-04') if subset=='FULL60' else (D/'fits'/f'{subset}_{seed}');fit=js(p/'COMPLETE.json');pool=js(p/'TRAIN_POOL.json');support=pool['valid_support'];scenes=pool['census'];labels={'subset':subset,'seed':seed,'train_original_scenes':pool['scene_count'],'train_unique_listed_inputs':len(scenes),'raw_valid_labels':sum(int(support[str(k)]) for k in range(3)),'raw_class_support':support,'updates':4000,'repeated_sampled_crop_pixels':4000*512*512,'scene_exposure':fit['scene_exposure'],'scene_exposure_min':min(fit['scene_exposure'].values()),'scene_exposure_max':max(fit['scene_exposure'].values()),'read_only236':subset=='FULL60'};labelrows.append(labels)
   for v in scenes:labelrows.append({'subset':subset,'seed':seed,'scene':v['scene'],'original_label_valid_pixels':v['valid_pixels'],'original_label_support':v['support'],'training_exposure':fit['scene_exposure'].get(v['scene'],0),'repeated_sampled_crop_pixels':fit['scene_exposure'].get(v['scene'],0)*512*512,'read_only236':subset=='FULL60'})
   row={k:labels[k] for k in ['subset','seed','train_original_scenes','raw_valid_labels','scene_exposure_min','scene_exposure_max','read_only236']}
   for st in ['E','W','P']:
    chosen=meta['val',subset,seed,st];row[st]={k:chosen[k] for k in ['step','alpha','source','pooled_mIoU']}
   t1.append(row)
   if subset!='FULL60':
    for step,v in fit['heads'].items():headrows.append({'subset':subset,'seed':seed,'step':int(step),**v,'head_path':f'fits/{subset}_{seed}/head_{step}.pth','resume_path':f'fits/{subset}_{seed}/resume_{step}.pth','retained_shared_not_transferred':True})
 for subset in ['FULL60',*SUBSETS]:
  for seed in SEEDS:
   for st in ['E','W','P','endpoint']:
    a=summary(groups['val',subset,seed,st]);b=summary(groups['test',subset,seed,st]);sv=summary(groups['val','Source',None,'Source']);ss=summary(groups['test','Source',None,'Source']);desc=meta['val',subset,seed,st];vg=a['mIoU']-sv['mIoU'];tg=b['mIoU']-ss['mIoU'];t2.append({'subset':subset,'seed':seed,'strategy':st,'step':desc['step'],'alpha':None if st=='endpoint' else desc['alpha'],'source':desc['source'],'val_mIoU':a['mIoU'],'val_mIoU_minus_Source_pp':vg,'test_mIoU_minus_Source_pp':tg,'test_scene_mean_mIoU_minus_Source_pp':b['scene_mean_mIoU']-ss['scene_mean_mIoU'],'val_positive_test_negative':bool(vg>0 and tg<0),**b,'read_only236':subset=='FULL60'})
 for split in ['val','test']:t2.append({'subset':'Source','seed':None,'strategy':'Source','split':split,**summary(groups[split,'Source',None,'Source']),'read_only236':False})
 scenes=sorted(groups['test','Source',None,'Source']);indices=np.random.default_rng(23701).integers(0,len(scenes),(10000,len(scenes)));np.savez_compressed(R/'BOOTSTRAP_INDICES.npz',indices=indices);bootstrap={}
 for (split,subset,seed,st),gs in groups.items():
  if split=='test' and st in ['E','W','P']:
   a=np.array([gs[g] for g in scenes]);bootstrap[subset,seed,st]=miou_batch(a[indices].sum(1))
 for comparison,left,right in [('E18−E60','E','E'),('W18−E18','W','E'),('W18−P18','W','P')]:
  differences=[];points=[]
  for subset in SUBSETS:
   per_subset=[]
   for seed in SEEDS:
    other='FULL60' if comparison=='E18−E60' else subset;g1=groups['test',subset,seed,left];g2=groups['test',other,seed,right];point=summary(g1)['mIoU']-summary(g2)['mIoU'];dist=bootstrap[subset,seed,left]-bootstrap[other,seed,right];points.append(point);differences.append(dist);per_subset.append(point);t3.append({'comparison':comparison,'subset':subset,'seed':seed,'delta_mIoU_pp':point,'aggregation':'individual fixed subset/seed difference','test_scenes':10})
    for g in scenes:
     a=metrics(g1[g]);b=metrics(g2[g]);scenedeltas.append({'comparison':comparison,'subset':subset,'seed':seed,'scene':g,**{k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in KEYS}})
   subsetdeltas.append({'comparison':comparison,'subset':subset,'two_seed_mean_delta_mIoU_pp':float(np.mean(per_subset)),'two_seed_range_pp':[min(per_subset),max(per_subset)],'no_population_inference_over_subsets':True})
  t3.append({'comparison':comparison,'subset':'ALL_THREE_FIXED_SUBSETS','seed':'six_difference_mean','delta_mIoU_pp':float(np.mean(points)),'conditional_95_pp':np.quantile(np.mean(differences,axis=0),[.025,.975]).tolist(),'six_difference_range_pp':[min(points),max(points)],'test_scenes':10,'shared_full60_E_references':2,'bootstrap_draws':10000,'bootstrap_seed':23701,'no_reselection':True})
 for population in ['FULL60',*SUBSETS]:
  rows=[v for v in t2 if v['subset']==population];signcounts.append({'subset':population,'val_positive_test_negative_count_all4_strategies':sum(v['val_positive_test_negative'] for v in rows),'rows_all4':len(rows),'val_positive_test_negative_count_E_W_P':sum(v['val_positive_test_negative'] for v in rows if v['strategy']!='endpoint'),'rows_E_W_P':sum(v['strategy']!='endpoint' for v in rows),'descriptive_only':True})
 for n,rows in [('TABLE1',t1),('TABLE2',t2),('TABLE3',t3),('SUBSET_TWO_SEED_DIFFERENCES',subsetdeltas),('VAL_ALL_CANDIDATES',allcandidates),('PER_SCENE',allscene),('PER_INPUT',allinput),('TEST_SCENE_DIFFERENCES',scenedeltas),('LABEL_BUDGET_EXPOSURE',labelrows),('HEAD_RESUME_INVENTORY',headrows),('SIGN_REVERSAL_COUNTS',signcounts)]:output(n,rows)
 assert time.monotonic()-start<7200 and cpu_peak()<16*1024**3;save('CPU_ANALYSIS_COMPLETE.json',{'status':'passed','CPU_seconds':time.monotonic()-start,'CPU_peak_bytes':cpu_peak(),'new_per_input_rows':len(allinput),'subset_choices':18,'full60_choices':6,'fixed_interval_count':3,'no_new_model_or_reference_pixel_access':True});print(json.dumps({'comparisons':[v for v in t3 if 'conditional_95_pp' in v],'subset_differences':subsetdeltas,'sign_reversals':signcounts}))
if __name__=='__main__':main()
