"""Independent scalar/scene/selection/bootstrap/training-ledger acceptance."""
from analyze_phase237 import ROOT,R,D,OLD,js,sha,save,KEYS,SUBSETS,SEEDS
from validate_phase232 import independent_metrics
from validate_phase233 import scalar
from analyze_phase230 import cpu_peak
import collections,json,math,time
import numpy as np
def idv(v):return tuple(v[k] for k in ['subset','seed','strategy','step','alpha','source'])
def metric(c):return {**scalar(c),**independent_metrics(c)}
def close(a,b):
 if a is None or b is None:assert a is b
 else:assert abs(a-b)<1e-9,(a,b)
def summed(cs):return [[sum(c[i][j] for c in cs) for j in range(3)] for i in range(3)]
def main():
 start=time.monotonic();assert js(R/'CPU_ANALYSIS_COMPLETE.json')['status']=='passed';C=js(D/'COMPLETE.json');TR=js(R/'TRANSFER_ACCEPTED.json');E=js(R/'EVAL_INPUTS.json');T=js(R/'SUBSET_INPUTS.json');expected=js(R/'EXPECTED_SOURCE236.json');raw={};counts=collections.Counter();sourcechecks=aliaschecks=0
 for n,v in TR['manifest'].items():assert (D/n).stat().st_size==v['bytes'] and sha(D/n)==v['SHA'],n
 for v in js(R/'PRIOR_IDENTITIES.json'):p=ROOT/v['path'];assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes']
 by={v['scene']:v for v in js(R/'FULL60_TRAIN_METADATA.json')};assert len(by)==60;rank=js(R/'DRAW_RANKINGS.json');overlap=js(R/'SUBSET_OVERLAP.json')
 for k,subset in enumerate(SUBSETS,1):
  order=sorted(by,key=lambda g:(__import__('hashlib').sha256(f'phase237|subset{k}|{g}'.encode()).hexdigest(),g));assert [v['scene'] for v in T[subset]]==order[:18] and [v['scene'] for v in rank[subset]]==order
  assert all(v==by[v['scene']] for v in T[subset])
  for other in SUBSETS:assert len({v['scene'] for v in T[subset]}&{v['scene'] for v in T[other]})==overlap[subset][other]
  for rows in E.values():assert not {v['scene'] for v in T[subset]}&{v['scene'] for v in rows}
 for split in ['val','test']:
  aliases={};meta={v['name']:v for v in E[split]};assert len(meta)==10
  with (D/f'{split}_PER_INPUT.jsonl').open() as f:
   for line in f:
    v=json.loads(line);cm=v['confusion'];ex=expected[split][v['name']];assert v['scene']==meta[v['name']]['scene'] and all(isinstance(x,int) and x>=0 for row in cm for x in row);assert [sum(row) for row in cm]==[sum(row) for row in ex['confusion']]
    if v['source']:assert cm==ex['confusion'] and v['prediction_SHA']==ex['prediction_SHA'];sourcechecks+=1
    alias=(v['name'],'Source') if v['source'] else (v['name'],v['subset'],v['seed'],v['step']) if v['strategy'] in ['E','endpoint'] or v['alpha']==1 else (v['name'],v['subset'],v['seed'],v['strategy'],v['alpha'])
    value=(v['prediction_SHA'],cm)
    if alias in aliases:assert aliases[alias]==value;aliaschecks+=1
    else:aliases[alias]=value
    key=split,idv(v);counts[key]+=1;raw.setdefault(key,{});raw[key].setdefault(v['scene'],[[0]*3 for _ in range(3)])
    for i in range(3):
     for j in range(3):raw[key][v['scene']][i][j]+=cm[i][j]
  for v in js(D/f'{split.upper()}_SUMMARY.json'):
   assert counts[split,idv(v)]==10 and raw[split,idv(v)]==v['scenes'];assert summed(v['scenes'].values())==v['confusion'];close(metric(v['confusion'])['mIoU'],v['pooled_mIoU'])
 lock=js(D/'VAL_SELECTION_LOCK.json');assert sha(D/'VAL_SELECTION_LOCK.json')==js(D/'VAL_SELECTION_SEAL.json')['SHA'];assert js(D/'TEST_ACCESS_STARTED.json')['epoch']>js(D/'VAL_SELECTION_SEAL.json')['epoch'];assert len(lock['subset_selections'])==18 and len(lock['full60_selections'])==6
 def order(v):return (-v['pooled_mIoU'],0,0) if v['source'] else (-v['pooled_mIoU'],1,v['step'] if v['strategy']=='E' else v['alpha'])
 for chosen in lock['subset_selections']:
  pool=[v for v in js(D/'VAL_SUMMARY.json') if (v['subset'],v['seed'],v['strategy'])==(chosen['subset'],chosen['seed'],chosen['strategy'])];assert len(pool)==5;winner=min(pool,key=order);assert idv(winner)==idv(chosen) and winner['confusion']==chosen['confusion']
 oldval=js(OLD/'VAL_ALL_CANDIDATES.json');oldtest=js(OLD/'delivery/SPARCS_test_SUMMARY.json');fullsel=js(R/'FULL60_VAL_SELECTION.json');assert lock['full60_selections']==fullsel
 for frozen in fullsel:
  pool=[v for v in oldval if (v['domain'],v['seed'],v['strategy'])==('SPARCS',frozen['seed'],frozen['strategy']) and v['lr'] in [None,1e-4]];assert pool==frozen['candidates'] and len(pool)==5;winner=min(pool,key=order);assert winner==frozen['selected'];assert any(all(v[k]==winner[k] for k in ['seed','strategy','step','alpha','source','lr']) for v in oldtest)
 scene_rows=js(R/'PER_SCENE.json');groups={}
 for v in scene_rows:
  key=v['split'],v['subset'],v['seed'],v['strategy'],v['step'],v['alpha'],v['source'];groups.setdefault(key,{})[v['scene']]=v['confusion'];m=metric(v['confusion'])
  for k in KEYS:close(m[k],v[k])
  for a,b in zip(m['class_IoU'],v['class_IoU']):close(a,b)
  if v['subset']!='FULL60':assert raw[v['split'],idv(v)][v['scene']]==v['confusion']
  else:
   ref=next(x for x in js(R/'FULL60_REFERENCES.json') if x['seed']==v['seed'] and x['strategy']==v['strategy'] and (v['strategy']!='endpoint' or x.get('split')==v['split']));data=ref['data'] if v['strategy']=='endpoint' else ref[v['split']];assert data['scenes'][v['scene']]==v['confusion']
 def chosen_groups(split,subset,seed,st):
  if st=='Source':return groups[split,'Source',None,'Source',None,None,True]
  if subset=='FULL60':
   rec=next(v for v in js(R/'FULL60_REFERENCES.json') if v['seed']==seed and v['strategy']==st and (st!='endpoint' or v.get('split')==split));data=rec['data'] if st=='endpoint' else rec[split];return data['scenes']
  if st=='endpoint':return next(gs for key,gs in groups.items() if key[:4]==(split,subset,seed,'E' if split=='val' else st) and key[4]==4000)
  choice=next(v for v in lock['subset_selections'] if (v['subset'],v['seed'],v['strategy'])==(subset,seed,st));return groups[(split,)+idv(choice)]
 for v in js(R/'TABLE2.json'):
  split=v.get('split','test');gs=chosen_groups(split,v['subset'],v['seed'],v['strategy']);cm=summed(gs.values());assert cm==v['confusion']
  for k in KEYS:
   close(metric(cm)[k],v[k]);vals=[metric(c)[k] for c in gs.values() if metric(c)[k] is not None];close(math.fsum(vals)/len(vals) if vals else None,v['scene_mean_'+k]);assert len(vals)==v[k+'_defined_scenes']
  if v['strategy']!='Source':
   valcm=summed(chosen_groups('val',v['subset'],v['seed'],v['strategy']).values());valbase=summed(chosen_groups('val','Source',None,'Source').values());testbase=summed(chosen_groups('test','Source',None,'Source').values());close(metric(valcm)['mIoU']-metric(valbase)['mIoU'],v['val_mIoU_minus_Source_pp']);close(metric(cm)['mIoU']-metric(testbase)['mIoU'],v['test_mIoU_minus_Source_pp']);assert v['val_positive_test_negative']==bool(v['val_mIoU_minus_Source_pp']>0 and v['test_mIoU_minus_Source_pp']<0)
 # All 6 new models and 2 same-seed references share each draw. Reused E60 is
 # not resampled as a separate reference for each training subset.
 idx=np.load(R/'BOOTSTRAP_INDICES.npz')['indices'];scenes=sorted(chosen_groups('test','Source',None,'Source'));assert np.array_equal(idx,np.random.default_rng(23701).integers(0,10,(10000,10)))
 pools={(s,seed,st):[chosen_groups('test',s,seed,st)[g] for g in scenes] for s in ['FULL60',*SUBSETS] for seed in SEEDS for st in ['E','W','P']};dist={c:[] for c in ['E18−E60','W18−E18','W18−P18']}
 def scalar_m(cm):
  valid=[]
  for i in range(3):
   u=sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]
   if u:valid.append(cm[i][i]/u)
  return math.fsum(valid)/len(valid)*100
 for draw in idx:
  multiplicity=np.bincount(draw,minlength=10).tolist();value={}
  for key,pool in pools.items():value[key]=scalar_m([[sum(multiplicity[g]*pool[g][i][j] for g in range(10)) for j in range(3)] for i in range(3)])
  dist['E18−E60'].append(math.fsum(value[s,seed,'E']-value['FULL60',seed,'E'] for s in SUBSETS for seed in SEEDS)/6)
  dist['W18−E18'].append(math.fsum(value[s,seed,'W']-value[s,seed,'E'] for s in SUBSETS for seed in SEEDS)/6)
  dist['W18−P18'].append(math.fsum(value[s,seed,'W']-value[s,seed,'P'] for s in SUBSETS for seed in SEEDS)/6)
 table=js(R/'TABLE3.json')
 for v in table:
  a,b={'E18−E60':('E','E'),'W18−E18':('W','E'),'W18−P18':('W','P')}[v['comparison']];cases=[(s,seed) for s in SUBSETS for seed in SEEDS] if isinstance(v['seed'],str) else [(v['subset'],v['seed'])];points=[scalar_m(summed(pools[s,seed,a]))-scalar_m(summed(pools['FULL60' if v['comparison']=='E18−E60' else s,seed,b])) for s,seed in cases];close(math.fsum(points)/len(points),v['delta_mIoU_pp'])
  if 'conditional_95_pp' in v:assert max(abs(float(x)-y) for x,y in zip(np.quantile(dist[v['comparison']],[.025,.975]),v['conditional_95_pp']))<1e-9
 for v in js(R/'SUBSET_TWO_SEED_DIFFERENCES.json'):
  points=[r['delta_mIoU_pp'] for r in table if (r['subset'],r['comparison'])==(v['subset'],v['comparison'])];assert len(points)==2;close(math.fsum(points)/2,v['two_seed_mean_delta_mIoU_pp']);assert [min(points),max(points)]==v['two_seed_range_pp']
 totalupdates=0;allinitial=[];statepairs=0
 for p in (D/'fits').glob('*/COMPLETE.json'):
  fit=js(p);name=p.parent.name;subset=fit['subset'];seed=fit['seed'];assert fit['updates']==fit['backwards']==4000 and not fit['forbidden_accesses'];pool=js(p.parent/'TRAIN_POOL.json');initial=js(p.parent/'INITIAL.json');allinitial.append(initial);assert initial['all_nonhead_Source_exact'] and initial['trainable']==1316516
  assert sha(p.parent/'SCHEDULE.jsonl')==fit['schedule_SHA'] and sha(p.parent/'LOSSES.jsonl')==fit['losses_SHA'];schedule=[json.loads(x) for x in (p.parent/'SCHEDULE.jsonl').read_text().splitlines()];losses=[json.loads(x) for x in (p.parent/'LOSSES.jsonl').read_text().splitlines()];assert len(schedule)==len(losses)==4000
  assert len(pool['census'])==len(pool['active_indices'])==18 and not pool['excluded_all_invalid'];assert {v['scene'] for v in pool['census']}=={v['scene'] for v in T[subset]};assert len(pool['original60_dataset_order'])==60
  filtered=[pool['original60_dataset_order'][i] for i in pool['selected_original_indices']];assert set(filtered)=={v['image_path'] for v in T[subset]};assert sorted(pool['selected_original_indices'])==pool['selected_original_indices']
  exposure=collections.Counter()
  for i,(data,loss) in enumerate(zip(schedule,losses),1):
   assert data['step']==loss['step']==i and data['data_seed']==seed*10000+i-1 and 0<=data['index']<18;assert data['scene'] in {v['scene'] for v in T[subset]};assert filtered[data['index']].endswith(data['name']);assert math.isfinite(loss['loss']) and all(math.isfinite(x) for x in loss['terms'].values());assert all(math.isfinite(x) and x>=0 for x in loss['lr_used']);exposure[data['scene']]+=1
  assert dict(exposure)==fit['scene_exposure'];assert losses[0]['lr_used'][0]==1e-5 and losses[-1]['lr_used'][0]==0
  ref=js(OLD/f'delivery/fits/SPARCS_{seed}_1e-04/INITIAL.json');assert initial['optimizer_groups']==ref['optimizer_groups'] and initial['optim_config']==ref['optim_config'] and initial['schedulers']==ref['schedulers'];assert initial['head_digest']==ref['head_digest'] and initial['nonhead_digest']==ref['nonhead_digest']
  assert [v['step'] for v in fit['nonhead_checks']]==list(range(100,4001,100))+[4000];assert all(v['nonhead_exact'] and v['nonhead_no_gradient'] and v['nonhead_digest']==initial['nonhead_digest'] for v in fit['nonhead_checks'])
  for step,v in fit['heads'].items():assert lock['heads_SHA'][f'fits/{name}/head_{step}.pth']==v['head_SHA'];assert v['nonhead_digest']==initial['nonhead_digest'];statepairs+=1
  totalupdates+=4000
 assert totalupdates==24000 and len(allinitial)==6 and statepairs==24
 ledger=[json.loads(x) for x in (D/'CALL_LEDGER.jsonl').read_text().splitlines()];assert [v['ordinal'] for v in ledger]==list(range(1,len(ledger)+1));assert len(ledger)==C['calls']==24580 and dict(collections.Counter(v['kind'] for v in ledger))==C['counts'];assert C['counts']['training_full_input']==24000 and C['counts']['evaluation_full_input']==580 and C['full60_new_calls']==C['full60_refits']==0
 costs=[js(p) for p in (D/'stage_costs').glob('*.json')];close(math.fsum(v['GPU_stage_seconds'] for v in costs),C['GPU_stage_seconds']);assert C['GPU_stage_seconds']<10800 and len(ledger)<26000
 for v in costs:
  for s in v['resources']:assert s['CPU_peak_bytes']<16*1024**3 and max(s['GPU_peak_allocated'],s['GPU_peak_reserved'])<20*1024**3 and s['remote_bytes']<4*1024**3
 assert TR['remote_total_bytes']<4*1024**3 and sum(v['bytes'] for v in TR['manifest'].values())<1024**3 and cpu_peak()<16*1024**3 and time.monotonic()-start<7200
 save('INDEPENDENT_ACCEPTANCE.json',{'status':'passed','new_per_input_rows':sum(counts.values()),'Source_prediction_SHA_cm_replays236':sourcechecks,'endpoint_Source_alias_checks':aliaschecks,'subset_choices_replayed':18,'full60_five_candidate_choices_replayed':6,'same_index_scalar_bootstrap_recomputations':30000,'all_24000_training_steps_and_nonhead_states':True,'same_optimizer_schedule_Source_as236':True,'frozen_subset_rank_membership_overlap_verified':True,'head_resume_pairs_in_inventory':24,'full60_new_calls':0,'ledger_full_input_forwards':24580,'CPU_seconds':time.monotonic()-start,'CPU_peak_bytes':cpu_peak()});print('Phase237 independent CPU acceptance passed')
if __name__=='__main__':main()
