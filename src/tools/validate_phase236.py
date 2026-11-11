"""Independent scalar counts, selection, training receipts and paired intervals."""
from analyze_phase236 import ROOT,R,D,js,save,sha,KEYS
from validate_phase232 import independent_metrics
from validate_phase233 import scalar
from analyze_phase230 import cpu_peak
import collections,math,time,json
import numpy as np
def ident(v):return (v['seed'],v['strategy'],v['lr'],v['step'],v['alpha'],v['source'])
def metric(cm):return {**scalar(cm),**independent_metrics(cm)}
def close(a,b):
 if a is None or b is None:assert a is b
 else:assert abs(a-b)<1e-9,(a,b)
def csum(cs):return [[sum(c[i][j] for c in cs) for j in range(3)] for i in range(3)]
def main():
 start=time.monotonic();assert js(R/'CPU_ANALYSIS_COMPLETE.json')['status']=='passed';T=js(R/'TRAIN_INPUTS.json');E=js(R/'EVAL_INPUTS.json');C=js(D/'COMPLETE.json');transfer=js(R/'TRANSFER_ACCEPTED.json')
 for n,v in transfer['manifest'].items():assert (D/n).stat().st_size==v['bytes'] and sha(D/n)==v['SHA'],n
 prior=js(ROOT/'outputs/phase235/phase235_20261027/EXPECTED_RAW234.json');native={(v['split'],v['name']):v for v in js(D/'EXPECTED_NATIVE234.json')};allgroups={};seen=collections.Counter();perchecks=0;sourcechecks=0;aliaschecks=0
 for domain in E:
  for split in ['val','test']:
   inputs={v['name']:v for v in E[domain][split]};aliases={};summ=js(D/f'{domain}_{split}_SUMMARY.json')
   with (D/f'{domain}_{split}_PER_INPUT.jsonl').open() as f:
    for line in f:
     v=json.loads(line);name=v['name'];cm=v['confusion'];assert v['scene']==inputs[name]['scene'] and len(cm)==3 and all(len(r)==3 and all(isinstance(x,int) and x>=0 for x in r) for r in cm)
     ex=prior[domain][split][name];assert [sum(row) for row in cm]==[sum(row) for row in ex['VALID']]
     if v['source']:
      assert cm==ex['VALID'] and v['prediction_SHA']==ex['prediction_SHA'];sourcechecks+=1
     # Endpoint identities and Source aliases must agree bit-for-bit even when
     # presented under a different strategy name.
     if v['source']:alias=(name,'Source')
     elif v['strategy'] in ['E','endpoint'] or v['alpha']==1:alias=(name,v['seed'],v['lr'],v['step'])
     else:alias=(name,v['seed'],v['strategy'],v['lr'],v['alpha'])
     val=(v['prediction_SHA'],cm)
     if alias in aliases:assert aliases[alias]==val;aliaschecks+=1
     else:aliases[alias]=val
     key=(domain,split,ident(v));seen[key]+=1;allgroups.setdefault(key,{});allgroups[key].setdefault(v['scene'],[[0]*3 for _ in range(3)])
     for i in range(3):
      for j in range(3):allgroups[key][v['scene']][i][j]+=cm[i][j]
     perchecks+=1
   for v in summ:
    key=(domain,split,ident(v));assert seen[key]==len(inputs);assert allgroups[key]==v['scenes'];assert csum(allgroups[key].values())==v['confusion'];close(metric(v['confusion'])['mIoU'],v['pooled_mIoU'])
   if domain=='L8':
    nr=js(D/f'{split.upper()}_NATIVE_ACCEPTANCE.json');assert len(nr)==len(inputs)
    for v in nr:
     ex=native[split,v['name']];assert v['native_SHA']==ex['native_array_SHA'] and v['valid_SHA']==ex['valid_array_SHA'];assert v['Fill']==ex['Fill_pixels'] and sum(v['support'])==ex['valid_pixels'] and v['old_PNG_exact']
  for split in ['val','test']:assert not {v['scene'] for v in T[domain]}&{v['scene'] for v in E[domain][split]}
 # Prior Source+B is a single frozen historical scalar, not a ninth fit.
 prior235=js(ROOT/'outputs/phase235/phase235_20261027/PER_SCENE.json')
 for v in prior235:
  if v['strategy']=='Source+B':allgroups.setdefault((v['domain'],v['split'],(None,'Source+B',None,None,None,False)),{})[v['scene']]=v['confusion']
 for v in js(R/'PER_SCENE.json'):
  key=(v['domain'],v['split'],ident(v));assert allgroups[key][v['scene']]==v['confusion'];m=metric(v['confusion'])
  for k in KEYS:close(m[k],v[k])
  for a,b in zip(m['class_IoU'],v['class_IoU']):close(a,b)
 lock=js(D/'VAL_SELECTION_LOCK.json');assert sha(D/'VAL_SELECTION_LOCK.json')==js(D/'VAL_SELECTION_SEAL.json')['SHA'];assert js(D/'TEST_ACCESS_STARTED.json')['epoch']>js(D/'VAL_SELECTION_SEAL.json')['epoch']
 for domain in E:
  for seed in [21701,21702]:
   for st in ['E','W','P']:
    pool=[v for v in js(D/f'{domain}_val_SUMMARY.json') if v['seed']==seed and v['strategy']==st];assert len(pool)==9
    def order(v):return (-v['pooled_mIoU'],0,0,0) if v['source'] else (-v['pooled_mIoU'],1,v['step'] if st=='E' else v['alpha'],v['lr'])
    expected=min(pool,key=order);chosen=next(v for v in lock['selections'] if (v['domain'],v['seed'],v['strategy'])==(domain,seed,st));assert ident(expected)==ident(chosen) and expected['confusion']==chosen['confusion']
 for v in js(R/'TABLE2.json'):
  if isinstance(v['seed'],str):
   rows=[x for x in js(R/'TABLE2.json') if x['domain']==v['domain'] and x['strategy']==v['strategy'] and isinstance(x['seed'],int) and (v['strategy']!='endpoint' or x['lr']==v['lr'])];assert len(rows)==2
   for k in KEYS:
    for kk in [k,'scene_mean_'+k]:close(math.fsum(x[kk] for x in rows)/2 if all(x[kk] is not None for x in rows) else None,v[kk])
   continue
  groups=allgroups[v['domain'],'test',ident(v)];cm=csum(groups.values());assert cm==v['confusion']
  for k in KEYS:
   close(metric(cm)[k],v[k]);vals=[metric(c)[k] for c in groups.values() if metric(c)[k] is not None];close(math.fsum(vals)/len(vals) if vals else None,v['scene_mean_'+k]);assert len(vals)==v[k+'_defined_scenes']
 for v in js(R/'L8_TEST_STRATA.json'):
  status={x['scene']:x['usgs_shadows'] for x in E['L8']['test']};groups={g:c for g,c in allgroups['L8','test',ident(v)].items() if v['stratum']=='FULL' or status[g]==v['stratum']};assert len(groups)==v['scene_count'] and csum(groups.values())==v['confusion']
  for k in KEYS:close(metric(v['confusion'])[k],v[k])
 # Independently aggregate every paired draw using integer multiplicities and
 # scalar IoU arithmetic. Same scene indices for both trained seeds.
 table3=js(R/'TABLE3.json');bootstrapchecks=0
 for domain in E:
  src=next(k for k in allgroups if k[:2]==(domain,'test') and k[2][1]=='Source');scenes=sorted(allgroups[src]);indices=np.load(R/f'BOOTSTRAP_INDICES_{domain}.npz')['indices'];assert np.array_equal(indices,np.random.default_rng(23601).integers(0,len(scenes),(10000,len(scenes))))
  def key(st,seed):return src if st=='Source' else next(k for k in allgroups if k[:2]==(domain,'test') and k[2][0]==seed and k[2][1]==st)
  pools={k:[allgroups[k][g] for g in scenes] for k in allgroups if k[:2]==(domain,'test') and k[2][1] in ['Source','E','W','P']}
  def m(cm):
   valid=[]
   for c in range(3):
    union=sum(cm[c])+sum(row[c] for row in cm)-cm[c][c]
    if union:valid.append(cm[c][c]/union)
   return math.fsum(valid)/len(valid)*100
  distributions={(a,b,s):[] for a,b in [('W','E'),('W','P'),('E','Source')] for s in [21701,21702]}
  for draw in indices:
   counts=np.bincount(draw,minlength=len(scenes)).tolist();values={}
   for k,cc in pools.items():
    cm=[[sum(counts[g]*cc[g][i][j] for g in range(len(scenes))) for j in range(3)] for i in range(3)];values[k]=m(cm)
   for (a,b,s),dist in distributions.items():dist.append(values[key(a,s)]-values[key(b,s)])
  for row in [v for v in table3 if v['domain']==domain]:
   a,b=row['comparison'].split('−');seeds=[21701,21702] if isinstance(row['seed'],str) else [row['seed']];dist=np.mean([distributions[a,b,s] for s in seeds],axis=0);ci=np.quantile(dist,[.025,.975]);assert max(abs(ci[j]-row['conditional_95_pp'][j]) for j in range(2))<1e-9
   point=math.fsum(m(csum(pools[key(a,s)]))-m(csum(pools[key(b,s)])) for s in seeds)/len(seeds);close(point,row['delta_mIoU_pp']);bootstrapchecks+=10000
 # Audit all optimizer steps, per-step RNG/input identities, saved state hashes,
 # LR ratios and immutable nonhead state receipts; never open a .pth here.
 fits={p.parent.name:js(p) for p in (D/'fits').glob('*/COMPLETE.json')};assert len(fits)==8;steps=0;losschecks=0;headchecks=0
 activation=js(D/'EVALUATION_RECOVERY03_ACTIVATION.json')['all_64_head_resume_SHA_unchanged'];initials=[]
 for name,fit in fits.items():
  p=D/'fits'/name;assert fit['updates']==fit['backwards']==4000 and not fit['forbidden_accesses'];assert fit['schedule_SHA']==sha(p/'SCHEDULE.jsonl') and fit['losses_SHA']==sha(p/'LOSSES.jsonl');initial=js(p/'INITIAL.json');initials.append(initial)
  assert initial['all_nonhead_Source_exact'] and initial['trainable']==1316516;assert [x['step'] for x in fit['nonhead_checks']]==list(range(100,4001,100))+[4000]
  assert all(v['nonhead_exact'] and v['nonhead_no_gradient'] and v['nonhead_digest']==initial['nonhead_digest'] for v in fit['nonhead_checks'])
  pool=js(p/'TRAIN_POOL.json');assert not pool['excluded_all_invalid'] and len(pool['active_indices'])==len(T[fit['domain']]);assert {v['scene'] for v in pool['census']}=={v['scene'] for v in T[fit['domain']]};assert sum(fit['scene_exposure'].values())==4000
  assert pool['config']['optim_wrapper']['optimizer']['lr']==fit['lr'];assert pool['config']['train_pipeline'][2]['type']=='RandomCrop';assert [v['type'] for v in pool['config']['param_scheduler']]==['LinearLR','PolyLR']
  schedule=[json.loads(line) for line in (p/'SCHEDULE.jsonl').read_text().splitlines()];losses=[json.loads(line) for line in (p/'LOSSES.jsonl').read_text().splitlines()];assert len(schedule)==len(losses)==4000
  for i,(data,loss) in enumerate(zip(schedule,losses),1):
   assert data['step']==loss['step']==i and data['data_seed']==fit['seed']*10000+i-1;assert math.isfinite(loss['loss']) and all(math.isfinite(v) for v in loss['terms'].values());assert all(math.isfinite(v) and v>=0 for v in loss['lr_used']);losschecks+=1
  assert losses[0]['lr_used'][0]==fit['lr']*.1 and losses[-1]['lr_used'][0]==0
  if fit['lr']==3e-5:
   partner=D/'fits'/f'{fit["domain"]}_{fit["seed"]}_1e-04';assert sha(p/'SCHEDULE.jsonl')==sha(partner/'SCHEDULE.jsonl');other=[json.loads(line) for line in (partner/'LOSSES.jsonl').read_text().splitlines()]
   assert all(len(a['lr_used'])==len(b['lr_used']) and max(abs(x-.3*y) for x,y in zip(a['lr_used'],b['lr_used']))<1e-12 for a,b in zip(losses,other))
  for step,rec in fit['heads'].items():
   for kind,filekey in [('head','head_SHA'),('resume','resume_SHA')]:assert activation[f'fits/{name}/{kind}_{step}.pth']==rec[filekey];headchecks+=1
   assert lock['model_SHA'][f'fits/{name}/head_{step}.pth']==rec['head_SHA']
  steps+=4000
 assert len({v['head_digest'] for v in initials})==len({v['nonhead_digest'] for v in initials})==1
 with (D/'CALL_LEDGER.jsonl').open() as f:ledger=[json.loads(line) for line in f]
 assert [v['ordinal'] for v in ledger]==list(range(1,len(ledger)+1));assert len(ledger)==C['calls']==75520;assert dict(collections.Counter(v['kind'] for v in ledger))==C['counts'];assert C['counts']['training_full_input']==steps+64
 costs=[js(p) for p in (D/'stage_costs').glob('*.json')];close(math.fsum(v['GPU_stage_seconds'] for v in costs),C['GPU_stage_seconds']);assert C['GPU_stage_seconds']<21600 and C['updates']==steps+63==32063
 for domain in E:
  for stage in ['val','test']:
   receipt=js(D/f'{stage.upper()}_COMPLETE.json')['pools'][domain];assert receipt['networks_per_input']<=(29 if stage=='val' else 9);assert receipt['inputs']==len(E[domain][stage])
 assert sum(len(E[d][s])*js(D/f'{s.upper()}_COMPLETE.json')['pools'][d]['networks_per_input'] for d in E for s in ['val','test'])==C['counts']['evaluation_full_input']
 for cost in costs:
  for v in cost['resources']:assert v['CPU_peak_bytes']<16*1024**3 and max(v['GPU_peak_allocated'],v['GPU_peak_reserved'])<20*1024**3 and v['remote_bytes']<8*1024**3
 assert transfer['remote_total_bytes']<8*1024**3 and sum(v['bytes'] for v in transfer['manifest'].values())<2*1024**3 and cpu_peak()<16*1024**3 and time.monotonic()-start<14400
 save('INDEPENDENT_ACCEPTANCE.json',{'status':'passed','per_input_count_checks':perchecks,'Source_prediction_SHA_and_cm_replays234':sourcechecks,'endpoint_and_Source_alias_checks':aliaschecks,'selections_replayed':12,'scalar_same_index_bootstrap_checks':bootstrapchecks,'complete_training_updates':steps,'failed_training_updates':63,'total_training_updates':32063,'loss_LR_steps_checked':losschecks,'head_resume_SHA_checks':headchecks,'nonhead_and_LR_pair_schedules_verified':True,'exact_call_ledger':len(ledger),'all_failed_calls_and_stage_costs_retained':True,'CPU_seconds':time.monotonic()-start,'CPU_peak_bytes':cpu_peak()});print('Phase236 independent CPU acceptance passed')
if __name__=='__main__':main()
