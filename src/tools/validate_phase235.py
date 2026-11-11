"""Independent count/metric/budget and same-scene-index interval verification."""
from analyze_phase235 import R,D,OLD,js,save,sha,KEYS,START
from validate_phase233 import scalar,eq
from validate_phase232 import independent_metrics
from analyze_phase230 import cpu_peak
import collections,json,time,math
import numpy as np
def m(c):
 return {**scalar(c),**independent_metrics(c)}
def main():
 assert js(R/'CPU_ANALYSIS_COMPLETE.json')['status']=='passed';F=js(D/'BIAS_FROZEN.json');assert sha(D/'BIAS_FROZEN.json')==js(D/'BIAS_SEAL.json')['SHA'];assert js(D/'INDEPENDENT_BIAS_ACCEPTANCE.json')['status']=='passed';assert js(D/'INDEPENDENT_BIAS_ACCEPTANCE.json')['absolute_difference']<1e-10;assert F['new_scalar_fits']==1 and F['biases']['SPARCS']['b']==1.5086972783056742
 expected=js(R/'EXPECTED_RAW234.json');b3=js(R/'EXPECTED_B3_232.json');raw={};sums={};checks=0
 with (D/'EVALUATION.jsonl').open() as f:
  for line in f:
   v=json.loads(line);key=(v['domain'],v['split'],v['scene'],'Source' if v['decision']=='Raw' else 'Source+B','NA');cm=v['confusion'];sums.setdefault(key,[[0]*3 for _ in range(3)])
   for i in range(3):
    for j in range(3):sums[key][i][j]+=cm[i][j]
   ex=expected[v['domain']][v['split']][v['name']];assert [sum(r) for r in cm]==[sum(r) for r in ex['VALID']]
   if v['decision']=='Raw':assert cm==ex['VALID'] and v['prediction_SHA']==ex['prediction_SHA']
   elif v['domain']=='SPARCS' and v['split']=='val':assert cm==b3[v['scene']]['confusion'] and v['prediction_SHA']==b3[v['scene']]['prediction_SHA']
   checks+=1
 assert checks==6300
 prior=js(OLD/'PER_SCENE.json');selection=js(OLD/'delivery/VAL_SELECTION_LOCK.json')
 for row in js(R/'PER_SCENE.json'):
  key=row['domain'],row['split'],row['scene'],row['strategy'],row['seed'];cm=row['confusion']
  if row['strategy'] in ['Source','Source+B']:assert sums[key]==cm
  else:
   st='W' if row['strategy']=='C' else row['strategy'];a=1 if row['strategy']=='C' else next(v['selected_alpha'] for v in selection['selections'] if v['domain']==row['domain'] and v['seed']==row['seed'] and v['strategy']==st);support='VALID' if row['domain']=='L8' else 'ORIGINAL';ex=next(v for v in prior if (v['domain'],v['split'],v['scene'],v['strategy'],v['seed'],v['alpha'],v['support'])==(row['domain'],row['split'],row['scene'],st,row['seed'],a,support));assert ex['confusion']==cm;sums[key]=cm
  for k in KEYS:eq(m(cm)[k],row[k])
  for a,b in zip(m(cm)['class_IoU'],row['class_IoU']):eq(a,b)
 for row in js(R/'TABLE2.json'):
  cs=[c for (d,s,g,st,seed),c in sums.items() if (d,s,st,seed)==(row['domain'],row['split'],row['strategy'],row['seed'])];cm=[[sum(c[i][j] for c in cs) for j in range(3)] for i in range(3)];assert cm==row['confusion']
  for k in KEYS:
   eq(m(cm)[k],row[k]);vals=[m(c)[k] for c in cs if m(c)[k] is not None];eq(math.fsum(vals)/len(vals) if vals else None,row['scene_mean_'+k]);assert len(vals)==row[k+'_defined_scenes']
 for row in js(R/'P_COMPARISON_POINTS.json'):
  d=row['domain'];seeds=[21701,21702] if row['reference_seed']=='two_seed_difference_mean' else [row['reference_seed']];scenes=sorted({k[2] for k in sums if k[0]==d and k[1]=='test'})
  def pooled(strategy,seed):return [[sum(sums[d,'test',g,strategy,seed][i][j] for g in scenes) for j in range(3)] for i in range(3)]
  point=math.fsum(m(pooled('Source+B','NA'))['mIoU']-m(pooled('P',seed))['mIoU'] for seed in seeds)/len(seeds);eq(point,row['delta_pooled_mIoU_pp'])
 E=js(R/'EVAL_INPUTS.json');table=js(R/'TABLE3.json');nboot=0
 for domain in E:
  scenes=sorted({v['scene'] for v in E[domain]['test']});indices=np.load(R/f'BOOTSTRAP_INDICES_{domain}.npz')['indices'];assert np.array_equal(indices,np.random.default_rng(23501).integers(0,len(scenes),(10000,len(scenes))))
  for row in [v for v in table if v['domain']==domain]:
   st=row['comparison'].replace('Source+B_minus_','');seed=row['reference_seed'];seeds=[21701,21702] if seed=='two_seed_difference_mean' else [seed];dist=[]
   for draw in indices:
    def pool(strategy,seed):return [[sum(sums[domain,'test',scenes[k],strategy,seed][i][j] for k in draw) for j in range(3)] for i in range(3)]
    b=m(pool('Source+B','NA'))['mIoU'];dist.append(math.fsum(b-m(pool(st,s))['mIoU'] for s in seeds)/len(seeds));nboot+=1
   ci=np.quantile(dist,[.025,.975]);assert np.max(np.abs(ci-row['descriptive_95_pp']))<1e-10
   def full(strategy,seed):return [[sum(sums[domain,'test',g,strategy,seed][i][j] for g in scenes) for j in range(3)] for i in range(3)]
   point=math.fsum(m(full('Source+B','NA'))['mIoU']-m(full(st,s))['mIoU'] for s in seeds)/len(seeds);eq(point,row['delta_pooled_mIoU_pp'])
 ledger=[json.loads(x) for x in (D/'CALL_LEDGER.jsonl').read_text().splitlines()];C=js(D/'COMPLETE.json');assert len(ledger)==C['total_calls'] and [v['ordinal'] for v in ledger]==list(range(1,len(ledger)+1));assert dict(collections.Counter(v['kind'] for v in ledger))==C['counts'];assert C['counts']['train_input']==1560 and C['counts']['evaluation_input']==3150 and C['total_calls']<=4800;assert C['GPU_stage_seconds']==sum(js(D/f'GPU_{s}_COMPLETE.json')['GPU_stage_seconds'] for s in ['extract','evaluate'])
 train=js(D/'TRAIN_SCENE_SUPPORT.json');cache=js(D/'CACHE_RECEIPTS.json');binding=js(D/'TRAIN_NATIVE_BINDING.json');assert len(train)==18 and len(cache)==len(binding)==1560
 for scene in train:
  rows=[v for v in cache if v['scene']==scene['scene']];assert len(rows)==scene['inputs'] and sum(v['valid_pixels'] for v in rows)==scene['valid_pixels'] and sum(v['Fill_pixels'] for v in rows)==scene['Fill_pixels'];assert [sum(v['support'][k] for v in rows) for k in range(3)]==scene['support']
 for row in js(R/'L8_STRATA.json'):
  gs=[g for g in {v['scene'] for v in E['L8'][row['split']]} if row['stratum']=='FULL' or next(v['usgs_shadows'] for v in E['L8'][row['split']] if v['scene']==g)==row['stratum']];assert len(gs)==row['scenes'];cm=[[sum(sums['L8',row['split'],g,row['strategy'],row['seed']][i][j] for g in gs) for j in range(3)] for i in range(3)];assert cm==row['confusion']
  for k in KEYS:eq(m(cm)[k],row['pooled'][k])
 assert cpu_peak()<16*1024**3;save('INDEPENDENT_ACCEPTANCE.json',{'status':'passed','per_input_records':checks,'all_Raw234_SHA_counts_exact':True,'SPval_B3_232_exact':True,'scene_scalar_metrics':True,'same_index_scalar_bootstrap_checks':nboot,'independent_bias_receipt_verified':True,'ledger_and_scene_support_reconciled':True,'SourceB_replicates':1,'CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak()});print('independent235 CPU acceptance passed')
if __name__=='__main__':main()
