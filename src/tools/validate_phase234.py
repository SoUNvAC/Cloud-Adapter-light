"""Independent old identity counts, support aggregation, selection and same-index CI replay."""
from analyze_phase234 import R,D,js,save,sha,START,SEEDS
from validate_phase233 import scalar,eq
from analyze_phase230 import cpu_peak
import collections,json,math,time
import numpy as np
def main():
 assert js(R/'CPU_ANALYSIS_COMPLETE.json')['status']=='passed';index=js(R/'EXPECTED_INDEX.json');per=js(R/'PER_SCENE.json');grid=js(R/'TABLE1.json');summaries={};checked=0;sha_missing=0;E=js(R/'EVAL_INPUTS.json');native={(v['split'],v['name']):v for v in js(D/'NATIVE_REFERENCES.json')}
 for domain in ['L8','SPARCS']:
  for split in ['val','test']:
   if not (D/f'{domain}_{split}_SUMMARY.json').exists():continue
   expected={k.rsplit('/',1)[1]:{v['name']:v for v in js(R/p)['per_input']} for k,p in index.items() if k.startswith(domain+'/'+split+'/')};groups={};byinput={};inputnames=set()
   with (D/f'evaluation/{domain}_{split}.jsonl').open() as f:
    for line in f:
     v=json.loads(line);inputnames.add(v['name']);k=(v['seed'],v['strategy'],float(v['alpha']));byinput.setdefault(v['name'],{})[k]=v
     if v['strategy']=='W':
      key='Source' if v['alpha']==0 else f"{v['seed']}_{v['alpha']:g}";ex=expected[key][v['name']];assert v['OLD']==ex['confusion'];h=ex.get('prediction_SHA',ex.get('prediction_sha256'))
      if h:assert v['prediction_SHA']==h
      else:sha_missing+=1
     if domain=='L8':
      ref=native[split,v['name']];assert sum(v['Fill_prediction_histogram'])==ref['Fill_pixels'];assert sum(map(sum,v['VALID']))==ref['valid_pixels'];assert ref['old_PNG_exact']
      for i in range(3):
       for j in range(3):assert v['OLD'][i][j]-v['VALID'][i][j]==(v['Fill_prediction_histogram'][j] if i==0 else 0)
     else:assert v['OLD']==v['VALID']
     for support in (['OLD','VALID'] if domain=='L8' else ['ORIGINAL']):
      cm=v['VALID' if support=='ORIGINAL' else support];key=(*k,support,v['scene']);groups.setdefault(key,[[0]*3 for i in range(3)])
      for i in range(3):
       for j in range(3):groups[key][i][j]+=cm[i][j]
     checked+=1
   assert inputnames=={v['name'] for v in E[domain][split]}
   for name,rows in byinput.items():
    assert len(rows)==20
    for seed in SEEDS:
     for alpha in [0.,1.]:
      a,b=rows[seed,'W',alpha],rows[seed,'P',alpha];assert a['prediction_SHA']==b['prediction_SHA'] and a['OLD']==b['OLD'] and a['VALID']==b['VALID']
    assert rows[21701,'W',0.]['prediction_SHA']==rows[21702,'W',0.]['prediction_SHA']
   for (seed,st,alpha,support,g),cm in groups.items():
    row=next(v for v in per if (v['domain'],v['split'],v['seed'],v['strategy'],float(v['alpha']),v['support'],v['scene'])==(domain,split,seed,st,alpha,support,g));assert row['confusion']==cm
    for k,v in scalar(cm).items():
     if isinstance(v,list):
      for a,b in zip(v,row[k]):eq(a,b)
     else:eq(v,row[k])
    summaries.setdefault((domain,split,seed,st,alpha,support),{})[g]=cm
   del byinput
 for row in grid:
  for split in ['val','test']:
   if row[split] is None:continue
   gs=summaries[row['domain'],split,row['seed'],row['strategy'],float(row['alpha']),row['support']];c=[[sum(v[i][j] for v in gs.values()) for j in range(3)] for i in range(3)];assert c==row[split]['confusion'];m=scalar(c)
   for k,v in m.items():
    if isinstance(v,list):
     for a,b in zip(v,row[split]['pooled'][k]):eq(a,b)
    else:eq(v,row[split]['pooled'][k])
   for k in ['mIoU','shadow_recall','nonshadow_FPR']:
    vals=[scalar(v)[k] for v in gs.values() if scalar(v)[k] is not None];eq(math.fsum(vals)/len(vals) if vals else None,row[split]['scene_mean_'+k])
 selection=js(D/'VAL_SELECTION_LOCK.json');assert sha(D/'VAL_SELECTION_LOCK.json')==js(D/'VAL_SELECTION_SEAL.json')['SHA']
 for v in selection['selections']:
  support='VALID' if v['domain']=='L8' else 'ORIGINAL';candidates=[]
  for alpha in [0.,.25,.5,.75,1.]:
   gs=summaries[v['domain'],'val',v['seed'],v['strategy'],alpha,support];cm=np.array(list(gs.values()),np.int64).sum(0);d=np.diag(cm);u=cm.sum(0)+cm.sum(1)-d;candidates.append((float(np.mean(d[u>0]/u[u>0])*100),alpha))
  best=max(candidates,key=lambda x:(x[0],-x[1]));assert best[1]==v['selected_alpha'] and best[0]==v['val_mIoU']
 bootstrapchecks=0;chosen=js(R/'TABLE2.json')
 for interval in js(R/'BOOTSTRAP_COMPARISONS.json'):
  domain,support=interval['domain'],interval['support'];gs=interval['scene_ids'];idx=np.load(R/f'BOOTSTRAP_INDICES_{domain}.npz')['indices'];assert np.array_equal(idx,np.random.default_rng(23401).integers(0,len(gs),(10000,len(gs))));comparison=interval['comparison'].replace('W_minus_','');dist=[]
  for draw in idx:
   values=[]
   for seed in SEEDS:
    row=next(v for v in chosen if v['domain']==domain and v['support']==support and v['seed']==seed);ak=(domain,'test',seed,'W',float(row['W_alpha']),support);bst,ba=('P',row['P_alpha']) if comparison=='P' else ('W',0 if comparison=='Source' else 1);bk=(domain,'test',seed,bst,float(ba),support)
    def pool(key):return [[sum(summaries[key][gs[k]][i][j] for k in draw) for j in range(3)] for i in range(3)]
    values.append(scalar(pool(ak))['mIoU']-scalar(pool(bk))['mIoU'])
   dist.append(math.fsum(values)/2);bootstrapchecks+=1
  ci=np.quantile(dist,[.025,.975]);assert np.max(np.abs(ci-interval['descriptive_95_pp']))<1e-10
 for v in js(R/'OUTPUT_WEIGHT_DIFFERENCES.json'):
  if v['alpha'] in [0,1]:assert all(x==0 for x in v['score_max_abs_difference_by_class']) and v['argmax_disagreement_percent']==0
 complete=js(D/'COMPLETE.json');ledger=[json.loads(line) for line in (D/'CALL_LEDGER.jsonl').read_text().splitlines()];assert len(ledger)==complete['total_calls'];assert [v['ordinal'] for v in ledger]==list(range(1,len(ledger)+1));assert dict(collections.Counter(v['kind'] for v in ledger))==complete['counts']
 if complete['status']=='complete':
  calls=collections.Counter(v['current'] for v in ledger if v['kind']=='main_complete_input');assert len(calls)==3150 and set(calls.values())=={9};assert complete['counts']['main_complete_input']==28350
 for row in js(R/'TABLE3_L8_STRATA.json'):
  gs=summaries['L8',row['split'],row['seed'],row['strategy'],float(row['alpha']),row['support']];subset={g:c for g,c in gs.items() if row['stratum']=='FULL' or next(v['usgs_shadows'] for v in E['L8'][row['split']] if v['scene']==g)==row['stratum']};assert len(subset)==row['scenes'];cm=[[sum(c[i][j] for c in subset.values()) for j in range(3)] for i in range(3)];assert cm==row['confusion']
  for k in ['mIoU','shadow_recall','nonshadow_FPR']:eq(scalar(cm)[k],row['pooled'][k])
 assert cpu_peak()<16*1024**3 and time.monotonic()-START<14400;save('INDEPENDENT_ACCEPTANCE.json',{'status':'passed','raw_grid_prediction_records_checked':checked,'alias_SHA_unavailable_confusion_only_rows':sha_missing,'selection_replays':8,'same_index_scalar_bootstrap_checks':bootstrapchecks,'endpoints_W_P_identical':True,'call_ledger_conservation':True,'L8_strata_membership_and_counts':True,'native_support_and_fill_histogram_count_checks':True,'CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak()});print('independent234 acceptance passed')
if __name__=='__main__':main()
