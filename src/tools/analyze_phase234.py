"""Fixed val-selected clean grid and output-mix evaluation from counts only."""
from pathlib import Path
import collections,csv,hashlib,json,time
import numpy as np
from analyze_phase229 import metrics,miou_batch
from analyze_phase230 import cpu_peak
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase234/phase234_20261027';D=R/'delivery';START=time.monotonic();ALPHAS=[0,.25,.5,.75,1];SEEDS=[21701,21702]
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def output(n,rows):
 save(n+'.json',rows);keys=list(dict.fromkeys(k for v in rows for k in v))
 with (R/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for v in rows:w.writerow({k:json.dumps(x) if isinstance(x,(list,dict)) else x for k,x in v.items()})
def summarized(groups):
 c=sum(groups.values());m=metrics(c);row={'pooled':m,'confusion':c.tolist()}
 for key in ['mIoU','shadow_recall','nonshadow_FPR']:
  vals=[metrics(c)[key] for c in groups.values() if metrics(c)[key] is not None];row['scene_mean_'+key]=float(np.mean(vals)) if vals else None;row[key+'_defined_scenes']=len(vals)
 return row
def main():
 assert not (R/'CPU_ANALYSIS_COMPLETE.json').exists();complete=js(D/'COMPLETE.json');assert complete['total_calls']<=30000 and complete['GPU_stage_seconds']<7200
 selection=js(D/'VAL_SELECTION_LOCK.json');assert sha(D/'VAL_SELECTION_LOCK.json')==js(D/'VAL_SELECTION_SEAL.json')['SHA'];assert selection['test_native_decoded_before_seal'] is False and selection['test_performance_results_read_before_seal'] is False;assert js(D/'TEST_ACCESS_STARTED.json')['epoch']>selection['epoch'];assert js(D/'TEST_ACCESS_STARTED.json')['selection_SHA']==sha(D/'VAL_SELECTION_LOCK.json')
 E=js(R/'EVAL_INPUTS.json');meta={(d,s,v['name']):v for d,sets in E.items() for s,rows in sets.items() for v in rows};data={};scene_rows=[];grid=[];table2=[];strata=[];boot=[];streams=[];selection_replay=[];observed={}
 fields=['domain','split','scene','name','usgs_shadows','seed','strategy','alpha','support','prediction_SHA','confusion','valid_pixels','mIoU','class_IoU','valid_classes','shadow_recall','nonshadow_FPR','positive_shadow_pixels','nonshadow_pixels','Fill_prediction_histogram']
 f=(R/'PER_INPUT.csv').open('x',encoding='utf-8-sig',newline='');writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();count=0
 for domain in ['L8','SPARCS']:
  for split in ['val','test']:
   path=D/f'evaluation/{domain}_{split}.jsonl';summary=D/f'{domain}_{split}_SUMMARY.json'
   if not summary.exists():continue
   assert sha(path)==js(summary)['per_input_SHA'];names=collections.defaultdict(set)
   with path.open() as inp:
    for line in inp:
     v=json.loads(line);ex=meta[domain,split,v['name']];assert v['scene']==ex['scene'];key=(domain,split,v['seed'],v['strategy'],float(v['alpha']));names[key].add(v['name'])
     for support in (['OLD','VALID'] if domain=='L8' else ['ORIGINAL']):
      cm=np.array(v['VALID' if support=='ORIGINAL' else support],np.int64);assert cm.shape==(3,3) and (cm>=0).all();data.setdefault((*key,support),{});gs=data[*key,support];gs.setdefault(v['scene'],np.zeros((3,3),np.int64));gs[v['scene']]+=cm;m=metrics(cm);row={k:v[k] for k in ['domain','split','scene','name','usgs_shadows','seed','strategy','alpha','prediction_SHA','Fill_prediction_histogram']};row.update(support=support,confusion=cm.tolist(),**m);writer.writerow({k:json.dumps(x) if isinstance(x,(dict,list)) else x for k,x in row.items()});count+=1
     if domain=='L8':a=np.array(v['OLD'])-np.array(v['VALID']);assert np.array_equal(a[0],v['Fill_prediction_histogram']) and not a[1:].any()
   assert len(names)==20 and all(len(v)==len(E[domain][split]) for v in names.values());streams.append({'domain':domain,'split':split,'SHA':sha(path),'inputs':len(E[domain][split]),'grid_records':sum(len(v) for v in names.values())})
 f.close()
 for key,gs in data.items():
  domain,split,seed,st,a,support=key;s=summarized(gs);observed[key]=s
  for g,c in sorted(gs.items()):scene_rows.append({'domain':domain,'split':split,'seed':seed,'strategy':st,'alpha':a,'support':support,'scene':g,'usgs_shadows':next(v.get('usgs_shadows','NA') for v in E[domain][split] if v['scene']==g),'confusion':c.tolist(),**metrics(c)})
 for sel in selection['selections']:
  d,seed,st=sel['domain'],sel['seed'],sel['strategy'];support='VALID' if d=='L8' else 'ORIGINAL';rows=[]
  for a in ALPHAS:
   c=np.asarray(observed[d,'val',seed,st,float(a),support]['confusion'],np.int64);diag=np.diag(c);union=c.sum(0)+c.sum(1)-diag;rows.append((a,float(np.mean(diag[union>0]/union[union>0])*100)))
  best=sorted(rows,key=lambda v:(-v[1],v[0]))[0];assert best[0]==sel['selected_alpha'] and best[1]==sel['val_mIoU'];selection_replay.append({'domain':d,'seed':seed,'strategy':st,'selected_alpha':best[0],'same':True})
 for domain in ['L8','SPARCS']:
  for support in (['OLD','VALID'] if domain=='L8' else ['ORIGINAL']):
   for seed in SEEDS:
    for st in ['W','P']:
     for a in ALPHAS:
      v=observed[domain,'val',seed,st,float(a),support];test=observed.get((domain,'test',seed,st,float(a),support));base=observed.get((domain,'test',seed,'W',0.,support));endpoint=observed.get((domain,'test',seed,'W',1.,support));grid.append({'domain':domain,'support':support,'seed':seed,'strategy':st,'alpha':a,'val':v,'test':test,'test_minus_Source_mIoU_pp':test['pooled']['mIoU']-base['pooled']['mIoU'] if test else None,'test_minus_endpoint_mIoU_pp':test['pooled']['mIoU']-endpoint['pooled']['mIoU'] if test else None})
   for seed in SEEDS:
    picks={st:next(v['selected_alpha'] for v in selection['selections'] if v['domain']==domain and v['seed']==seed and v['strategy']==st) for st in ['W','P']};k=lambda st,a:(domain,'test',seed,st,float(a),support)
    if k('W',picks['W']) not in observed:continue
    get=lambda st,a:observed[k(st,a)]['pooled']['mIoU'];wv,pv,src,end=get('W',picks['W']),get('P',picks['P']),get('W',0),get('W',1);table2.append({'domain':domain,'support':support,'seed':seed,'W_alpha':picks['W'],'P_alpha':picks['P'],'W_test_mIoU':wv,'P_test_mIoU':pv,'Source_test_mIoU':src,'endpoint_test_mIoU':end,'W_minus_P_pp':wv-pv,'W_minus_Source_pp':wv-src,'W_minus_endpoint_pp':wv-end,'P_minus_Source_pp':pv-src,'P_minus_endpoint_pp':pv-end})
   rows=[v for v in table2 if v['domain']==domain and v['support']==support]
   if len(rows)==2:table2.append({'domain':domain,'support':support,'seed':'two_seed_metric_mean',**{k:float(np.mean([v[k] for v in rows])) for k in ['W_test_mIoU','P_test_mIoU','Source_test_mIoU','endpoint_test_mIoU','W_minus_P_pp','W_minus_Source_pp','W_minus_endpoint_pp','P_minus_Source_pp','P_minus_endpoint_pp']}})
  main_support='VALID' if domain=='L8' else 'ORIGINAL';rows=[v for v in table2 if v['domain']==domain and v['support']==main_support and isinstance(v['seed'],int)]
  if len(rows)==2:
   scenes=sorted(data[domain,'test',21701,'W',float(rows[0]['W_alpha']),main_support]);indices=np.random.default_rng(23401).integers(0,len(scenes),(10000,len(scenes)));np.savez_compressed(R/f'BOOTSTRAP_INDICES_{domain}.npz',indices=indices)
   for comparison in ['P','Source','endpoint']:
    diffs=[]
    for row in rows:
     seed=row['seed'];a=np.array([data[domain,'test',seed,'W',float(row['W_alpha']),main_support][g] for g in scenes]);st,alpha=('P',row['P_alpha']) if comparison=='P' else ('W',0 if comparison=='Source' else 1);b=np.array([data[domain,'test',seed,st,float(alpha),main_support][g] for g in scenes]);diffs.append(miou_batch(a[indices].sum(1))-miou_batch(b[indices].sum(1)))
    ci=np.quantile(np.mean(diffs,axis=0),[.025,.975]).tolist();boot.append({'domain':domain,'support':main_support,'comparison':'W_minus_'+comparison,'two_seed_mean_delta_pp':float(np.mean([r['W_minus_'+comparison+'_pp'] for r in rows])),'descriptive_95_pp':ci,'scene_ids':scenes,'draws':10000,'seed':23401,'same_indices_both_seeds':True,'alpha_fixed_not_reselected':True,'conditional_on_selected_models_not_selection_randomness':True})
 for key,gs in data.items():
  domain,split,seed,st,a,support=key
  if domain!='L8':continue
  statuses={g:next(v['usgs_shadows'] for v in E['L8'][split] if v['scene']==g) for g in gs}
  if split=='test':assert len(gs)==16 and list(statuses.values()).count('yes')==6 and list(statuses.values()).count('no')==10
  selected=next(v['selected_alpha'] for v in selection['selections'] if v['domain']=='L8' and v['seed']==seed and v['strategy']==st)==a
  for stratum in ['FULL','yes','no']:
   subset={g:c for g,c in gs.items() if stratum=='FULL' or statuses[g]==stratum}
   if subset:strata.append({'split':split,'seed':seed,'strategy':st,'alpha':a,'support':support,'stratum':stratum,'scenes':len(subset),'selected_by_VALID_val':selected,**summarized(subset)})
 gaps=js(D/'SCORE_DIFFERENCES.json');agg={}
 for v in gaps:
  key=(v['domain'],v['split'],v['seed'],v['alpha']);a=agg.setdefault(key,{'sum':np.zeros(3),'max':np.zeros(3),'pixels':0,'disagree':0});a['sum']+=v['score_abs_sum_by_class'];a['max']=np.maximum(a['max'],[x or 0 for x in v['score_abs_max_by_class']]);a['pixels']+=v['valid_pixels'];a['disagree']+=v['argmax_disagree_pixels']
 diffsummary=[{'domain':d,'split':s,'seed':seed,'alpha':alpha,'valid_pixels':v['pixels'],'score_mean_abs_difference_by_class':(v['sum']/v['pixels']).tolist() if v['pixels'] else [None]*3,'score_max_abs_difference_by_class':v['max'].tolist(),'argmax_disagreement_percent':100*v['disagree']/v['pixels'] if v['pixels'] else None} for (d,s,seed,alpha),v in agg.items()]
 for row in grid:
  for split in ['val','test']:
   item=row[split]
   for key in ['mIoU','class_IoU','shadow_recall','nonshadow_FPR','valid_pixels','valid_classes']:
    row[split+'_'+key]=item['pooled'][key] if item else None
   row[split+'_scene_mean_mIoU']=item['scene_mean_mIoU'] if item else None
   row[split+'_scene_mean_shadow_recall']=item['scene_mean_shadow_recall'] if item else None
   row[split+'_scene_mean_nonshadow_FPR']=item['scene_mean_nonshadow_FPR'] if item else None
 for n,v in [('TABLE1',grid),('TABLE2',table2),('TABLE3_L8_STRATA',strata),('PER_SCENE',scene_rows),('BOOTSTRAP_COMPARISONS',boot),('OUTPUT_WEIGHT_DIFFERENCES',diffsummary),('SELECTION_REPLAY',selection_replay)]:output(n,v)
 save('INPUTS_USED.json',streams);save('CPU_ANALYSIS_COMPLETE.json',{'status':'passed','partial_GPU_delivery':complete['status']!='complete','CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'per_input_support_rows':count,'selection_replayed_all8':len(selection_replay)==8,'new_forward_fit_or_parameter_search':0});assert cpu_peak()<16*1024**3 and time.monotonic()-START<14400;print(json.dumps({'selected':table2,'intervals':boot}))
if __name__=='__main__':main()
