"""L8-only paired scene Fill-support sensitivity, without fitting."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
from analyze_phase229 import metrics,miou_batch
from analyze_phase230 import cpu_peak
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase233/phase233_20261027';D=R/'delivery';OLD=REPO/'outputs/phase229/phase229_20261026';START=time.monotonic();NAMES=['Source','L8_21701','L8_21702','SPARCS_21701','SPARCS_21702']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def output(n,rows):
 save(n+'.json',rows);keys=list(dict.fromkeys(k for v in rows for k in v))
 with (R/(n+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for v in rows:w.writerow({k:json.dumps(x) if isinstance(x,(dict,list)) else x for k,x in v.items()})
def main():
 assert not (R/'CPU_ANALYSIS_COMPLETE.json').exists() and (D/'EXIT_CODE').read_text().strip()=='0';c=js(D/'COMPLETE.json');assert c['counts']['matrix_complete_input']==4815 and c['total_calls']<=5200 and c['GPU_stage_seconds']<3600
 for n,h in js(R/'LOCK.json')['identities'].items():assert sha(REPO/n)==h,n
 ref=js(D/'REFERENCE_ACCEPTED.json');assert ref['Fill_pixels']==33961651 and ref['OLD_pixels']==252444672
 data={};inputs=[];scenes=[];t1=[];t2=[];t3=[];diff=[];used=[];evals={};expectedrows=js(R/'L8_VAL_INPUTS.json')
 for model in NAMES:
  v=js(D/f'evaluation/{model}.json');evals[model]=v;old={x['name']:x for x in js(OLD/f'delivery/evaluation/L8_{model}.json')['per_input']};assert [(x['name'],x['scene']) for x in v['per_input']]==[(x['name'],x['scene']) for x in expectedrows];groups={'OLD':{},'VALID':{}};fills={}
  for x,ex in zip(v['per_input'],expectedrows):
   a=np.array(x['OLD'],np.int64);b=np.array(x['VALID'],np.int64);h=np.array(x['Fill_prediction_histogram'],np.int64);assert x['OLD']==old[x['name']]['confusion'] and x['prediction_SHA']==old[x['name']]['prediction_SHA'];assert (a>=0).all() and (b>=0).all() and np.array_equal((a-b)[0],h) and not (a-b)[1:].any() and h.sum()==x['Fill_pixels'];assert x['image_SHA']==ex['image_sha256'] and x['mask_SHA']==ex['mask_sha256'];g=x['scene'];fills.setdefault(g,np.zeros(3,np.int64));fills[g]+=h
   for mode,cm in [('OLD',a),('VALID',b)]:groups[mode].setdefault(g,np.zeros((3,3),np.int64));groups[mode][g]+=cm;inputs.append({'model':model,'support':mode,'scene':g,'name':x['name'],'Fill_pixels':x['Fill_pixels'],'Fill_prediction_histogram':h.tolist(),'confusion':cm.tolist(),**metrics(cm)})
  for mode,gs in groups.items():
   assert len(gs)==8;data[model,mode]=gs;total=sum(gs.values());assert int(total.sum())==(252444672 if mode=='OLD' else 218483021)
   for g,cm in sorted(gs.items()):assert cm.tolist()==v['per_scene'][g][mode];scenes.append({'model':model,'support':mode,'scene':g,'Fill_pixels':int(fills[g].sum()),'Fill_prediction_histogram':fills[g].tolist(),'confusion':cm.tolist(),**metrics(cm)})
   ms=[metrics(cm) for cm in gs.values()];row={'model':model,'support':mode,**metrics(total),'confusion':total.tolist()}
   for k in ['mIoU','shadow_recall','nonshadow_FPR']:
    vals=[x[k] for x in ms if x[k] is not None];row['scene_mean_'+k]=float(np.mean(vals)) if vals else None;row[k+'_defined_scenes']=len(vals)
   t1.append(row)
  used.append({'model':model,'evaluation_SHA':sha(D/f'evaluation/{model}.json'),'old_evaluation_SHA':sha(OLD/f'delivery/evaluation/L8_{model}.json')})
 gs=sorted(data['Source','OLD']);idx=np.random.default_rng(22901).integers(0,8,size=(10000,8));np.savez_compressed(R/'BOOTSTRAP_FIXED_INDICES.npz',scene_indices=idx);boot=[]
 for label,a_domain,b_domain in [('SPARCS_minus_L8','SPARCS','L8'),('SPARCS_minus_Source','SPARCS','Source'),('L8_minus_Source','L8','Source')]:
  seedrows=[];boots=[]
  for seed in [21701,21702]:
   am=f'{a_domain}_{seed}';bm='Source' if b_domain=='Source' else f'{b_domain}_{seed}';vals={}
   for mode in ['OLD','VALID']:
    a,b=data[am,mode],data[bm,mode];vals[mode]=metrics(sum(a.values()))['mIoU']-metrics(sum(b.values()))['mIoU']
    for g in gs:diff.append({'comparison':label,'seed':seed,'support':mode,'scene':g,'delta_mIoU_pp':metrics(a[g])['mIoU']-metrics(b[g])['mIoU']})
   row={'comparison':label,'seed':seed,'OLD_delta_mIoU_pp':vals['OLD'],'VALID_delta_mIoU_pp':vals['VALID'],'delta_change_VALID_minus_OLD_pp':vals['VALID']-vals['OLD']};t2.append(row);seedrows.append(row);a=np.array([data[am,'VALID'][g] for g in gs]);b=np.array([data[bm,'VALID'][g] for g in gs]);boots.append(miou_batch(a[idx].sum(1))-miou_batch(b[idx].sum(1)))
  ci=np.quantile(np.mean(boots,axis=0),[.025,.975]).tolist();mean={'comparison':label,'seed':'two_seed_metric_difference_mean',**{k:float(np.mean([v[k] for v in seedrows])) for k in ['OLD_delta_mIoU_pp','VALID_delta_mIoU_pp','delta_change_VALID_minus_OLD_pp']},'VALID_paired_scene_bootstrap_95_descriptive_pp':ci};t2.append(mean);boot.append({'comparison':label,'scene_ids':gs,'draws':10000,'seed':22901,'shared_indices_both_seeds':True,'percentiles':ci})
 for g in gs:
  a=data['Source','OLD'][g];b=data['Source','VALID'][g];h=evals['Source']['per_scene'][g]['Fill_prediction_histogram'];row={'scene':g,'Fill_pixels':sum(h),'OLD_pixels':int(a.sum()),'Fill_fraction':sum(h)/int(a.sum()),'OLD_Source_mIoU':metrics(a)['mIoU'],'VALID_Source_mIoU':metrics(b)['mIoU']}
  for mode in ['OLD','VALID']:
   ds=[next(v['delta_mIoU_pp'] for v in diff if v['scene']==g and v['seed']==s and v['support']==mode and v['comparison']=='SPARCS_minus_L8') for s in [21701,21702]];row[mode+'_SPARCS_minus_L8_seed21701']=ds[0];row[mode+'_SPARCS_minus_L8_seed21702']=ds[1];row[mode+'_SPARCS_minus_L8_seed_mean']=float(np.mean(ds))
  for model in NAMES:row[model+'_Fill_prediction_histogram']=evals[model]['per_scene'][g]['Fill_prediction_histogram']
  t3.append(row)
 for n,v in [('TABLE1',t1),('TABLE2',t2),('TABLE3',t3),('PER_SCENE',scenes),('PER_INPUT',inputs),('SCENE_DIFFERENCES',diff)]:output(n,v)
 historical=next(v for v in js(OLD/'TABLE2.json') if v['domain']=='SPARCS');save('HISTORICAL_SPARCS_REFERENCE.json',{'source_SHA':sha(OLD/'TABLE2.json'),'entry':historical,'SPARCS_pixels_opened':False});save('BOOTSTRAP_RECEIPT.json',boot);save('INPUTS_USED.json',used);save('CPU_ANALYSIS_COMPLETE.json',{'status':'passed','CPU_seconds':time.monotonic()-START,'CPU_peak_bytes':cpu_peak(),'OLD_all4815_SHA_confusions_exact229':True,'scene_bootstrap_not_patch_sampling':True,'training_updates':0});assert cpu_peak()<16*1024**3;print(json.dumps({'TABLE2':t2}))
if __name__=='__main__':main()
