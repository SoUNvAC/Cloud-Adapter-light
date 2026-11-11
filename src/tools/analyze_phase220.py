"""Fixed confidence x class direct transitions. Pure count analysis, no fits/bootstrap."""
from pathlib import Path
import json,csv,hashlib,time,collections
import numpy as np
from analyze_phase216 import peak
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase220/phase220_20261024';RAW=OUT/'delivery';CLASSES=['Surface','Cloud','Shadow'];BANDS=['high','low'];AXES=['confidence_band','reference_class','teacher_prediction','D_prediction','G_prediction','S_prediction']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def write(n,rows):
 with (OUT/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)) or ['no_rows']);w.writeheader();w.writerows(rows)
def table(j,seed,roi):
 ds,gs,ss=np.indices((3,3,3));out=[];flows=[]
 for bi,band in enumerate(BANDS):
  for c,cl in enumerate(CLASSES):
   for tier,keep in [('all',[0,1,2]),('correct',[c]),('wrong',[t for t in range(3) if t!=c])]:
    x=j[bi,c,keep].sum(0);n=int(x.sum());de=int(x[ds!=c].sum());ge=int(x[gs!=c].sum());se=int(x[ss!=c].sum());fixed=int(x[(ds!=c)&(gs==c)].sum());broken=int(x[(ds==c)&(gs!=c)].sum());assert ge-de==broken-fixed
    r={'seed':seed,'ROI':roi,'band':band,'reference_class':cl,'teacher_reference_correctness':tier,'valid_pixels':n,'D_wrong_G_right':fixed,'D_right_G_wrong':broken,'G_minus_D_errors':ge-de,'D_errors':de,'G_errors':ge,'S_errors':se,'D_error_rate':de/n if n else None,'G_error_rate':ge/n if n else None,'S_error_rate':se/n if n else None,'S_minus_D_errors':se-de};out.append(r)
    dests=[0,1] if c==2 else [2]
    for dest in dests:
     removed=int(x[(ds==dest)&(gs!=dest)].sum());added=int(x[(ds!=dest)&(gs==dest)].sum());flow={'seed':seed,'ROI':roi,'band':band,'reference_class':cl,'teacher_reference_correctness':tier,'flow':cl+'->'+CLASSES[dest],'valid_reference_pixels':n,'D_flow_pixels':int(x[ds==dest].sum()),'G_flow_pixels':int(x[gs==dest].sum()),'S_flow_pixels':int(x[ss==dest].sum()),'D_flow_removed_by_G':removed,'G_flow_added_from_other_D_predictions':added,'G_minus_D_flow':added-removed};assert flow['G_minus_D_flow']==flow['G_flow_pixels']-flow['D_flow_pixels'];flows.append(flow)
 return out,flows
def main():
 start=time.monotonic();assert js(RAW/'COMPLETE.json')['units']==4800;raw=lines(RAW/'val/JOINT_COUNTS.jsonl');assert len(raw)==1070;aggregate={};names=set()
 for r in raw:
  key=r['seed'],r['ROI'];j=np.array(r['J'],np.int64);assert j.shape==(2,3,3,3,3,3) and (j>=0).all() and j.sum()==r['valid_pixels'];assert (r['seed'],r['name']) not in names;names.add((r['seed'],r['name']));aggregate[key]=aggregate.get(key,np.zeros(j.shape,np.int64))+j
 assert len(aggregate)==214;rows=[];flows=[];cohort=[];focus=js(RAW/'FOCUS_LOCK.json')
 for s in [21701,21702]:
  ids=sorted(k for seed,k in aggregate if seed==s);assert len(ids)==107
  for roi in ids+['__pooled__']:
   j=sum(aggregate[s,k] for k in ids) if roi=='__pooled__' else aggregate[s,roi];a,b=table(j,s,roi);rows.extend(a);flows.extend(b)
  for label,ids in [('historical_four',focus['historical_four']),('posthoc_both_seeds_gt1pp_13',focus['both_seeds_G_minus_D_mIoU_below_minus1pp_13'])]:
   a,_=table(sum(aggregate[s,k] for k in ids),s,label);cohort.extend({'cohort_group_count':len(ids),**v} for v in a)
 write('TABLE_A_POOLED.csv',[v for v in rows if v['ROI']=='__pooled__']);write('TABLE_A_ALL_ROIS.csv',[v for v in rows if v['ROI']!='__pooled__']);write('DIRECT_ERROR_FLOWS.csv',flows);write('FOUR_FOCUS_ROIS.csv',[v for v in rows if v['ROI'] in focus['historical_four']]);write('THIRTEEN_POSTHOC_DAMAGE_ROIS.csv',[v for v in rows if v['ROI'] in focus['both_seeds_G_minus_D_mIoU_below_minus1pp_13']]);write('COHORT_SUMMARIES.csv',cohort)
 exposure=[];perimage=[];roigroups=collections.defaultdict(lambda:collections.Counter());globalgroups=collections.defaultdict(lambda:collections.Counter())
 for r in lines(RAW/'val/VAL_WEIGHT_EXPOSURE.jsonl'):
  for v in r['weight_exposure']:
   key=('source_val535','evaluation_allFalse',v['basis'],'unique535',v['class']);roikey=key+(r['ROI'],)
   for k in ['support','G_high_pixels','S_high_pixels','G_weight_sum','S_weight_sum']:globalgroups[key][k]+=v[k];roigroups[roikey][k]+=v[k]
   perimage.append({'pool':'source_val535','runtime':'evaluation_allFalse','name':r['name'],'ROI':r['ROI'],'sample_id':r['sample_id'],'class':CLASSES[v['class']],**{k:x for k,x in v.items() if k!='class'}})
 for r in lines(RAW/'replay/REPLAY_WEIGHTS.jsonl'):
  for v in r['class_exposure']:
   for method,mult in [('unique256',1),('original4000_sequence',r['repeat_count'])]:
    key=('replay256','training_deterministic_warnOnly','teacher_prediction_geometry',method,v['teacher_prediction_class'])
    for k in ['support','G_high_pixels','S_high_pixels','G_weight_sum','S_weight_sum']:globalgroups[key][k]+=v[k]*mult
   perimage.append({'pool':'replay256','runtime':'training_deterministic_warnOnly','name':r['name'],'sample_id':r['sample_id'],'class':CLASSES[v['teacher_prediction_class']],'basis':'teacher_prediction_geometry','repeat_count':r['repeat_count'],**{k:x for k,x in v.items() if k!='teacher_prediction_class'}})
 def weightrow(key,v,roi=None):
  pool,runtime,basis,method,cl=key;n=v['support'];return {'pool':pool,'runtime':runtime,'basis':basis,'repetition':method,'class':CLASSES[cl],**({'ROI':roi} if roi is not None else {}),**dict(v),'high_confidence_fraction':v['G_high_pixels']/n if n else None,'G_mean_weight':v['G_weight_sum']/n if n else None,'S_mean_weight':v['S_weight_sum']/n if n else None,'S_high_fraction':v['S_high_pixels']/n if n else None,'interpretation':'validation reconstruction only; not actual training gradients' if pool=='source_val535' else 'teacher predicted class only; no replay references read'}
 for k,v in sorted(globalgroups.items()):exposure.append(weightrow(k,v))
 write('TABLE_B_WEIGHT_ALLOCATION.csv',exposure);write('VAL_WEIGHT_ALLOCATION_ALL_ROIS.csv',[weightrow(k[:5],v,k[5]) for k,v in sorted(roigroups.items())]);write('PER_IMAGE_CLASS_WEIGHT_EXPOSURE.csv',perimage)
 seq=js(RAW/'SOURCE_SEQUENCE_LOCK.json');freq=collections.Counter(v['source_sample_id'] for v in seq);replay=js(RAW/'SUPPORT_LOCK.json')['replay'];write('REPLAY_SEQUENCE_MULTIPLICITY.csv',[{'sample_id':v['sample_id'],'name':v['name'],'ROI':v['ROI'],'repeat_count':freq[v['sample_id']],'image_SHA':v['image_SHA']} for v in replay]);assert sum(freq.values())==4000 and len(freq)==256
 # Compact answer facts preserve high/low and teacher-correct strata separately.
 facts={}
 for s in [21701,21702]:
  rr=[v for v in rows if v['seed']==s and v['ROI']=='__pooled__' and v['reference_class']=='Shadow' and v['teacher_reference_correctness']=='all'];facts[str(s)]={'Shadow_net_errors':sum(v['G_minus_D_errors'] for v in rr),'by_band':{v['band']:{k:v[k] for k in ['valid_pixels','D_wrong_G_right','D_right_G_wrong','G_minus_D_errors','D_errors','G_errors','S_errors','D_error_rate','G_error_rate','S_error_rate']} for v in rr},'teacher_correct_Shadow_by_band':{v['band']:{k:v[k] for k in ['valid_pixels','D_wrong_G_right','D_right_G_wrong','G_minus_D_errors']} for v in rows if v['seed']==s and v['ROI']=='__pooled__' and v['reference_class']=='Shadow' and v['teacher_reference_correctness']=='correct'}}
 save('ANSWER_FACTS.json',facts);save('JOINT_AXES.json',{'axes':AXES,'shape':[2,3,3,3,3,3],'values':{'confidence_band':BANDS,'reference_and_predictions':CLASSES},'flatten_order':'C row-major','support':'original source-val valid reference pixels; confidence ranking uses whole512x512 geometry','seed_units':'two paired models on same107 ROIs, not214 independent groups','zero_support_rates':None,'direct_transitions':'D/G/S co-observed on same pixel; directly reconstructed from J, unlike219 marginal net counts'})
 save('ANALYSIS_RECEIPT.json',{'status':'passed','seconds':time.monotonic()-start,'CPU_peak_bytes':peak(),'joint_image_seed_rows':len(raw),'all107_ROIs_each_seed':True,'no_training_fits_parameter_or_alpha_selection':True,'new_bootstrap':0,'new_model_calls':0,'code_SHA':sha(Path(__file__))});assert peak()<16*1024**3
 print(json.dumps(facts,indent=2))
if __name__=='__main__':main()
