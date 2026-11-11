"""All107 ROI diagnosis; no inference, threshold selection or new bootstrap."""
from pathlib import Path
import json,csv,collections,time,hashlib
import numpy as np
from analyze_phase216 import metrics,miou,peak
from analyze_phase218 import aggregate,group_stats
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase223/phase223_20261024';D=O/'delivery';CS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(v) for v in p.read_text(encoding='utf-8-sig').splitlines()]
def write(n,rows):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rows for k in v)) or ['no_rows']);w.writeheader();w.writerows(rows)
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def flat(v):return {k:x for k,x in v.items() if not isinstance(x,(list,dict))}
def main():
 clock=time.monotonic();assert js(D/'COMPLETE.json')['total_units']==5360 and js(D/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed';lock=js(D/'INPUT_MODEL_LOCK.json');data=lines(D/'PER_IMAGE.jsonl');assert len(data)==4280;by=collections.defaultdict(list);roiby=collections.defaultdict(list);per=[];classes=[];states=[];flows=[];shadow=[]
 for v in data:
  by[v['model']].append(v);roiby[v['model'],v['ROI']].append(v);meta={k:v[k] for k in ['model','target','seed','arm','alpha','ROI','name']};per.append({**meta,**flat(v),'reference_valid_pixels':v['reference_valid']['pixels'],'reference_KL':v['reference_valid']['KL_mean'],'reference_errors':v['reference_valid']['student_errors']})
  classes.extend({**meta,'reference_class':CS[q['class']],**flat(q)} for q in v['reference_classes']);states.extend({**meta,**q} for q in v['correctness']);flows.extend({**meta,'flow':CS[q['reference']]+'->'+CS[q['student']],**q} for q in v['error_flows']);shadow.extend({**meta,**q} for q in v['teacher_correct_shadow_destinations'])
 write('PER_IMAGE.csv',per);write('PER_IMAGE_REFERENCE_CLASSES.csv',classes);write('PER_IMAGE_CORRECTNESS.csv',states);write('PER_IMAGE_ERROR_FLOW_KL.csv',flows);write('PER_IMAGE_TEACHER_CORRECT_SHADOW.csv',shadow)
 source=js(D/'expected/Source.json');sourcegs={g:np.asarray(c,np.int64) for g,c in source['per_group'].items()};pools={k:aggregate(v) for k,v in by.items()};rois={k:aggregate(v) for k,v in roiby.items()};arows=[];roirows=[];classrows=[];staterows=[];flowrows=[];shadowrows=[];worst=[]
 for model,meta in lock['models'].items():
  v=pools[model];gids=sorted(sourcegs);loss={g:float(miou(sourcegs[g])-miou(np.asarray(rois[model,g]['confusion']))) for g in gids};ordered=sorted(gids,key=lambda g:(-loss[g],g));cm=np.asarray(v['confusion'],np.int64);a={k:meta[k] for k in ['target','seed','arm']};arows.append({'model':model,**a,'alpha':.25,'KL_mean_geometry':v['KL_mean_geometry'],'KL_mean_reference':v['reference_valid']['KL_mean'],'teacher_disagreement_rate_geometry':v['teacher_disagreement_rate_geometry'],'source_pool_loss_pp':float(miou(np.asarray(source['confusion']))-miou(cm)),'T11_pp':float(np.mean([loss[g] for g in ordered[:11]])),'worst_ROI':ordered[0],'worst_loss_pp':loss[ordered[0]],'error_pixels':int(cm.sum()-np.trace(cm)),'reference_valid_pixels':int(cm.sum()),**metrics(cm)})
  for g in gids+['__pooled__']:
   rr=by[model] if g=='__pooled__' else roiby[model,g];v=pools[model] if g=='__pooled__' else rois[model,g];c=np.asarray(v['confusion']);b=np.asarray(source['confusion']) if g=='__pooled__' else sourcegs[g];m={'model':model,**a,'ROI':g,'alpha':.25};roirows.append({**m,**flat(v),'reference_KL':v['reference_valid']['KL_mean'],'reference_errors':v['reference_valid']['student_errors'],'source_mIoU':float(miou(b)),'loss_pp':float(miou(b)-miou(c)),**metrics(c),'confusion':json.dumps(c.tolist())});classrows.extend({**m,'reference_class':CS[q['class']],**flat(q)} for q in v['reference_classes']);staterows.extend({**m,**q} for q in v['correctness'])
   for c0 in range(3):
    for dst in range(3):
     if c0!=dst:flowrows.append({**m,'flow':CS[c0]+'->'+CS[dst],**group_stats([next(q for q in x['error_flows'] if q['reference']==c0 and q['student']==dst) for x in rr])})
   for dst in [0,1]:
    qs=[next(q for q in x['teacher_correct_shadow_destinations'] if q['destination']==dst) for x in rr];q=group_stats(qs);conf=sum(q['teacher_confidence_sum'] for q in qs);shadowrows.append({**m,'destination':CS[dst],**q,'teacher_confidence_sum':conf,'teacher_confidence_mean':conf/q['pixels'] if q['pixels'] else None})
  for rank,g in enumerate(ordered[:11],1):worst.append({'model':model,**a,'rank':rank,'ROI':g,'loss_pp':loss[g]})
 write('TABLE_A_ALL_SOURCE.csv',arows);write('PER_ROI.csv',roirows);write('PER_ROI_REFERENCE_CLASSES.csv',classrows);write('PER_ROI_CORRECTNESS.csv',staterows);write('PER_ROI_ERROR_FLOW_KL.csv',flowrows);write('TEACHER_CORRECT_SHADOW_DESTINATIONS.csv',shadowrows);write('ORIGINAL_WORST11_ROIS.csv',worst)
 # Four-axis joint projections are direct paired counts, not old confusion subtraction.
 joints={}
 for v in lines(D/'JOINT_COUNTS.jsonl'):
  j=np.asarray(v['J'],np.int64);assert j.shape==(3,3,3,3) and j.sum()==v['valid_pixels'];key=v['target'],v['seed'],v['ROI'];joints[key]=joints.get(key,np.zeros_like(j))+j
 cindex,dindex=np.indices((3,3));direct=[];crows=[];pairs=[];classdiff=[];focus=[]
 for target in ['SPARCS','L8']:
  for seed in [21701,21702]:
   ck=f'{target}_{seed}_C';dk=f'{target}_{seed}_D';a=next(q for q in arows if q['model']==ck);b=next(q for q in arows if q['model']==dk)
   for cl in range(3):
    c=pools[ck]['reference_classes'][cl];d=pools[dk]['reference_classes'][cl];classdiff.append({'target':target,'seed':seed,'reference_class':CS[cl],'C_KL':c['KL_mean'],'D_KL':d['KL_mean'],'D_minus_C_KL':None if c['KL_mean'] is None else d['KL_mean']-c['KL_mean'],'C_errors':c['student_errors'],'D_errors':d['student_errors'],'D_minus_C_errors':d['student_errors']-c['student_errors']})
   for g in sorted(sourcegs)+['__pooled__']:
    c=pools[ck] if g=='__pooled__' else rois[ck,g];d=pools[dk] if g=='__pooled__' else rois[dk,g];j=sum(joints[target,seed,h] for h in sourcegs) if g=='__pooled__' else joints[target,seed,g];teacher_correct_fewer_wrong=0;teacher_wrong_fewer_corrections=0
    for ref in range(3):
     for t in range(3):
      x=j[ref,t]
      if t==ref:teacher_correct_fewer_wrong+=int(x[cindex!=ref].sum()-x[dindex!=ref].sum())
      else:teacher_wrong_fewer_corrections+=int(x[cindex==ref].sum()-x[dindex==ref].sum())
     for tier,keep in [('all',[0,1,2]),('correct',[ref]),('wrong',[t for t in range(3) if t!=ref])]:
      x=j[ref,keep].sum(0);direct.append({'target':target,'seed':seed,'ROI':g,'reference_class':CS[ref],'teacher_correctness':tier,'pixels':int(x.sum()),'C_wrong_D_right':int(x[(cindex!=ref)&(dindex==ref)].sum()),'C_right_D_wrong':int(x[(cindex==ref)&(dindex!=ref)].sum()),'D_minus_C_errors':int(x[dindex!=ref].sum()-x[cindex!=ref].sum())})
    cv=c['reference_valid'];dv=d['reference_valid'];delta=dv['student_errors']-cv['student_errors'];assert delta==teacher_wrong_fewer_corrections-teacher_correct_fewer_wrong;crows.append({'target':target,'seed':seed,'ROI':g,'teacher_correct_D_fewer_errors':teacher_correct_fewer_wrong,'teacher_wrong_D_fewer_corrections':teacher_wrong_fewer_corrections,'D_minus_C_errors':delta})
    ccm=np.asarray(c['confusion']);dcm=np.asarray(d['confusion']);scm=np.asarray(source['confusion']) if g=='__pooled__' else sourcegs[g];row={'target':target,'seed':seed,'ROI':g,'Source_mIoU':float(miou(scm)),'C_mIoU':float(miou(ccm)),'D_mIoU':float(miou(dcm)),'D_minus_C_mIoU_pp':float(miou(dcm)-miou(ccm)),'C_KL':c['KL_mean_geometry'],'D_KL':d['KL_mean_geometry'],'D_minus_C_KL':d['KL_mean_geometry']-c['KL_mean_geometry'],'C_Shadow_KL':c['reference_classes'][2]['KL_mean'],'D_Shadow_KL':d['reference_classes'][2]['KL_mean'],'C_errors':cv['student_errors'],'D_errors':dv['student_errors'],'D_minus_C_errors':delta,'D_KL_lower_errors_more':d['KL_mean_geometry']<c['KL_mean_geometry'] and delta>0,'C_Shadow_recall':metrics(ccm)['Shadow_recall'],'D_Shadow_recall':metrics(dcm)['Shadow_recall'],'Source_Shadow_to_Surface':int(scm[2,0]),'C_Shadow_to_Surface':int(ccm[2,0]),'D_Shadow_to_Surface':int(dcm[2,0]),'Source_Shadow_to_Cloud':int(scm[2,1]),'C_Shadow_to_Cloud':int(ccm[2,1]),'D_Shadow_to_Cloud':int(dcm[2,1])};pairs.append(row)
    if g=='ROI_0124':focus.append(row)
 write('TABLE_A_CLASS_DIFFERENCES.csv',classdiff);write('TABLE_B_FOCUS_ROI0124.csv',focus);write('ALL_ROI_PAIRED_KL_ERRORS.csv',pairs);write('KL_LOWER_ERRORS_MORE_ROIS.csv',[v for v in pairs if v['ROI']!='__pooled__' and v['D_KL_lower_errors_more']]);write('TABLE_C_CORRECTNESS.csv',crows);write('DIRECT_C_TO_D_TRANSITIONS.csv',direct)
 diffs=[]
 for target in ['SPARCS','L8']:
  for s in [21701,21702]:
   c=next(v for v in arows if v['model']==f'{target}_{s}_C');d=next(v for v in arows if v['model']==f'{target}_{s}_D');diffs.append({'target':target,'seed':s,'D_minus_C_KL':d['KL_mean_geometry']-c['KL_mean_geometry'],'D_minus_C_errors':d['error_pixels']-c['error_pixels'],'D_minus_C_pool_loss_pp':d['source_pool_loss_pp']-c['source_pool_loss_pp'],'D_minus_C_T11_pp':d['T11_pp']-c['T11_pp'],'KL_lower_errors_more_ROI_count':sum(v['target']==target and v['seed']==s and v['ROI']!='__pooled__' and v['D_KL_lower_errors_more'] for v in pairs)})
 write('TABLE_A_DIFFERENCES.csv',diffs);gap=lines(D/'SIGNED_GAP_SUMMARY.jsonl');write('SIGNED_GAP_QUANTILES.csv',[{'model':v['model'],'ROI':v['ROI'],'condition':v['condition'],'gap':v['gap'],'pixels':v['stats']['teacher']['pixels'],**{kind+'_'+k:x for kind in ['teacher','student'] for k,x in v['stats'][kind].items() if k!='pixels'}} for v in gap]);save('AGGREGATE_STATISTICS.json',{'pools':pools,'ROIs':{k+'/'+g:v for (k,g),v in rois.items()}});assert peak()<16*1024**3;save('ANALYSIS_RECEIPT.json',{'seconds':time.monotonic()-clock,'CPU_peak_bytes':peak(),'new_model_calls':0,'new_bootstrap':0,'all107_ROIs_each_target_seed':True,'per_image':len(data),'model_ROIs':len(rois),'code_SHA':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()});print(json.dumps({'differences':diffs,'focus':focus},ensure_ascii=False))
if __name__=='__main__':main()
