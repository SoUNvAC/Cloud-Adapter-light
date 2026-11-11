"""Frozen218 descriptive tables. No new predictions, labels, selection or inference."""
from pathlib import Path
import collections,csv,json,hashlib,time
import numpy as np
from analyze_phase216 import metrics,miou,peak
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase218/phase218_20261023';RAW=O/'delivery';LABELS=['Surface','Cloud','Shadow']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(n,rows):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def group_stats(rr):
 n=sum(r['pixels'] for r in rr);k=sum(r['KL_sum'] for r in rr);return rates({'pixels':n,'KL_sum':k,'KL_mean':k/n if n else None,'teacher_disagreements':sum(r['teacher_disagreements'] for r in rr),'student_errors':sum(r['student_errors'] for r in rr),'teacher_errors':sum(r['teacher_errors'] for r in rr)})
def rates(r):
 n=r['pixels'];return {**r,'teacher_disagreement_rate':r['teacher_disagreements']/n if n else None,'student_reference_error_rate':r['student_errors']/n if n else None,'teacher_reference_error_rate':r['teacher_errors']/n if n else None}
def aggregate(rr):
 g=sum(r['geometry_pixels'] for r in rr);k=sum(r['KL_sum_geometry'] for r in rr);ds=sum(r['teacher_disagreements_geometry'] for r in rr);v=[r['KL_mean_geometry'] for r in rr]
 a={'images':len(rr),'ROIs':len({r['ROI'] for r in rr}),'geometry_pixels':g,'KL_sum_geometry':k,'KL_mean_geometry':k/g,'image_KL_median':float(np.median(v)),'image_KL_P90':float(np.quantile(v,.9)),'image_KL_max':max(v),'teacher_disagreements_geometry':ds,'teacher_disagreement_rate_geometry':ds/g,'negative_KL_pixels':sum(r['negative_KL_pixels'] for r in rr),'negative_KL_sum':sum(r['negative_KL_sum'] for r in rr),'KL_min':min(r['KL_min'] for r in rr),'KL_max':max(r['KL_max'] for r in rr)}
 a['teacher_student_geometry_counts']=sum(np.asarray(r['teacher_student_geometry_counts'],np.int64) for r in rr).tolist()
 if rr[0]['pool']=='Source_val535':
  a['reference_valid']=group_stats([r['reference_valid'] for r in rr]);a['reference_ignore_pixels']=sum(r['reference_ignore_pixels'] for r in rr);a['reference_classes']=[{'class':i,**group_stats([r['reference_classes'][i] for r in rr])} for i in range(3)]
  a['correctness']=[{'teacher_correct':t,'student_correct':s,**group_stats([next(q for q in r['correctness'] if q['teacher_correct']==t and q['student_correct']==s) for r in rr])} for t in [True,False] for s in [True,False]]
  a['both_wrong_subtypes']=[{'same_error_class':b,**group_stats([next(q for q in r['both_wrong_subtypes'] if q['same_error_class']==b) for r in rr])} for b in [True,False]]
  tr=sum(np.asarray(r['reference_teacher_student_counts'],np.int64) for r in rr);a['reference_teacher_student_counts']=tr.tolist();a['confusion']=tr.sum(1).tolist();a['teacher_confusion']=tr.sum(2).tolist()
 return a
def flatten(a):return {k:v for k,v in a.items() if not isinstance(v,(list,dict))}
def main():
 start=time.monotonic();assert js(RAW/'COMPLETE.json')['total_units']==4758 and js(RAW/'REMOTE_CPU_ACCEPTANCE.json')['status']=='passed'
 rows=[json.loads(x) for x in (RAW/'PER_IMAGE.jsonl').read_text().splitlines()];assert len(rows)==3164
 bg=collections.defaultdict(list);rg=collections.defaultdict(list);per=[];classes=[];states=[];wrong=[]
 for r in rows:
  key=(r['seed'],r['arm'],r['pool']);bg[key].append(r);rg[(*key,r['ROI'])].append(r);meta={k:r[k] for k in ['seed','arm','pool','ROI','name']};per.append({**meta,**flatten(r),'reference_valid_pixels':r.get('reference_valid',{}).get('pixels'),'KL_mean_reference':r.get('reference_valid',{}).get('KL_mean'),'reference_student_errors':r.get('reference_valid',{}).get('student_errors'),'reference_teacher_errors':r.get('reference_valid',{}).get('teacher_errors')})
  if r['pool']=='Source_val535':
   classes.extend({**meta,'reference_class':LABELS[x['class']],**{k:v for k,v in rates(x).items() if k!='class'}} for x in r['reference_classes']);states.extend({**meta,**x} for x in r['correctness']);wrong.extend({**meta,**x} for x in r['both_wrong_subtypes'])
 write('PER_IMAGE.csv',per);write('PER_IMAGE_REFERENCE_CLASSES.csv',classes);write('PER_IMAGE_CORRECTNESS.csv',states);write('PER_IMAGE_BOTH_WRONG.csv',wrong)
 pools={k:aggregate(v) for k,v in bg.items()};rois={k:aggregate(v) for k,v in rg.items()};table_a=[{'seed':s,'arm':a,'pool':p,**flatten(v),'reference_valid_pixels':v.get('reference_valid',{}).get('pixels'),'reference_KL_mean':v.get('reference_valid',{}).get('KL_mean'),'reference_ignore_pixels':v.get('reference_ignore_pixels')} for (s,a,p),v in sorted(pools.items())]
 write('TABLE_A_RETENTION.csv',table_a);diff=[];paired=[]
 for s in [21701,21702]:
  for p in ['Replay256','Source_val535']:
   c=pools[s,'C',p];d=pools[s,'D',p];byname={r['name']:r for r in bg[s,'C',p]};dd=[]
   for r in bg[s,'D',p]:
    delta=r['KL_mean_geometry']-byname[r['name']]['KL_mean_geometry'];dd.append(delta);paired.append({'seed':s,'pool':p,'name':r['name'],'ROI':r['ROI'],'C_KL':byname[r['name']]['KL_mean_geometry'],'D_KL':r['KL_mean_geometry'],'D_minus_C_KL':delta})
   diff.append({'seed':s,'pool':p,'D_minus_C_pooled_KL':d['KL_mean_geometry']-c['KL_mean_geometry'],'D_minus_C_image_median_KL':d['image_KL_median']-c['image_KL_median'],'D_minus_C_image_P90_KL':d['image_KL_P90']-c['image_KL_P90'],'D_minus_C_image_max_KL':d['image_KL_max']-c['image_KL_max'],'D_minus_C_disagreement_rate':d['teacher_disagreement_rate_geometry']-c['teacher_disagreement_rate_geometry'],'paired_image_delta_KL_median':float(np.median(dd)),'paired_image_delta_KL_P90':float(np.quantile(dd,.9)),'paired_image_delta_KL_max':max(dd)})
 write('TABLE_A_DIFFERENCES.csv',diff);write('PER_IMAGE_PAIRED_DIFFERENCES.csv',paired)
 roi_rows=[];pc=[];ps=[];pw=[];poolclasses=[];poolstates=[];poolwrong=[];b=[]
 source=js(RAW/'expected217/Source__source.json');sourcegs={k:np.asarray(v,np.int64) for k,v in source['per_group'].items()}
 for k,v in sorted(rois.items()):
  s,a,p,g=k;meta={'seed':s,'arm':a,'pool':p,'ROI':g};row={**meta,**flatten(v)}
  if p=='Source_val535':
   row.update({'reference_valid_pixels':v['reference_valid']['pixels'],'KL_mean_reference':v['reference_valid']['KL_mean'],'teacher_disagreement_rate_reference':v['reference_valid']['teacher_disagreements']/v['reference_valid']['pixels'],'reference_student_errors':v['reference_valid']['student_errors'],'reference_teacher_errors':v['reference_valid']['teacher_errors'],**metrics(np.asarray(v['confusion']))})
   pc.extend({**meta,'reference_class':LABELS[x['class']],**{k:z for k,z in x.items() if k!='class'}} for x in v['reference_classes']);ps.extend({**meta,**x} for x in v['correctness']);pw.extend({**meta,**x} for x in v['both_wrong_subtypes'])
  roi_rows.append(row)
 # Replay and val have different reference columns; use a stable union of headers.
 fields=list(dict.fromkeys(k for r in roi_rows for k in r));roi_rows=[{k:r.get(k) for k in fields} for r in roi_rows]
 write('PER_ROI.csv',roi_rows);write('PER_ROI_REFERENCE_CLASSES.csv',pc);write('PER_ROI_CORRECTNESS.csv',ps);write('PER_ROI_BOTH_WRONG.csv',pw)
 for (s,a,p),v in sorted(pools.items()):
  if p!='Source_val535':continue
  c=np.asarray(v['confusion'],np.int64);m=metrics(c);loss=sorted((float(miou(sourcegs[g])-miou(np.asarray(rois[s,a,p,g]['confusion']))) for g in sourcegs),reverse=True);valid=v['reference_valid'];b.append({'seed':s,'arm':a,'mIoU':m['mIoU'],'Source_mIoU':float(miou(np.asarray(source['confusion']))),'pool_loss_pp':float(miou(np.asarray(source['confusion'])))-m['mIoU'],'T11_pp':sum(loss[:11])/11,'Surface_IoU':m['Surface_IoU'],'Cloud_IoU':m['Cloud_IoU'],'Shadow_IoU':m['Shadow_IoU'],'error_rate':valid['student_errors']/valid['pixels'],'teacher_error_rate':valid['teacher_errors']/valid['pixels'],'geometry_pixels':v['geometry_pixels'],'reference_valid_pixels':valid['pixels'],'reference_ignore_pixels':v['reference_ignore_pixels'],'KL_mean_geometry':v['KL_mean_geometry'],'KL_mean_reference':valid['KL_mean'],'teacher_disagreement_rate_reference':valid['teacher_disagreements']/valid['pixels']})
  meta={'seed':s,'arm':a};poolclasses.extend({**meta,'reference_class':LABELS[x['class']],**{k:z for k,z in x.items() if k!='class'}} for x in v['reference_classes']);poolstates.extend({**meta,**x} for x in v['correctness']);poolwrong.extend({**meta,**x} for x in v['both_wrong_subtypes'])
 write('TABLE_B_REFERENCE.csv',b);write('POOLED_REFERENCE_CLASSES.csv',poolclasses);write('POOLED_CORRECTNESS.csv',poolstates);write('POOLED_BOTH_WRONG.csv',poolwrong)
 focus=js(RAW/'INPUT_MODEL_LOCK.json')['focus_ROIs'];allc=[]
 for s in [21701,21702]:
  for g in sorted(sourcegs):
   c=rois[s,'C','Source_val535',g];d=rois[s,'D','Source_val535',g];cm=metrics(np.asarray(c['confusion']));dm=metrics(np.asarray(d['confusion']));cv=c['reference_valid'];dv=d['reference_valid']
   row={'seed':s,'ROI':g,'focus':g in focus,'Source_mIoU':float(miou(sourcegs[g])),'C_mIoU':cm['mIoU'],'D_mIoU':dm['mIoU'],'D_minus_C_mIoU_pp':dm['mIoU']-cm['mIoU'],'C_KL_mean':cv['KL_mean'],'D_KL_mean':dv['KL_mean'],'D_minus_C_KL':dv['KL_mean']-cv['KL_mean'],'C_teacher_disagreement_rate':cv['teacher_disagreements']/cv['pixels'],'D_teacher_disagreement_rate':dv['teacher_disagreements']/dv['pixels'],'C_errors':cv['student_errors'],'D_errors':dv['student_errors'],'D_minus_C_errors':dv['student_errors']-cv['student_errors'],'D_KL_lower_and_errors_more':dv['KL_mean']<cv['KL_mean'] and dv['student_errors']>cv['student_errors']}
   for i,label in enumerate(LABELS):
    for arm,v in [('C',c),('D',d)]:row[arm+'_'+label+'_KL']=v['reference_classes'][i]['KL_mean']
   for i,x in enumerate(LABELS):
    for j,y in enumerate(LABELS):
     if i!=j:
      for arm,v in [('C',c),('D',d)]:row[arm+'_'+x+'_to_'+y]=v['confusion'][i][j]
   allc.append(row)
 write('TABLE_C_ALL_ROI.csv',allc);write('TABLE_C_FOCUS.csv',[r for r in allc if r['focus']]);save('AGGREGATE_STATISTICS.json',{'pools':{'/'.join(map(str,k)):v for k,v in pools.items()},'ROIs':{'/'.join(map(str,k)):v for k,v in rois.items()}})
 assert peak()<16*1024**3;save('ANALYSIS_RECEIPT.json',{'seconds':time.monotonic()-start,'peak_working_set_bytes':peak(),'model_calls':0,'no_bootstrap_or_significance_or_new_selection':True,'per_image_rows':3164,'per_ROI_rows':1424,'all_val_ROI_pair_rows':214,'code_SHA':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 print(json.dumps({'tableA':table_a,'tableB':b,'focus':[{k:r[k] for k in ['seed','ROI','C_KL_mean','D_KL_mean','C_errors','D_errors','D_KL_lower_and_errors_more']} for r in allc if r['focus']]},indent=2))
if __name__=='__main__':main()
