"""Full support decision statistics and disk-exact ranking, no new model calls."""
from phase241_ranking import *
import csv,sys,traceback
CLASSES=('Surface','Cloud','Shadow')
def ratio(a,b):return float(a/b) if b else None
def cmmetrics(c):
 c=np.asarray(c,np.int64);ref=c.sum(1);pred=c.sum(0);diag=c.diagonal();union=ref+pred-diag;iou=[ratio(diag[i],union[i]) for i in range(3)]
 o={'reference':ref.tolist(),'predicted':pred.tolist(),'IoU':iou,'mIoU_union':float(np.mean([iou[i] for i in range(3) if union[i]>0])) if union.sum() else None,'mIoU_reference_present':float(np.mean([iou[i] for i in range(3) if ref[i]>0])) if ref.sum() else None,'mIoU_fixed3':float(np.mean(iou)) if all(v is not None for v in iou) else None,'Cloud_recall':ratio(diag[1],ref[1]),'Shadow_recall':ratio(diag[2],ref[2]),'Surface_to_Cloud_FPR':ratio(c[0,1],ref[0]),'nonShadow_to_Shadow_FPR':ratio(c[0,2]+c[1,2],ref[0]+ref[1]),'absent_class_FP':{CLASSES[i]:int(pred[i]) for i in range(3) if ref[i]==0}}
 for i in range(3):
  for j in range(3):
   if i!=j:o[f'{CLASSES[i]}_to_{CLASSES[j]}']=int(c[i,j]);o[f'{CLASSES[i]}_to_{CLASSES[j]}_rate']=ratio(c[i,j],ref[i])
 return o
def writecsv(path,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows([{k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows])
def mean(v):
 a=[x for x in v if x is not None];return float(np.mean(a)) if a else None

def main():
 global cpu
 cpu=CPU('analyze');assert js(R/'TEST_COMPLETE.json')['passed'];assert not (R/'ANALYSIS_STARTED.json').exists();save('ANALYSIS_STARTED.json',{'epoch':time.time(),'PID':os.getpid()})
 fits={(v['domain'],v['model']):v for v in js(R/'CALIBRATION_LOCK.json')['fits']};thresholds={(v['domain'],v['model'],v['state']):v for v in js(R/'THRESHOLD_LOCK.json')['thresholds']};records=[];ranking=[];pooled=[];missing=[]
 for split in ('val','test'):
  for domain in ('L8','SPARCS'):
   for model in ('Source','C21701','C21702'):
    groups=grouped(split,domain,model);primary=[s for s,vs in groups.items() if domain!='L8' or split=='val' or vs[0]['usgs_shadows']=='yes'];supports={'main':primary}
    if domain=='L8' and split=='test':supports['nominal_full16']=list(groups);supports['nominal_no10']=[s for s in groups if s not in primary]
    for state in ('raw','calibrated'):
     fit=fits[domain,model]
     if state=='calibrated' and not fit['success']:missing.append({'split':split,'domain':domain,'model':model,'state':state,'reason':'calibration_failed'});continue
     threshold=thresholds[domain,model,state]['threshold'];base=[]
     for scene,rows in groups.items():
      cms=[np.zeros((3,3),np.int64),np.zeros((3,3),np.int64)];N=0;nll=0.;brier=0.;score_sum=np.zeros(3);score_min=np.full(3,np.inf);score_max=np.full(3,-np.inf)
      for v in rows:
       with arrays(v) as (p,y):
        a=calibrated(p,fit['bias']) if state=='calibrated' else p;valid=y!=255;truth=y[valid].astype(np.int64);flat=np.asarray(a[:,valid],np.float64);n=truth.size;N+=n
        if n:
         nll-=float(np.log(np.maximum(flat[truth,np.arange(n)],1e-12)).sum());brier+=float((flat*flat).sum()-2*flat[truth,np.arange(n)].sum()+n);score_sum+=flat.sum(1);score_min=np.minimum(score_min,flat.min(1));score_max=np.maximum(score_max,flat.max(1))
        argmax=a.argmax(0);decision=a[:2].argmax(0);decision[np.asarray(a[2],np.float64)>=threshold]=2
        for j,pr in enumerate((argmax,decision)):cms[j]+=np.bincount(3*truth+pr[valid],minlength=9).reshape(3,3)
        if state=='raw':assert np.array_equal(np.bincount(3*truth+argmax[valid],minlength=9).reshape(3,3),v['confusion'])
      for dec,cm in zip(('argmax','frozen_threshold'),cms):
       r={'split':split,'domain':domain,'model':model,'state':state,'scene':scene,'main':scene in primary,'quality':'nominal_unknown_shadow_completeness' if domain=='L8' and split=='test' and scene not in primary else 'published_reference_not_independent_truth','decision':dec,'threshold':threshold if dec=='frozen_threshold' else None,'confusion':cm.tolist(),'valid':N,'NLL':ratio(nll,N),'Brier':ratio(brier,N),'score_mean':(score_sum/N).tolist() if N else None,'score_min':score_min.tolist() if N else None,'score_max':score_max.tolist() if N else None,**cmmetrics(cm)};base.append(r);records.append(r)
      cpu.check()
     for support,scenes in supports.items():
      for decision in ('argmax','frozen_threshold'):
       sub=[v for v in base if v['scene'] in scenes and v['decision']==decision];cm=np.sum([v['confusion'] for v in sub],axis=0);N=sum(v['valid'] for v in sub);rates=[v['nonShadow_to_Shadow_FPR'] for v in sub if v['nonShadow_to_Shadow_FPR'] is not None]
       pooled.append({'split':split,'domain':domain,'model':model,'state':state,'support':support,'decision':decision,'scenes':len(sub),'confusion':cm.tolist(),**cmmetrics(cm),'scene_mean_mIoU_union':mean([v['mIoU_union'] for v in sub]),'scene_mean_mIoU_reference_present':mean([v['mIoU_reference_present'] for v in sub]),'scene_mean_Shadow_recall':mean([v['Shadow_recall'] for v in sub]),'scene_mean_nonShadow_to_Shadow_FPR':mean(rates),'FPR_scenes_above_1pct':sum(x>.01 for x in rates),'worst_scene_FPR':max(rates) if rates else None,'NLL':sum(v['NLL']*v['valid'] for v in sub if v['NLL'] is not None)/N if N else None,'Brier':sum(v['Brier']*v['valid'] for v in sub if v['Brier'] is not None)/N if N else None,'threshold':threshold if decision=='frozen_threshold' else None})
     for task in TASKS:
      folder=R/'scratch'/f'rank_{split}_{domain}_{model}_{state}_{task}';folder.mkdir(parents=True,exist_ok=False);runs={};scene_metrics={}
      for si,(scene,rows) in enumerate(groups.items()):
       ss=[];ys=[]
       for v in rows:
        with arrays(v) as (p,y):
         a=calibrated(p,fit['bias']) if state=='calibrated' else p;s,t=score_task(a,y,task);ss.append(s);ys.append(t)
       s=np.concatenate(ss);t=np.concatenate(ys);del ss,ys;runs[scene]=sorted_run(folder,str(si),s,t);del s,t
       m=merge([runs[scene]],folder,str(si));scene_metrics[scene]=m;ranking.append({'split':split,'domain':domain,'model':model,'state':state,'scope':'scene','support':'main' if scene in primary else 'nominal_no10','scene':scene,'task':task,**m});cpu.check()
      for support,scenes in supports.items():
       result=merge([runs[s] for s in scenes],folder,'pool_'+support);ranking.append({'split':split,'domain':domain,'model':model,'state':state,'scope':'pooled','support':support,'scene':'ALL','task':task,'evaluable_scenes':sum(scene_metrics[s]['AUC'] is not None for s in scenes),**result})
       ranking.append({'split':split,'domain':domain,'model':model,'state':state,'scope':'scene_mean','support':support,'scene':'ALL','task':task,'evaluable_scenes':sum(scene_metrics[s]['AUC'] is not None for s in scenes),'AP':mean([scene_metrics[s]['AP'] for s in scenes]),'AUC':mean([scene_metrics[s]['AUC'] for s in scenes]),'positive':sum(scene_metrics[s]['positive'] for s in scenes),'negative':sum(scene_metrics[s]['negative'] for s in scenes)})
      clean(folder);save('ANALYSIS_PROGRESS.json',{'split':split,'domain':domain,'model':model,'state':state,'task':task,'epoch':time.time()})
     save('CHECKPOINT_METRICS.json',{'per_scene':records,'ranking':ranking,'pooled':pooled,'missing':missing})
 save('METRICS.json',{'per_scene':records,'ranking':ranking,'pooled':pooled,'missing':missing});writecsv(R/'PER_SCENE.csv',records);writecsv(R/'RANKING.csv',ranking);writecsv(R/'POOLED.csv',pooled);bootstrap(records,ranking);save('ANALYSIS_COMPLETE.json',{'passed':True,'epoch':time.time(),'metrics_SHA':sha(R/'METRICS.json')});cpu.finish()

def bootstrap(records,ranking):
 output=[];indices={}
 for domain in ('L8','SPARCS'):
  base=sorted([v['scene'] for v in records if v['domain']==domain and v['split']=='test' and v['model']=='Source' and v['state']=='raw' and v['decision']=='argmax' and v['main']]);rng=np.random.default_rng(24101);idx=rng.integers(len(base),size=(10000,len(base)));indices[domain]={'scenes':base,'indices':idx.tolist()}
  cms={};auc={}
  for model in ('Source','C21701','C21702'):
   for state in ('raw','calibrated'):
    for decision in ('argmax','frozen_threshold'):
     lookup={v['scene']:v for v in records if v['domain']==domain and v['split']=='test' and v['model']==model and v['state']==state and v['decision']==decision}
     if not all(s in lookup for s in base):continue
     c=np.asarray([lookup[s]['confusion'] for s in base],np.float64);total=c[idx].sum(1);diag=np.diagonal(total,axis1=1,axis2=2);ref=total.sum(2);pred=total.sum(1);union=ref+pred-diag
     cms[model,state,decision]={'mIoU':np.mean(np.divide(diag,union,out=np.full_like(diag,np.nan),where=union>0),axis=1),'Shadow_recall':np.divide(diag[:,2],ref[:,2],out=np.full(10000,np.nan),where=ref[:,2]>0),'FPR':np.divide(total[:,0,2]+total[:,1,2],ref[:,0]+ref[:,1],out=np.full(10000,np.nan),where=ref[:,0]+ref[:,1]>0)}
    lookup={v['scene']:v['AUC'] for v in ranking if v['domain']==domain and v['split']=='test' and v['model']==model and v['state']==state and v['scope']=='scene' and v['task']=='Shadow_Surface'}
    if all(s in lookup for s in base):
     a=np.asarray([np.nan if lookup[s] is None else lookup[s] for s in base]);sample=a[idx];n=np.isfinite(sample).sum(1);auc[model,state]=np.divide(np.nansum(sample,axis=1),n,out=np.full(10000,np.nan),where=n>0)
  def entry(metric,contrast,seed,values,decision=None):
   ok=np.isfinite(values);output.append({'domain':domain,'metric':metric,'contrast':contrast,'seed':seed,'decision':decision,'bootstrap_mean':float(np.mean(values[ok])) if ok.any() else None,'CI95':np.quantile(values[ok],[.025,.975]).tolist() if ok.any() else None,'NA_replicates':int((~ok).sum()),'conditional':'fixed models, calibration and thresholds; paired scenes; no training randomness coverage'})
  for metric in ('Shadow_Surface_AUC','mIoU','Shadow_recall','FPR'):
   decisions=[None] if metric=='Shadow_Surface_AUC' else ['argmax','frozen_threshold']
   for decision in decisions:
    allseed=collections.defaultdict(list)
    for seed in (21701,21702):
     try:
      A=auc['Source','raw'] if decision is None else cms['Source','raw',decision][metric];B=auc['Source','calibrated'] if decision is None else cms['Source','calibrated',decision][metric];C=auc['C'+str(seed),'raw'] if decision is None else cms['C'+str(seed),'raw',decision][metric];D=auc['C'+str(seed),'calibrated'] if decision is None else cms['C'+str(seed),'calibrated',decision][metric]
     except KeyError:continue
     for name,v in {'B-A':B-A,'D-C':D-C,'C-A':C-A,'D-B':D-B,'(B-A)-(D-C)':(B-A)-(D-C)}.items():entry(metric,name,seed,v,decision);allseed[name].append(v)
    for name,vs in allseed.items():
     if len(vs)==2:entry(metric,name,'two_seed_difference_mean',(vs[0]+vs[1])/2,decision)
 save('BOOTSTRAP_INDICES.json',{'seed':24101,'replicates':10000,'domains':indices});save('BOOTSTRAP.json',output);writecsv(R/'BOOTSTRAP.csv',output)
if __name__=='__main__':
 try:main()
 except BaseException:
  save('ANALYSIS_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'cpu' in globals():cpu.finish('failed')
  raise
