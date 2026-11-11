"""Fixed CPU-only matched-pixel and directed-scene-pair diagnosis."""
from phase242_common import *
import sys,traceback
CLASSES=('Surface','Cloud','Shadow')
def close(a,b):return (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=1e-10)
def avg(a):
 a=[v for v in a if v is not None];return float(np.mean(a)) if a else None
def decision(p,y,shift,threshold,raw=False):
 if raw:a=p;pr=p.argmax(0)
 else:
  z=np.log(np.maximum(np.asarray(p,np.float64),1e-12));z[2]-=shift;pr=z.argmax(0);z-=z.max(0);a=np.exp(z);a/=a.sum(0)
 t=a[:2].argmax(0);t[np.asarray(a[2],np.float64)>=threshold]=2
 return {'argmax':metrics(cm(y,pr)),'frozen_threshold':metrics(cm(y,t))}
def old_scene(split,domain,model,scene,decision_name):return next(v for v in js(R/'PRIOR_METRICS.json')['per_scene'] if v['split']==split and v['domain']==domain and v['model']==model and v['scene']==scene and v['state']=='raw' and v['decision']==decision_name)

def pixel_scene(split,domain,scene,cpu):
 path=R/'scene_receipts'/split/domain/(scene+'.json');assert not path.exists();p,y,rows=scene_arrays(split,domain,scene,'Source');s=logscore(p);source_counts=np.bincount(y,minlength=3);cache={};out={'split':split,'domain':domain,'scene':scene,'reference':source_counts.tolist(),'valid':len(y),'Source':{},'C':{}}
 # Source is saved once per scene. Sorting is exact and lossless; no copied FP32 pool.
 for label,truth in ((1,2),(0,0)):cache[str(label)]=sortedfile(cachepath(split,domain,'Source',scene,label),s[y==truth])
 threshold=js(R/'RAW_THRESHOLDS.json')[domain]['Source'];out['Source']={'cache':cache,'decision':decision(p,y,0.,threshold,True)}
 for dec,v in out['Source']['decision'].items():assert v['confusion']==old_scene(split,domain,'Source',scene,dec)['confusion']
 for seed in (21701,21702):
  model='C'+str(seed);q,t,rs=scene_arrays(split,domain,scene,model);assert np.array_equal(y,t);assert [(v['name'],v['label_SHA']) for v in rows]==[(v['name'],v['label_SHA']) for v in rs]
  sc=logscore(q);delta=sc-s;median=float(np.median(delta));stats=[]
  for name,mask in [('all_valid',None)]+[(CLASSES[k],y==k) for k in range(3)]:
   ss=s if mask is None else s[mask];cs=sc if mask is None else sc[mask];ds=delta if mask is None else delta[mask];v={'reference_stratum':name,'s_Source':distribution(ss),'s_C':distribution(cs),'delta':distribution(ds),'p':[]}
   for k in range(3):
    a=p[k] if mask is None else p[k,mask];b=q[k] if mask is None else q[k,mask];v['p'].append({'class':CLASSES[k],'Source_mean':float(np.mean(a,dtype=np.float64)) if len(a) else None,'C_mean':float(np.mean(b,dtype=np.float64)) if len(b) else None,'Source_median':float(np.median(a.astype(np.float64))) if len(a) else None,'C_median':float(np.median(b.astype(np.float64))) if len(b) else None,'paired_delta':distribution(b.astype(np.float64)-a.astype(np.float64)),'Source_below_1e_minus12':int((a<1e-12).sum()),'C_below_1e_minus12':int((b<1e-12).sum())})
   stats.append(v);cpu.check()
  ps=p.argmax(0);pc=q.argmax(0);transition=np.bincount(9*y.astype(np.int64)+3*ps+pc,minlength=27).reshape(3,3,3);right_wrong=[]
  for k in range(3):right_wrong.append({'reference_class':CLASSES[k],'Source_correct_C_wrong':int(((y==k)&(ps==k)&(pc!=k)).sum()),'Source_wrong_C_correct':int(((y==k)&(ps!=k)&(pc==k)).sum()),'Shadow_to_Surface':int(((y==2)&(pc==0)).sum()) if k==2 else None,'Shadow_to_Cloud':int(((y==2)&(pc==1)).sum()) if k==2 else None})
  ccache={str(label):sortedfile(cachepath(split,domain,model,scene,label),sc[y==truth]) for label,truth in ((1,2),(0,0))}
  out['C'][model]={'m':median,'median_source':'all matched valid positions, no class conditioning','cache':ccache,'statistics':stats,'transition_reference_Source_C':transition.tolist(),'changes':right_wrong};del q,t,sc,delta,ps,pc,stats
  cpu.check()
 save(path,out);return out

def matrix(split,domain,model,scenes,receipts,shifts,cpu):
 cells=[]
 def files(scene,label):return cachepath(split,domain,model,scene,label)
 for i in scenes:
  for j in scenes:
   cpu.current={'split':split,'domain':domain,'matrix_model':model,'positive_scene':i,'negative_scene':j};a=ranked([(files(i,1),1,shifts.get(i,0)),(files(j,0),0,shifts.get(j,0))],cpu);cells.append({'positive_scene':i,'negative_scene':j,**a});cpu.check()
 totalP=sum(receipts[s]['reference'][2] for s in scenes);totalN=sum(receipts[s]['reference'][0] for s in scenes)
 for c in cells:c['weight']=float(c['P'])*c['N']/(float(totalP)*totalN) if totalP and totalN else 0.
 pool=ranked([(files(s,label),label,shifts.get(s,0)) for s in scenes for label in (1,0)],cpu);summary=matrix_summary(cells,scenes);assert close(pool['AUC'],summary['pooled_AUC']);summary['pooled_AP']=pool['AP'];summary['scene_mean_AP']=avg([c['AP'] for c in cells if c['positive_scene']==c['negative_scene']]);summary['P']=totalP;summary['N']=totalN
 return {'cells':cells,'summary':summary}

def matrix_summary(cells,scenes):
 selected=[c for c in cells if c['positive_scene'] in scenes and c['negative_scene'] in scenes];positives={c['positive_scene']:c['P'] for c in selected};negatives={c['negative_scene']:c['N'] for c in selected};total=float(sum(positives.values()))*sum(negatives.values());within=0.;cross=0.;wm=0.;cmass=0.;diag=[]
 for c in selected:
  w=c['P']*float(c['N'])/total if total else 0.
  if c['AUC'] is None:continue
  if c['positive_scene']==c['negative_scene']:within+=w*c['AUC'];wm+=w;diag.append(c['AUC'])
  else:cross+=w*c['AUC'];cmass+=w
 return {'pooled_AUC':within+cross if total else None,'scene_mean_AUC':avg(diag),'scene_evaluable':len(diag),'within_weight_mass':wm,'within_contribution':within,'within_weighted_AUC':ratio(within,wm),'cross_weight_mass':cmass,'cross_contribution':cross,'cross_AUC':ratio(cross,cmass)}

def compare_cells(source,other):
 ss={(v['positive_scene'],v['negative_scene']):v for v in source['cells']};out=[]
 for c in other['cells']:
  a=ss[c['positive_scene'],c['negative_scene']];d=None if c['AUC'] is None or a['AUC'] is None else c['AUC']-a['AUC'];out.append({**c,'Source_AUC':a['AUC'],'C_AUC':c['AUC'],'delta_AUC':d,'weighted_delta':None if d is None else c['weight']*d})
 return out

def signal(cells,medians,excluded=None):
 rows=[]
 for c in cells:
  i=c['positive_scene'];j=c['negative_scene']
  if i==j or excluded in (i,j) or c['delta_AUC'] is None:continue
  rows.append({'positive_scene':i,'negative_scene':j,'T':medians[i]-medians[j],'delta_AUC':c['delta_AUC']})
 x=np.array([v['T'] for v in rows]);y=np.array([v['delta_AUC'] for v in rows]);nonzero=(x!=0)&(y!=0)
 def ranks(a):return np.array([np.count_nonzero(a<v)+(np.count_nonzero(a==v)+1)/2 for v in a],np.float64)
 rho=float(np.corrcoef(ranks(x),ranks(y))[0,1]) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None
 return {'Spearman':rho,'evaluable_pairs':len(rows),'non_tie_pairs':int(nonzero.sum()),'direction_agreement':ratio(int(((x*y>0)&nonzero).sum()),int(nonzero.sum())),'T_zero':int((x==0).sum()),'delta_zero':int((y==0).sum()),'both_zero':int(((x==0)&(y==0)).sum()),'scatter':rows,'direction':'larger T predicts more favorable C-Source AUC; zero pairs separate; no p value'}

def main():
 global cpu
 split=sys.argv[1];assert split in ('val','test');cpu=CPU(split);assert not (R/f'{split.upper()}_STARTED.json').exists()
 if split=='test':
  seal=js(R/'VAL_SEAL.json')
  for n,h in seal['files'].items():assert sha(R/n)==h
  assert seal['test_new_diagnosis_not_started'];save('TEST_ACCESS_STARTED.json',{'epoch':time.time(),'val_seal_SHA':sha(R/'VAL_SEAL.json')})
 save(split.upper()+'_STARTED.json',{'epoch':time.time(),'PID':os.getpid()});results={};globals_lock={};allreceipts=[]
 for domain in ('L8','SPARCS'):
  scenes=list(groups(split,domain,'Source'));receipts={}
  for scene in scenes:
   cpu.current={'split':split,'domain':domain,'pixel_scene':scene};receipts[scene]=pixel_scene(split,domain,scene,cpu);allreceipts.append(receipts[scene]);cpu.check()
  if split=='val':g={model:float(np.median([receipts[s]['C'][model]['m'] for s in scenes])) for model in ('C21701','C21702')};globals_lock[domain]=g
  else:g=js(R/'VAL_LOCK.json')['g'][domain]
  matrices={'Source:raw':matrix(split,domain,'Source',scenes,receipts,{},cpu)};comparisons={};signals={};decisions=[];loo=[];contributions=[];top=[];decompositions=[]
  for scene in scenes:
   for dec,v in receipts[scene]['Source']['decision'].items():decisions.append({'scene':scene,'model':'Source','state':'raw','decision':dec,**v})
  for model in ('C21701','C21702'):
   medians={s:receipts[s]['C'][model]['m'] for s in scenes};raw=matrix(split,domain,model,scenes,receipts,{},cpu);matrices[model+':raw']=raw;shifted=matrix(split,domain,model,scenes,receipts,medians,cpu);matrices[model+':J1']=shifted
   # A common log-score translation cancels in every pair. Independently merge actual shifted pool and every diagonal below.
   common=ranked([(cachepath(split,domain,model,s,l),l,g[model]) for s in scenes for l in (1,0)],cpu);assert close(common['AP'],raw['summary']['pooled_AP']) and close(common['AUC'],raw['summary']['pooled_AUC']);J0={'cells':[dict(c) for c in raw['cells']],'summary':dict(raw['summary']),'global_log_shift':g[model],'global_translation_pair_identity':True};matrices[model+':J0']=J0
   for c in raw['cells']:
    if c['positive_scene']!=c['negative_scene']:continue
    s=c['positive_scene'];q=ranked([(cachepath(split,domain,model,s,l),l,g[model]) for l in (1,0)],cpu);j=next(v for v in shifted['cells'] if v['positive_scene']==s and v['negative_scene']==s);assert close(q['AP'],c['AP']) and close(q['AUC'],c['AUC']) and close(j['AP'],c['AP']) and close(j['AUC'],c['AUC'])
   comparisons[model]=compare_cells(matrices['Source:raw'],raw);sig=signal(comparisons[model],medians);sig['leave_one_scene']=[{'excluded':s,**{k:v for k,v in signal(comparisons[model],medians,s).items() if k!='scatter'}} for s in scenes];vals=[v['Spearman'] for v in sig['leave_one_scene'] if v['Spearman'] is not None];sig['leave_one_rho_range']=[min(vals),max(vals)] if vals else None;signals[model]=sig
   for scene in scenes:
    p,y,rows=scene_arrays(split,domain,scene,model,False);t=js(R/'RAW_THRESHOLDS.json')[domain][model]
    for state,shift in [('raw',0),('J0',g[model]),('J1',medians[scene])]:
     ds=decision(p,y,shift,t,state=='raw')
     for dec,v in ds.items():
      if state=='raw':assert v['confusion']==old_scene(split,domain,model,scene,dec)['confusion']
      decisions.append({'scene':scene,'model':model,'state':state,'decision':dec,'shift':shift,'raw_frozen_threshold':t,**v})
    del p,y;cpu.check()
   for excluded in scenes:
    ss=matrix_summary(matrices['Source:raw']['cells'],[s for s in scenes if s!=excluded]);cc=matrix_summary(raw['cells'],[s for s in scenes if s!=excluded]);loo.append({'model':model,'excluded':excluded,'Source':ss,'C':cc,'pooled_delta':None if ss['pooled_AUC'] is None or cc['pooled_AUC'] is None else cc['pooled_AUC']-ss['pooled_AUC'],'scene_mean_delta':None if ss['scene_mean_AUC'] is None or cc['scene_mean_AUC'] is None else cc['scene_mean_AUC']-ss['scene_mean_AUC']})
   for scene in scenes:contributions.append({'model':model,'scene':scene,'Shadow_row_contribution':sum(v['weighted_delta'] or 0. for v in comparisons[model] if v['positive_scene']==scene),'Surface_column_contribution':sum(v['weighted_delta'] or 0. for v in comparisons[model] if v['negative_scene']==scene)})
  # Two-seed average is not two more scenes or a third model.
  a=comparisons['C21701'];b=comparisons['C21702'];average=[]
  for x,y in zip(a,b):
   assert (x['positive_scene'],x['negative_scene'])==(y['positive_scene'],y['negative_scene']);average.append({**x,'AUC':None if x['C_AUC'] is None else (x['C_AUC']+y['C_AUC'])/2,'C_AUC':None if x['C_AUC'] is None else (x['C_AUC']+y['C_AUC'])/2,'AP':None if x['AP'] is None else (x['AP']+y['AP'])/2,'delta_AUC':None if x['delta_AUC'] is None else (x['delta_AUC']+y['delta_AUC'])/2,'weighted_delta':None if x['weighted_delta'] is None else (x['weighted_delta']+y['weighted_delta'])/2})
  comparisons['two_seed_mean']=average
  if split=='test':
   cross=[v for v in average if v['positive_scene']!=v['negative_scene'] and v['weighted_delta'] is not None]
   selected=[('damage',v) for v in sorted([v for v in cross if v['weighted_delta']<0],key=lambda v:(v['weighted_delta'],v['positive_scene'],v['negative_scene']))[:3]]+[('benefit',v) for v in sorted([v for v in cross if v['weighted_delta']>0],key=lambda v:(-v['weighted_delta'],v['positive_scene'],v['negative_scene']))[:3]]
   for kind,v in selected:
    top.append({'kind':kind,**v});i=v['positive_scene'];j=v['negative_scene']
    for model in ('C21701','C21702'):
     z={}
     for label,pm,nm in [('SS','Source','Source'),('CS',model,'Source'),('SC','Source',model),('CC',model,model)]:z[label]=ranked([(cachepath(split,domain,pm,i,1),1,0),(cachepath(split,domain,nm,j,0),0,0)],cpu)['AUC']
     shadow=.5*((z['CS']-z['SS'])+(z['CC']-z['SC']));surface=.5*((z['SC']-z['SS'])+(z['CC']-z['CS']));assert abs(shadow+surface-(z['CC']-z['SS']))<1e-12;decompositions.append({'kind':kind,'model':model,'positive_scene':i,'negative_scene':j,**z,'Shadow_component':shadow,'Surface_component':surface,'sum':shadow+surface,'limitation':'symmetric mixed-score arithmetic; not causal feature attribution; cross-model scale included'})
  # Compare raw ranking and supports with the frozen Phase241 full reference.
  for model in ('Source','C21701','C21702'):
   for scope,key in [('pooled','pooled_AUC'),('scene_mean','scene_mean_AUC')]:
    prior=next(v for v in js(R/'PRIOR_METRICS.json')['ranking'] if v['split']==split and v['domain']==domain and v['model']==model and v['state']=='raw' and v['scope']==scope and v['support']=='main' and v['task']=='Shadow_Surface');assert close(matrices[model+':raw']['summary'][key],prior['AUC'])
   priorAP=next(v for v in js(R/'PRIOR_METRICS.json')['ranking'] if v['split']==split and v['domain']==domain and v['model']==model and v['state']=='raw' and v['scope']=='pooled' and v['support']=='main' and v['task']=='Shadow_Surface');assert close(matrices[model+':raw']['summary']['pooled_AP'],priorAP['AP'])
  results[domain]={'scenes':scenes,'g':g,'matrices':matrices,'comparisons':comparisons,'signals':signals,'decisions':decisions,'leave_one':loo,'contributions':contributions,'top_pairs':top,'decompositions':decompositions};save(split.upper()+'_'+domain+'_RESULTS.json',results[domain]);cpu.check()
 save(split.upper()+'_COMPLETE.json',{'passed':True,'epoch':time.time(),'domains':{d:sha(R/(split.upper()+'_'+d+'_RESULTS.json')) for d in results},'network_calls':0})
 if split=='val':
  save('VAL_LOCK.json',{'g':globals_lock,'g_rule':'median of all-valid matched-pixel delta medians across val scenes; equal scenes','signal':'T_ij=m_i-m_j, larger means favorable delta AUC','no_label_features':'class truth not used for m/T/g; published Fill valid domain retained','no_new_threshold':True,'no_signal_selection':True,'test_historically_exposed':True})
  files=['VAL_LOCK.json','VAL_COMPLETE.json','KERNEL_ACCEPTANCE.json','RAW_THRESHOLDS.json','INDEX_REFERENCES.json','MODEL_IDENTITIES.json','LOCK.json']+[p.relative_to(R).as_posix() for p in (R/'code').glob('*') if p.is_file()];save('VAL_SEAL.json',{'epoch':time.time(),'files':{n:sha(R/n) for n in files},'test_new_diagnosis_not_started':not (R/'TEST_STARTED.json').exists()})
 cpu.finish('passed')
if __name__=='__main__':
 try:main()
 except BaseException:
  save(sys.argv[1].upper()+'_FAILURE.json',{'traceback':traceback.format_exc()})
  if 'cpu' in globals():cpu.finish('failed')
  raise
