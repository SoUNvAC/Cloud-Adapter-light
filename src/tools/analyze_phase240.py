"""CPU-only aggregation of fixed choices and full-scene trade-offs; no refitting."""
from pathlib import Path
import collections,csv,hashlib,json,time
import numpy as np
from phase231_metrics import metrics
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase240/phase240_20261030';D=R/('delivery_recovery03' if (R/'delivery_recovery03').exists() else 'delivery_recovery02' if (R/'delivery_recovery02').exists() else 'delivery');Q=D/('recovery03' if (D/'recovery03').exists() else 'recovery02')
KEYS=['mIoU','Cloud_recall','Surface_to_Cloud_FPR','Shadow_recall','nonShadow_to_Shadow_FPR']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def emit(name,rows):
 save(name+'.json',rows)
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (R/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def identity(v):return (v['branch'],v['strategy'],v['step'],v['source'])
def summarize(gs):
 out=metrics(sum(np.asarray(c,np.int64) for c in gs.values()))
 for k in KEYS:
  vals=[metrics(c)[k] for c in gs.values() if metrics(c)[k] is not None]
  out['scene_mean_'+k]=float(np.mean(vals)) if vals else None;out[k+'_defined_scenes']=len(vals)
 return out
def main():
 start=time.monotonic();assert not (R/'CPU_ANALYSIS_COMPLETE.json').exists()
 assert ((Q/'EXIT_CODE') if (Q/'EXIT_CODE').exists() else (D/'EXIT_CODE')).read_text().strip() in ['0','2']
 lock=js(D/'VAL_SELECTION_LOCK.json');assert sha(D/'VAL_SELECTION_LOCK.json')==js(D/'VAL_SELECTION_SEAL.json')['SHA']
 assert js(D/'TEST_ACCESS_STARTED.json')['epoch']>js(D/'VAL_SELECTION_SEAL.json')['epoch']
 if (Q/'VAL_SELECTION_LOCK.json').exists():
  assert sha(Q/'VAL_SELECTION_LOCK.json')==js(Q/'VAL_SELECTION_SEAL.json')['SHA'];assert js(Q/'TEST_ACCESS_STARTED.json')['epoch']>js(Q/'VAL_SELECTION_SEAL.json')['epoch'];lock['selections']+=js(Q/'VAL_SELECTION_LOCK.json')['selections']
 groups={};meta={};allinput=[];allscene=[];candidates=[];t1=[];t2=[];t3=[];strata=[];budgets=[];heads=[];exposures=[]
 for domain in ['L8','SPARCS']:
  for split in ['val','test']:
   raw={};counts=collections.Counter()
   paths=[D/f'{domain}_{split}_PER_INPUT.jsonl']
   if (Q/f'{domain}_{split}_PER_INPUT.jsonl').exists():paths.append(Q/f'{domain}_{split}_PER_INPUT.jsonl')
   lines=[];table=[]
   for p in paths:
    assert sha(p)==js(p.parent/f'{split.upper()}_COMPLETE.json')['pools'][domain]['per_input_SHA']
    lines.extend(line for line in p.read_text().splitlines() if not (p.parent==Q and split=='test' and json.loads(line)['branch']=='Source'))
    table.extend(row for row in js(p.parent/f'{domain}_{split}_SUMMARY.json') if not (p.parent==Q and split=='test' and row['branch']=='Source'))
   for line in lines:
    v=json.loads(line);key=identity(v);c=np.asarray(v['confusion'],np.int64);assert c.shape==(3,3) and (c>=0).all()
    raw.setdefault(key,{});raw[key].setdefault(v['scene'],np.zeros((3,3),np.int64));raw[key][v['scene']]+=c;counts[key]+=1;allinput.append(v)
   for row in table:
    key=identity(row);assert counts[key]==len(js(R/'EVAL_INPUTS.json')[domain][split])
    assert {g:c.tolist() for g,c in raw[key].items()}==row['scenes'] and sum(raw[key].values()).tolist()==row['confusion']
    assert abs(metrics(row['confusion'])['mIoU']-row['pooled_mIoU'])<1e-10
    if split=='val':candidates.append({k:v for k,v in row.items() if k!='scenes'})
    for g,c in raw[key].items():allscene.append(dict(domain=domain,split=split,branch=row['branch'],strategy=row['strategy'],step=row['step'],source=row['source'],scene=g,**metrics(c)))
   if split=='val':
    for chosen in [z for z in lock['selections'] if z['domain']==domain]:
     pool=[v for v in table if v['branch']==chosen['branch']];assert len(pool)==5
     winner=min(pool,key=lambda c:(-c['pooled_mIoU'],0 if c['source'] else 1,c['step'] or 0));assert identity(winner)==identity(chosen)
     groups[domain,split,chosen['branch']]=raw[identity(chosen)];meta[domain,chosen['branch']]=chosen
   else:
    for row in table:
     role=row['strategy'];groups[domain,split,role]=raw[identity(row)]
  old=ROOT/'outputs/phase236/phase236_20261027/delivery'
  for split in ['val','test']:
   cache=js(old/f'{domain}_{split}_SUMMARY.json');src=next(v for v in cache if v['source'])
   groups[domain,split,'Source']=src['scenes']
   for role in ['C_selected','C4000']:
    ref=next(v for v in js(R/'C_REFERENCES.json') if (v['domain'],v['split'],v['role'])==(domain,split,role));groups[domain,split,role]=ref['data']['scenes']
  C=next(v for v in js(R/'C_BASELINE_VAL_SELECTION.json') if v['domain']==domain)
  for role in ['Source','C_selected','C4000','A','A_endpoint','B','B_endpoint','G','G_endpoint']:
   if (domain,'test',role) not in groups:
    t1.append(dict(domain=domain,role=role,status='NA_missing_training_or_evaluation'));continue
   m=summarize(groups[domain,'test',role]);base=summarize(groups[domain,'test','C_selected']);source=summarize(groups[domain,'test','Source'])
   if role=='Source':step=None;val=summarize(groups[domain,'val','Source'])['mIoU'];params=0
   elif role.startswith('C'):step=C['selected']['step'] if role=='C_selected' else 4000;val=summarize(groups[domain,'val',role])['mIoU'];params=1316516
   else:
    arm=role[0];step=4000 if role.endswith('_endpoint') else meta[domain,arm]['step'];params=516 if arm=='A' else 1316516
    if role.endswith('_endpoint'):
     row=next(z for z in candidates if z['domain']==domain and z['branch']==arm and z['step']==4000);val=row['pooled_mIoU']
    else:val=meta[domain,arm]['pooled_mIoU']
   t1.append(dict(domain=domain,role=role,status='available',trainable_parameters=params,val_step=step,val_mIoU=val,delta_C_pp=m['mIoU']-base['mIoU'],delta_Source_pp=m['mIoU']-source['mIoU'],reference_budget='Source no target fit; C reused236 singleLR five-candidate; new arms4000steps and five-candidate selection',**m))
  # Every test scene, all six directed off-diagonal error flows.
  scenes=list(groups[domain,'test','Source'])
  for g in scenes:
   sc=metrics(groups[domain,'test','Source'][g]);cc=metrics(groups[domain,'test','C_selected'][g]);entry=dict(domain=domain,scene=g,Source_mIoU=sc['mIoU'],C_mIoU=cc['mIoU'],reference_all_Cloud=sc['Cloud_pixels']>0 and sc['Surface_pixels']==0 and sc['Shadow_pixels']==0)
   if domain=='L8':entry['usgs_shadows']=next(v['usgs_shadows'] for v in js(R/'EVAL_INPUTS.json')['L8']['test'] if v['scene']==g)
   for arm in ['A','B','G']:
    if (domain,'test',arm) not in groups:entry[arm+'_mIoU']=None;continue
    c=np.asarray(groups[domain,'test',arm][g]);b=np.asarray(groups[domain,'test','C_selected'][g]);assert np.array_equal(c.sum(1),b.sum(1))
    entry[arm+'_mIoU']=metrics(c)['mIoU'];entry[arm+'_delta_C_pp']=metrics(c)['mIoU']-cc['mIoU']
    entry[arm+'_error_flow_delta_vs_C']={f'{a}->{bname}':int(c[i,j]-b[i,j]) for i,a in enumerate(['Surface','Cloud','Shadow']) for j,bname in enumerate(['Surface','Cloud','Shadow']) if i!=j}
   t3.append(entry)
  if domain=='L8':
   for flag in ['yes','no']:
    ids={v['scene'] for v in js(R/'EVAL_INPUTS.json')['L8']['test'] if v['usgs_shadows']==flag}
    for (d,split,role),gs in groups.items():
     if d==domain and split=='test':strata.append(dict(domain=d,stratum=flag,role=role,no_not_physical_absence=True,**summarize({g:c for g,c in gs.items() if g in ids})))
 for fit in list((D/'fits').glob('*/COMPLETE.json'))+list((Q/'fits').glob('*/COMPLETE.json')):
  c=js(fit);arm=c['branch'];ini=js(fit.parent/'INITIAL.json');cfg=js(fit.parent/'BRANCH_CONFIGURATION.json');lossrows=[json.loads(l) for l in (fit.parent/'LOSSES.jsonl').read_text().splitlines()]
  assert len(lossrows)==c['updates']==4000;assert len(c['nonhead_checks'])>=40 and all(z['nonhead_exact'] and z['nonhead_no_gradient'] for z in c['nonhead_checks'])
  dist=np.asarray([z['augmented_class_counts'] for z in lossrows]);valid=dist[:,:3].sum(1);p=np.divide(dist[:,:3],valid[:,None],out=np.zeros_like(dist[:,:3],dtype=float),where=valid[:,None]>0)
  observation=dict(domain=c['domain'],branch=arm,trainable_parameters=ini['trainable'],frozen_checks=len(c['nonhead_checks']),schedule_exact_original236=c['paired_LR']['exact'],matching=cfg['matching'],training_weights=cfg['training_loss_weights'],class_fraction_quantiles={k:np.quantile(p[:,i],[0,.25,.5,.75,1]).tolist() for i,k in enumerate(['Surface','Cloud','Shadow'])},empty_augmented_crops=int((valid==0).sum()))
  if arm=='B':observation['all_aux_Dice_loss_zero']=all(v==0 for row in lossrows for k,v in row['terms'].items() if k.endswith('loss_dice'));assert observation['all_aux_Dice_loss_zero']
  if arm=='G':
   stats=[z['gradient_stats'] for z in lossrows];observation['gradient_branches']=dict(collections.Counter(z['branch'] for z in stats));cos=[z['cos'] for z in stats if z['cos'] is not None];observation['cos_quantiles']=np.quantile(cos,[0,.25,.5,.75,1]).tolist() if cos else None;observation['gradient_gate']=js(fit.parent/'G_GRADIENT_GATE.json')
  t2.append(observation)
  budgets.append(dict(domain=c['domain'],branch=arm,updates=c['updates'],backward_passes=c['backwards'],raw_train_pool=js(fit.parent/'TRAIN_POOL.json')['valid_support']))
  exposures.extend(dict(domain=c['domain'],branch=arm,scene=g,exposures=n) for g,n in c['scene_exposure'].items())
  for step,h in c['heads'].items():heads.append({**h,'domain':c['domain'],'branch':arm,'step':int(step),'weights_and_resume_not_transferred':True})
 diagnostics=js(D/'DIAGNOSTICS_COMPLETE.json') if (D/'DIAGNOSTICS_COMPLETE.json').exists() else {'gaps':['diagnostic_failed_see_preserved_receipt']}
 if 'rows' in diagnostics:
  for row in diagnostics['rows']:
   if row['state']=='A4000':assert row['A_mask_Source_exact']
 gradsummary=[];masksummary=[]
 for domain in ['L8','SPARCS']:
  for state in ['Source','C2000','C4000']:
   ds=[z['gradient'] for z in diagnostics.get('rows',[]) if z['domain']==domain and z['state']==state]
   cos=[z['cos'] for z in ds if z['cos'] is not None]
   gradsummary.append(dict(domain=domain,state=state,samples=len(ds),conflicts=sum(z['branch']=='conflict' for z in ds),cos_mean=float(np.mean(cos)) if cos else None,cos_values=cos))
  ar=[z for z in diagnostics.get('rows',[]) if z['domain']==domain and z['state']=='A4000']
  masksummary.append(dict(domain=domain,samples=len(ar),mask_exact=sum(z['A_mask_Source_exact'] for z in ar),class_changed=sum(z['A_class_Source_changed'] for z in ar)))
 emit('GRADIENT_DIAGNOSTIC_SUMMARY',gradsummary);emit('A_MASK_INVARIANCE',masksummary)
 emit('TABLE1',t1);emit('TABLE2',t2);emit('TABLE3_ALL_SCENES',t3);emit('VAL_ALL_CANDIDATES',candidates);emit('PER_SCENE',allscene);emit('L8_TEST_STRATA',strata);emit('TRAIN_BUDGET',budgets);emit('SCENE_EXPOSURE',exposures);emit('HEAD_RESUME_IDENTITIES',heads);save('DIAGNOSTICS.json',diagnostics)
 save('COST.json',js(Q/'COMPLETE.json') if (Q/'COMPLETE.json').exists() else js(D/'COMPLETE.json'));assert time.monotonic()-start<10800
 save('CPU_ANALYSIS_COMPLETE.json',{'status':'passed','seconds':time.monotonic()-start,'new_model_calls':0,'no_new_bootstrap':True,'data_and_interpretation_separate':True})
 print('Phase240 CPU tables complete')
if __name__=='__main__':main()
