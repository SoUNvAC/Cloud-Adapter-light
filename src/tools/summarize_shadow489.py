from pathlib import Path
import json,csv
import numpy as np
R=Path('/home/scv/shared/phase489_20261108');read=lambda p:json.loads(p.read_text());refs=read(R/'PREFLIGHT.json')['refs']
def directory(seed,method):
 if method=='N' and seed!=45201:return R/f'seed_{seed}'/'B/network_run'
 return Path(refs[f'{seed}/{method}']['directory'])
def validate(p):
 v=read(p);rows=v['per_image'];r=v['result'];assert len(rows)==535 and len({x['group'] for x in rows})==107
 cm=sum((np.array(x['confusion'],dtype=np.int64) for x in rows),start=np.zeros((3,3),np.int64));diag=np.diag(cm);iou=100*diag/(cm.sum(0)+cm.sum(1)-diag)
 assert np.array_equal(cm,r['confusion']) and np.max(np.abs(iou-r['IoU']))<1e-9
 assert abs(iou.mean()-r['mIoU'])<1e-9 and r['state_unchanged'];return r
seeds=[45201,45202,45203];rows=[];metrics={};onoff=[];texture_onoff=[]
for seed in seeds:
 for method in ['S','N']:
  d=directory(seed,method);f=read(d/'FINAL_RESULTS.json');assert f['passed'] and f['target_label_supervision_percent']==0
  for step in set(f['choices'].values()):validate(d/f'INDEPENDENT_{step}.json')
  for selection in ['last_6000','best_shadow','best_protected']:
   step=f['choices'][selection];r=validate(d/f'INDEPENDENT_{step}.json');v=np.array(r['IoU']+[r['mIoU']]);metrics[seed,method,selection]=v
   rows.append(dict(seed=seed,method=method,selection=selection,step=step,ground=v[0],cloud=v[1],shadow=v[2],mIoU=v[3],protected450=r['protected'],target_supervision_percent=0))
  if method=='N':
   step=f['choices']['best_shadow'];on=validate(d/f'INDEPENDENT_{step}.json');off=validate(d/f'OFF_INDEPENDENT_{step}.json')
   onoff.append(dict(seed=seed,step=step,IoU_delta=(np.array(on['IoU'])-off['IoU']).tolist(),mIoU_delta=on['mIoU']-off['mIoU']))
   texture_off=validate(d/f'REGION_OFF_INDEPENDENT_{step}.json')
   texture_onoff.append(dict(seed=seed,step=step,IoU_delta=(np.array(on['IoU'])-texture_off['IoU']).tolist(),mIoU_delta=on['mIoU']-texture_off['mIoU']))
summary=[];paired=[]
for selection in ['last_6000','best_shadow','best_protected']:
 for method in ['S','N']:
  v=np.stack([metrics[s,method,selection] for s in seeds]);z=dict(method=method,selection=selection,n=3)
  for i,name in enumerate(['ground','cloud','shadow','mIoU']):z[name+'_mean']=float(v[:,i].mean());z[name+'_sd']=float(v[:,i].std(ddof=1))
  summary.append(z)
 for a,b in [('N','S')]:
  v=np.stack([metrics[s,a,selection]-metrics[s,b,selection] for s in seeds]);paired.append(dict(comparison=f'{a}-{b}',selection=selection,seeds=seeds,delta_by_seed=v.tolist(),mean=v.mean(0).tolist(),sample_sd=v.std(0,ddof=1).tolist(),positive_count=(v>0).sum(0).tolist(),all_classes_improve_count=int((v[:,:3]>0).all(1).sum())))
for n,items in [('MAIN.csv',rows),('SUMMARY.csv',summary)]:
 with (R/n).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
(R/'MULTISEED_RESULTS.json').write_text(json.dumps(dict(passed=True,methods={'S':'482B fine redistribution baseline','N':'488B raw-gradient texture evidence'},metric_order=['ground','cloud','shadow','mIoU'],seeds=seeds,primary='last_6000',sample_sd_ddof=1,summary=summary,paired=paired,on_off_best_shadow=onoff,texture_on_off_best_shadow=texture_onoff,independent_validation=dict(images=535,ROI_groups=107,all_metrics_recomputed=True),fixed_source_checkpoint=True),indent=2));print('MULTISEED_RESULTS passed')
