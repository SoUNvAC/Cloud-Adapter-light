from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path('/home/scv/shared/phase481_20261108');read=lambda p:json.loads(p.read_text())
def directory(seed,arm):
 if seed==45201:return Path('/home/scv/shared/phase477_20261107/A/network_run') if arm=='C' else Path('/home/scv/shared/phase480_20261108')/arm/'network_run'
 return R/f'seed_{seed}'/arm/'network_run'
def validate(p):
 v=read(p);rows=v['per_image'];r=v['result'];assert len(rows)==535 and len({x['group'] for x in rows})==107
 cm=sum((np.array(x['confusion'],dtype=np.int64) for x in rows),start=np.zeros((3,3),np.int64))
 diag=np.diag(cm);iou=100*diag/(cm.sum(0)+cm.sum(1)-diag)
 assert np.max(np.abs(iou-r['IoU']))<1e-9
 assert abs(iou.mean()-r['mIoU'])<1e-9 and r['state_unchanged']
 return r
rows=[];onoff=[];metrics={};seeds=[45201,45202,45203]
for seed in seeds:
 for arm in ['C','A','B']:
  d=directory(seed,arm);f=read(d/'FINAL_RESULTS.json');assert f['passed'] and f['target_label_supervision_percent']==0
  for step in set(f['choices'].values()):validate(d/f'INDEPENDENT_{step}.json')
  for selection in ['last_6000','best_shadow','best_protected']:
   step=f['choices'][selection];r=validate(d/f'INDEPENDENT_{step}.json');v=np.array(r['IoU']+[r['mIoU']]);metrics[seed,arm,selection]=v
   rows.append(dict(seed=seed,arm=arm,selection=selection,step=step,ground=v[0],cloud=v[1],shadow=v[2],mIoU=v[3],protected=r['protected'],target_supervision_percent=0))
  if arm!='C':
   step=f['choices']['best_shadow'];on=validate(d/f'INDEPENDENT_{step}.json');off=validate(d/f'OFF_INDEPENDENT_{step}.json')
   onoff.append(dict(seed=seed,arm=arm,step=step,IoU_delta=(np.array(on['IoU'])-off['IoU']).tolist(),mIoU_delta=on['mIoU']-off['mIoU']))
summary=[];paired=[]
for selection in ['last_6000','best_shadow','best_protected']:
 for arm in ['C','A','B']:
  v=np.stack([metrics[s,arm,selection] for s in seeds]);z=dict(arm=arm,selection=selection,n=3)
  for i,name in enumerate(['ground','cloud','shadow','mIoU']):z[name+'_mean']=float(v[:,i].mean());z[name+'_sd']=float(v[:,i].std(ddof=1))
  summary.append(z)
 for a,b in [('A','C'),('B','C'),('B','A')]:
  v=np.stack([metrics[s,a,selection]-metrics[s,b,selection] for s in seeds])
  paired.append(dict(comparison=f'{a}-{b}',selection=selection,seeds=seeds,delta_by_seed=v.tolist(),mean=v.mean(0).tolist(),sample_sd=v.std(0,ddof=1).tolist(),positive_count=(v>0).sum(0).tolist()))
def csvwrite(name,items):
 with (R/name).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
csvwrite('MAIN.csv',rows);csvwrite('SUMMARY.csv',summary)
result=dict(passed=True,metric_order=['ground','cloud','shadow','mIoU'],seeds=seeds,primary='last_6000',sample_sd_ddof=1,fixed_source_checkpoint=True,summary=summary,paired=paired,on_off_best_shadow=onoff,independent_validation=dict(images=535,ROI_groups=107,all_selected_confusion_and_metrics_recomputed=True))
(R/'MULTISEED_RESULTS.json').write_text(json.dumps(result,indent=2));print('MULTISEED_RESULTS passed')
