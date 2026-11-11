from pathlib import Path
import json
R=Path('/home/scv/shared/phase463_20261106')
for arm in ['B']:
 d=R/arm/'network_run';out={'arm':arm}
 for name in ['STATUS','HISTORY','COMPLETE','FAILURE']:
  p=d/(name+'.json')
  if not p.exists():continue
  x=json.loads(p.read_text())
  if name=='STATUS':out.update(step=x['step'],seconds=round(x['seconds']))
  if name=='HISTORY':out['latest_validation']={k:x[-1][k] for k in ['step','IoU','mIoU','protected']}
  if name=='COMPLETE':out['training_complete']=True
  if name=='FAILURE':out['failure']=x
 out['independent_complete']=(d/'FINAL_RESULTS.json').exists();print(json.dumps(out))
