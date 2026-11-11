from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import json
r=Path('/home/scv/shared/phase452_20261106');out={}
for a in ['A','B']:
 d=r/a/'network_run';v={}
 for n in ['STATUS','FAILURE','UPDATE_CHECK','COMPLETE']:
  p=d/(n+'.json')
  if p.exists():
   x=json.loads(p.read_text());v[n]=x if n!='COMPLETE' else dict(passed=x['passed'],seconds=x['seconds'])
 p=d/'HISTORY.json'
 if p.exists():v['curve']=[dict(step=x['step'],IoU=x['IoU'],protected=x['protected']) for x in json.loads(p.read_text())]
 p=d/'LOCK.json'
 if p.exists():
  x=json.loads(p.read_text());v['parameters']={k:x[k] for k in ['branch_parameters','task_parameters']}
 v['independent_complete']=(d/'FINAL_RESULTS.json').exists();out[a]=v
print(json.dumps(out))
''').decode())
