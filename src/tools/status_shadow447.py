from collect_phase241 import ssh
print(ssh('''from pathlib import Path
import json
r=Path('/home/scv/shared/phase447_20261106');out={}
for a in ['A','B']:
 d=r/a/'network_run';v={}
 for n in ['STATUS','FAILURE','COMPLETE','FINAL_RESULTS']:
  p=d/(n+'.json')
  if p.exists():
   z=json.loads(p.read_text());v[n]=z if n in ['STATUS','FAILURE','FINAL_RESULTS'] else dict(passed=z['passed'],seconds=z['seconds'])
 p=d/'HISTORY.json'
 if p.exists():v['curve']=[dict(step=x['step'],IoU=x['IoU'],protected=x['protected']) for x in json.loads(p.read_text())]
 out[a]=v
print(json.dumps(out))
''').decode())
