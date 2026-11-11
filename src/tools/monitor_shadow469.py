from collect_phase241 import ssh
print(ssh("""from pathlib import Path
import json
r=Path('/home/scv/shared/phase469_20261107');out={}
for a in ['A','B']:
 d=r/a/'network_run';v={}
 for n in ['STATUS','FAILURE','INITIAL_REPRODUCTION','UPDATE_CHECK','COMPLETE','FINAL_RESULTS','ABLATION_OFF','ABLATION_RELATIVE_OFF']:
  p=d/(n+'.json')
  if p.exists():
   data=json.loads(p.read_text());v[n]=data if n in ['STATUS','FAILURE','UPDATE_CHECK'] else True
 p=d/'HISTORY.json'
 if p.exists():v['history']=[dict(step=x['step'],IoU=x['IoU'],protected=x['protected']) for x in json.loads(p.read_text())]
 out[a]=v
print(json.dumps(out))
""").decode())
