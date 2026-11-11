"""Compact read-only snapshot; no retries or launches."""
from pathlib import Path
import json,os
R=Path('/home/scv/shared/phase222_20261024');v={}
for n in ['CONTROLLER_STATUS.json','EXIT_CODE','CONTROLLER_FAILURE.txt']:
 p=R/n
 if p.exists():
  v[n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-3000:]
s=v.get('CONTROLLER_STATUS.json',{});child=s.get('child_root')
for label,pid in [('controller',s.get('PID')),('worker',s.get('child_PID'))]:
 if pid:
  try:os.kill(pid,0);v[label+'_alive']=True
  except ProcessLookupError:v[label+'_alive']=False
if child:
 for n in ['RUN_STATUS.json','FAILURE.txt','FAILURE_COST.json','RESOURCE_FAILURE.txt']:
  p=R/child/n
  if p.exists():v[child+'/'+n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-2500:]
print(json.dumps(v,indent=2))
