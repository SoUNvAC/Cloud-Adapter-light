"""Read one compact snapshot; no launches."""
from pathlib import Path
import json,os
R=Path('/home/scv/shared/phase224_20261024');v={}
for n in ['CONTROLLER_STATUS.json','EXIT_CODE','CONTROLLER_FAILURE.txt']:
 p=R/n
 if p.exists():v[n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-3000:]
st=v.get('CONTROLLER_STATUS.json',{})
for kind,pid in [('controller',st.get('PID')),('worker',st.get('child_PID'))]:
 if pid:
  try:os.kill(pid,0);v[kind+'_alive']=True
  except ProcessLookupError:v[kind+'_alive']=False
if st.get('child_root'):
 for n in ['RUN_STATUS.json','FAILURE.txt','FAILURE_COST.json','RESOURCE_FAILURE.txt']:
  p=R/st['child_root']/n
  if p.exists():v[st['child_root']+'/'+n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-2500:]
print(json.dumps(v,indent=2))
