"""Compact single Phase223 status read, no launches."""
from pathlib import Path
import json,os
R=Path('/home/scv/shared/phase223_20261024');out={}
for n in ['PID','RUN_STATUS.json','EXIT_CODE','FAILURE.txt','FAILURE_COST.json','RESOURCE_FAILURE.txt']:
 p=R/n
 if p.exists():out[n]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-4000:]
if 'PID' in out:
 try:os.kill(int(out['PID']),0);out['alive']=True
 except ProcessLookupError:out['alive']=False
print(json.dumps(out,indent=2))
