"""Live read-only monitor; never restarts or modifies training."""
import base64,subprocess
script='''
from pathlib import Path
import json,time,re
r=Path('/home/scv/shared/phase497_20261110')
while True:
 state=json.loads((r/'STATUS.json').read_text())
 out={'state':state['state']}
 for v in ['texture']:
  p=r/(v+'.log')
  if p.exists():
   text=p.read_text(errors='replace')
   steps=re.findall(r'Iter\\(train\\) \\[\\s*(\\d+)/40000\\].*?eta: ([^ ]+).*?time: ([0-9.]+)',text)
   vals=re.findall(r'Iter\\(val\\) \\[268/268\\].*?mIoU: ([0-9.]+)',text)
   out[v]={'step':steps[-1][0] if steps else 0,'eta':steps[-1][1] if steps else None,'last_mIoU':vals[-1] if vals else None}
 print(json.dumps(out),flush=True)
 if state['state'] in ['complete','failed']:break
 time.sleep(45)
'''
encoded=base64.b64encode(script.encode()).decode()
command=f'/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -u -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'
raise SystemExit(subprocess.call(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=6','gzs',command]))
