"""Fetch compact progress and audit receipts, without transferring model weights."""
import json
from pathlib import Path
from collect_phase241 import ssh

script = """from pathlib import Path
import json,os
r=Path('/home/scv/shared/phase428_20261101');out={}
for name in ['LAUNCH.json','PREFLIGHT.json','COMPLETE.json','EXIT_CODE','FAILURE.txt','work_dirs/integrated/ACTIVE_SCOPE.json','work_dirs/integrated/UPDATE_CHECK.json']:
 p=r/name
 if p.exists():
  v=p.read_text()
  out[name]=json.loads(v) if name.endswith('.json') else v
if 'LAUNCH.json' in out:
 try:os.kill(out['LAUNCH.json']['PID'],0);out['alive']=True
 except ProcessLookupError:out['alive']=False
if (r/'console.log').exists():out['log_tail']=(r/'console.log').read_text(errors='replace')[-5000:]
print(json.dumps(out))
"""
if __name__ == '__main__':
    data = json.loads(ssh(script))
    root = Path(__file__).resolve().parents[2] / 'outputs/phase428/phase428_20261101'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'LAST_STATUS.json').write_text(json.dumps(data, indent=2))
    for k, v in data.items():
        if k.endswith('.json'):
            (root / Path(k).name).write_text(json.dumps(v, indent=2))
    if 'work_dirs/integrated/ACTIVE_SCOPE.json' in data:
        data['work_dirs/integrated/ACTIVE_SCOPE.json'].pop('trainable_names', None)
    print(json.dumps(data, indent=2))
