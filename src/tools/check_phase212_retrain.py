"""One compact snapshot; no waits, restarts, inference or training."""
from pathlib import Path
import json
import os

ROOT=Path('/home/scv/shared/phase212_retrain_20261020_run02')
out={}
for n in ['LAUNCH.json','CONTROLLER_STATUS.json','EXIT_CODE','RETRAINING_COMPLETE.json']:
    p=ROOT/n
    if p.is_file():
        s=p.read_text();out[n]=json.loads(s) if p.suffix=='.json' else s.strip()
status=out.get('CONTROLLER_STATUS.json',{});target=status.get('target')
if target:
    for n in ['RUN_STATUS.json','INITIAL_IDENTITY.json','EXIT_CODE','FAILURE.txt','RESOURCE_FAILURE.txt']:
        p=ROOT/target/n
        if p.is_file():
            s=p.read_text();out[target+'/'+n]=json.loads(s) if p.suffix=='.json' else s[-4000:]
    for n in [target+'.error.log',target+'.console.log']:
        p=ROOT/n
        if p.is_file():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-2500));out[n]=f.read().decode(errors='replace')
out['new_persistent_bytes']=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
print(json.dumps(out,ensure_ascii=False))
