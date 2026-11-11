"""Single snapshot and, only at EXIT0, seal small shared text delivery once."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path('/home/scv/shared/phase212_retrain_20261020_run02');OUT=ROOT/'interpolation'
out={}
for name in ['EVALUATION_LAUNCH.json']:
    p=ROOT/name
    if p.is_file():out[name]=json.loads(p.read_text())
for name in ['RUN_STATUS.json','COMPLETED.json','EXIT_CODE','FAILURE.txt','FAILURE_COST.json','ENDPOINT_REPRODUCTION.json']:
    p=OUT/name
    if p.is_file():out[name]=json.loads(p.read_text()) if p.suffix=='.json' else p.read_text()[-3000:]
if (OUT/'EXIT_CODE').is_file() and (OUT/'EXIT_CODE').read_text().strip()=='0':
    package=ROOT/'small_evaluation_delivery.zip'
    if not package.exists():
        paths=[(p,p.name) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl','.txt']]
        paths.append((OUT/'EXIT_CODE','EXIT_CODE'))
        for target in ['l8','sparcs']:
            for name in ['COMPLETED.json','INITIAL_IDENTITY.json','FROZEN_FINAL_AUDIT.json','CALL_LEDGER.jsonl','EXIT_CODE']:
                p=ROOT/target/name;paths.append((p,'training_receipts/'+target+'/'+name))
        paths.append((ROOT/'RETRAINING_COMPLETE.json','training_receipts/RETRAINING_COMPLETE.json'))
        manifest={name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,name in paths}
        with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,name in paths:z.write(p,name)
            z.writestr('TEXT_DELIVERY_MANIFEST.json',json.dumps(manifest,indent=2))
    out['package']={'path':str(package),'bytes':package.stat().st_size,'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'weights_logits_pixels_included':False}
else:
    p=ROOT/'evaluation.error.log'
    if p.is_file():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));out['stderr_tail']=f.read().decode(errors='replace')
print(json.dumps(out,ensure_ascii=False))
