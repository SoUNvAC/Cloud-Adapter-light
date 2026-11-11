"""Collect only small Phase213 shared text results; no checkpoints or pixels."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path('/home/scv/shared/phase213_20261021_run04');OUT=ROOT/'evaluation';PKG=ROOT/'phase213_text_delivery.zip'
assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
assert json.loads((OUT/'REMOTE_CPU_ACCEPTANCE.json').read_text())['status']=='passed'
if PKG.exists():print((ROOT/'DELIVERY_RECEIPT.json').read_text());raise SystemExit(0)
files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log','.py'] and p.name!='DELIVERY_RECEIPT.json']+[ROOT/'EXIT_CODE']
assert sum(p.stat().st_size for p in files)<25_000_000
manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
mp=ROOT/'TEXT_DELIVERY_MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(PKG,'x',zipfile.ZIP_DEFLATED) as z:
    for p in files+[mp]:z.write(p,str(p.relative_to(ROOT)))
r={'path':str(PKG),'bytes':PKG.stat().st_size,'sha256':hashlib.sha256(PKG.read_bytes()).hexdigest(),'files':len(files)+1,'contents':'shared text and code only; no weights, pixels, arrays or work_dirs transfer'}
(ROOT/'DELIVERY_RECEIPT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
