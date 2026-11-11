"""Shared235 identity, exact bounded statistics and cumulative stage accounting."""
from pathlib import Path
import hashlib,json,os,threading,time,collections
R=Path('/home/scv/shared/phase235_20261027');SRC=Path('/home/scv/Cloud-Adapter-light/src');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp_'+str(os.getpid())+'_'+str(threading.get_ident()));t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def check_bundle():
 for n,h in js(R/'BUNDLE_SHA.json').items():assert sha(R/n)==h,n
def prior_gpu():return [js(p) for p in R.glob('GPU_*_COMPLETE.json')]
def prior_seconds():return sum(v['GPU_stage_seconds'] for v in prior_gpu())
def ledger():return [json.loads(line) for line in (R/'CALL_LEDGER.jsonl').read_text().splitlines()] if (R/'CALL_LEDGER.jsonl').exists() else []
def bytes_cache():return sum(p.stat().st_size for p in (R/'cache').rglob('*') if p.is_file()) if (R/'cache').exists() else 0
def bytes_formal():return sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and 'cache' not in p.relative_to(R).parts)
