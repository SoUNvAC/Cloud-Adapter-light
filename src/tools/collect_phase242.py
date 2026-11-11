"""One bounded query, then only verified small formal delivery if complete."""
from pathlib import Path
import sys,json,hashlib,io,tarfile
from collect_phase241 import ssh
R=Path(__file__).resolve().parents[2]/'outputs/phase242/phase242_20261031'
REMOTE='''from pathlib import Path
import json,os
r=Path('/home/scv/shared/phase242_20261031');out={}
for n in ('LAUNCH.json','CONTROLLER_STATUS.json','WORKER.json','RUN_STATUS.json','KERNEL_ACCEPTANCE.json','CPU_SMOKE_ACCEPTED.json','VAL_COMPLETE.json','VAL_SEAL.json','TEST_COMPLETE.json','INDEPENDENT_ACCEPTANCE.json','DELIVERY_COMPLETE.json'):
 if (r/n).exists():
  v=json.loads((r/n).read_text());out[n]={k:v[k] for k in ('PID','stage','epoch','current','cumulative_CPU_seconds','peak_RSS_bytes','new_bytes','free_bytes','passed','checks','formal_bytes','manifest_SHA','network_calls') if k in v}
for n in ('CONTROLLER_STATUS.json','WORKER.json'):
 if n in out and 'PID' in out[n]:
  try:os.kill(out[n]['PID'],0);out[n]['alive']=True
  except ProcessLookupError:out[n]['alive']=False
if (r/'EXIT_CODE').exists():out['EXIT_CODE']=(r/'EXIT_CODE').read_text().strip()
if out.get('EXIT_CODE') not in (None,'0'):
 for n in ('CONTROLLER_FAILURE.json','VAL_FAILURE.json','TEST_FAILURE.json','VALIDATION_FAILURE.json'):
  if (r/n).exists():out[n]=(r/n).read_text()[-5000:]
 stage=out.get('WORKER.json',{}).get('stage')
 if stage and (r/(stage+'.log')).exists():out['worker_log_tail']=(r/(stage+'.log')).read_text(errors='replace')[-5000:]
print(json.dumps(out))
'''
def main():
 v=json.loads(ssh(REMOTE));(R/'LAST_STATUS.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
 if v.get('EXIT_CODE')!='0' or (R/'TRANSFER_ACCEPTED.json').exists():return
 raw=ssh("from pathlib import Path\nimport io,tarfile,sys\nr=Path('/home/scv/shared/phase242_20261031/delivery');assert r.is_dir();assert sum(p.stat().st_size for p in r.rglob('*') if p.is_file())<=500*1024**2\nb=io.BytesIO()\nwith tarfile.open(fileobj=b,mode='w:gz') as t:t.add(r,arcname='delivery')\nsys.stdout.buffer.write(b.getvalue())")
 (R/'formal_delivery.tar.gz').write_bytes(raw)
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
  assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in t.getmembers());t.extractall(R,filter='data')
 d=R/'delivery';manifest=json.loads((d/'SHA256SUMS.json').read_text());assert hashlib.sha256((d/'SHA256SUMS.json').read_bytes()).hexdigest()==v['DELIVERY_COMPLETE.json']['manifest_SHA']
 for n,h in manifest.items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h,n
 assert json.loads((d/'INDEPENDENT_ACCEPTANCE.json').read_text())['passed'];(R/'RESULT.md').write_bytes((d/'RESULT.md').read_bytes());(R/'TRANSFER_ACCEPTED.json').write_text(json.dumps({'passed':True,'files':len(manifest),'archive_SHA':hashlib.sha256(raw).hexdigest(),'score_or_weight_arrays_transferred':False},indent=2));print('TRANSFER_ACCEPTED')
if __name__=='__main__':main()
