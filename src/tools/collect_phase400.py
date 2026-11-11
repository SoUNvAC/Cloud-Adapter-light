"""One compact SSH query. Receive only small shared reports after complete."""
from pathlib import Path
import json,hashlib,io,tarfile
from collect_phase241 import ssh
L=Path(__file__).resolve().parents[2]/'outputs/phase400/phase400_20261031'
REMOTE="""from pathlib import Path
import json,os
r=Path('/home/scv/shared/phase400_20261031');v={}
for n in ('LAUNCH.json','CONTROLLER_STATUS.json','WORKER.json','RUN_STATUS.json','SHORT_RUN.json','FORMAL_ADMISSION.json','DELIVERY_COMPLETE.json'):
 if (r/n).exists():
  z=json.loads((r/n).read_text());v[n]={k:z[k] for k in ('PID','stage','epoch','current','calls_in_stage','updates_in_stage','cumulative_GPU_stage_seconds','passed','normal_plan_estimated_seconds','manifest_SHA','formal_bytes') if k in z}
for n in ('LAUNCH.json','WORKER.json'):
 if n in v and 'PID' in v[n]:
  try:os.kill(v[n]['PID'],0);v[n]['alive']=True
  except ProcessLookupError:v[n]['alive']=False
if (r/'EXIT_CODE').exists():v['EXIT_CODE']=(r/'EXIT_CODE').read_text().strip()
if v.get('EXIT_CODE') not in (None,'0'):
 if (r/'CONTROLLER_FAILURE.json').exists():v['failure']=json.loads((r/'CONTROLLER_FAILURE.json').read_text())
 s=v.get('WORKER.json',{}).get('stage');p=r/'logs'/(str(s)+'.log')
 if p.exists():v['log_tail']=p.read_text(errors='replace')[-7000:]
print(json.dumps(v))
"""
def main():
 v=json.loads(ssh(REMOTE));(L/'LAST_STATUS.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
 if v.get('EXIT_CODE')!='0' or (L/'TRANSFER_ACCEPTED.json').exists():return
 b=ssh("from pathlib import Path\nimport io,tarfile,sys\nr=Path('/home/scv/shared/phase400_20261031/delivery');assert sum(p.stat().st_size for p in r.rglob('*') if p.is_file())<128*1024**2\ns=io.BytesIO()\nwith tarfile.open(fileobj=s,mode='w:gz') as t:t.add(r,arcname='delivery')\nsys.stdout.buffer.write(s.getvalue())")
 (L/'formal_delivery.tar.gz').write_bytes(b)
 with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
  assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in t.getmembers());t.extractall(L,filter='data')
 d=L/'delivery';manifest=json.loads((d/'SHA256SUMS.json').read_text());assert hashlib.sha256((d/'SHA256SUMS.json').read_bytes()).hexdigest()==v['DELIVERY_COMPLETE.json']['manifest_SHA']
 for n,h in manifest.items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h,n
 assert json.loads((d/'INDEPENDENT_ACCEPTANCE.json').read_text())['passed'];(L/'README.md').write_bytes((d/'README.md').read_bytes());(L/'TRANSFER_ACCEPTED.json').write_text(json.dumps({'passed':True,'files':len(manifest),'archive_SHA':hashlib.sha256(b).hexdigest(),'no_weights_arrays_or_work_dirs':True},indent=2));print('TRANSFER_ACCEPTED')
if __name__=='__main__':main()
