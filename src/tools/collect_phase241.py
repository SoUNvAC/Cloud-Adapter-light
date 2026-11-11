"""One compact status query; only formal small delivery after completion."""
from pathlib import Path
import base64,hashlib,json,subprocess,time,io,tarfile
R=Path(__file__).resolve().parents[2]/'outputs/phase241/phase241_20261030'
REMOTE='''from pathlib import Path
import json,os
r=Path('/home/scv/shared/phase241_20261030');out={}
for n in ('LAUNCH.json','CONTROLLER_STATUS.json','WORKER.json','RUN_STATUS.json','CPU_STATUS.json','ANALYSIS_PROGRESS.json','VAL_COMPLETE.json','ALIGNMENT_SEAL.json','TEST_COMPLETE.json','ANALYSIS_COMPLETE.json','INDEPENDENT_ACCEPTANCE.json','DELIVERY_COMPLETE.json'):
 p=r/n
 if p.exists():
  v=json.loads(p.read_text());out[n]={k:v[k] for k in ('PID','stage','epoch','current','cumulative_calls','cumulative_GPU_stage_seconds','seconds','passed','inputs_models','checks','formal_bytes','manifest_SHA','split','domain','model','state','task') if k in v}
for n in ('CONTROLLER_STATUS.json','WORKER.json'):
 if n in out and 'PID' in out[n]:
  try:os.kill(out[n]['PID'],0);out[n]['alive']=True
  except ProcessLookupError:out[n]['alive']=False
if (r/'EXIT_CODE').exists():out['EXIT_CODE']=(r/'EXIT_CODE').read_text().strip()
if out.get('EXIT_CODE') not in (None,'0'):
 for n in ('CONTROLLER_FAILURE.json','VAL_FAILURE.json','CALIBRATION_FAILURE.json','TEST_FAILURE.json','ANALYSIS_FAILURE.json','VALIDATION_FAILURE.json'):
  if (r/n).exists():out[n]=(r/n).read_text()[-5000:]
 stage=out.get('WORKER.json',{}).get('stage')
 if stage and (r/(stage+'.log')).exists():out['worker_log_tail']=(r/(stage+'.log')).read_text(errors='replace')[-5000:]
print(json.dumps(out))
'''
def ssh(script):
 cmd="/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c \"import base64;exec(base64.b64decode('"+base64.b64encode(script.encode()).decode()+"'))\""
 for a in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],capture_output=True,timeout=90)
  if not p.returncode:return p.stdout
  if p.stdout or b'Traceback' in p.stderr:raise RuntimeError(p.stderr.decode())
  if a<3:time.sleep(5)
 raise RuntimeError(p.stderr.decode())
def main():
 v=json.loads(ssh(REMOTE));(R/'LAST_STATUS.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
 if v.get('EXIT_CODE')!='0' or (R/'TRANSFER_ACCEPTED.json').exists():return
 b=ssh("from pathlib import Path\nimport io,tarfile,sys\nr=Path('/home/scv/shared/phase241_20261030/delivery');assert r.is_dir();assert sum(p.stat().st_size for p in r.rglob('*') if p.is_file())<1024**3\ns=io.BytesIO()\nwith tarfile.open(fileobj=s,mode='w:gz') as t:t.add(r,arcname='delivery')\nsys.stdout.buffer.write(s.getvalue())")
 (R/'formal_delivery.tar.gz').write_bytes(b)
 with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
  assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in t.getmembers());t.extractall(R,filter='data')
 d=R/'delivery';manifest=json.loads((d/'SHA256SUMS.json').read_text());assert hashlib.sha256((d/'SHA256SUMS.json').read_bytes()).hexdigest()==v['DELIVERY_COMPLETE.json']['manifest_SHA']
 for n,h in manifest.items():assert hashlib.sha256((d/n).read_bytes()).hexdigest()==h,n
 assert json.loads((d/'INDEPENDENT_ACCEPTANCE.json').read_text())['passed'];(R/'RESULT.md').write_bytes((d/'RESULT.md').read_bytes());(R/'TRANSFER_ACCEPTED.json').write_text(json.dumps({'passed':True,'files':len(manifest),'archive_SHA':hashlib.sha256(b).hexdigest(),'weights_or_score_arrays_transferred':False},indent=2));print('TRANSFER_ACCEPTED')
if __name__=='__main__':main()
