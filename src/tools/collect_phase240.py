"""One bounded status query; completed formal counts/receipts only, not weights."""
from pathlib import Path
import base64,gzip,hashlib,json,subprocess,time
O=Path(__file__).resolve().parents[2]/'outputs/phase240/phase240_20261030'
REMOTE='''from pathlib import Path
import json,os,gzip,base64,hashlib
r=Path('/home/scv/shared/phase240_20261030');names=['CONTROLLER_STATUS.json','RUN_STATUS.json','EXIT_CODE','CONTROLLER_FAILURE.json','DIAGNOSTICS_FAILURE.json','FIT_PROGRESS.json','VAL_COMPLETE.json','TEST_COMPLETE.json','recovery01/LAUNCH.json','recovery01/EXIT_CODE','recovery01/tf32_off_PARITY.json','recovery02/LAUNCH.json','recovery02/CONTROLLER_STATUS.json','recovery02/FIT_PROGRESS.json','recovery02/EXIT_CODE','recovery02/CONTROLLER_FAILURE.json','recovery03/LAUNCH.json','recovery03/CONTROLLER_STATUS.json','recovery03/FIT_PROGRESS.json','recovery03/EXIT_CODE','recovery03/CONTROLLER_FAILURE.json']
out={'status':{n:(r/n).read_text() for n in names if (r/n).exists()}}
current=r/'recovery03' if (r/'recovery03/CONTROLLER_STATUS.json').exists() else r/'recovery02' if (r/'recovery02/CONTROLLER_STATUS.json').exists() else r
if (current/'CONTROLLER_STATUS.json').exists():
 s=json.loads((current/'CONTROLLER_STATUS.json').read_text());pid=s['child_PID']
 try:os.kill(pid,0);out['status']['child_alive']=True
 except ProcessLookupError:out['status']['child_alive']=False
 p=current/(s['stage']+'.error.log')
 if p.exists():
  tail=p.read_text()[-4500:]
  if 'Traceback' in tail or 'Error:' in tail:out['status']['current_error_tail']=tail
if (r/'EXIT_CODE').exists() and not ((r/'recovery03/LAUNCH.json').exists() and not (r/'recovery03/EXIT_CODE').exists()) and not ((r/'recovery02/LAUNCH.json').exists() and not (r/'recovery02/EXIT_CODE').exists()) and not ((r/'recovery01/LAUNCH.json').exists() and not (r/'recovery01/EXIT_CODE').exists()):
 files={};weights={}
 for p in r.rglob('*'):
  if not p.is_file():continue
  n=p.relative_to(r).as_posix()
  if p.suffix=='.pth':weights[n]={'bytes':p.stat().st_size};continue
  if p.suffix=='.zip':continue
  assert p.stat().st_size<200_000_000
  b=p.read_bytes();files[n]={'SHA':hashlib.sha256(b).hexdigest(),'bytes':len(b),'data':base64.b64encode(b).decode()}
 out.update(files=files,weights_not_transferred=weights,remote_total_bytes=sum(p.stat().st_size for p in r.rglob('*') if p.is_file()))
printdata=gzip.compress(json.dumps(out).encode(),compresslevel=4)
import sys
sys.stdout.buffer.write(printdata)
'''
def main():
 cmd='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c "import base64;exec(base64.b64decode(\''+base64.b64encode(REMOTE.encode()).decode()+'\'))"'
 for attempt in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],capture_output=True,timeout=180)
  if p.returncode==0:break
  if attempt<3:time.sleep(5)
 assert p.returncode==0,p.stderr.decode();v=json.loads(gzip.decompress(p.stdout));(O/'REMOTE_STATUS.json').write_text(json.dumps(v['status'],indent=2));print(json.dumps(v['status']))
 if 'files' not in v:return
 d=O/('delivery_recovery03' if 'recovery03/LAUNCH.json' in v['status'] else 'delivery_recovery02' if 'recovery02/LAUNCH.json' in v['status'] else 'delivery_recovery01' if 'recovery01/LAUNCH.json' in v['status'] else 'delivery');d.mkdir(exist_ok=True);manifest={}
 for n,x in v.pop('files').items():
  pp=d/n;assert pp.resolve().is_relative_to(d.resolve());b=base64.b64decode(x['data']);assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['SHA'];pp.parent.mkdir(parents=True,exist_ok=True)
  if pp.exists():assert pp.read_bytes()==b,'Existing receipt preserved'
  else:pp.write_bytes(b)
  manifest[n]={'SHA':x['SHA'],'bytes':len(b)}
 v['manifest']=manifest;v['compressed_bytes']=len(p.stdout);(O/'TRANSFER_ACCEPTED.json').write_text(json.dumps(v,indent=2));print('formal transfer SHA accepted; no weights/arrays/work_dirs transferred')
if __name__=='__main__':main()
