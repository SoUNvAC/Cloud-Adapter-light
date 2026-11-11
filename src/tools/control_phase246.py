"""Start or inspect only the authorized Phase246 task; no recurring monitor."""
from pathlib import Path
import base64, json, hashlib, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase246/phase246_20261031'
REMOTE_ROOT='/home/scv/shared/phase246_20261031'

def main():
 mode=sys.argv[1] if len(sys.argv)>1 else 'status';OUT.mkdir(parents=True,exist_ok=True)
 code=(ROOT/('src/tools/phase246_remote_v2.py' if mode=='recover_parallel' else 'src/tools/phase246_accept_data.py' if mode=='accept' else 'src/tools/phase246_recover_metadata.py' if mode=='native_metadata' else 'src/tools/phase246_remote.py')).read_bytes()
 program="""
from pathlib import Path
import os,json,subprocess,sys,base64,hashlib
root=Path('/home/scv/shared/phase246_20261031')
out={}
"""
 if mode=='recover_parallel':
  program+='''
import signal,time
oldpid=int((root/'TASK_PID').read_text());cmd=(Path('/proc')/str(oldpid)/'cmdline').read_bytes()
if str(root/'phase246_remote.py').encode() not in cmd:raise RuntimeError('PID identity mismatch; no signal sent')
started=(root/'TASK_PID').stat().st_mtime
(root/'START_EPOCH').write_text(str(started))
snapshot=json.loads((root/'STATUS.json').read_text()) if (root/'STATUS.json').exists() else {}
os.kill(oldpid,signal.SIGTERM)
time.sleep(2)
budgets={};previous={}
for filename in ['p26_r46.tar.gz','p35_r42.tar.gz','p39_r25.tar.gz']:
 p=root/('DOWNLOAD_'+filename+'.json');prior=json.loads(p.read_text()) if p.exists() else []
 current=snapshot.get('attempt',0) if snapshot.get('scene','').startswith({'p26_r46.tar.gz':'Irish_026046','p35_r42.tar.gz':'Irish_035042','p39_r25.tar.gz':'Irish_039025'}[filename]) else 0
 used=max(len(prior),current);budgets[filename]=max(0,3-used);previous[filename]={'failures':prior,'requests_used_including_interrupted':used}
(root/'REMAINING_REQUESTS.json').write_text(json.dumps(budgets))
script=root/'phase246_remote_v2.py';payload=base64.b64decode(PAYLOAD);script.write_bytes(payload)
with (root/'task_v2.log').open('ab') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
(root/'TASK_PID').write_text(str(p.pid));recovery={'old_pid':oldpid,'new_pid':p.pid,'original_start_epoch':started,'prior_requests':previous,'remaining_requests':budgets,'partial_files_preserved':True,'script_SHA256':hashlib.sha256(payload).hexdigest()}
(root/'RECOVERY_01.json').write_text(json.dumps(recovery));out['RECOVERY_01.json']=recovery
'''.replace('PAYLOAD',repr(base64.b64encode(code).decode()))
 elif mode=='postflight':
  program+='''
receipt={'files_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),'task_processes':{}}
for name in ['TASK_PID','ACCEPT_PID']:
 p=root/name
 if p.exists():
  pid=int(p.read_text());proc=Path('/proc')/str(pid)
  receipt['task_processes'][name]={'pid':pid,'active':proc.exists() and bool((proc/'cmdline').read_bytes())}
out['POSTFLIGHT.json']=receipt
'''
 elif mode=='native_metadata':
  program+='''
script=root/'phase246_recover_metadata.py';payload=base64.b64decode(PAYLOAD)
script.write_bytes(payload)
p=subprocess.run(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B',str(script)],capture_output=True,text=True,timeout=50)
out['metadata_recovery_exit']=p.returncode
out['metadata_recovery_error']=p.stderr[-1500:]
'''.replace('PAYLOAD',repr(base64.b64encode(code).decode()))
 elif mode=='accept':
  program+='''
script=root/'phase246_accept_data.py';payload=base64.b64decode(PAYLOAD)
if script.exists() and script.read_bytes()!=payload:raise RuntimeError('existing acceptance script differs; no overwrite')
script.write_bytes(payload)
if not (root/'ACCEPT_PID').exists():
 with (root/'accept.log').open('ab') as log:
  p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','GDAL_CACHEMAX':'64','PYTHONDONTWRITEBYTECODE':'1'})
 (root/'ACCEPT_PID').write_text(str(p.pid));out['accept_started_pid']=p.pid
else:out['accept_existing_pid']=(root/'ACCEPT_PID').read_text()
out['accept_script_SHA256']=hashlib.sha256(payload).hexdigest()
'''.replace('PAYLOAD',repr(base64.b64encode(code).decode()))
 elif mode=='prefix':
  program+='''
import tarfile
prefix={}
for p in root.glob('*.part'):
 if not p.name.endswith('.tar.gz.part'):continue
 members=[];error=None
 try:
  with tarfile.open(p,'r|gz') as tar:
   for m in tar:
    members.append({'name':m.name,'bytes':m.size})
    if len(members)>=32:break
 except Exception as e:error=type(e).__name__+': '+str(e)
 prefix[p.name]={'members':members,'incomplete_archive_error':error}
out['PREFIX_METADATA']=prefix
'''
 elif mode=='start':
  program+='''
root.mkdir(parents=True,exist_ok=True)
script=root/'phase246_remote.py'
payload=base64.b64decode(PAYLOAD)
if script.exists() and script.read_bytes()!=payload:raise RuntimeError('existing task script differs; no overwrite')
script.write_bytes(payload)
if not (root/'TASK_PID').exists():
 with (root/'task.log').open('ab') as log:
  p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
 (root/'TASK_PID').write_text(str(p.pid));out['started_pid']=p.pid
else:out['existing_task_pid']=(root/'TASK_PID').read_text()
out['script_SHA256']=hashlib.sha256(payload).hexdigest()
'''.replace('PAYLOAD',repr(base64.b64encode(code).decode()))
 program+='''
for name in ['STATUS.json','ARCHIVE_SUMMARY.json','DATA_ACCEPTANCE.json','RECOVERY_01.json','METADATA_RECOVERY.json']:
 p=root/name
 if p.is_file():out[name]=json.loads(p.read_text())
if root.exists():
 out['scene_status']={p.name:json.loads(p.read_text()) for p in root.glob('STATUS_Irish*.json')}
 out['archives']={p.name:p.stat().st_size for p in root.iterdir() if p.name.endswith(('.tar.gz','.part'))}
 out['archive_receipts']={p.name:json.loads(p.read_text()) for p in root.glob('*_ARCHIVE.json')}
 if (root/'task.log').exists():out['log_tail']=(root/'task.log').read_text(errors='replace')[-3000:]
 if (root/'task_v2.log').exists():out['v2_log_tail']=(root/'task_v2.log').read_text(errors='replace')[-3000:]
 if (root/'accept.log').exists():out['accept_log_tail']=(root/'accept.log').read_text(errors='replace')[-3000:]
print(json.dumps(out))
'''
 attempts=[]
 for i in range(4):
  try:
   r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs','/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -'],input=program,text=True,encoding='utf-8',capture_output=True,timeout=70)
   attempts.append({'attempt':i+1,'exit':r.returncode,'stderr':r.stderr[-1000:]})
   if r.returncode==0:
    v=json.loads(r.stdout);(OUT/'REMOTE_SNAPSHOT.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    for name,receipt in v.get('archive_receipts',{}).items():(OUT/name).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for name in ['ARCHIVE_SUMMARY.json','DATA_ACCEPTANCE.json','RECOVERY_01.json','PREFIX_METADATA','METADATA_RECOVERY.json','POSTFLIGHT.json']:
     if name in v:(OUT/name).write_text(json.dumps(v[name],indent=2),encoding='utf-8')
    brief={k:x for k,x in v.items() if k not in ['archive_receipts','DATA_ACCEPTANCE.json','METADATA_RECOVERY.json']}
    if 'DATA_ACCEPTANCE.json' in v:brief['DATA_ACCEPTANCE_summary']=[{k:r.get(k) for k in ['scene','status','same_grid','joint_classes','MTL_identity_matches']} for r in v['DATA_ACCEPTANCE.json']['results']]
    print(json.dumps(brief,indent=2));break
  except Exception as e:attempts.append({'attempt':i+1,'error':str(e)[:1000]})
  if i<3:time.sleep(3)
 (OUT/('SSH_'+mode+'_'+str(time.time_ns())+'.json')).write_text(json.dumps(attempts,indent=2),encoding='utf-8')

if __name__=='__main__':main()
