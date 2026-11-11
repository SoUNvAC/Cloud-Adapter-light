"""Bounded SSH launcher and small-file collector for the fixed L7 diagnostic."""
from pathlib import Path
import base64,json,subprocess,sys,time,hashlib
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/phase247/phase247_20261031';REMOTE='/home/scv/shared/phase247_20261031';OUT.mkdir(parents=True,exist_ok=True)
def main():
 mode=sys.argv[1] if len(sys.argv)>1 else 'status';program='from pathlib import Path\nimport json,os,subprocess,base64,hashlib\nr=Path('+repr(REMOTE)+')\nout={}\n'
 if mode=='start':
  code=(BASE/'src/tools/phase247_remote.py').read_bytes();compile(code,'phase247_remote.py','exec')
  protocol=(BASE/'src/research_plans/PHASE247_L7_FROZEN_HEAD_EXTERNAL_DIAGNOSIS.md').read_bytes();ident=(BASE/'outputs/phase241/phase241_preparation_20261030/MODEL_IDENTITIES.json').read_bytes()
  files={'phase247_remote.py':code,'PROTOCOL.md':protocol,'MODEL_IDENTITIES.json':ident}
  files['LOCK.json']=json.dumps({'files':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},'created_epoch':time.time(),'primary':'full native 64 vs 128 pair score AUC','old_L8_heads_external_L7_extrapolation':True,'three_groups_only':True,'training':0,'fit':0,'threshold_selection':0}).encode()
  for k,v in files.items():(OUT/k).write_bytes(v)
  payload={k:base64.b64encode(v).decode() for k,v in files.items()}
  program+='r.mkdir(parents=True,exist_ok=True)\nfiles='+repr(payload)+'\n'
  program+='''
for name,b64 in files.items():
 p=r/name;raw=base64.b64decode(b64)
 if p.exists() and p.read_bytes()!=raw:raise RuntimeError('existing file differs; no overwrite '+name)
 p.write_bytes(raw)
if (r/'TASK_PID').exists():raise RuntimeError('already started; use status')
with (r/'run.log').open('ab') as log:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'phase247_remote.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
(r/'TASK_PID').write_text(str(p.pid));out['started_PID']=p.pid
'''
 program+='''
if r.exists():
 for name in ['STATUS.json','FAILURE.json','DATA_SUPPORT_ACCEPTANCE.json','INFERENCE_COMPLETE.json','RESULT_TABLES.json','VALIDATION.json','INDEPENDENT_VALIDATION.json','RECOVERY_COMPLETE.json','SCORE_INDEX.json','SAMPLE_INDEX.json','CONFUSIONS.json','LOCK.json']:
  if (r/name).exists():out[name]=json.loads((r/name).read_text())
 if (r/'run.log').exists():out['log_tail']=(r/'run.log').read_text(errors='replace')[-2000:]
 if (r/'TASK_PID').exists():
  pid=(r/'TASK_PID').read_text().strip();out['PID']=pid;out['process_exists']=Path('/proc',pid).exists()
print(json.dumps(out))
'''
 attempts=[]
 for i in range(4):
  try:
   p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs','/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -'],input=program,capture_output=True,text=True,encoding='utf-8',timeout=65)
   attempts.append({'attempt':i+1,'exit':p.returncode,'stderr':p.stderr[-1500:]})
   if p.returncode==0:
    v=json.loads(p.stdout)
    for k,x in v.items():
     if k.endswith('.json'):(OUT/k).write_text(json.dumps(x,indent=2),encoding='utf-8')
    (OUT/'LATEST_REMOTE.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    print(json.dumps({k:x for k,x in v.items() if k in ['started_PID','STATUS.json','FAILURE.json','INFERENCE_COMPLETE.json','VALIDATION.json','log_tail','PID','process_exists']},indent=2));break
  except Exception as e:attempts.append({'attempt':i+1,'error':str(e)})
  # Mutating start is never automatically retried after an uncertain response.
  if mode=='start':break
  if i<3:time.sleep(3)
 (OUT/('SSH_'+mode+'_'+str(time.time_ns())+'.json')).write_text(json.dumps(attempts,indent=2),encoding='utf-8')
if __name__=='__main__':main()
