from pathlib import Path
import json,subprocess,base64,sys,time,hashlib,zipfile
B=Path(__file__).resolve().parents[2];O=B/'outputs/phase248/phase248_20261031';O.mkdir(parents=True,exist_ok=True);R='/home/scv/shared/phase248_20261031'
mode=sys.argv[1] if len(sys.argv)>1 else 'status';code='from pathlib import Path\nimport json,base64,subprocess,os,hashlib\nr=Path('+repr(R)+')\nout={}\n'
if mode=='start':
 files={'phase248_cases_remote.py':(B/'src/tools/phase248_cases_remote.py').read_bytes(),'PROTOCOL.md':(B/'src/research_plans/PHASE248_FIXED_CASE_AND_LANDSAT_ELIGIBILITY_AUDIT.md').read_bytes()}
 for n,v in files.items():
  if n.endswith('.py'):compile(v,n,'exec')
  (O/n).write_bytes(v)
 (O/'CODE_LOCK.json').write_text(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in files.items()},indent=2),encoding='utf-8')
 code+='r.mkdir(parents=True,exist_ok=True)\nfiles='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+'''
assert not (r/'TASK_PID').exists(),'already launched'
for n,b64 in files.items():
 p=r/n;raw=base64.b64decode(b64)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
with (r/'run.log').open('ab') as log:p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'phase248_cases_remote.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(r/'TASK_PID').write_text(str(p.pid));out['started_PID']=p.pid
'''
code+='''
for name in ['STATUS.json','FAILURE.json','STAGE_A_READY.json','TECHNICAL_AUDIT.json','STAGE_A_CASES.json']:
 if (r/name).exists():out[name]=json.loads((r/name).read_text())
if (r/'run.log').exists():out['log_tail']=(r/'run.log').read_text(errors='replace')[-2500:]
print(json.dumps(out))
'''
for attempt in range(4):
 try:
  p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs','/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=65)
  if p.returncode==0:
   d=json.loads(p.stdout)
   for n,v in d.items():
    if n.endswith('.json'):(O/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
   (O/'LATEST_STAGE_A_REMOTE.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
   print(json.dumps({n:v for n,v in d.items() if n not in ['TECHNICAL_AUDIT.json','STAGE_A_CASES.json']},indent=2));break
  print(p.stderr[-1500:])
 except Exception as e:print(type(e).__name__,str(e))
 if mode=='start':break
 if attempt<3:time.sleep(3)
else:raise RuntimeError('SSH unavailable')
if mode=='collectA':
 assert d.get('STAGE_A_READY.json',{}).get('ready')
 p=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10','gzs:'+R+'/stageA_visuals.zip',str(O/'stageA_visuals.zip')],capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stderr
 assert hashlib.sha256((O/'stageA_visuals.zip').read_bytes()).hexdigest()==d['STAGE_A_READY.json']['package_SHA256']
 dest=O/'stageA';dest.mkdir(exist_ok=True)
 with zipfile.ZipFile(O/'stageA_visuals.zip') as z:
  for n in z.namelist():assert Path(n).name==n,'unsafe zip entry'
  z.extractall(dest)
 print('Stage A visual package verified and extracted')
