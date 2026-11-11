from pathlib import Path
import json,hashlib,subprocess,base64,zipfile,sys
B=Path(__file__).resolve().parents[2];O=B/'outputs/phase248/phase248_20261031';R='/home/scv/shared/phase248_20261031';mode=sys.argv[1] if len(sys.argv)>1 else 'status'
code='from pathlib import Path\nimport json,hashlib,base64,subprocess,os\nr=Path('+repr(R)+')\n'
if mode=='start':
 files={n:(O/n).read_bytes() for n in ['STAGE_A_REVIEW_LOCKED.json','STAGE_A_REVIEW_SEAL.json']};files['phase248_stage_b_remote.py']=(B/'src/tools/phase248_stage_b_remote.py').read_bytes();compile(files['phase248_stage_b_remote.py'],'phase248_stage_b_remote.py','exec')
 code+='files='+repr({n:base64.b64encode(v).decode() for n,v in files.items()})+'\n'+'''
assert not (r/'STAGE_B_PID').exists()
for n,b64 in files.items():
 p=r/n;raw=base64.b64decode(b64)
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw)
assert hashlib.sha256((r/'STAGE_A_REVIEW_LOCKED.json').read_bytes()).hexdigest()==json.loads((r/'STAGE_A_REVIEW_SEAL.json').read_text())['sha256']
with (r/'stageB.log').open('ab') as log:p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'phase248_stage_b_remote.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(r/'STAGE_B_PID').write_text(str(p.pid))
'''
code+='''
out={}
if (r/'STAGE_B_READY.json').exists():out['ready']=json.loads((r/'STAGE_B_READY.json').read_text())
if (r/'stageB.log').exists():out['log']=(r/'stageB.log').read_text(errors='replace')[-2500:]
print(json.dumps(out))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs','/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -'],input=code,capture_output=True,text=True,encoding='utf-8',timeout=65);assert p.returncode==0,p.stderr;d=json.loads(p.stdout);print(json.dumps(d,indent=2))
if 'ready' in d:(O/'STAGE_B_READY.json').write_text(json.dumps(d['ready'],indent=2),encoding='utf-8')
if mode=='collect':
 assert d.get('ready',{}).get('ready');p=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10','gzs:'+R+'/stageB_visuals.zip',str(O/'stageB_visuals.zip')],capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stderr
 assert hashlib.sha256((O/'stageB_visuals.zip').read_bytes()).hexdigest()==d['ready']['package_SHA256']
 dest=O/'stageB';dest.mkdir(exist_ok=True)
 with zipfile.ZipFile(O/'stageB_visuals.zip') as z:
  for n in z.namelist():assert Path(n).name==n
  z.extractall(dest)
 print('Stage B package SHA verified')
