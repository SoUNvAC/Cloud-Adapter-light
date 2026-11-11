"""Deploy CPU-only code and immutable cache references into fresh shared root."""
from pathlib import Path
import json,hashlib,base64,subprocess,time
R=Path(__file__).resolve().parents[2]/'outputs/phase242/phase242_20261031'
REMOTE='''from pathlib import Path
import hashlib,io,json,os,shutil,subprocess,sys,zipfile
r=Path('/home/scv/shared/phase242_20261031');prior=Path('/home/scv/shared/phase241_20261030');assert not r.exists(),'Existing run: inspect, do not relaunch';b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==EXPECTED;assert shutil.disk_usage(r.parent).free>50*1024**3
r.mkdir();z=zipfile.ZipFile(io.BytesIO(b));assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(r)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
for n,h in js(r/'BUNDLE_SHA.json').items():assert sha(r/n)==h,n
for n,h in js(r/'LOCK.json')['prior_files'].items():assert sha(prior/n)==h,n
assert js(prior/'INDEPENDENT_ACCEPTANCE.json')['passed'];assert js(prior/'DELIVERY_COMPLETE.json')['passed']
for split,rows in js(r/'INDEX_REFERENCES.json').items():
 for v in rows:
  assert split!='test' or v['domain']!='L8' or v['usgs_shadows']=='yes'
  for key in ('score_file','label_file'):
   p=(prior/v[key]).resolve();assert p.is_relative_to(prior) and p.exists(),str(p)
for p in (r/'code').glob('*.py'):compile(p.read_text(),str(p),'exec')
p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'code/controller_phase242.py')],cwd=r/'code',stdin=subprocess.DEVNULL,stdout=(r/'controller.console.log').open('x'),stderr=(r/'controller.error.log').open('x'),start_new_session=True)
v={'PID':p.pid,'root':str(r),'bundle_SHA':EXPECTED,'network_calls':0,'launched':True,'read_only_prior':True};(r/'LAUNCH.json').write_text(json.dumps(v));print(json.dumps(v))
'''
def main():
 assert not (R/'LAUNCH.json').exists();assert json.loads((R/'KERNEL_ACCEPTANCE.json').read_text())['passed'] and json.loads((R/'CPU_SMOKE_ACCEPTED.json').read_text())['passed'];b=(R/'bootstrap.zip').read_bytes();script=REMOTE.replace('EXPECTED',repr(hashlib.sha256(b).hexdigest()));cmd="/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c \"import base64;exec(base64.b64decode('"+base64.b64encode(script.encode()).decode()+"'))\""
 for attempt in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],input=b,capture_output=True,timeout=120);(R/f'launch_attempt{attempt}.log').write_bytes(p.stdout+p.stderr)
  if not p.returncode:v=json.loads(p.stdout);(R/'LAUNCH.json').write_text(json.dumps(v,indent=2));print(json.dumps(v));return
  if p.stdout or b'Traceback' in p.stderr or b'Existing run' in p.stderr:raise RuntimeError(p.stderr.decode())
  if attempt<3:time.sleep(5)
 raise RuntimeError(p.stderr.decode())
if __name__=='__main__':main()
