"""SHA-deploy a fresh one-shot shared run; never retries an ambiguous launch."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
R=Path(__file__).resolve().parents[2]/'outputs/phase241/phase241_20261030'
REMOTE='''from pathlib import Path
import hashlib,io,json,os,subprocess,sys,zipfile,shutil
r=Path('/home/scv/shared/phase241_20261030');assert not r.exists(),'Existing run: inspect, never relaunch';b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==EXPECTED
assert shutil.disk_usage(r.parent).free>=80*1024**3;r.mkdir();z=zipfile.ZipFile(io.BytesIO(b));assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(r)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(4194304),b''):h.update(part)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
for n,h in js(r/'BUNDLE_SHA.json').items():assert sha(r/n)==h,n
for dep in js(r/'DEPENDENCY_LOCK.json').values():assert sha(dep['path'])==dep['SHA'],dep['path']
for n,h in js(r/'CONFIG_LOCK.json').items():assert sha(Path('/home/scv/Cloud-Adapter-light/src')/n)==h,n
ids=js(r/'MODEL_IDENTITIES.json');assert sha(ids['Source_remote_path'])==ids['Source_SHA256']
for h in ids['C']:assert sha(h['remote_path'])==h['head_SHA256']
E=js(r/'EVAL_INPUTS.json')
for pools in E.values():
 for rows in pools.values():
  for v in rows:
   for k in ('image_path','mask_path'):assert Path(v[k]).exists(),v[k]
for v in js(r/'L8_VAL_NATIVE_REFERENCES.json'):assert Path(v['file']).exists(),v['file']
for v in js(r/'NATIVE_PLAN.json')['test'].values():assert Path(v['path']).exists(),v['path']
for p in (r/'code').glob('*.py'):compile(p.read_text(),str(p),'exec')
p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-B',str(r/'code/controller_phase241.py')],cwd=r/'code',stdin=subprocess.DEVNULL,stdout=(r/'controller.console.log').open('x'),stderr=(r/'controller.error.log').open('x'),start_new_session=True)
receipt={'PID':p.pid,'root':str(r),'bundle_SHA':EXPECTED,'launched':True,'only_one_shot_no_cron':True};(r/'LAUNCH.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
def main():
 assert not (R/'LAUNCH.json').exists();assert json.loads((R/'LOCAL_KERNEL_TEST.json').read_text())['passed'];b=(R/'bootstrap.zip').read_bytes();s=REMOTE.replace('EXPECTED',repr(hashlib.sha256(b).hexdigest()));cmd="/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c \"import base64;exec(base64.b64decode('"+base64.b64encode(s.encode()).decode()+"'))\""
 for a in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],input=b,capture_output=True,timeout=120);(R/f'launch_attempt{a}.log').write_bytes(p.stdout+p.stderr)
  if p.returncode==0:
   receipt=json.loads(p.stdout);(R/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));return
  if p.stdout or b'Existing run' in p.stderr or b'Traceback' in p.stderr:raise RuntimeError(p.stderr.decode())
  if a<3:time.sleep(5)
 raise RuntimeError(p.stderr.decode())
if __name__=='__main__':main()
