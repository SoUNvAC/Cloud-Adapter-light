"""Deploy only locally verified stage code into a fresh shared directory."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
R=Path(__file__).resolve().parents[2]/'outputs/phase237/phase237_20261028'
REMOTE='''from pathlib import Path
import hashlib,io,json,os,subprocess,sys,zipfile
r=Path('/home/scv/shared/phase237_20261028');assert not r.exists(),'Existing run: inspect, never relaunch';b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==EXPECTED;r.mkdir();z=zipfile.ZipFile(io.BytesIO(b));assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(r)
for n,h in json.loads((r/'BUNDLE_SHA.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert os.statvfs(r).f_bavail*os.statvfs(r).f_frsize>4*1024**3
source=Path('/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
assert file_sha(source)==json.loads((r/'LOCK.json').read_text())['Source_SHA256']
for dep in json.loads((r/'DEPENDENCY_LOCK.json').read_text()).values():assert file_sha(dep['path'])==dep['SHA'],dep['path']
for name,h in json.loads((r/'CONFIG_LOCK.json').read_text()).items():assert file_sha(Path('/home/scv/Cloud-Adapter-light/src')/name)==h,name
train=json.loads((r/'SUBSET_INPUTS.json').read_text());evals=json.loads((r/'EVAL_INPUTS.json').read_text());paths={}
for rows in list(train.values())+list(evals.values()):
 for v in rows:
  for k in ['image','mask']:paths[v[k+'_path']]=v[k+'_sha256']
for path,h in paths.items():
 pp=Path(path).resolve();assert pp.is_relative_to(Path('/home/scv/Cloud-Adapter-light')) or pp.is_relative_to(Path('/home/scv/shared'));assert file_sha(pp)==h,path
(r/'REMOTE_PREFLIGHT.json').write_text(json.dumps({'status':'passed','Source_exact':True,'dependency_and_config_SHA_exact236':True,'input_file_SHA_count':len(paths),'test_reference_hashed_not_decoded':True,'new_neural_calls':0,'full60_new_calls':0},indent=2))
p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-B',str(r/'code/controller_phase237.py')],cwd=r/'code',stdin=subprocess.DEVNULL,stdout=(r/'controller.console.log').open('x'),stderr=(r/'controller.error.log').open('x'),start_new_session=True)
print(json.dumps({'PID':p.pid,'root':str(r),'bootstrap_SHA':EXPECTED,'launched':True}))
'''
def main():
 assert not (R/'LAUNCH.json').exists();b=(R/'bootstrap.zip').read_bytes();s=REMOTE.replace('EXPECTED',repr(hashlib.sha256(b).hexdigest()));cmd="/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c \"import base64;exec(base64.b64decode('"+base64.b64encode(s.encode()).decode()+"'))\""
 for a in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],input=b,capture_output=True,timeout=70);(R/f'launch_attempt{a}.log').write_bytes(p.stdout+p.stderr)
  if p.returncode==0:
   v=json.loads(p.stdout);(R/'LAUNCH.json').write_text(json.dumps(v,indent=2));print(json.dumps(v));return
  # Remote execution may have begun: retry only an SSH pre-auth connection closure.
  if p.stdout or b'Existing run' in p.stderr:raise RuntimeError(p.stderr.decode())
  if a<3:time.sleep(5)
 raise RuntimeError(p.stderr.decode())
if __name__=='__main__':main()
