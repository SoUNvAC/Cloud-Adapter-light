"""Seal and launch a fresh Phase400 only; no git or historic output mutation."""
from pathlib import Path
import sys,json,hashlib,io,tarfile,base64,ast,subprocess
from collect_phase241 import ssh
ROOT=Path(__file__).resolve().parents[2];L=ROOT/'outputs/phase400/phase400_20261031';RR='/home/scv/shared/phase400_20261031'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=['phase400_common.py','phase400_model.py','phase400_runtime.py','admit_phase400.py','gate_phase400.py','worker_phase400.py','controller_phase400.py','report_phase400.py']
 files={'code/'+n:(ROOT/'src/tools'/n).read_bytes() for n in names}
 for n,b in files.items():ast.parse(b,filename=n)
 deps=['src/cloud_adapter/models/backbones/target_msre_dinov2.py','src/cloud_adapter/models/backbones/utils.py','src/cloud_adapter/datasets/phase65_catalogue.py','src/cloud_adapter/optimizers/peft_optimizer_constructor.py','src/configs/protocol/phase65a_explore_rgb.py','src/configs/protocol/phase64_source_parent_rgb.py','src/tools/audit_phase52_sparse_msre.py','src/tools/audit_phase64_model_matrix.py','src/tests/test_phase52_sparse_msre.py']
 lock={'new_code':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},'existing_dependencies':{n:digest(ROOT/n) for n in deps},'existing_dependencies_LF':{n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in deps},'protocol_SHA':digest(ROOT/'src/research_plans/phase400.md'),'old_code_unmodified':True,'source_training':False}
 matrix={'S':{'trainable':0,'M':False,'H':False},'MH':{'trainable':380577,'M':True,'H':True},'M':{'trainable':354433,'M':True,'H':False},'H':{'trainable':26144,'M':False,'H':True},'formal_seed':40001,'formal_steps_each':4000,'short_steps_each':120,'data':'153 fit representatives/32 exposed development; Source val535 only','normal_GPU_seconds':36000,'all_attempts_GPU_seconds':43200,'max_directed_repairs':2}
 files.update({'CODE_LOCK.json':json.dumps(lock,indent=2).encode(),'EXPERIMENT_MATRIX.json':json.dumps(matrix,indent=2).encode(),'PREFLIGHT_READONLY.json':(L/'PREFLIGHT_READONLY.json').read_bytes(),'PROTOCOL.md':(ROOT/'src/research_plans/phase400.md').read_bytes()})
 files['BUNDLE_SHA.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()}).encode()
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w:gz') as t:
  for n,b in files.items():i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
 payload=buf.getvalue();(L/'BOOTSTRAP.tar.gz').write_bytes(payload)
 ssh("from pathlib import Path;assert not Path('/home/scv/shared/phase400_20261031').exists();assert not Path('/home/scv/shared/phase400_bootstrap_20261031_v2.tar.gz').exists();print('fresh')")
 subprocess.run(['scp','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',str(L/'BOOTSTRAP.tar.gz'),'gzs:/home/scv/shared/phase400_bootstrap_20261031_v2.tar.gz'],check=True,timeout=180)
 script="""from pathlib import Path
import base64,hashlib,tarfile,io,json,subprocess,os
r=Path(%r);assert not r.exists(),'Phase400 already exists; do not restart'
lock=json.loads(%r)
remote_deps={}
for n,h in lock['existing_dependencies_LF'].items():
 raw=(Path('/home/scv/Cloud-Adapter-light')/n).read_bytes();assert hashlib.sha256(raw.replace(bytes([13,10]),bytes([10]))).hexdigest()==h,n;remote_deps[n]=hashlib.sha256(raw).hexdigest()
b=Path(%r).read_bytes();assert hashlib.sha256(b).hexdigest()==%r
r.mkdir()
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.isfile() for m in t.getmembers());t.extractall(r)
for n,h in json.loads((r/'BUNDLE_SHA.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
(r/'REMOTE_DEPENDENCY_SHA.json').write_text(json.dumps(remote_deps,indent=2))
with (r/'controller.log').open('x') as f:
 p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'code/controller_phase400.py')],cwd=r/'code',stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
v={'PID':p.pid,'bundle_SHA':hashlib.sha256(b).hexdigest(),'single_controller':True};(r/'LAUNCH.json').write_text(json.dumps(v));print(json.dumps(v))
"""%(RR,json.dumps(lock),'/home/scv/shared/phase400_bootstrap_20261031_v2.tar.gz',hashlib.sha256(payload).hexdigest())
 result=json.loads(ssh(script));(L/'LAUNCH.json').write_text(json.dumps(result,indent=2));(L/'CODE_LOCK.json').write_bytes(files['CODE_LOCK.json']);(L/'EXPERIMENT_MATRIX.json').write_bytes(files['EXPERIMENT_MATRIX.json']);print(json.dumps(result))
if __name__=='__main__':main()
