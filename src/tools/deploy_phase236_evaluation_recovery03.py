"""Preserve schema failure and continue fixed evaluation without new updates."""
from pathlib import Path
import ast,base64,hashlib,io,json,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase236/phase236_20261027'
def main():
 assert not (O/'EVALUATION_RECOVERY03_LAUNCH.json').exists();files={}
 for n in ['evaluate_phase236_recovery03.py','controller_phase236_evaluation_recovery03.py']:
  p=ROOT/'src/tools'/n;ast.parse(p.read_text());files['code/'+n]=p.read_bytes()
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
  for n,data in files.items():z.writestr(n,data)
 blob=b.getvalue();h=hashlib.sha256(blob).hexdigest();(O/'evaluation_recovery03.zip').write_bytes(blob)
 script='''from pathlib import Path
import os,sys,json,zipfile,io,hashlib,shutil,subprocess,time
r=Path('/home/scv/shared/phase236_20261027');assert not (r/'EVALUATION_RECOVERY03_STARTED.json').exists();assert (r/'EXIT_CODE').read_text().strip()=='1'
for pid in [1821430,1823938]:
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Previous process still alive')
fits=[json.loads(p.read_text()) for p in (r/'fits').glob('*/COMPLETE.json')];assert len(fits)==8 and sum(v['updates'] for v in fits)==32000
assert json.loads((r/'stage_costs/evaluate_val.json').read_text())['calls_delta']=={}
assert (r/'L8_val_PER_INPUT.jsonl').stat().st_size==0 and not (r/'VAL_SELECTION_LOCK.json').exists() and not (r/'TEST_STARTED.json').exists()
before=(r/'CALL_LEDGER.jsonl').read_bytes();assert len(before.splitlines())==32064
costs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'stage_costs').glob('*.json')};heads={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'fits').glob('*/*.pth')};assert len(heads)==64
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==EXPECTED;z=zipfile.ZipFile(io.BytesIO(b));assert all(n.startswith('code/') and '..' not in Path(n).parts for n in z.namelist())
old=r/'failure_evaluation01';assert not old.exists();old.mkdir();shutil.copy2(r/'code/evaluate_phase236.py',old/'evaluate_phase236.py');shutil.copy2(r/'BUNDLE_SHA.json',old/'BUNDLE_SHA.json')
for n in ['EXIT_CODE','PID','CONTROLLER_STATUS.json','RUN_STATUS.json','CONTROLLER_FAILURE.json','VAL_STARTED.json','VAL_FAILURE.json','L8_val_PER_INPUT.jsonl','val.console.log','val.error.log']:
 if (r/n).exists():shutil.move(r/n,old/n)
z.extractall(r);seal=json.loads((r/'BUNDLE_SHA.json').read_text())
for n in z.namelist():seal[n]=hashlib.sha256((r/n).read_bytes()).hexdigest()
(r/'BUNDLE_SHA.json').write_text(json.dumps(seal,indent=2))
for n,h in seal.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert (r/'CALL_LEDGER.jsonl').read_bytes()==before
for n,h in costs.items():assert hashlib.sha256((r/'stage_costs'/n).read_bytes()).hexdigest()==h
for n,h in heads.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
(r/'EVALUATION_RECOVERY03_ACTIVATION.json').write_text(json.dumps({'epoch':time.time(),'failure':'native binding field name mismatch before any evaluation forward','original_attempt_preserved':str(old),'calls_before':32064,'training_updates_total':32063,'all_64_head_resume_SHA_unchanged':heads,'stage_costs_preserved':costs,'no_refit_or_retraining':True,'package_SHA':EXPECTED},indent=2))
p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-B',str(r/'code/controller_phase236_evaluation_recovery03.py')],cwd=r/'code',stdin=subprocess.DEVNULL,stdout=(r/'recovery03.controller.console.log').open('x'),stderr=(r/'recovery03.controller.error.log').open('x'),start_new_session=True)
print(json.dumps({'PID':p.pid,'root':str(r),'package_SHA':EXPECTED,'evaluation_only':True,'training_updates_unchanged':32063,'prior_calls_preserved':32064}))
'''.replace('EXPECTED',repr(h))
 cmd='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c "import base64;exec(base64.b64decode(\''+base64.b64encode(script.encode()).decode()+'\'))"'
 p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],input=blob,capture_output=True,timeout=120);(O/'EVALUATION_RECOVERY03_LAUNCH_LOG.txt').write_bytes(p.stdout+p.stderr);assert p.returncode==0,p.stderr.decode();result=json.loads(p.stdout);(O/'EVALUATION_RECOVERY03_LAUNCH.json').write_text(json.dumps(result,indent=2))
 old=O/'failure_evaluation01';old.mkdir(exist_ok=True)
 for n in ['delivery','TRANSFER_ACCEPTED.json','REMOTE_STATUS.json']:
  if (O/n).exists():assert not (old/n).exists();shutil.move(O/n,old/n)
 for n,data in files.items():dest=O/n;assert not dest.exists();dest.write_bytes(data)
 print(json.dumps(result))
if __name__=='__main__':main()
