"""Activate the tested Fill correction and explicitly authorized 32063 budget."""
from pathlib import Path
import ast,base64,hashlib,io,json,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase236/phase236_20261027'
def main():
 assert not (O/'RECOVERY02_LAUNCH.json').exists();files={}
 for n in ['phase236_common.py','phase236_valid_mask.py','train_phase236_recovery02.py','evaluate_phase236.py','controller_phase236_recovery02.py']:
  p=ROOT/'src/tools'/n;ast.parse(p.read_text(encoding='utf-8'));files['code/'+n]=p.read_bytes()
 accepted=json.loads((O/'fill_fix02_receipts/FILL_FIX02_ACCEPTED.json').read_text());assert accepted['status']=='passed'
 assert hashlib.sha256(files['code/phase236_valid_mask.py']).hexdigest()==accepted['module_SHA']
 auth={'user_message':'允许32063次更新','allowed_total_updates':32063,'allowed_total_training_backwards':32063,'previous_failed_updates':63,'fresh_fits':8,'fresh_updates':32000,'maximum_full_input_forwards':85000,'maximum_GPU_seconds':21600,'remaining_scientific_settings':'unchanged original Phase236'}
 files['RECOVERY02_AUTHORIZATION.json']=json.dumps(auth,ensure_ascii=False,indent=2).encode()
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
  for n,data in files.items():z.writestr(n,data)
 blob=b.getvalue();h=hashlib.sha256(blob).hexdigest();(O/'recovery02.zip').write_bytes(blob)
 script='''from pathlib import Path
import os,sys,json,zipfile,io,hashlib,shutil,subprocess,time
r=Path('/home/scv/shared/phase236_20261027');assert not (r/'CONTROLLER_RECOVERY02_STARTED.json').exists();assert (r/'EXIT_CODE').read_text().strip()=='1'
for pid in [1820757,1820850,1820989]:
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Old process still alive')
gate=json.loads((r/'FILL_FIX02_ACCEPTED.json').read_text());assert gate['status']=='passed';assert len(gate['sparse_singleton_cases'])==3
fit=r/'fits/L8_21701_1e-04';assert len((fit/'LOSSES.jsonl').read_text().splitlines())==63;assert not list(fit.glob('head_*.pth'))
ledger=(r/'CALL_LEDGER.jsonl').read_bytes();costs={p.name:p.read_bytes() for p in (r/'stage_costs').glob('*.json')};assert len(ledger.splitlines())==64
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==EXPECTED;z=zipfile.ZipFile(io.BytesIO(b));assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
old=r/'failure_training01';assert not old.exists();old.mkdir();shutil.copytree(r/'code',old/'code');shutil.copy2(r/'BUNDLE_SHA.json',old/'BUNDLE_SHA.json');shutil.move(fit,old/'fit')
for n in ['CONTROLLER_STARTED.json','CONTROLLER_STATUS.json','RUN_STATUS.json','CONTROLLER_FAILURE.json','controller.console.log','controller.error.log','EXIT_CODE','PID','FILL_GATE_ACCEPTED.json','FILL_IMPLEMENTATION_DIFF.json','controller_stages','train_L8_21701_1e-4.console.log','train_L8_21701_1e-4.error.log']:
 if (r/n).exists():shutil.move(r/n,old/n)
z.extractall(r)
assert hashlib.sha256((r/'code/phase236_valid_mask.py').read_bytes()).hexdigest()==gate['module_SHA']
shutil.copy2(r/'FILL_FIX02_ACCEPTED.json',r/'FILL_GATE_ACCEPTED.json');shutil.copy2(r/'FILL_FIX02_IMPLEMENTATION_DIFF.json',r/'FILL_IMPLEMENTATION_DIFF.json')
seal=json.loads((r/'BUNDLE_SHA.json').read_text())
for n in z.namelist():seal[n]=hashlib.sha256((r/n).read_bytes()).hexdigest()
(r/'BUNDLE_SHA.json').write_text(json.dumps(seal,indent=2))
for n,h in seal.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert (r/'CALL_LEDGER.jsonl').read_bytes()==ledger
for n,b in costs.items():assert (r/'stage_costs'/n).read_bytes()==b
(r/'RECOVERY02_ACTIVATION.json').write_text(json.dumps({'epoch':time.time(),'prior_updates':63,'prior_calls':64,'failed_attempt_preserved':str(old),'all_original_costs_and_calls_unchanged':True,'accepted_gate_SHA':hashlib.sha256((r/'FILL_GATE_ACCEPTED.json').read_bytes()).hexdigest(),'package_SHA':EXPECTED},indent=2))
p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-B',str(r/'code/controller_phase236_recovery02.py')],cwd=r/'code',stdin=subprocess.DEVNULL,stdout=(r/'recovery02.controller.console.log').open('x'),stderr=(r/'recovery02.controller.error.log').open('x'),start_new_session=True)
print(json.dumps({'PID':p.pid,'root':str(r),'package_SHA':EXPECTED,'total_updates_authorized':32063,'prior_failed_calls_preserved':64}))
'''.replace('EXPECTED',repr(h))
 cmd='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c "import base64;exec(base64.b64decode(\''+base64.b64encode(script.encode()).decode()+'\'))"'
 p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],input=blob,capture_output=True,timeout=90);(O/'RECOVERY02_LAUNCH_LOG.txt').write_bytes(p.stdout+p.stderr);assert p.returncode==0,p.stderr.decode();result=json.loads(p.stdout);(O/'RECOVERY02_LAUNCH.json').write_text(json.dumps(result,indent=2))
 old=O/'failure_training01';old.mkdir(exist_ok=True)
 if (O/'delivery').exists():assert not (old/'delivery').exists();shutil.move(O/'delivery',old/'delivery')
 for n in ['TRANSFER_ACCEPTED.json','REMOTE_STATUS.json']:
  if (O/n).exists():shutil.move(O/n,old/n)
 for n,data in files.items():
  dest=O/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():archive=old/n;archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();shutil.copy2(dest,archive)
  dest.write_bytes(data)
 print(json.dumps(result))
if __name__=='__main__':main()
