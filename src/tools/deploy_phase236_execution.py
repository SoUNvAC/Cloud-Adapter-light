"""Preserve exited gate attempt; deploy one SHA-sealed execution package."""
from pathlib import Path
import ast,base64,hashlib,io,json,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase236/phase236_20261027'
def main():
 assert not (O/'EXECUTION_LAUNCH.json').exists();files={}
 for n in ['phase236_common.py','phase236_valid_mask.py','gate_phase236.py','train_phase236.py','evaluate_phase236.py','controller_phase236.py']:
  p=ROOT/'src/tools'/n;ast.parse(p.read_text());files['code/'+n]=p.read_bytes()
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
  for n,data in files.items():z.writestr(n,data)
 blob=b.getvalue();h=hashlib.sha256(blob).hexdigest();(O/'execution.zip').write_bytes(blob)
 script='''from pathlib import Path
import os,sys,json,zipfile,io,hashlib,shutil,subprocess,time
r=Path('/home/scv/shared/phase236_20261027');assert not (r/'CONTROLLER_STARTED.json').exists();assert not (r/'FILL_GATE_ACCEPTED.json').exists()
try:os.kill(1820641,0)
except ProcessLookupError:pass
else:raise RuntimeError('Original gate still alive')
assert (r/'FILL_GATE_FAILURE.json').exists();old=r/'failure_gate01';assert not old.exists();old.mkdir();shutil.copytree(r/'code',old/'code');shutil.copy2(r/'BUNDLE_SHA.json',old/'BUNDLE_SHA.json')
for n in ['FILL_GATE_FAILURE.json','FILL_GATE_STARTED.json','gate.console.log','gate.error.log']:
 if (r/n).exists():shutil.move(r/n,old/n)
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==EXPECTED;z=zipfile.ZipFile(io.BytesIO(b));assert all(n.startswith('code/') and '..' not in Path(n).parts for n in z.namelist());z.extractall(r)
seal=json.loads((r/'BUNDLE_SHA.json').read_text())
for n in z.namelist():seal[n]=hashlib.sha256((r/n).read_bytes()).hexdigest()
(r/'BUNDLE_SHA.json').write_text(json.dumps(seal,indent=2))
(r/'ENGINEERING_RECOVERY01.json').write_text(json.dumps({'previous_gate_exited':True,'failure':'source-text indentation matcher before synthetic loss or any fit','old_attempt_preserved':str(old),'all_failed_calls_and_GPU_costs_preserved':True,'recovery_package_SHA':EXPECTED,'no_research_configuration_change':True},indent=2))
for n,h in seal.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-B',str(r/'code/controller_phase236.py')],cwd=r/'code',stdin=subprocess.DEVNULL,stdout=(r/'controller.console.log').open('x'),stderr=(r/'controller.error.log').open('x'),start_new_session=True)
print(json.dumps({'PID':p.pid,'root':str(r),'package_SHA':EXPECTED}))
'''.replace('EXPECTED',repr(h))
 cmd='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c "import base64;exec(base64.b64decode(\''+base64.b64encode(script.encode()).decode()+'\'))"'
 p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],input=blob,capture_output=True,timeout=90);(O/'EXECUTION_LAUNCH_LOG.txt').write_bytes(p.stdout+p.stderr);assert p.returncode==0,p.stderr.decode();result=json.loads(p.stdout);(O/'EXECUTION_LAUNCH.json').write_text(json.dumps(result,indent=2))
 for n,data in files.items():
  dest=O/n
  if dest.exists():
   archived=O/'failure_gate01'/n;archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,archived)
  dest.write_bytes(data)
 print(json.dumps(result))
if __name__=='__main__':main()
