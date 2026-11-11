"""Deploy only locally verified stage code into a fresh shared directory."""
from pathlib import Path
import base64,hashlib,json,subprocess,time
R=Path(__file__).resolve().parents[2]/'outputs/phase233/phase233_20261027'
REMOTE='''from pathlib import Path
import hashlib,io,json,os,subprocess,sys,zipfile
r=Path('/home/scv/shared/phase233_20261027');assert not r.exists(),'Existing run: inspect, never relaunch';b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==EXPECTED;r.mkdir();z=zipfile.ZipFile(io.BytesIO(b));assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(r)
for n,h in json.loads((r/'BUNDLE_SHA.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert os.statvfs(r).f_bavail*os.statvfs(r).f_frsize>4*1024**3
p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-B',str(r/'code/execute_phase233.py')],cwd=r/'code',stdin=subprocess.DEVNULL,stdout=(r/'controller.console.log').open('x'),stderr=(r/'controller.error.log').open('x'),start_new_session=True)
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
