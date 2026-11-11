from pathlib import Path
import base64,hashlib,json,subprocess,time
R=Path(__file__).resolve().parents[2]/'outputs/phase232/phase232_20261027'
def main():
 data=(R/'code/controller_phase232_recovery01.py').read_bytes();digest=hashlib.sha256(data).hexdigest()
 source='''from pathlib import Path
import base64,hashlib,json,subprocess,time
r=Path('/home/scv/shared/phase232_20261027');assert not (r/'GPU_STARTED.json').exists() and not (r/'CONTROLLER_STARTED.json').exists()
assert not Path('/proc/1809723').exists(),'Original process not exited'
failure=(r/'controller.error.log').read_text();assert 'phase232_20261026/PID' in failure
code=base64.b64decode(DATA);assert hashlib.sha256(code).hexdigest()==DIGEST;p=r/'code/controller_phase232_recovery01.py';assert not p.exists();p.write_bytes(code)
record={'epoch':time.time(),'reason':'controller date path mismatch before any worker start','original_PID':1809723,'original_exited':True,'consumed_model_calls':0,'original_code_SHA':hashlib.sha256((r/'code/controller_phase232.py').read_bytes()).hexdigest(),'recovery_code_SHA':DIGEST};(r/'LAUNCH_RECOVERY01.json').write_text(json.dumps(record,indent=2))
p=subprocess.Popen(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-u','-B',str(p)],cwd=r/'code',stdin=subprocess.DEVNULL,stdout=(r/'recovery01.console.log').open('x'),stderr=(r/'recovery01.error.log').open('x'),start_new_session=True);print(json.dumps({'PID':p.pid,**record}))
'''.replace('DATA',repr(base64.b64encode(data).decode())).replace('DIGEST',repr(digest))
 cmd="/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python -B -c \"import base64;exec(base64.b64decode('"+base64.b64encode(source.encode()).decode()+"'))\""
 for attempt in range(4):
  p=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','gzs',cmd],capture_output=True,timeout=70);(R/f'recovery01_attempt{attempt}.log').write_bytes(p.stdout+p.stderr)
  if p.returncode==0:break
  if b'Traceback' in p.stderr:raise RuntimeError(p.stderr.decode())
  if attempt<3:time.sleep(5)
 assert p.returncode==0,p.stderr.decode();v=json.loads(p.stdout);(R/'LAUNCH_RECOVERY01.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
