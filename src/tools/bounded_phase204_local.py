"""Bounded local cache analysis, no remote connection or model invocation."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from bounded_phase202_local import usage
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase204/phase204_20261019'
def write(v):
 p=O/'RUN_STATUS.json';t=p.with_suffix('.writing');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
 O.mkdir(parents=True,exist_ok=True);p=O/'LAUNCH.json'
 with p.open('x') as f:json.dump(dict(PID=os.getpid(),epoch=time.time()),f)
 s=dict(status='running',peak_bytes=0,PID=os.getpid());q=None;start=time.time();write(s)
 try:
  for script in ('analyze_phase204_local.py','check_phase204_local.py'):
   q=subprocess.Popen([sys.executable,'-B',str(R/'src/tools'/script)],cwd=R)
   while q.poll() is None:
    s['peak_bytes']=max(s['peak_bytes'],usage(q.pid));assert s['peak_bytes']<=16*1024**3;assert time.time()-start<=7200;assert sum(p.stat().st_size for p in O.rglob('*') if p.is_file())<=4*1024**3;time.sleep(5)
   assert q.returncode==0,(script,q.returncode)
  s.update(status='analysis_complete_pending_report',seconds=time.time()-start);write(s);(O/'EXIT_CODE').write_text('0\n')
 except BaseException:
  if q is not None and q.poll() is None:q.kill();q.wait()
  s.update(status='failed',error=traceback.format_exc(),seconds=time.time()-start);write(s);(O/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
