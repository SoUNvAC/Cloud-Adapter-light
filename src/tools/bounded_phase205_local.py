"""One resource-bounded local reference join and CPU evaluation."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from bounded_phase202_local import usage
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase205/phase205_20261019'
def write(v):
 p=O/'LOCAL_STATUS.json';t=p.with_suffix('.writing');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 with (O/'LOCAL_LAUNCH.json').open('x') as f:json.dump(dict(PID=os.getpid(),epoch=time.time()),f)
 s=dict(status='running',PID=os.getpid(),peak_bytes=0);q=None;start=time.time();write(s)
 try:
  q=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/analyze_phase205_local.py')],cwd=R)
  while q.poll() is None:
   s['peak_bytes']=max(s['peak_bytes'],usage(q.pid));assert s['peak_bytes']<=16*1024**3 and time.time()-start<7200;s['seconds']=time.time()-start;write(s);time.sleep(5)
  assert q.returncode==0 and (O/'CPU_ACCEPTANCE.json').exists();s.update(status='CPU_complete_pending_final_delivery',seconds=time.time()-start);write(s);(O/'LOCAL_EXIT_CODE').write_text('0\n')
 except BaseException:
  if q is not None and q.poll() is None:q.kill();q.wait()
  s.update(status='failed',failure=traceback.format_exc(),seconds=time.time()-start);write(s);(O/'LOCAL_EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
