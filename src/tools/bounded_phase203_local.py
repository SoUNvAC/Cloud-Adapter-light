"""One local CPU evaluation with16GiB process guard; never neural inference."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from bounded_phase202_local import usage
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase203/phase203_20261018'
def write(v):
 p=O/'LOCAL_ANALYSIS_STATUS.json';t=p.with_suffix('.writing');t.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8');t.replace(p)
def main():
 assert not (O/'LOCAL_ANALYSIS_LAUNCH.json').exists();(O/'LOCAL_ANALYSIS_LAUNCH.json').write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time()))+'\n');s=dict(status='running',peak_bytes=0,PID=os.getpid());write(s);q=None;started=time.time()
 try:
  q=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/analyze_phase203_local.py')],cwd=R)
  while q.poll() is None:
   s['peak_bytes']=max(s['peak_bytes'],usage(q.pid));assert s['peak_bytes']<=16*1024**3;time.sleep(2)
  assert q.returncode==0,q.returncode;assert (O/'LOCAL_CPU_ACCEPTANCE.json').exists();s.update(status='completed_pending_summary_delivery',seconds=time.time()-started);write(s);(O/'LOCAL_ANALYSIS_EXIT_CODE').write_text('0\n')
 except BaseException:
  if q is not None and q.poll() is None:q.kill();q.wait()
  s.update(status='failed',error=traceback.format_exc(),seconds=time.time()-started);write(s);(O/'LOCAL_ANALYSIS_EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
