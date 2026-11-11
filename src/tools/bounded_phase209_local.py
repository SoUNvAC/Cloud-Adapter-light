"""Bounded local CPU stage; one exclusive run per stage, no timer/SSH."""
import json,os,subprocess,sys,time,shutil
from pathlib import Path
from bounded_phase202_local import usage
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase209/phase209_20261020'
def main():
 stage=sys.argv[1];assert stage in ['baseline_A','calibrate','evaluate_B'];folder=O/('cpu_'+stage);folder.mkdir(exist_ok=False);(folder/'PID').write_text(str(os.getpid()));start=time.monotonic();peak=0
 with (folder/'console.log').open('x',encoding='utf8') as out,(folder/'error.log').open('x',encoding='utf8') as err:
  p=subprocess.Popen([sys.executable,str(R/'src/tools/analyze_phase209.py'),stage],cwd=R,stdout=out,stderr=err)
  while True:
   rss=usage(p.pid) if p.poll() is None else 0
   peak=max(peak,rss);seconds=time.monotonic()-start;status=dict(stage=stage,status='running' if p.poll() is None else 'completed' if p.returncode==0 else 'failed',worker_pid=p.pid,wrapper_pid=os.getpid(),seconds=seconds,peak_bytes=peak)
   (folder/'STATUS.json').write_text(json.dumps(status),encoding='utf8')
   if peak>16*1024**3 or seconds>7200 or shutil.disk_usage('D:/').free<4000000000:
    p.kill();p.wait();(folder/'RESOURCE_FAILURE.json').write_text(json.dumps(status));break
   if p.poll() is not None:break
   time.sleep(2)
 (folder/'EXIT_CODE').write_text(str(p.returncode));sys.exit(p.returncode)
if __name__=='__main__':main()
