"""One independently bounded CPU acceptance, no inference or scheduling."""
import json,os,subprocess,sys,time,shutil
from pathlib import Path
from bounded_phase202_local import usage
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase209/phase209_20261020'
def main():
 folder=O/'cpu_validation';folder.mkdir(exist_ok=False);start=time.monotonic();peak=0
 with (folder/'console.log').open('x') as out,(folder/'error.log').open('x') as err:
  p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/validate_phase209.py')],cwd=R,stdout=out,stderr=err)
  while True:
   peak=max(peak,usage(p.pid) if p.poll() is None else 0);elapsed=time.monotonic()-start;state=dict(status='running' if p.poll() is None else 'completed' if p.returncode==0 else 'failed',worker_pid=p.pid,wrapper_pid=os.getpid(),seconds=elapsed,peak_bytes=peak);(folder/'STATUS.json').write_text(json.dumps(state))
   if peak>16*1024**3 or elapsed>7200 or shutil.disk_usage('D:/').free<4000000000:p.kill();p.wait();break
   if p.poll() is not None:break
   time.sleep(2)
 (folder/'EXIT_CODE').write_text(str(p.returncode));sys.exit(p.returncode)
if __name__=='__main__':main()
