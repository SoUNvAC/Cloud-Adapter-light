"""One local CPU stage with existing native Windows memory accounting."""
import json,os,subprocess,sys,time,shutil
from pathlib import Path
from bounded_phase202_local import usage
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase210/phase210_20261020'
def main():
 stage=sys.argv[1];assert stage in ['analysis','validation'];folder=O/('cpu_'+stage);folder.mkdir(exist_ok=False);start=time.monotonic();peak=0;target='analyze_phase210.py' if stage=='analysis' else 'validate_phase210.py'
 with (folder/'console.log').open('x') as out,(folder/'error.log').open('x') as err:
  p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools'/target)],cwd=R,stdout=out,stderr=err)
  while True:
   peak=max(peak,usage(p.pid) if p.poll() is None else 0);seconds=time.monotonic()-start;status=dict(stage=stage,status='running' if p.poll() is None else 'completed' if p.returncode==0 else 'failed',worker_pid=p.pid,wrapper_pid=os.getpid(),seconds=seconds,peak_bytes=peak);(folder/'STATUS.json').write_text(json.dumps(status))
   if peak>16*1024**3 or seconds>7200 or shutil.disk_usage('D:/').free<4000000000:p.kill();p.wait();break
   if p.poll() is not None:break
   time.sleep(2)
 (folder/'EXIT_CODE').write_text(str(p.returncode));sys.exit(p.returncode)
if __name__=='__main__':main()
