"""Start one detached Phase200 stage. Never duplicate an existing stage."""
import argparse,subprocess,sys,time
from phase200_common import *
def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['acquire','infer']);args=a.parse_args()
    ROOT.mkdir(exist_ok=True);d=ROOT/args.stage;d.mkdir(exist_ok=False)
    from install_phase111_runtime import host_free
    free=host_free();assert free>=load(LOCK)['host_min_free_bytes']
    for p,h in load(LOCK)['source_sha256'].items():assert sha(REPO/p)==h,p
    if args.stage=='infer':assert load(ROOT/'acquire/STATUS.json')['status']=='prepared' and (ROOT/'POINTS_SEALED.json').exists()
    with (d/'console.log').open('xb') as f:
        p=subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools/run_phase200.py'),args.stage],cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    receipt=dict(PID=p.pid,stage=args.stage,launch_epoch=time.time(),host_free_bytes=free,plan_sha256=sha(LOCK),poll_hours=1)
    write(d/'LAUNCH.json',receipt);(d/'PID').write_text(str(p.pid));print(json.dumps(receipt))
if __name__=='__main__':main()
