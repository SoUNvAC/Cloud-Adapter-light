"""One fresh 40K experiment, compared to the completed same-seed Phase496."""
import json,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/home/scv/shared/phase500_20261111')
def status(state,**kw):
    (R/'STATUS.json').write_text(json.dumps(dict(state=state,time=time.time(),**kw),indent=2))
def run(module,work,args=None):
    with (R/(work+'.log')).open('x') as f:
        cmd=[sys.executable,'-u','-m',module]
        cmd+=args if args is not None else ['texture','--work-dir',str(R/work)]
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        status(work,pid=p.pid)
        assert p.wait()==0,work
def main():
    run('phase500.verify','unit',[])
    run('phase500.preflight','preflight_texture')
    # Verify every parameter of the new early projections actually changes in real training.
    from .verify import check_smoke
    check_smoke()
    start=time.time()
    run('phase500.train','texture')
    status('evaluating',training_wall_seconds=time.time()-start)
    from .evaluate import main as evaluate
    evaluate()
    status('complete',total_wall_seconds=time.time()-start)
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'FAILURE.txt').write_text(traceback.format_exc());status('failed');raise
