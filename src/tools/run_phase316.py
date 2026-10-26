"""Serial one-shot CPU audit under8GiB/1hour/free4GiB/300MB persistence caps."""
import subprocess
import traceback
from phase316_common import *
def main():
    assert not (ROOT/'RUN_LOCK.json').exists(),'Existing run; inspect instead of retry'
    resource_check();write(ROOT/'RUN_LOCK.json',dict(pid=os.getpid(),git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),code={p.name:sha(p) for p in (R/'src/tools').glob('*phase316*.py')}))
    for p in (R/'src/tools').glob('*phase316*.py'):shutil.copyfile(p,ROOT/'code'/p.name)
    stages=[]
    try:
        for script,args in [('audit_phase316.py',[]),('verify_phase316.py',['--independent']),('report_phase316.py',[])]:
            resource_check();start=time.monotonic()
            with (ROOT/(script+'.log')).open('x') as f:
                p=subprocess.Popen([sys.executable,'-B',str(R/'src/tools'/script),*args],cwd=R,stdout=f,stderr=subprocess.STDOUT)
                try:
                    while p.poll() is None:
                        resource_check()
                        size=sum(q.stat().st_size for q in ROOT.rglob('*') if q.is_file());assert 4*size+10000000<=300000000
                        time.sleep(1)
                except BaseException:
                    p.terminate();p.wait(timeout=15);raise
            stages.append(dict(script=script,exit=p.returncode,seconds=time.monotonic()-start));write(ROOT/'STAGES.json',stages)
            assert p.returncode==0,script+' failed; preserve log, never retry blindly'
        write(ROOT/'STATUS.json',dict(status='completed',stages=len(stages),elapsed_since_start_seconds=time.time()-load(ROOT/'START.json')['unix'],resources=resource_check(),training_updates=0,neural_forwards=0,new_data_requests=0))
        (ROOT/'EXIT_CODE').write_text('0\n');print('Phase316 all CPU stages passed')
    except BaseException:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),stages=stages));(ROOT/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
