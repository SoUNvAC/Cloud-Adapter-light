from pathlib import Path
import os,sys,subprocess,json,traceback
R=Path('/home/scv/shared/phase444_retrain_20261103')
def one(arm):
    with (R/(arm+'.log')).open('w') as log:return subprocess.Popen([sys.executable,str(R/'train_phase444.py'),arm],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
def main():
    subprocess.run([sys.executable,str(R/'check_phase444.py')],check=True)
    workers={a:one(a) for a in ['baseline429','source431']};(R/'WORKERS.json').write_text(json.dumps({a:p.pid for a,p in workers.items()}))
    failed=[]
    for arm,p in workers.items():
        code=p.wait()
        if code:failed.append(dict(model=arm,exit=code));continue
        with (R/(arm+'_eval.log')).open('w') as log:
            code=subprocess.run([sys.executable,str(R/'eval_phase444.py'),arm],stdout=log,stderr=subprocess.STDOUT).returncode
        if code:failed.append(dict(model=arm,evaluation_exit=code))
    assert not failed,failed
    subprocess.run([sys.executable,str(R/'report_phase444.py')],check=True);(R/'EXIT_CODE').write_text('0')
if __name__=='__main__':
    try:main()
    except BaseException:(R/'EXIT_CODE').write_text('1');(R/'FAILURE.txt').write_text(traceback.format_exc());raise
