from pathlib import Path
import os,json,sys,subprocess,time,traceback
R=Path('/home/scv/shared/phase441_20261103')
def main():
    env=dict(os.environ,**json.loads(Path('/home/scv/shared/phase429_20261102/launch_env.json').read_text()))
    env.update(PYTHONPATH='/home/scv/shared/phase431_20261102/code:/home/scv/shared/phase429_20261102/dinov3',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
    subprocess.run([sys.executable,str(R/'prepare_phase441.py')],env=env,check=True)
    jobs={}
    for arm in ['source','adapted','official']:
        with (R/(arm+'_extract.log')).open('x') as f:jobs[arm]=subprocess.Popen([sys.executable,'-u',str(R/'extract_phase441.py'),arm],env=env,stdout=f,stderr=subprocess.STDOUT)
    while any(p.poll() is None for p in jobs.values()):
        (R/'STATUS.json').write_text(json.dumps(dict(stage='extract',jobs={a:dict(pid=p.pid,code=p.poll()) for a,p in jobs.items()})));time.sleep(10)
    assert all(p.returncode==0 for p in jobs.values())
    sigs=[json.loads((R/a/'SIGNATURES.json').read_text()) for a in jobs]
    assert all(all(s[i]['input_tensor_sha']==sigs[0][i]['input_tensor_sha'] for s in sigs) for i in range(567))
    (R/'EXTRACTION_COMPLETE.json').write_text(json.dumps(dict(passed=True,forwards=1701,identical_inputs=True)))
    # Analysis is delivered separately while the frozen feature extraction runs.
    (R/'STATUS.json').write_text(json.dumps(dict(stage='extraction_complete',next='analyze_phase441.py')))
if __name__=='__main__':
    try:main()
    except BaseException:(R/'FAILURE.json').write_text(json.dumps(dict(error=traceback.format_exc())));raise
