"""CPU report queue only; never launches or repeats training."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback
phase=int(sys.argv[1]);r=Path(f'/home/scv/shared/phase{phase}_20261101')
start=time.monotonic()
try:
 while not (r/'EXIT_CODE').exists():
  assert time.monotonic()-start<10800
  time.sleep(15)
 assert (r/'EXIT_CODE').read_text()=='0','training did not complete successfully'
 with (r/'report.log').open('x') as f:
  p=subprocess.run(['/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python','-B','-u',str(r/'report_phase417.py'),str(phase)],cwd=r,stdout=f,stderr=subprocess.STDOUT,timeout=1800,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 assert p.returncode==0,p.returncode
except BaseException:
 (r/'REPORT_FAILURE.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
