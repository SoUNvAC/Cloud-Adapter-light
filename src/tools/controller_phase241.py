"""One-shot pipeline; no scheduled remote task or automatic restart."""
from phase241_common import *
import sys,traceback

def main():
 assert not (R/'CONTROLLER_STARTED.json').exists();check_bundle();assert __import__('shutil').disk_usage(R).free>=80*1024**3
 save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'expected_peak_GiB':40,'free_bytes':__import__('shutil').disk_usage(R).free})
 stages=[('compile',['/usr/bin/g++','-O3','-std=c++17',str(R/'code/phase241_rank_merge.cpp'),'-o',str(R/'code/rank_merge')]),('cpu_smoke',[PY,str(R/'code/test_phase241_cpu.py')]),('val',[PY,str(R/'code/infer_phase241.py'),'val']),('calibrate',[PY,str(R/'code/calibrate_phase241.py')]),('test',[PY,str(R/'code/infer_phase241.py'),'test']),('analyze',[PY,str(R/'code/analyze_phase241.py')]),('validate',[PY,str(R/'code/validate_phase241.py')]),('finalize',[PY,str(R/'code/finalize_phase241.py')])]
 for stage,cmd in stages:
  save('CONTROLLER_STATUS.json',{'PID':os.getpid(),'stage':stage,'epoch':time.time()})
  started=time.monotonic()
  with (R/(stage+'.log')).open('x') as out:
   worker=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,cwd=SRC,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONPATH':str(R/'code')+':'+str(SRC)+':'+str(SRC/'tools'),'PYTHONDONTWRITEBYTECODE':'1','PHASE241_TEST_ROOT':str(R/'CPU_SYNTHETIC_TEST')});save('WORKER.json',{'PID':worker.pid,'stage':stage});rc=worker.wait()
  if stage in ('compile','cpu_smoke'):save(R/'cpu_costs'/f'{stage}.json',{'status':'passed' if rc==0 else 'failed','seconds':time.monotonic()-started,'CPU_peak_bytes':None,'resource_note':'tiny synthetic/compile child, RSS not sampled'})
  if rc:raise RuntimeError(f'{stage} exited {rc}; do not blindly restart')
 save('CONTROLLER_STATUS.json',{'PID':os.getpid(),'stage':'complete','epoch':time.time()});(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  save('CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc()});(R/'EXIT_CODE').write_text('2\n');raise
