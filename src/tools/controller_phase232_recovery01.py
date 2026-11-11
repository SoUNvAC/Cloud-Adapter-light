"""One GPU extraction, then one CPU fitting process, never validation."""
from pathlib import Path
import json,os,subprocess,time,traceback
R=Path('/home/scv/shared/phase232_20261027');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def save(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
def main():
 global p
 assert not (R/'CONTROLLER_STARTED.json').exists();(R/'PID').write_text(str(os.getpid()));save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time()})
 for stage,name,limit in [('GPU','execute_phase232.py',1830),('FIT','fit_phase232.py',3600)]:
  with (R/(stage+'.console.log')).open('x') as out,(R/(stage+'.error.log')).open('x') as err:
   p=subprocess.Popen([PY,'-u','-B',str(R/'code'/name)],stdout=out,stderr=err,cwd=R/'code');save('CONTROLLER_STATUS.json',{'stage':stage,'PID':os.getpid(),'child_PID':p.pid});rc=p.wait(timeout=limit)
  assert rc==0,(stage,rc)
 assert (R/'FIT_EXIT_CODE').read_text().strip()=='0';save('CONTROLLER_STATUS.json',{'stage':'biases_sealed','PID':os.getpid(),'validation_not_read':True});(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  if 'p' in globals() and p.poll() is None:p.terminate()
  (R/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());(R/'EXIT_CODE').write_text('1\n');raise
