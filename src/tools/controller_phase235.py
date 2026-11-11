"""Separate GPU/CPU stage timers, cumulative call ledger, no retries or resets."""
from phase235_common import *
import subprocess,traceback
def main():
 global child
 assert not (R/'CONTROLLER_STARTED.json').exists();save('CONTROLLER_STARTED.json',{'epoch':time.time(),'PID':os.getpid()});(R/'PID').write_text(str(os.getpid()));check_bundle()
 for stage,script,args,timeout in [('extract','execute_phase235.py',['extract'],3650),('fit','fit_phase235.py',[],14400),('evaluate','execute_phase235.py',['evaluate'],3650)]:
  started=time.monotonic()
  with (R/f'{stage}.console.log').open('x') as out,(R/f'{stage}.error.log').open('x') as err:
   child=subprocess.Popen([PY,'-u','-B',str(R/'code'/script),*args],cwd=R/'code',stdin=subprocess.DEVNULL,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':stage,'PID':os.getpid(),'child_PID':child.pid});rc=child.wait(timeout=timeout)
  save(f'CONTROLLER_{stage}_WALL.json',{'stage':stage,'wall_seconds':time.monotonic()-started,'returncode':rc,'PID':child.pid});assert rc==0,(stage,rc)
 assert len(ledger())<=4800 and prior_seconds()<3600;save('COMPLETE.json',{'status':'complete','counts':dict(collections.Counter(v['kind'] for v in ledger())),'total_calls':len(ledger()),'GPU_stage_seconds':prior_seconds(),'temporary_cache_bytes':bytes_cache(),'formal_bytes':bytes_formal(),'new_network_updates':0,'new_fits':1});save('CONTROLLER_STATUS.json',{'stage':'complete','PID':os.getpid()});(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  if 'child' in globals() and child.poll() is None:child.terminate()
  (R/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());save('FAILURE_COST.json',{'ledger_calls':len(ledger()),'completed_GPU_seconds':prior_seconds(),'controller_stage_wall_receipts_include_failed_stage':True});(R/'EXIT_CODE').write_text('1\n');raise
