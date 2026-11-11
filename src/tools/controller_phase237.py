"""Six fixed fits, then all val selections, then fixed test; no retries."""
from phase237_common import *
import subprocess,traceback
def main():
 global child
 assert not (R/'CONTROLLER_STARTED.json').exists();check_bundle();save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time()});(R/'PID').write_text(str(os.getpid()));stages=[]
 for subset in ['SUBSET_1','SUBSET_2','SUBSET_3']:
  for seed in [21701,21702]:stages.append((f'train_{subset}_{seed}','train_phase237.py',[subset,str(seed)]))
 stages.extend([('val','evaluate_phase237.py',['val']),('test','evaluate_phase237.py',['test'])])
 for name,script,args in stages:
  check_bundle();start=time.monotonic()
  with (R/f'{name}.console.log').open('x') as out,(R/f'{name}.error.log').open('x') as err:
   child=subprocess.Popen([PY,'-u','-B',str(R/'code'/script),*args],cwd=R/'code',stdin=subprocess.DEVNULL,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':name,'PID':os.getpid(),'child_PID':child.pid,'epoch':time.time()});rc=child.wait(timeout=10900)
  save(R/'controller_stages'/f'{name}.json',{'returncode':rc,'PID':child.pid,'wall_seconds':time.monotonic()-start});assert rc==0,(name,rc)
 fits=[js(p) for p in (R/'fits').glob('*/COMPLETE.json')];assert len(fits)==6 and sum(v['updates'] for v in fits)==sum(v['backwards'] for v in fits)==24000
 assert len(ledger())<=26000 and prior_seconds()<10800;save('COMPLETE.json',{'status':'complete','fits':6,'updates':24000,'training_backwards':24000,'calls':len(ledger()),'counts':dict(collections.Counter(v['kind'] for v in ledger())),'GPU_stage_seconds':prior_seconds(),'remote_new_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'full60_new_calls':0,'full60_refits':0});(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  if 'child' in globals() and child.poll() is None:child.terminate()
  save('CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc(),'calls':len(ledger()),'completed_GPU_seconds':prior_seconds(),'no_automatic_retry':True});(R/'EXIT_CODE').write_text('1\n');raise
