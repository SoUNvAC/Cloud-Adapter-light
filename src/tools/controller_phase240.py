"""Six authorized independent fits; no automatic retry or budget expansion."""
from phase240_common import *
import traceback


def run(name, script, args):
 global child
 check_bundle();start=time.monotonic()
 with (R/f'{name}.console.log').open('x') as out,(R/f'{name}.error.log').open('x') as err:
  child=subprocess.Popen([PY,'-u','-B',str(R/'code'/script),*args],cwd=R/'code',stdin=subprocess.DEVNULL,stdout=out,stderr=err)
  save('CONTROLLER_STATUS.json',{'stage':name,'PID':os.getpid(),'child_PID':child.pid,'epoch':time.time()})
  rc=child.wait(timeout=14500)
 save(R/'controller_stages'/f'{name}.json',{'returncode':rc,'PID':child.pid,'wall_seconds':time.monotonic()-start})
 return rc


def main():
 assert not (R/'CONTROLLER_STARTED.json').exists();check_bundle();save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time()});(R/'PID').write_text(str(os.getpid()))
 failed=[]
 for domain in ['L8','SPARCS']:
  for branch in ['A','B','G']:
   if prior_seconds()>=14400 or len(ledger())>=52000:failed.append({'domain':domain,'branch':branch,'reason':'budget_exhausted'});continue
   rc=run(f'train_{domain}_{branch}','train_phase240.py',[domain,branch])
   if rc:failed.append({'domain':domain,'branch':branch,'returncode':rc,'no_auto_retry':True})
   save('FIT_PROGRESS.json',{'failures':failed,'completed':[p.parent.name for p in (R/'fits').glob('*/COMPLETE.json')]})
 # Diagnostics never justify refitting an existing C or cancelling completed arms.
 rc=run('diagnostics','diagnose_phase240.py',[])
 if rc:failed.append({'stage':'diagnostics','returncode':rc,'counts_preserved':True})
 assert run('val','evaluate_phase240.py',['val'])==0
 assert run('test','evaluate_phase240.py',['test'])==0
 fits=[js(p) for p in (R/'fits').glob('*/COMPLETE.json')];compute=collections.Counter(z['kind'] for z in compute_ledger())
 assert len(ledger())<=52000 and prior_seconds()<14400 and compute['optimizer_update']<=26000 and compute['backward_or_autograd']<=38000
 save('COMPLETE.json',{'status':'partial' if failed else 'complete','fits_completed':len(fits),'successful_updates':sum(v['updates'] for v in fits),'failures':failed,'calls':len(ledger()),'counts':dict(collections.Counter(v['kind'] for v in ledger())),'compute_counts':dict(compute),'GPU_stage_seconds':prior_seconds(),'remote_new_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'historical_C_refits':0,'Source_data_access':False})
 (R/'EXIT_CODE').write_text('2\n' if failed else '0\n')


if __name__=='__main__':
 try:main()
 except BaseException:
  if 'child' in globals() and child.poll() is None:child.terminate()
  save('CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc(),'calls':len(ledger()),'completed_GPU_seconds':prior_seconds(),'no_automatic_retry':True});(R/'EXIT_CODE').write_text('1\n');raise
