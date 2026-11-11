"""Resume only zero-output evaluation; never re-enter any training process."""
from phase236_common import *
import subprocess,traceback
def main():
 global child
 assert not (R/'EVALUATION_RECOVERY03_STARTED.json').exists();check_bundle()
 fits=[js(p) for p in (R/'fits').glob('*/COMPLETE.json')];assert len(fits)==8 and sum(v['updates'] for v in fits)==32000
 assert js(R/'RECOVERY02_AUTHORIZATION.json')['allowed_total_updates']==32063
 save('EVALUATION_RECOVERY03_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'all_completed_training_preserved':True});(R/'PID').write_text(str(os.getpid()))
 for stage in ['val','test']:
  name='recovery03_'+stage;start=time.monotonic()
  with (R/f'{name}.console.log').open('x') as out,(R/f'{name}.error.log').open('x') as err:
   child=subprocess.Popen([PY,'-u','-B',str(R/'code/evaluate_phase236_recovery03.py'),stage],cwd=R/'code',stdin=subprocess.DEVNULL,stdout=out,stderr=err)
   save('CONTROLLER_STATUS.json',{'stage':name,'PID':os.getpid(),'child_PID':child.pid,'epoch':time.time()});rc=child.wait(timeout=22000)
  save(R/'controller_stages'/f'{name}.json',{'returncode':rc,'PID':child.pid,'wall_seconds':time.monotonic()-start});assert rc==0,(stage,rc)
 assert len(ledger())<=85000 and prior_seconds()<21600
 save('COMPLETE.json',{'status':'complete','calls':len(ledger()),'counts':dict(collections.Counter(v['kind'] for v in ledger())),'GPU_stage_seconds':prior_seconds(),'updates':32063,'backwards':32063,'new_completed_fit_updates':32000,'prior_failed_updates':63,'fits':8,'all_selections_before_test':True,'evaluation_schema_recovery_no_additional_training':True,'remote_new_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())});(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  if 'child' in globals() and child.poll() is None:child.terminate()
  save('CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc(),'calls':len(ledger()),'completed_GPU_stage_seconds':prior_seconds()});(R/'EXIT_CODE').write_text('1\n');raise
