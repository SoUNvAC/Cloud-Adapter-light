"""Single serial controller. No restarts, no result-dependent arm changes."""
from phase236_common import *
import subprocess,traceback
def main():
 global child
 assert not (R/'CONTROLLER_STARTED.json').exists();save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time()});(R/'PID').write_text(str(os.getpid()));check_bundle()
 stages=[('fill_gate_recovery01','gate_phase236.py',[])]
 for domain in ['L8','SPARCS']:
  for seed in [21701,21702]:
   for lr in ['1e-4','3e-5']:stages.append((f'train_{domain}_{seed}_{lr}','train_phase236.py',[domain,str(seed),lr]))
 stages.extend([('val','evaluate_phase236.py',['val']),('test','evaluate_phase236.py',['test'])])
 for name,script,args in stages:
  check_bundle();start=time.monotonic()
  with (R/f'{name}.console.log').open('x') as cout,(R/f'{name}.error.log').open('x') as cerr:
   child=subprocess.Popen([PY,'-u','-B',str(R/'code'/script),*args],cwd=R/'code',stdin=subprocess.DEVNULL,stdout=cout,stderr=cerr)
   save('CONTROLLER_STATUS.json',{'stage':name,'PID':os.getpid(),'child_PID':child.pid,'epoch':time.time()});rc=child.wait(timeout=22000)
  save(R/'controller_stages'/f'{name}.json',{'stage':name,'PID':child.pid,'returncode':rc,'wall_seconds':time.monotonic()-start});assert rc==0,(name,rc)
  if name.startswith('fill_gate'):assert js(R/'FILL_GATE_ACCEPTED.json')['status']=='passed'
 assert len(ledger())<=85000 and prior_seconds()<21600
 fits=[js(p) for p in (R/'fits').glob('*/COMPLETE.json')];assert len(fits)==8 and sum(v['updates'] for v in fits)==32000 and sum(v['backwards'] for v in fits)==32000
 save('COMPLETE.json',{'status':'complete','calls':len(ledger()),'counts':dict(collections.Counter(v['kind'] for v in ledger())),'GPU_stage_seconds':prior_seconds(),'updates':32000,'backwards':32000,'fits':8,'all_selections_before_test':True,'remote_new_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())});(R/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException:
  if 'child' in globals() and child.poll() is None:child.terminate()
  save('CONTROLLER_FAILURE.json',{'traceback':traceback.format_exc(),'calls':len(ledger()),'completed_GPU_stage_seconds':prior_seconds(),'remaining_partial_costs_in_RUN_STATUS':True});(R/'EXIT_CODE').write_text('1\n');raise
