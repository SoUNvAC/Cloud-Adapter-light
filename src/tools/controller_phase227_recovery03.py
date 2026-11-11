"""No training: verify all eight endpoints, resume only missing evaluations."""
from pathlib import Path
import json,hashlib,subprocess,os,time,traceback
R=Path('/home/scv/shared/phase227_20261025');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python';S=Path('/home/scv/Cloud-Adapter-light/src')
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(R/n).write_text(json.dumps(v,indent=2))
def main():
 assert not (R/'RECOVERY03_STARTED.json').exists();save('RECOVERY03_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'training_repeats':0,'planned_total_units':102752,'extra_teacher_heads':535});(R/'PID').write_text(str(os.getpid()));prior=js(R/'RESUME03_BUDGET.json')
 try:
  for n,h in js(R/'RECOVERY03_CODE_LOCK.json').items():assert sha(R/'code'/n)==h,n
  jobs=js(R/'EXPERIMENT_LOCK.json')['jobs'];assert len(jobs)==4
  for job in jobs:
   assert js(R/job/'TRAINING_COMPLETE.json')['total_units']==8002 and (R/job/'EXIT_CODE').read_text().strip()=='0'
   for arm in ['W1','W10']:assert sha(R/job/f'heads/{arm}_iter4000.pth')==js(R/job/f'{arm}_TRAIN_COMPLETE.json')['head_SHA']
  with (R/'evaluation_recovery03.console.log').open('x') as out,(R/'evaluation_recovery03.error.log').open('x') as err:
   p=subprocess.Popen([PY,'-u','-B',str(R/'code/evaluate_phase227_recovery03.py')],cwd=S,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':'evaluate','child_root':'SPARCS/evaluation','child_PID':p.pid,'completed_units':prior['units'],'completed_GPU_seconds':prior['GPU_seconds'],'PID':os.getpid()});rc=p.wait()
  assert rc==0 and (R/'EVALUATION_EXIT_CODE').read_text().strip()=='0';ev=js(R/'EVALUATION_COMPLETE.json');assert ev['units']==102752 and ev['GPU_seconds_cumulative']<21600
  from controller_phase227 import host
  save('COMPLETE.json',{'units':102752,'GPU_seconds':ev['GPU_seconds_cumulative'],'new_fits':8,'training_units':32008,'evaluation_units_including_failure':70744,'training_source_RGB_reads':0,'training_source_label_reads':0,'training_teacher_forward':0,'evaluation_teacher_head_forward_including_failure':1070,'extra_teacher_head_calls':535,'failed_SP_source_units_preserved':10170,'host_volumes_final':host()});save('CONTROLLER_STATUS.json',{'stage':'complete','units':102752,'PID':os.getpid()});(R/'EXIT_CODE').write_text('0\n')
 except BaseException:
  (R/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());save('CONTROLLER_STATUS.json',{'stage':'failed','completed_units':prior['units'],'completed_GPU_seconds':prior['GPU_seconds'],'PID':os.getpid()});(R/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
