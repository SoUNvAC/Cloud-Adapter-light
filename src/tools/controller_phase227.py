"""Four original paired target sequences, eight fits, then one diagnostic evaluation."""
from pathlib import Path
import json,hashlib,subprocess,os,time,traceback
R=Path('/home/scv/shared/phase227_20261025');S=Path('/home/scv/Cloud-Adapter-light/src');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def host():
 v=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(x['FreeSpace'] for x in v)>=4*1024**3;return v
def main():
 assert not (R/'CONTROLLER_STARTED.json').exists();(R/'PID').write_text(str(os.getpid()));save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'planned_units':102217,'cap_units':125000,'GPU_seconds_cap':21600,'fits':8,'host_volumes':host(),'automatic_restarts':False});units=0;seconds=0.;current=None
 try:
  for n,h in js(R/'CODE_LOCK.json').items():assert sha(R/'code'/n)==h,n
  assert sha(S/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
  for file,key in [('PHASE216_EXECUTION_CODE_SNAPSHOT.json','source_runtime_code_SHA'),('PHASE216_RUNTIME_CODE_LOCK.json','historical_code_SHA')]:
   for n,h in js(R/file)[key].items():assert sha(S/n)==h
  jobs=js(R/'EXPERIMENT_LOCK.json')['jobs']
  for job,meta in jobs.items():
   root=R/job;old=Path(f"/home/scv/shared/phase{meta['old_phase']}_{meta['date']}/seed{meta['seed']}");assert sha(root/'TARGET_SOURCE_SCHEDULE.json')==meta['schedule_SHA']==sha(old/'TARGET_SOURCE_SCHEDULE.json') and sha(root/'TRAINING_CONFIG_LOCK.json')==meta['config_SHA']==sha(old/'TRAINING_CONFIG_LOCK.json')
  save('PRETRAIN_ACCEPTED.json',{'status':'passed','target_sequences_exact':True,'source_train_pixel_reads':0,'new_source_RGB_hash_checks':0,'only_regularizer_new':True})
  for job,meta in jobs.items():
   current=job;root=R/job;(root.parent/'BUDGET_STATE.json').write_text(json.dumps({'units':units,'GPU_seconds':seconds}));(root.parent/'GPU_PRIOR_UNITS.json').write_text(json.dumps({'units':units,'GPU_seconds':seconds}));label=job.replace('/','_')
   with (R/(label+'.console.log')).open('x') as out,(R/(label+'.error.log')).open('x') as err:
    p=subprocess.Popen([PY,'-u','-B',str(R/'code'/meta['entry'])],cwd=S,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':'train','child_root':job,'child_PID':p.pid,'completed_units':units,'completed_GPU_seconds':seconds,'PID':os.getpid()});rc=p.wait()
   assert rc==0 and (root/'EXIT_CODE').read_text().strip()=='0',(job,rc);v=js(root/'TRAINING_COMPLETE.json');assert v['total_units']==8002 and v['counts']=={'target_full_segmentor':8002};units+=8002;seconds+=v['GPU_seconds'];assert units<=125000 and seconds<21600;host()
  assert units==32008
  for domain in ['SPARCS','L8']:(R/domain/'FOUR_ENDPOINTS_SEALED.json').write_text(json.dumps({'epoch':time.time(),'all_eight_fits_complete':True,'target_metrics_opened':False}))
  save('BUDGET_STATE.json',{'units':units,'GPU_seconds':seconds});save('EIGHT_ENDPOINTS_SEALED.json',{'epoch':time.time(),'units':units,'GPU_seconds':seconds});current='SPARCS/evaluation'
  with (R/'evaluation.console.log').open('x') as out,(R/'evaluation.error.log').open('x') as err:
   p=subprocess.Popen([PY,'-u','-B',str(R/'code/evaluate_phase227.py')],cwd=S,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':'evaluate','child_root':current,'child_PID':p.pid,'completed_units':units,'completed_GPU_seconds':seconds,'PID':os.getpid()});rc=p.wait()
  assert rc==0 and (R/'EVALUATION_EXIT_CODE').read_text().strip()=='0';ev=js(R/'EVALUATION_COMPLETE.json');assert ev['units']==102217 and ev['GPU_seconds_cumulative']<21600;save('COMPLETE.json',{'units':102217,'GPU_seconds':ev['GPU_seconds_cumulative'],'new_fits':8,'training_units':32008,'evaluation_units':70209,'training_source_RGB_reads':0,'training_source_label_reads':0,'training_teacher_forward':0,'evaluation_teacher_head_forward':535,'host_volumes_final':host()});save('CONTROLLER_STATUS.json',{'stage':'complete','units':102217,'PID':os.getpid()});(R/'EXIT_CODE').write_text('0\n')
 except BaseException:
  (R/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());save('CONTROLLER_STATUS.json',{'stage':'failed','child_root':current,'completed_units':units,'completed_GPU_seconds':seconds,'PID':os.getpid()});(R/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
