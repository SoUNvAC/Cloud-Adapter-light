"""One sequential bounded run: four fixed H fits then all finite evaluation points."""
from pathlib import Path
import json,hashlib,os,subprocess,time,traceback
R=Path('/home/scv/shared/phase224_20261024');S=Path('/home/scv/Cloud-Adapter-light/src');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def host():
 v=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(x['FreeSpace'] for x in v)>=4*1024**3;return v
def main():
 assert not (R/'CONTROLLER_STARTED.json').exists();lock=js(R/'EXPERIMENT_LOCK.json');save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'planned_units':100460,'cap':120000,'GPU_seconds_cap':21600,'host_volumes':host(),'automatic_restarts':False});(R/'PID').write_text(str(os.getpid()));units=0;seconds=0.;current=None
 try:
  for n,h in js(R/'CODE_LOCK.json').items():assert sha(R/'code'/n)==h,n
  assert sha(S/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
  for data,k in [('PHASE216_EXECUTION_CODE_SNAPSHOT.json','source_runtime_code_SHA'),('PHASE216_RUNTIME_CODE_LOCK.json','historical_code_SHA')]:
   for n,h in js(R/data)[k].items():assert sha(S/n)==h,n
  for rows in js(R/'INPUT_SUPPORT_LOCK.json').values():
   for v in rows:assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']
  seq=js(R/'SOURCE_SEQUENCE_LOCK.json');unique_inputs={};jobs=lock['jobs']
  for job,v in jobs.items():
   root=R/job;assert sha(root/'TARGET_SOURCE_SCHEDULE.json')==v['original_schedule_SHA'];original=Path(v['historical_root'])/f"seed{v['seed']}";assert sha(original/'TARGET_SOURCE_SCHEDULE.json')==v['original_schedule_SHA'] and sha(original/'GATE_ACCEPTED.json')==v['historical_gate_SHA'];assert sha(v['historical_D_path'])==v['historical_D_SHA']
   schedule=js(root/'TARGET_SOURCE_SCHEDULE.json');assert [{'source_sample_id':x['source_sample_id'],'source_png_name':x['source_png_name']} for x in schedule]==seq
   cfg=js(root/'TRAINING_CONFIG_LOCK.json');old=js(root/'HISTORICAL_TRAINING_CONFIG_LOCK.json');old['KL']['lambda']=4;assert cfg==old
   for row in schedule:unique_inputs[row['image_path']]=row['mask_path']
  for image,mask in unique_inputs.items():assert Path(image).is_file() and Path(mask).is_file()
  for v in js(R/'PHASE216_SOURCE_POOL_REQUEST.json')['selected']:assert sha(S/'data/cloudsen12_high_l1c/img_dir/train'/v['png_name'])==v['png_sha256']
  save('PRETRAIN_ACCEPTED.json',{'status':'passed','epoch':time.time(),'four_original_sequences_SHA':{j:v['original_schedule_SHA'] for j,v in jobs.items()},'only_change_lambda4':True,'source_replay_reselected':False,'new_updates':0})
  def run(entry,child):
   nonlocal current
   current=child;save('BUDGET_STATE.json',{'units':units,'GPU_seconds':seconds})
   with (R/(child+'.console.log')).open('x') as out,(R/(child+'.error.log')).open('x') as err:
    p=subprocess.Popen([PY,'-u','-B',str(R/'code'/entry)],cwd=S,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':'evaluate' if child=='evaluation' else 'train','child_root':child,'child_PID':p.pid,'completed_units':units,'completed_GPU_seconds':seconds,'PID':os.getpid()});code=p.wait()
   assert code==0 and (R/child/'EXIT_CODE').read_text().strip()=='0',(child,code)
  for job in jobs:
   run(f'train_{job}.py',job);v=js(R/job/'TRAINING_COMPLETE.json');assert v['total_units']==16004;units+=v['total_units'];seconds+=v['GPU_seconds'];assert units<=120000 and seconds<21600;host()
  save('FOUR_ENDPOINTS_SEALED.json',{'epoch':time.time(),'jobs':{j:js(R/j/'TRAINING_COMPLETE.json') for j in jobs},'target_metrics_opened':False});run('evaluate_phase224.py','evaluation');ev=js(R/'evaluation/COMPLETE.json');assert ev['total_units']==36444
  save('COMPLETE.json',{'status':'complete','units':units+ev['total_units'],'GPU_seconds':seconds+ev['GPU_seconds'],'training_units':units,'eval_units':ev['total_units'],'new_fits':4,'host_volumes_final':host()});assert units+ev['total_units']==100460;save('CONTROLLER_STATUS.json',{'stage':'complete','units':100460,'PID':os.getpid()});(R/'EXIT_CODE').write_text('0\n')
 except BaseException:(R/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());save('CONTROLLER_STATUS.json',{'stage':'failed','child_root':current,'completed_units':units,'completed_GPU_seconds':seconds,'PID':os.getpid()});(R/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
