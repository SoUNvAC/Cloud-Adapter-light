"""One-shot sequential four-fit Phase222 with cumulative resource accounting."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,traceback
ROOT=Path('/home/scv/shared/phase222_20261024');SRC=Path('/home/scv/Cloud-Adapter-light/src');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=ROOT/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,default=str));t.replace(p)
def host():
 v=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(x['FreeSpace'] for x in v)>4*1024**3;return v
def main():
 assert not (ROOT/'CONTROLLER_STARTED.json').exists();save('CONTROLLER_STARTED.json',{'epoch':time.time(),'PID':os.getpid(),'planned_units':99190,'cap':120000,'GPU_seconds_cap':21600,'seeds':[21701,21702],'model_fits_authorized':4,'automatic_restarts':False,'host_volumes':host()});save('PID',os.getpid());units=0;gpu_seconds=0.;current=None
 try:
  assert sha(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
  for n,h in js(ROOT/'CODE_LOCK.json').items():assert sha(ROOT/'code'/n)==h,n
  assert all(sha(SRC/n)==v for n,v in js(ROOT/'PHASE216_EXECUTION_CODE_SNAPSHOT.json')['source_runtime_code_SHA'].items())
  assert all(sha(SRC/n)==v for n,v in js(ROOT/'PHASE216_RUNTIME_CODE_LOCK.json')['historical_code_SHA'].items())
  if not (ROOT/'INPUT_PREFLIGHT_ACCEPTED.json').exists():
   with (ROOT/'preflight.console.log').open('x') as out,(ROOT/'preflight.error.log').open('x') as err:subprocess.run([PY,'-u','-B',str(ROOT/'code/preflight_phase222.py')],cwd=SRC,stdout=out,stderr=err,check=True)
  else:
   accepted=js(ROOT/'INPUT_PREFLIGHT_ACCEPTED.json');assert sha(ROOT/'INPUT_SUPPORT_LOCK.json')==accepted['input_lock_SHA'] and sha(ROOT/'L8_TRAIN_INPUT_LOCK.json')==accepted['train_input_lock_SHA']
   for rows in list(js(ROOT/'INPUT_SUPPORT_LOCK.json').values())+[js(ROOT/'L8_TRAIN_INPUT_LOCK.json')]:
    for v in rows:assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']
  assert js(ROOT/'INPUT_PREFLIGHT_ACCEPTED.json')['status']=='passed'
  for seed in [21701,21702]:
   for prefix in ['preparation_seed','seed']:
    d=ROOT/f'{prefix}{seed}';d.mkdir(exist_ok=False)
    for old,new in [('PHASE216_SOURCE_POOL_REQUEST.json','SOURCE_POOL_REQUEST.json'),('PHASE216_SOURCE_IDENTITY_ACCEPTED.json','SOURCE_IDENTITY_ACCEPTED.json')]:shutil.copy2(ROOT/old,d/new)
  def run(stage,seed=None):
   nonlocal current
   current='evaluation' if stage=='evaluate' else f'preparation_seed{seed}' if stage=='prepare' else f'seed{seed}';save('BUDGET_STATE.json',{'units':units,'GPU_seconds':gpu_seconds});entry=f'{stage}_phase222.py' if seed is None else f'{stage}_phase222_seed{seed}.py'
   with (ROOT/f'{stage}_{seed}.console.log').open('x') as out,(ROOT/f'{stage}_{seed}.error.log').open('x') as err:
    proc=subprocess.Popen([PY,'-u','-B',str(ROOT/'code'/entry)],cwd=SRC,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':stage,'seed':seed,'child_root':current,'child_PID':proc.pid,'completed_units':units,'completed_GPU_seconds':gpu_seconds,'PID':os.getpid()});code=proc.wait()
   assert code==0 and (ROOT/current/'EXIT_CODE').read_text().strip()=='0',(current,code)
  for s in [21701,21702]:run('prepare',s)
  assert sha(ROOT/'preparation_seed21701/TARGET_SOURCE_SCHEDULE.json')!=sha(ROOT/'preparation_seed21702/TARGET_SOURCE_SCHEDULE.json')
  for s in [21701,21702]:
   for n in ['TARGET_SOURCE_SCHEDULE.json','TRAINING_CONFIG_LOCK.json']:shutil.copy2(ROOT/f'preparation_seed{s}'/n,ROOT/f'seed{s}'/n)
   assert [{'source_sample_id':r['source_sample_id'],'source_png_name':r['source_png_name']} for r in js(ROOT/f'seed{s}/TARGET_SOURCE_SCHEDULE.json')]==js(ROOT/'SOURCE_SEQUENCE_LOCK.json')
  save('BOTH_TARGET_SEQUENCES_LOCKED.json',{'epoch':time.time(),'seeds':{str(s):sha(ROOT/f'seed{s}/TARGET_SOURCE_SCHEDULE.json') for s in [21701,21702]},'source_sequence_SHA':sha(ROOT/'SOURCE_SEQUENCE_LOCK.json'),'no_fit_started':True})
  for s in [21701,21702]:
   run('train',s);r=js(ROOT/f'seed{s}/TRAINING_COMPLETE.json');assert r['total_units']==20005;units+=r['total_units'];gpu_seconds+=r['GPU_seconds'];assert units<=120000 and gpu_seconds<21600;host()
  save('FOUR_ENDPOINTS_SEALED.json',{'epoch':time.time(),'seeds':{str(s):js(ROOT/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]},'source_selection_started':False,'target_eval_started':False});run('evaluate');ev=js(ROOT/'evaluation/COMPLETE.json');assert ev['total_new_units']==59180 and ev['cumulative_units']==99190
  save('COMPLETE.json',{'status':'all_four_fits_and_fixed_points_complete','units':ev['cumulative_units'],'GPU_seconds':gpu_seconds+ev['GPU_seconds'],'evaluation':ev,'host_volumes_final':host(),'persistent_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())});save('CONTROLLER_STATUS.json',{'stage':'complete','units':99190,'PID':os.getpid()});(ROOT/'EXIT_CODE').write_text('0\n')
 except BaseException:
  (ROOT/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());save('CONTROLLER_STATUS.json',{'stage':'failed','child_root':current,'completed_units':units,'completed_GPU_seconds':gpu_seconds,'PID':os.getpid()});(ROOT/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
