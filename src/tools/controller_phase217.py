"""Sequential bounded217: prepare both sequences, four fixed fits, fixed evaluation."""
from pathlib import Path
import collections,hashlib,json,os,shutil,subprocess,time,traceback
ROOT=Path('/home/scv/shared/phase217_20261023');OLD=Path('/home/scv/shared/phase216_20261023');SRC=Path('/home/scv/Cloud-Adapter-light/src');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def js(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 p=ROOT/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,default=str));t.replace(p)
def host():
 v=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(x['FreeSpace'] for x in v)>4*1024**3;return v
def main():
 assert not (ROOT/'CONTROLLER_STARTED.json').exists()
 save('CONTROLLER_STARTED.json',{'epoch':time.time(),'PID':os.getpid(),'planned_units':48893,'cap':60000,'GPU_seconds_cap':14400,'seeds':[21701,21702],'model_fits_authorized':4,'automatic_restarts':False,'host_volumes':host()})
 save('PID',os.getpid());units=0;gpu_seconds=0.;current=None
 try:
  # Preserve216 read-only. Check runtime and all exact image hashes, but never open CloudSEN test here.
  assert sha(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
  assert all(sha(SRC/n)==v for n,v in js(ROOT/'PHASE216_EXECUTION_CODE_SNAPSHOT.json')['source_runtime_code_SHA'].items())
  assert all(sha(SRC/n)==v for n,v in js(ROOT/'PHASE216_RUNTIME_CODE_LOCK.json')['historical_code_SHA'].items())
  for row in js(ROOT/'PHASE216_SOURCE_POOL_REQUEST.json')['selected']:assert sha(SRC/'data/cloudsen12_high_l1c/img_dir/train'/row['png_name'])==row['png_sha256']
  support=js(ROOT/'PHASE216_EVAL_INPUT_LOCK.json')
  for pool,rs in support.items():
   for r in rs:assert sha(Path(r['image_path']))==r['image_sha256'] and sha(Path(r['mask_path']))==r['mask_sha256']
  save('INPUT_RUNTIME_PREFLIGHT_ACCEPTED.json',{'status':'passed','Source256_reselected':False,'source_test_images_or_labels_read':False,'source_eval_and_target_val_test_byte_identities_exact':True,'prior216_code_snapshot_exact':True,'new_model_calls':0})
  for seed in [21701,21702]:
   for prefix in ['preparation_seed','seed']:
    d=ROOT/f'{prefix}{seed}';d.mkdir(exist_ok=False)
    for old,new in [('PHASE216_SOURCE_POOL_REQUEST.json','SOURCE_POOL_REQUEST.json'),('PHASE216_SOURCE_IDENTITY_ACCEPTED.json','SOURCE_IDENTITY_ACCEPTED.json')]:shutil.copy2(ROOT/old,d/new)
  def run(stage,seed):
   nonlocal current
   current=f'preparation_seed{seed}' if stage=='prepare' else f'seed{seed}'
   save('BUDGET_STATE.json',{'units':units,'GPU_seconds':gpu_seconds})
   with (ROOT/f'{stage}_{seed}.console.log').open('x') as out,(ROOT/f'{stage}_{seed}.error.log').open('x') as err:
    proc=subprocess.Popen([PY,'-u','-B',str(ROOT/f'code/{stage}_phase217_seed{seed}.py')],cwd=SRC,stdout=out,stderr=err)
    save('CONTROLLER_STATUS.json',{'stage':stage,'seed':seed,'child_root':current,'child_PID':proc.pid,'completed_units':units,'completed_GPU_seconds':gpu_seconds,'PID':os.getpid()})
    code=proc.wait()
   assert code==0 and (ROOT/current/'EXIT_CODE').read_text().strip()=='0',(current,code)
  # Both locked target sequences must exist and differ before the first new fit.
  for seed in [21701,21702]:run('prepare',seed)
  assert sha(ROOT/'preparation_seed21701/TARGET_SOURCE_SCHEDULE.json')!=sha(ROOT/'preparation_seed21702/TARGET_SOURCE_SCHEDULE.json')
  for seed in [21701,21702]:
   for n in ['TARGET_SOURCE_SCHEDULE.json','TRAINING_CONFIG_LOCK.json']:shutil.copy2(ROOT/f'preparation_seed{seed}'/n,ROOT/f'seed{seed}'/n)
   seq=js(ROOT/f'seed{seed}/TARGET_SOURCE_SCHEDULE.json');assert [{'source_sample_id':r['source_sample_id'],'source_png_name':r['source_png_name']} for r in seq]==js(ROOT/'SOURCE_SEQUENCE_LOCK.json')
  save('BOTH_TARGET_SEQUENCES_LOCKED.json',{'epoch':time.time(),'seeds':{str(s):sha(ROOT/f'seed{s}/TARGET_SOURCE_SCHEDULE.json') for s in [21701,21702]},'source_sequence_SHA':sha(ROOT/'SOURCE_SEQUENCE_LOCK.json'),'no_fit_started':True,'old216_input_replay':js(ROOT/'PHASE216_INPUT_REPLAY_ACCEPTED.json')})
  for seed in [21701,21702]:
   run('train',seed);receipt=js(ROOT/f'seed{seed}/TRAINING_COMPLETE.json');assert receipt['total_units']==20005
   units+=receipt['total_units'];gpu_seconds+=receipt['GPU_seconds'];assert units<=60000 and gpu_seconds<14400;host()
  save('FOUR_ENDPOINTS_SEALED.json',{'epoch':time.time(),'seeds':{str(s):js(ROOT/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]},'source_selection_started':False,'target_eval_started':False})
  save('BUDGET_STATE.json',{'units':units,'GPU_seconds':gpu_seconds})
  assert (ROOT/'code/evaluate_phase217.py').is_file()
  with (ROOT/'evaluate.console.log').open('x') as out,(ROOT/'evaluate.error.log').open('x') as err:
   proc=subprocess.Popen([PY,'-u','-B',str(ROOT/'code/evaluate_phase217.py')],cwd=SRC,stdout=out,stderr=err)
   current='evaluation';save('CONTROLLER_STATUS.json',{'stage':'evaluate','child_PID':proc.pid,'child_root':current,'completed_units':units,'completed_GPU_seconds':gpu_seconds,'PID':os.getpid()});code=proc.wait()
  assert code==0 and (ROOT/'evaluation/EXIT_CODE').read_text().strip()=='0',('evaluation',code)
  ev=js(ROOT/'evaluation/COMPLETE.json');assert ev['total_new_units']==8883 and ev['cumulative_units']==48893
  save('COMPLETE.json',{'status':'all_four_fits_and_fixed_points_complete','units':ev['cumulative_units'],'GPU_seconds':gpu_seconds+ev['GPU_seconds'],'head_receipts':js(ROOT/'FOUR_ENDPOINTS_SEALED.json'),'evaluation':ev,'host_volumes_final':host(),'persistent_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())})
  save('CONTROLLER_STATUS.json',{'stage':'complete','units':48893,'PID':os.getpid()});(ROOT/'EXIT_CODE').write_text('0\n')
 except BaseException:
  (ROOT/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());save('CONTROLLER_STATUS.json',{'stage':'failed','child_root':current,'completed_units':units,'completed_GPU_seconds':gpu_seconds,'PID':os.getpid()});(ROOT/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
