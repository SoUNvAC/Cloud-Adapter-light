"""One sequential219 budget ledger; no automatic model retries or historical refits."""
from pathlib import Path
import json,hashlib,subprocess,os,time,traceback
R=Path('/home/scv/shared/phase221_20261024');SRC=Path('/home/scv/Cloud-Adapter-light/src');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def host():
 v=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(x['FreeSpace'] for x in v)>4*1024**3;return v
def main():
 assert not (R/'CONTROLLER_STARTED.json').exists();save('PID',os.getpid());save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'expected_units':75018,'cap':85000,'GPU_seconds_cap':14400,'new_fits':4,'C_D_refit':False,'host':host()});units=0;gpu=0.;childroot=None
 try:
  assert all(sha(R/'code'/n)==h for n,h in js(R/'CODE_LOCK.json').items())
  assert sha(SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
  for n,h in js(R/'PHASE216_EXECUTION_CODE_SNAPSHOT.json')['source_runtime_code_SHA'].items():assert sha(SRC/n)==h
  for n,h in js(R/'PHASE216_RUNTIME_CODE_LOCK.json')['historical_code_SHA'].items():assert sha(SRC/n)==h
  original=Path('/home/scv/shared/phase217_20261023')
  for s in [21701,21702]:
   for n in ['TARGET_SOURCE_SCHEDULE.json','TRAINING_CONFIG_LOCK.json']:assert sha(R/f'seed{s}'/n)==sha(original/f'seed{s}'/n)
  def run(label,script,subroot):
   nonlocal childroot
   childroot=subroot;save('BUDGET_STATE.json',{'units':units,'GPU_seconds':gpu})
   with (R/f'{label}.console.log').open('x') as out,(R/f'{label}.error.log').open('x') as err:
    proc=subprocess.Popen([PY,'-u','-B',str(R/'code'/script)],cwd=SRC,stdin=subprocess.DEVNULL,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':label,'child_root':subroot,'child_PID':proc.pid,'completed_units':units,'GPU_seconds':gpu,'PID':os.getpid()});code=proc.wait()
   assert code==0 and (R/subroot/'EXIT_CODE').read_text().strip()=='0',(label,code)
  run('weight_preparation','precompute_phase221.py','weight_preparation');w=js(R/'weight_preparation/COMPLETE.json');units+=w['units'];gpu+=w['GPU_seconds'];assert units==512
  save('PRETRAIN_LOCK.json',{'epoch':time.time(),'weight_manifest_SHA':sha(R/'weight_preparation/WEIGHT_MANIFEST.json'),'numpy':'1.26.4','p':'exp(log_softmax(S_Source)), same219 FP32 KL','teacher_class':'original Source sum-normalized semantic argmax','normalization':'CPUfloat64 mean(g|teacherclass), divide then float32; no epsilon/clip/re-scaling; mean tolerances1e-6','N':'G divided by per-image teacher-predicted-class mean','P':'same teacher-class histogram, local PCG64 SHA256 phase221|sample_id|class first8 big-endian; fixed class-row-major permutation','no_Global_RNG_use':True,'seed_schedules':{str(s):sha(R/f'seed{s}/TARGET_SOURCE_SCHEDULE.json') for s in [21701,21702]},'code_lock_SHA':sha(R/'CODE_LOCK.json'),'no_fits_started':True})
  for s in [21701,21702]:
   run(f'train_{s}',f'train_phase221_seed{s}.py',f'seed{s}');c=js(R/f'seed{s}/TRAINING_COMPLETE.json');assert c['total_units']==32008;units+=c['total_units'];gpu+=c['GPU_seconds'];assert units<=85000 and gpu<14400;host()
  assert units==64528;save('FOUR_ENDPOINTS_SEALED.json',{'epoch':time.time(),'Source_only_selection_not_started':True,'seeds':{str(s):js(R/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]}})
  run('evaluate','evaluate_phase221.py','evaluation');ev=js(R/'evaluation/COMPLETE.json');units+=ev['total_new_units'];gpu+=ev['GPU_seconds'];assert units==75018 and gpu<14400
  save('COMPLETE.json',{'status':'complete','units':units,'GPU_seconds':gpu,'weight_preparation':w,'training':{str(s):js(R/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]},'evaluation':ev,'host_final':host(),'new_fits':4,'C_D_refit':False});save('CONTROLLER_STATUS.json',{'stage':'complete','units':units,'GPU_seconds':gpu,'PID':os.getpid()});(R/'EXIT_CODE').write_text('0\n')
 except BaseException:
  (R/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());save('CONTROLLER_STATUS.json',{'stage':'failed','child_root':childroot,'completed_units':units,'completed_GPU_seconds':gpu,'PID':os.getpid()});(R/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
