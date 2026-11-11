"""Two frozen diagnostic passes with one cumulative ledger. Never retries model calls."""
from pathlib import Path
import json,hashlib,subprocess,os,time,traceback
R=Path('/home/scv/shared/phase220_20261024');SRC=Path('/home/scv/Cloud-Adapter-light/src');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def host():
 x=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(v['FreeSpace'] for v in x)>4*1024**3;return x
def main():
 assert not (R/'STARTED.json').exists();save('PID',os.getpid());save('STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'planned_units':4800,'cap':6000,'GPU_seconds_cap':3600,'no_training_or_target_inference':True,'host':host()});units=0;gpu=0.;mode=None
 try:
  for name,h in js(R/'CODE_LOCK.json').items():assert sha(R/'code'/name)==h
  model=js(R/'MODEL_LOCK.json');assert sha(model['Source']['path'])==model['Source']['SHA']
  for v in model['students'].values():assert sha(v['path'])==v['SHA']
  for n,h in js(R/'PHASE216_EXECUTION_CODE_SNAPSHOT.json')['source_runtime_code_SHA'].items():assert sha(SRC/n)==h
  for n,h in js(R/'PHASE216_RUNTIME_CODE_LOCK.json')['historical_code_SHA'].items():assert sha(SRC/n)==h
  assert sha(R.parent/'phase219_20261023/weight_preparation/WEIGHT_MANIFEST.json')==sha(R/'WEIGHT_MANIFEST.json')
  for mode in ['val','replay']:
   save('BUDGET_STATE.json',{'units':units,'GPU_seconds':gpu})
   with (R/f'{mode}.console.log').open('x') as out,(R/f'{mode}.error.log').open('x') as err:
    proc=subprocess.Popen([PY,'-u','-B',str(R/'code/execute_phase220.py'),'--mode',mode],cwd=SRC,stdin=subprocess.DEVNULL,stdout=out,stderr=err);save('RUN_STATUS.json',{'stage':mode,'PID':os.getpid(),'child_PID':proc.pid,'completed_units':units,'GPU_seconds':gpu});code=proc.wait()
   assert code==0 and (R/mode/'EXIT_CODE').read_text().strip()=='0',(mode,code)
   c=js(R/mode/'COMPLETE.json');units+=c['new_units'];gpu+=c['GPU_seconds'];assert units<=6000 and gpu<3600;host()
  assert units==4800;save('COMPLETE.json',{'units':units,'GPU_seconds':gpu,'val':js(R/'val/COMPLETE.json'),'replay':js(R/'replay/COMPLETE.json'),'host':host(),'training_steps':0,'fits':0,'backwards':0,'target_inference':0});save('RUN_STATUS.json',{'stage':'complete','PID':os.getpid(),'units':units,'GPU_seconds':gpu});(R/'EXIT_CODE').write_text('0\n')
 except BaseException:
  (R/'FAILURE.txt').write_text(traceback.format_exc());save('RUN_STATUS.json',{'stage':'failed','child_root':mode,'completed_units':units,'completed_GPU_seconds':gpu,'PID':os.getpid()});(R/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
