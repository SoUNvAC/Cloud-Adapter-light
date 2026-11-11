"""Sequential two-seed R/Q pipeline; no automatic training restarts."""
from pathlib import Path
import json,hashlib,os,subprocess,time,traceback
R=Path('/home/scv/shared/phase225_20261024');S=Path('/home/scv/Cloud-Adapter-light/src');PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=R/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def host():
 v=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True));assert min(x['FreeSpace'] for x in v)>=4*1024**3;return v
def main():
 assert not (R/'CONTROLLER_STARTED.json').exists();save('CONTROLLER_STARTED.json',{'PID':os.getpid(),'epoch':time.time(),'planned_units':124268,'cap':140000,'GPU_seconds_cap':21600,'four_fits4000':True,'host_volumes':host(),'automatic_restarts':False});(R/'PID').write_text(str(os.getpid()));units=0;seconds=0.;current=None
 try:
  for n,h in js(R/'CODE_LOCK.json').items():assert sha(R/'code'/n)==h,n
  assert sha(S/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')=='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
  for data,k in [('PHASE216_EXECUTION_CODE_SNAPSHOT.json','source_runtime_code_SHA'),('PHASE216_RUNTIME_CODE_LOCK.json','historical_code_SHA')]:
   for n,h in js(R/data)[k].items():assert sha(S/n)==h
  for rows in js(R/'INPUT_SUPPORT_LOCK.json').values():
   for v in rows:assert sha(v['image_path'])==v['image_sha256'] and sha(v['mask_path'])==v['mask_sha256']
  for v in js(R/'PHASE216_SOURCE_POOL_REQUEST.json')['selected']:assert sha(S/'data/cloudsen12_high_l1c/img_dir/train'/v['png_name'])==v['png_sha256']
  sys_path=__import__('sys').path;sys_path.insert(0,str(R/'code'));from phase225_weights import load
  receipts={}
  for s in [21701,21702]:
   seq=js(R/f'seed{s}/TARGET_SOURCE_SCHEDULE.json');old=Path('/home/scv/shared/phase222_20261024')/f'seed{s}';assert sha(R/f'seed{s}/TARGET_SOURCE_SCHEDULE.json')==sha(old/'TARGET_SOURCE_SCHEDULE.json') and sha(R/f'seed{s}/TRAINING_CONFIG_LOCK.json')==sha(old/'TRAINING_CONFIG_LOCK.json');_,receipts[str(s)]=load(R/'weights',seq)
  save('PRETRAIN_ACCEPTED.json',{'epoch':time.time(),'status':'passed','weights':receipts,'weight_generation_new_neural_calls':0,'no_source_reference_labels_read':True,'frozen_weights_not_reranked':True})
  def run(entry,child):
   nonlocal current
   current=child;save('BUDGET_STATE.json',{'units':units,'GPU_seconds':seconds})
   with (R/(child+'.console.log')).open('x') as out,(R/(child+'.error.log')).open('x') as err:
    p=subprocess.Popen([PY,'-u','-B',str(R/'code'/entry)],cwd=S,stdout=out,stderr=err);save('CONTROLLER_STATUS.json',{'stage':'evaluate' if child=='evaluation' else 'train','child_root':child,'child_PID':p.pid,'completed_units':units,'completed_GPU_seconds':seconds,'PID':os.getpid()});code=p.wait()
   assert code==0 and (R/child/'EXIT_CODE').read_text().strip()=='0',(child,code)
  for s in [21701,21702]:
   run(f'train_phase225_seed{s}.py',f'seed{s}');v=js(R/f'seed{s}/TRAINING_COMPLETE.json');assert v['total_units']==32008;units+=v['total_units'];seconds+=v['GPU_seconds'];assert units<=140000 and seconds<21600;host()
  save('FOUR_ENDPOINTS_SEALED.json',{'epoch':time.time(),'seeds':{str(s):js(R/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]},'target_metrics_opened':False});run('evaluate_phase225.py','evaluation');ev=js(R/'evaluation/COMPLETE.json');assert ev['total_new_units']==60252 and ev['cumulative_units']==124268
  save('COMPLETE.json',{'units':124268,'GPU_seconds':seconds+ev['GPU_seconds'],'training_units':units,'eval_units':60252,'new_fits':4,'host_volumes_final':host()});save('CONTROLLER_STATUS.json',{'stage':'complete','units':124268,'PID':os.getpid()});(R/'EXIT_CODE').write_text('0\n')
 except BaseException:(R/'CONTROLLER_FAILURE.txt').write_text(traceback.format_exc());save('CONTROLLER_STATUS.json',{'stage':'failed','child_root':current,'completed_units':units,'completed_GPU_seconds':seconds,'PID':os.getpid()});(R/'EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
