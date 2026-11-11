"""Fresh Phase400 paths, wallclock ledger and explicit GPU/resource caps."""
from pathlib import Path
import collections,hashlib,json,os,subprocess,time,threading
R=Path(os.environ.get('PHASE400_ROOT','/home/scv/shared/phase400_20261031'));SRC=Path(os.environ.get('PHASE400_SRC','/home/scv/Cloud-Adapter-light/src'));PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python';SOURCE=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
EXPECTED_SOURCE='64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p=p if p.is_absolute() else R/p;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False,default=str));tmp.replace(p)
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def state_sha(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def hostfree():
 a=json.loads(subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True,timeout=45));a=[a] if isinstance(a,dict) else a;assert min(x['FreeSpace'] for x in a)>10*1024**3;return a
def prior_gpu_seconds():return sum(js(p)['seconds'] for p in (R/'costs').glob('*.json') if js(p).get('GPU_stage'))
class Budget:
 def __init__(self,stage):
  import torch
  self.torch=torch;self.stage=stage;self.start=time.monotonic();self.prior=prior_gpu_seconds();self.path=R/'costs'/(stage+'.json');assert not self.path.exists();self.current=None;self.calls=0;self.updates=0;self.last_host=0;self.samples=[];self.stop=threading.Event();self.check();threading.Thread(target=self.guard,daemon=True).start()
 def count(self,kind):
  self.calls+=1
  with (R/'CALL_LEDGER.jsonl').open('a') as f:f.write(json.dumps({'stage':self.stage,'ordinal_within_stage':self.calls,'kind':kind,'elapsed':time.monotonic()-self.start,'current':self.current})+'\n')
 def check(self):
  import shutil,resource
  assert self.prior+time.monotonic()-self.start<43200,'12h cumulative GPU-stage wallclock'
  assert shutil.disk_usage(R).free>10*1024**3
  if time.monotonic()-self.last_host>180:self.host_volumes=hostfree();self.last_host=time.monotonic()
  t=self.torch;total=t.cuda.get_device_properties(0).total_memory if t.cuda.is_available() else 0;peak=max(t.cuda.max_memory_allocated(),t.cuda.max_memory_reserved()) if t.cuda.is_initialized() else 0
  if total:assert peak<=.90*total,'retain 10% physical GPU memory'
  sample={'elapsed':time.monotonic()-self.start,'CPU_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_allocated':t.cuda.max_memory_allocated() if t.cuda.is_initialized() else 0,'GPU_peak_reserved':t.cuda.max_memory_reserved() if t.cuda.is_initialized() else 0};self.samples.append(sample);save('RUN_STATUS.json',{'PID':os.getpid(),'stage':self.stage,'current':self.current,'calls_in_stage':self.calls,'updates_in_stage':self.updates,'cumulative_GPU_stage_seconds':self.prior+sample['elapsed'],'resource':sample})
 def guard(self):
  import traceback
  while not self.stop.wait(30):
   try:self.check()
   except BaseException:save('RESOURCE_FAILURE.json',{'stage':self.stage,'error':traceback.format_exc()});self.finish('resource_failure');os._exit(71)
 def finish(self,status):
  self.stop.set();save(self.path,{'stage':self.stage,'status':status,'GPU_stage':True,'seconds':time.monotonic()-self.start,'calls':self.calls,'updates':self.updates,'resources':self.samples,'host_volumes':getattr(self,'host_volumes',None)})
def setup():
 import sys,torch
 os.chdir(SRC);sys.path[:0]=[str(R/'code'),str(SRC),str(SRC/'tools')];os.environ['XFORMERS_DISABLED']='1';torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
 from mmseg.utils import register_all_modules
 register_all_modules();import cloud_adapter.models,cloud_adapter.datasets.phase65_catalogue,phase400_model
 # No network downloads may be triggered by initialization.
 torch.hub.download_url_to_file=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Phase400 forbids downloads'))
 return torch
def config(arm,source=False):
 from mmengine.config import Config
 cfg=Config.fromfile(str(SRC/'configs/protocol/phase65a_explore_rgb.py'));cfg.model.backbone.type='Phase400TargetMsRECloudAdapter';cfg.model.backbone.update(msre_enabled=arm in ('MH','M'),head_delta_enabled=arm in ('MH','H'),msre_init_seed=4000101,head_init_seed=4000102);cfg.model.backbone.target_enabled=not source
 return cfg
def build(arm,budget,legacy=False,baseline=False):
 import torch
 from mmseg.registry import MODELS
 from mmengine.config import Config
 cfg=Config.fromfile(str(SRC/'configs/protocol/phase64_source_parent_rgb.py')) if baseline else config(arm)
 if legacy:cfg.model.backbone.type='TargetMsRECloudAdapterDinoVisionTransformer';[cfg.model.backbone.pop(k,None) for k in ('msre_enabled','head_delta_enabled','msre_init_seed','head_init_seed')]
 torch.manual_seed(40001);model=MODELS.build(cfg.model);model.init_weights();state=torch.load(SOURCE,map_location='cpu')['state_dict'];missing,unexpected=model.load_state_dict(state,strict=False);permitted=('backbone.target_msre.','backbone.target_head_delta.');assert not unexpected and all(n.startswith(permitted) for n in missing)
 if baseline:assert not missing
 else:assert missing and all(n.startswith(permitted) for n in missing)
 model.train();model.cuda();return model,cfg,{'missing':missing,'unexpected':unexpected,'nontrainable':{n:tensor_sha(t) for n,t in model.state_dict().items() if not n.startswith(permitted)}}
