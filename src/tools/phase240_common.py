from pathlib import Path
import collections,hashlib,json,os,subprocess,threading,time
R=Path('/home/scv/shared/phase240_20261030')
OLD=Path('/home/scv/shared/phase236_20261027')
SRC=Path('/home/scv/Cloud-Adapter-light/src')
PY='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
SOURCE=SRC/'work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth'
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p=p if p.is_absolute() else R/p;p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False,default=str));t.replace(p)
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def digest(d):
 h=hashlib.sha256()
 for n,t in sorted(d.items()):
  h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def ledger():return [json.loads(l) for l in (R/'CALL_LEDGER.jsonl').read_text().splitlines()] if (R/'CALL_LEDGER.jsonl').exists() else []
def prior_seconds():return sum(js(p)['GPU_stage_seconds'] for p in (R/'stage_costs').glob('*.json')) if (R/'stage_costs').exists() else 0
def check_bundle():
 for n,h in js(R/'BUNDLE_SHA.json').items():assert sha(R/n)==h,n
def hostfree():
 raw=subprocess.check_output(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,FreeSpace | ConvertTo-Json -Compress"],text=True)
 rows=json.loads(raw);rows=[rows] if isinstance(rows,dict) else rows
 assert min(v['FreeSpace'] for v in rows)>4*1024**3;return rows
class Budget:
 def __init__(self,stage):
  import torch
  self.torch=torch;self.stage=stage;self.start=time.monotonic();self.prior=prior_seconds();self.ordinal=len(ledger());self.counts=collections.Counter();self.current=None;self.done=threading.Event();self.samples=[];self.free=hostfree();self.last_host=time.monotonic();self.log=(R/'CALL_LEDGER.jsonl').open('a',buffering=1)
  self.path=R/'stage_costs'/f'{stage}.json';assert not self.path.exists()
  prior=compute_ledger();self.compute_counts=collections.Counter(z['kind'] for z in prior);self.compute_log=(R/'COMPUTE_LEDGER.jsonl').open('a',buffering=1)
  self.check();threading.Thread(target=self.guard,daemon=True).start()
 def count(self,kind):
  assert self.ordinal<52000 and self.prior+time.monotonic()-self.start<14400
  self.ordinal+=1;self.counts[kind]+=1;self.log.write(json.dumps({'ordinal':self.ordinal,'stage':self.stage,'kind':kind,'current':self.current,'elapsed':time.monotonic()-self.start})+'\n')
 def compute(self,kind):
  cap={'optimizer_update':26000,'backward_or_autograd':38000}[kind];assert self.compute_counts[kind]<cap
  self.compute_counts[kind]+=1;self.compute_log.write(json.dumps({'stage':self.stage,'kind':kind,'ordinal':self.compute_counts[kind],'current':self.current})+'\n')
 def check(self):
  import resource
  assert self.prior+time.monotonic()-self.start<14400
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert rss<16*1024**3
  alloc=self.torch.cuda.max_memory_allocated() if self.torch.cuda.is_initialized() else 0
  reserve=self.torch.cuda.max_memory_reserved() if self.torch.cuda.is_initialized() else 0
  assert max(alloc,reserve)<20*1024**3
  size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());assert size<6*1024**3
  if time.monotonic()-self.last_host>180:self.free=hostfree();self.last_host=time.monotonic()
  self.samples.append({'elapsed':time.monotonic()-self.start,'CPU_peak_bytes':rss,'GPU_peak_allocated':alloc,'GPU_peak_reserved':reserve,'remote_bytes':size})
  save('RUN_STATUS.json',{'PID':os.getpid(),'stage':self.stage,'current':self.current,'cumulative_calls':self.ordinal,'stage_counts':dict(self.counts),'cumulative_GPU_stage_seconds':self.prior+time.monotonic()-self.start,'resource':self.samples[-1]})
 def guard(self):
  import traceback
  while not self.done.wait(20):
   try:self.check()
   except BaseException:
    save('RESOURCE_FAILURE.json',{'stage':self.stage,'error':traceback.format_exc()});self.finish('resource_failure');os._exit(71)
 def finish(self,status):
  self.done.set();save(self.path,{'stage':self.stage,'status':status,'GPU_stage_seconds':time.monotonic()-self.start,'calls_delta':dict(self.counts),'cumulative_calls':self.ordinal,'resources':self.samples,'host_volumes':self.free,'compute_counts':dict(self.compute_counts)});self.log.close();self.compute_log.close()
def build(budget):
 import argparse,torch
 from export_phase12_onnx import build_wrapper
 depth=[0]
 def pre(m,a):
  if depth[0]==0:budget.count('construction')
  depth[0]+=1
 def post(m,a,o):depth[0]-=1
 a=torch.nn.modules.module.register_module_forward_pre_hook(pre);b=torch.nn.modules.module.register_module_forward_hook(post)
 try:w,cfg=build_wrapper(argparse.Namespace(config='configs/protocol/phase64_source_parent_rgb.py',checkpoint=str(SOURCE),precision='fp32',active_block_indices=None))
 finally:a.remove();b.remove()
 return w,cfg

def compute_ledger():
 return [json.loads(l) for l in (R/'COMPUTE_LEDGER.jsonl').read_text().splitlines()] if (R/'COMPUTE_LEDGER.jsonl').exists() else []
