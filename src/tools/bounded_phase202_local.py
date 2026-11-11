"""Local background CPU sequence; resource monitoring is not SSH polling."""
import ctypes,hashlib,json,os,subprocess,sys,time,traceback
from ctypes import wintypes
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase202/phase202_20261017'
def save(p,v):
 t=p.with_suffix('.writing');t.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8');t.replace(p)
class Memory(ctypes.Structure):
 _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
def usage(pid):
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.restype=wintypes.HANDLE;k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.CloseHandle.argtypes=[wintypes.HANDLE]
 h=k.OpenProcess(0x1000|0x10,False,pid)
 if not h:return 0
 m=Memory();m.cb=ctypes.sizeof(m);f=ctypes.WinDLL('psapi').GetProcessMemoryInfo;f.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD];f.restype=wintypes.BOOL;assert f(h,ctypes.byref(m),m.cb);k.CloseHandle(h);return max(m.PeakWorkingSetSize,m.PeakPagefileUsage)
def main():
 assert not (O/'LOCAL_LAUNCH.json').exists();(O/'LOCAL_LAUNCH.json').write_text(json.dumps(dict(PID=os.getpid(),epoch=time.time()))+'\n');state=dict(status='running',completed=[],PID=os.getpid(),peak_bytes=0);proc=None;started=time.time()
 try:
  state['stage']='verify_transferred_inputs';save(O/'LOCAL_STATUS.json',state)
  ix=json.loads((O/'local_inputs/INPUT_INDEX.json').read_text());assert len(ix['rows'])==185 and ix['run_status']['status']=='completed'
  # Wait on immutable manifest completion, not cross-session process visibility.
  while True:
   missing=[row for row in ix['rows'] if not (O/'local_inputs'/row['local_name']).is_file() or (O/'local_inputs'/row['local_name']).stat().st_size!=row['bytes']]
   if not missing and (O/'local_inputs/PIXBOX_INPUTS_PRIVATE.json').is_file():break
   assert time.time()<1792425600,'2026-10-20 Beijing stop; no late start'
   state.update(stage='waiting_transfer',files_remaining=len(missing));save(O/'LOCAL_STATUS.json',state);time.sleep(60)
  state['stage']='verify_transferred_inputs';save(O/'LOCAL_STATUS.json',state)
  for row in ix['rows']:
   h=hashlib.sha256()
   with (O/'local_inputs'/row['local_name']).open('rb') as f:
    for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
   assert h.hexdigest()==row['sha256'],row['product']
  assert hashlib.sha256((O/'local_inputs/PIXBOX_INPUTS_PRIVATE.json').read_bytes()).hexdigest()==ix['point_inputs_sha256']
  for stage,done in [('baseline','BASELINE_REPRODUCED.json'),('fit','MODELS_FROZEN.json'),('calibrate','THRESHOLDS_FROZEN.json'),('evaluate','CPU_ACCEPTANCE.json')]:
   assert not (O/done).exists(),'fresh sequence only; accepted stages must not be rerun'
   state.update(stage=stage);save(O/'LOCAL_STATUS.json',state)
   proc=subprocess.Popen([sys.executable,'-B',str(R/'src/tools/analyze_phase202_local.py'),stage],cwd=R)
   while proc.poll() is None:
    state['peak_bytes']=max(state['peak_bytes'],usage(proc.pid));assert state['peak_bytes']<=16*1024**3,'CPU memory limit';time.sleep(2)
   assert proc.returncode==0,(stage,proc.returncode);assert (O/done).exists();state['completed'].append(stage);state['seconds']=time.time()-started;save(O/'LOCAL_STATUS.json',state)
  state.update(status='completed',seconds=time.time()-started);save(O/'LOCAL_STATUS.json',state);(O/'LOCAL_EXIT_CODE').write_text('0\n')
 except BaseException:
  if proc is not None and proc.poll() is None:proc.kill();proc.wait()
  state.update(status='failed',error=traceback.format_exc(),seconds=time.time()-started);save(O/'LOCAL_STATUS.json',state);(O/'LOCAL_EXIT_CODE').write_text('1\n');raise
if __name__=='__main__':main()
