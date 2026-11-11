"""Measure/guard the same deterministic CPU replay with Windows process counters."""
import ctypes,os,runpy,sys,threading,time
from ctypes import wintypes
from pathlib import Path
from prepare_phase201 import R,O,write
class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
def main():
    assert os.name=='nt';start=time.monotonic();cpu=time.process_time();done=threading.Event();state={'peak_working_set_bytes':0}
    api=ctypes.WinDLL('psapi');kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];api.GetProcessMemoryInfo.restype=wintypes.BOOL
    def sample():
        c=Counters();c.cb=ctypes.sizeof(c);assert api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb)
        state['peak_working_set_bytes']=max(state['peak_working_set_bytes'],int(c.PeakWorkingSetSize))
        if c.PeakWorkingSetSize>16*1024**3 or time.monotonic()-start>7200:
            write(O/'CPU_GUARD.json',dict(state,status='stopped_limit'));os._exit(1)
    def monitor():
        while not done.wait(.05):sample()
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        sample();runpy.run_path(str(R/'src/tools/run_phase201.py'),run_name='__main__');runpy.run_path(str(R/'src/tools/verify_phase201.py'),run_name='__main__');sample()
    finally:done.set();thread.join()
    write(O/'CPU_GUARD.json',dict(state,status='passed',wall_seconds=time.monotonic()-start,process_CPU_seconds=time.process_time()-cpu,memory_scope='Windows GetProcessMemoryInfo PeakWorkingSetSize entire replay+independent review process',cap_bytes=16*1024**3,cap_seconds=7200,deterministic_resource_replay=True,model_calls=0))
if __name__=='__main__':main()
