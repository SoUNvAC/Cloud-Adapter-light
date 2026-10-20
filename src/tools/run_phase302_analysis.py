"""Windows CPU admission/monitor: wait for memory, never analyze remotely."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parents[2]
ROOT=R/'outputs/phase302/phase302_20261020'
PYTHON=Path('C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
MIN_FREE=4000000000
# Estimate, not a measured peak: pooled AP sorting plus two sets of cached scores.
ESTIMATED_WORKING_BYTES=2500000000


class Memory(ctypes.Structure):
    _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in
        ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]


class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]


def free_memory():
    value=Memory();value.length=ctypes.sizeof(value)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return int(value.avail_phys)


def write(name,v):(ROOT/name).write_text(json.dumps(v,indent=2)+'\n')


def main():
    assert sys.platform=='win32'
    ROOT.mkdir(parents=True,exist_ok=True)
    available=free_memory()
    if (ROOT/'analysis/CPU_ACCEPTANCE.json').exists():
        print(json.dumps(dict(status='already_completed')));return
    if available<MIN_FREE+ESTIMATED_WORKING_BYTES:
        value=dict(status='waiting_for_local_memory',available_physical_bytes=available,
            required_host_free_bytes=MIN_FREE,estimated_analysis_working_bytes=ESTIMATED_WORKING_BYTES,
            estimate_is_not_measurement=True,launch_admission_bytes=MIN_FREE+ESTIMATED_WORKING_BYTES,
            reason='Reserve estimated CPU working memory while maintaining protocol host-free minimum',
            neural_calls=0,remote_analysis_allowed=False)
        write('LOCAL_ANALYSIS_WAIT.json',value);print(json.dumps(value));return
    if '--check' in sys.argv:
        print(json.dumps(dict(status='ready',available_physical_bytes=available)));return
    receipt=json.loads((ROOT/'BACKUP_RECEIPT.json').read_text())
    assert receipt['all_archive_members_verified'] and receipt['exit_code']=='0'
    assert not (ROOT/'LOCAL_ANALYSIS_START.json').exists(), 'Preserve prior CPU attempt; inspect before retry'
    assert PYTHON.exists()
    write('LOCAL_ANALYSIS_START.json',dict(epoch=time.time(),available_physical_bytes=available,
        runtime=str(PYTHON),estimated_working_bytes=ESTIMATED_WORKING_BYTES))
    started=time.monotonic();peak=0;minimum=available;failure=None
    with (ROOT/'analysis.console.log').open('xb') as f:
        proc=subprocess.Popen([str(PYTHON),'-B',str(R/'src/tools/analyze_phase302.py')],cwd=R,
            stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        handle=kernel.OpenProcess(0x1000|0x0010,False,proc.pid)
        assert handle
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        try:
            while proc.poll() is None:
                counter=Counters();counter.cb=ctypes.sizeof(counter)
                assert ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(counter),counter.cb)
                peak=max(peak,int(counter.PeakWorkingSetSize));minimum=min(minimum,free_memory())
                if peak>16*1024**3:failure='CPU exceeds16GiB'
                if minimum<MIN_FREE:failure='Host free below4GB'
                if failure:proc.terminate();break
                time.sleep(.25)
            result=proc.wait()
        finally:kernel.CloseHandle(handle)
    value=dict(status='completed' if result==0 and failure is None else 'failed',exit_code=result,
        failure=failure,peak_working_set_bytes=peak,minimum_available_physical_bytes=minimum,
        seconds=time.monotonic()-started,neural_calls=0,execution='local Windows CPU')
    write('LOCAL_ANALYSIS_RESOURCE.json',value);print(json.dumps(value))
    assert value['status']=='completed'


if __name__=='__main__':main()
