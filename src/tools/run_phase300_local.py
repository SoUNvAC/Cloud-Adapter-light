"""Bound the independent local CPU evaluator's memory and elapsed time."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import ctypes
from ctypes import wintypes


class Memory(ctypes.Structure):
    _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]


def usage(pid):
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000|0x10,False,pid)
    if not handle: return 0
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    getter = ctypes.WinDLL('psapi').GetProcessMemoryInfo
    getter.argtypes = [wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    getter.restype = wintypes.BOOL
    try:
        assert getter(handle,ctypes.byref(memory),memory.cb)
        return max(memory.PeakWorkingSetSize,memory.PeakPagefileUsage)
    finally: kernel.CloseHandle(handle)

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/phase300/phase300_20261020'


def save(data):
    path = ROOT/'LOCAL_STATUS.json'
    temp = path.with_suffix('.writing')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)


def main():
    assert not (ROOT/'CPU_ACCEPTANCE.json').exists(), 'Analysis already accepted; do not silently rerun'
    status = dict(status='running',PID=os.getpid(),peak_bytes=0)
    start = time.time()
    with (ROOT/'local_analysis.console.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools/analyze_phase300.py')],
                                   cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                status.update(seconds=time.time()-start,peak_bytes=max(status['peak_bytes'],usage(process.pid)))
                assert status['peak_bytes'] <= 16*1024**3 and status['seconds'] <= 7200
                save(status)
                time.sleep(5)
            assert process.returncode==0 and (ROOT/'CPU_ACCEPTANCE.json').exists()
            status.update(status='completed',seconds=time.time()-start)
            save(status)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            status.update(status='failed',failure=traceback.format_exc(),seconds=time.time()-start)
            save(status)
            raise


if __name__ == '__main__': main()
