"""Local CPU/disk guard using native process counters; no system changes."""
import os,shutil,threading,time,traceback
from pathlib import Path
def peak_rss():
    if os.name!='nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
            ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    k=ctypes.WinDLL('kernel32',use_last_error=True);p=ctypes.WinDLL('psapi',use_last_error=True)
    k.GetCurrentProcess.restype=wintypes.HANDLE;p.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    c=Counters();c.cb=ctypes.sizeof(c);assert p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(c),c.cb)
    return int(c.peak_working)
def start(root,paths,save):
    stop=threading.Event();stats=dict(model_calls=0,started=time.time())
    def check():
        peak=peak_rss();free=shutil.disk_usage(root).free
        size=sum(f.stat().st_size for p in paths if p.exists() for f in p.rglob('*') if f.is_file())
        assert peak<=16*1024**3 and free>=4000000000 and size<=4000000000
        stats.update(CPU_peak_bytes=peak,local_min_free_bytes=min(stats.get('local_min_free_bytes',free),free),intermediate_peak_bytes=max(stats.get('intermediate_peak_bytes',0),size))
    check()
    def monitor():
        try:
            while not stop.wait(5):check()
        except Exception:
            save('LOCAL_RESOURCE_FAILURE.json',dict(failure=traceback.format_exc(),**stats));os._exit(1)
    threading.Thread(target=monitor,daemon=True).start()
    def finish():
        stop.set();check();stats['elapsed_seconds']=time.time()-stats.pop('started');save('LOCAL_CPU_RESOURCE_RECEIPT.json',stats)
    return finish
