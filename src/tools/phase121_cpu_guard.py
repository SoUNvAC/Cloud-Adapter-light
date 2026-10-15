"""CPU-only evaluation resource receipt; no neural model construction."""
import os,resource,threading,time,traceback
from phase121_common import write
_stop=threading.Event();_start=time.time();_stats={}
def start_guard(root):
    def check():
        from install_phase111_runtime import host_free
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        free=host_free();size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        assert rss<=16*1024**3 and free>=4000000000 and size<=4000000000
        _stats.update(rss_peak_bytes=rss,host_min_free_bytes=min(_stats.get('host_min_free_bytes',free),free),
            intermediate_peak_bytes=max(_stats.get('intermediate_peak_bytes',0),size),model_calls=0)
    check()
    def monitor():
        try:
            while not _stop.wait(10):check()
        except Exception:
            write(root/'Catalogue_CPU_RESOURCE_FAILURE.json',dict(failure=traceback.format_exc(),**_stats));os._exit(1)
    threading.Thread(target=monitor,daemon=True).start()
def finish_guard(root):
    _stop.set();_stats.update(elapsed_seconds=time.time()-_start,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    assert _stats['rss_peak_bytes']<=16*1024**3
    write(root/'Catalogue_CPU_RESOURCE_RECEIPT.json',_stats)
