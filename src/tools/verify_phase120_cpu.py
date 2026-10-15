"""Guarded CPU-only replay of all five Catalogue metrics from fixed caches."""
import csv,gc,os,resource,threading,time,traceback
from pathlib import Path
from phase120_common import *

def main():
    setup();start=time.time();finished=threading.Event();state={'status':'replaying','model_calls':0}
    path=ROOT/'Catalogue_CPU_REPLAY_RESOURCE.json'
    from install_phase111_runtime import host_free
    assert host_free()>=4000000000
    def guard():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        state.update(rss_peak_bytes=rss,elapsed_seconds=time.time()-start)
        assert rss<=16*1024**3
        assert time.time()-load(ROOT/'CONTROLLER_STATUS.json')['started_epoch']<=14400
    def monitor():
        try:
            while not finished.wait(2):guard();write(path,state)
        except Exception:
            state.update(status='resource_failure',failure=traceback.format_exc());write(path,state);os._exit(1)
    threading.Thread(target=monitor,daemon=True).start()
    import numpy as np
    from phase120_metrics import evaluate
    plan=[r for r in load(ROOT/'INPUT_PATCH_PLAN.json')['rows'] if r['pool']=='Catalogue']
    support=load(ROOT/'Catalogue_COMMON_SUPPORT_LOCK.json')['rows'];by={r['product']:r for r in support}
    oldroot=ROOT.parent/'phase119_20261015';groups={}
    for row in plan:
        pid=row['product_id'];maskpath=ROOT/'common_support'/('Catalogue_'+pid+'.npy')
        assert sha(maskpath)==by[pid]['mask_sha256'];mask=np.load(maskpath,allow_pickle=False)
        with np.load(OLD/'scores'/('O_'+pid+'.npz'),allow_pickle=False) as z:
            g={'products':pid,'truth':z['truth'][mask]==2,'E':z['scores'][mask]}
        for p in ('R','N','C','D'):
            source=oldroot/p if p in ('R','N') else ROOT/(p+'_Catalogue')
            file=source/'scores/Catalogue'/(pid+'.npz');assert sha(file)==by[pid]['score_bindings'][p]
            with np.load(file,allow_pickle=False) as z:g[p]=z['scores'][mask]
        groups[row['group']['group_id']]=g;guard()
    per,summary,cmp=evaluate(groups,'Catalogue')
    with (ROOT/'aggregate/Catalogue_summary.csv').open() as f:old=list(csv.DictReader(f))
    for a,b in zip(summary,old):
        for key,v in a.items():
            if isinstance(v,(int,float)):assert abs(v-float(b[key]))<1e-10,(key,v,b[key])
            else:assert str(v)==b[key]
    expected=load(ROOT/'aggregate/Catalogue_comparisons.json')['rows'];assert cmp==expected
    with (ROOT/'aggregate/Catalogue_per_group.csv').open() as f:old=list(csv.DictReader(f))
    for a,b in zip(per,old):
        for key,v in a.items():
            if isinstance(v,bool):assert str(v)==b[key]
            elif isinstance(v,(int,float)):assert abs(v-float(b[key]))<1e-10
            elif v is None:assert b[key]==''
            else:assert str(v)==b[key]
    guard();finished.set();state.update(status='passed',all_five_pool_and_group_metrics_reproduced=True,
        all_three_bootstrap_comparisons_exact=True,first_CPU_pass_resource_peak_not_instrumented=True,
        this_complete_cache_replay_resource_guarded=True,host_free_after=host_free())
    write(path,state);print(state,flush=True)
if __name__=='__main__':main()
