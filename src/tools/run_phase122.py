"""One fit, full held-out calibration, two frozen thresholds, then evaluation."""
import datetime,os,sys,time,subprocess,traceback
from phase122_common import *
def main():
    assert datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date()<datetime.date(2026,10,20)
    cfg=setup();ROOT.mkdir(exist_ok=False);(ROOT/'PID').write_text(str(os.getpid()))
    from install_phase111_runtime import host_free
    import numpy as np
    assert host_free()>=4000000000
    m=load(REPO/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json')
    assert sha(REPO/'src/work_dirs/phase65a/split_lock.json')==cfg['original_split_sha256']
    fit=m['fit'];cal=m['calibration'];assert len(fit)==100 and len(cal)==53
    assert not ({g['group_id'] for g in fit}&{g['group_id'] for g in cal})
    write(ROOT/'FIT_INPUT_LOCK.json',dict(bindings=fit,training_groups=100,no_calibration_training=True))
    rng=np.random.default_rng(12201)
    schedule=[dict(fit_index=int(rng.integers(100)),top=int(rng.integers(511)),left=int(rng.integers(511)),horizontal_flip=bool(rng.integers(2))) for _ in range(1000)]
    write(ROOT/'PAIRED_CROP_SCHEDULE.json',dict(seed=12201,fit_order=[g['product'] for g in fit],rows=schedule,before_any_model_construction=True))
    old=ROOT.parent/'phase119_20261015/INPUT_PATCH_PLAN.json';assert sha(old)==cfg['phase119_plan_sha256']
    plan=load(old);coords=[[0,1000,0,1000],[0,1000,22,1022],[22,1022,0,1000],[22,1022,22,1022]]
    assert all(g['actual_patch_coordinates']==coords for g in plan['rows'] if g['pool']=='Catalogue')
    plan['rows'] += [dict(pool='Calibration',product_id=g['product'],group=g,actual_patch_coordinates=coords) for g in cal]
    write(ROOT/'INPUT_PATCH_PLAN.json',plan)
    c=dict(status='running',calls_completed=0,started_epoch=time.time(),completed=[]);write(ROOT/'CONTROLLER_STATUS.json',c)
    def run(script,args,label,worker=None):
        with (ROOT/(label+'.console.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools'/script),*args],stdout=log,stderr=subprocess.STDOUT)
            assert p.wait()==0,label
        if worker:c['calls_completed']+=load(ROOT/worker/'RUN_STATUS.json')['all_calls']
        c['completed'].append(label);c['elapsed_seconds']=time.time()-c['started_epoch']
        assert c['calls_completed']<=3500 and c['elapsed_seconds']<=7200;write(ROOT/'CONTROLLER_STATUS.json',c)
    run('train_phase122.py',['C'],'train_C','train_C')
    write(ROOT/'TRAINING_SEAL.json',dict(models={'C':sha(ROOT/'C_decoder_final.safetensors')},receipts={'C':sha(ROOT/'train_C/TRAINING_RECEIPT.json')},
        source_lock_sha256=sha(LOCK),fixed1000_endpoint_before_calibration=True,calibration_groups_not_read_for_training=True))
    for p in ('R','C'):run('infer_phase122.py',[p,'Calibration'],p+'_Calibration',p+'_Calibration')
    run('calibrate_phase122.py',[],'calibration_CPU')
    for pool in ('Catalogue','PixBox'):run('infer_phase122.py',['C',pool],'C_'+pool,'C_'+pool)
    run('seal_phase122.py',[],'support_and_scores_seal')
    c.update(status='completed_pending_local_reference_join',elapsed_seconds=time.time()-c['started_epoch'])
    write(ROOT/'CONTROLLER_STATUS.json',c);(ROOT/'EXIT_CODE').write_text('0\n')
if __name__=='__main__':
    if ROOT.exists():raise SystemExit('Phase122 root already exists; refusing duplicate controller')
    try:main()
    except Exception:
        ROOT.mkdir(exist_ok=True);write(ROOT/'CONTROLLER_FAILURE.json',dict(failure=traceback.format_exc(),status='paused_engineering_not_scientific_failure'))
        (ROOT/'EXIT_CODE').write_text('1\n');traceback.print_exc();sys.exit(1)
