"""Fixed ordered Phase120 execution; no scientific retries or checkpoint selection."""
import os,sys,time,subprocess,traceback
from pathlib import Path
from phase120_common import *

def main():
    cfg=setup();ROOT.mkdir(exist_ok=False);(ROOT/'PID').write_text(str(os.getpid()))
    from install_phase111_runtime import host_free
    assert host_free()>=4000000000
    import numpy as np
    split=REPO/'src/work_dirs/phase65a/split_lock.json';assert sha(split)==cfg['original_split_sha256']
    proposed=load(split)['proposed_rows']
    fit=[g for g in load(REPO/'src/research_results/phase117_20261014/INPUT_GATE.json')['bindings'] if g['split']=='fit']
    assert len(fit)==153 and {(g['group_id'],g['product']) for g in fit}=={(g['group_id'],g['representative']) for g in proposed if g['split']=='fit'}
    write(ROOT/'FIT_INPUT_LOCK.json',{'bindings':fit,'original_split_sha256':sha(split),'only_fit_training':True})
    rng=np.random.default_rng(12001)
    schedule=[{'fit_index':int(rng.integers(153)),'top':int(rng.integers(511)),'left':int(rng.integers(511)),
        'horizontal_flip':bool(rng.integers(2))} for _ in range(1000)]
    write(ROOT/'PAIRED_CROP_SCHEDULE.json',{'seed':12001,'fit_order':[g['product'] for g in fit],'rows':schedule,'before_any_model_construction':True})
    oldplan=ROOT.parent/'phase119_20261015/INPUT_PATCH_PLAN.json'
    assert sha(oldplan)==cfg['phase119_plan_sha256']
    plan=load(oldplan);write(ROOT/'INPUT_PATCH_PLAN.json',plan)
    c={'status':'running','calls_completed':0,'started_epoch':time.time(),'completed':[]};write(ROOT/'CONTROLLER_STATUS.json',c)
    def run(script,args,label,status):
        with (ROOT/(label+'.console.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools'/script),*args],stdout=log,stderr=subprocess.STDOUT)
            assert p.wait()==0,label
        s=load(ROOT/status/'RUN_STATUS.json');c['calls_completed']+=s['all_calls'];c['completed'].append(label)
        c['elapsed_seconds']=time.time()-c['started_epoch'];assert c['calls_completed']<=8000 and c['elapsed_seconds']<=14400
        write(ROOT/'CONTROLLER_STATUS.json',c)
    for mode in ('C','D'):run('train_phase120.py',[mode],'train_'+mode,'train_'+mode)
    write(ROOT/'TRAINING_SEAL.json',{'models':{p:sha(ROOT/(p+'_decoder_final.safetensors')) for p in ('C','D')},
        'receipts':{p:sha(ROOT/('train_'+p)/'TRAINING_RECEIPT.json') for p in ('C','D')},
        'source_lock_sha256':sha(LOCK),'both_fixed1000_endpoints_before_evaluation':True,'no_checkpoint_selection':True})
    for mode in ('C','D'):run('infer_phase120.py',[mode,'Catalogue'],mode+'_Catalogue',mode+'_Catalogue')
    with (ROOT/'Catalogue_CPU.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools/evaluate_phase120_catalogue.py')],stdout=log,stderr=subprocess.STDOUT)
        assert p.wait()==0,'Catalogue CPU acceptance'
    c['Catalogue_evaluation_complete_before_PixBox']=sha(ROOT/'Catalogue_CPU_ACCEPTANCE.json');write(ROOT/'CONTROLLER_STATUS.json',c)
    for mode in ('C','D','R'):run('infer_phase120.py',[mode,'PixBox'],mode+'_PixBox',mode+'_PixBox')
    write(ROOT/'ALL_SCORES_SEALED.json',{'training':sha(ROOT/'TRAINING_SEAL.json'),
        'settings':{p+'_'+pool:sha(ROOT/(p+'_'+pool)/'SCORES_SEALED.json') for p in ('C','D') for pool in ('Catalogue','PixBox')},
        'PixBox_labels_joined':False,'actual_calls':c['calls_completed']})
    c.update(status='completed_pending_local_PixBox_reference_join',elapsed_seconds=time.time()-c['started_epoch'])
    write(ROOT/'CONTROLLER_STATUS.json',c);(ROOT/'EXIT_CODE').write_text('0\n')

if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'CONTROLLER_FAILURE.json',{'failure':traceback.format_exc(),'status':'paused_engineering_no_scientific_verdict'})
        (ROOT/'EXIT_CODE').write_text('1\n');traceback.print_exc();sys.exit(1)
