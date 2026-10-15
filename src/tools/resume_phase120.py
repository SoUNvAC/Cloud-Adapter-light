"""Resume Phase120 after pre-D JSON graph comparison failure; reuse sealed C."""
import os,subprocess,sys,time,traceback
from pathlib import Path
from phase120_common import *

def main():
    setup()
    assert (ROOT/'EXIT_CODE').read_text().strip()=='1'
    for p in (ROOT/'PID',ROOT/'train_D/PID'):
        pid=int(p.read_text());proc=Path('/proc')/str(pid)
        assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z','previous process still alive'
    failure=load(ROOT/'train_D/RUN_STATUS.json');assert failure['step']==0 and failure['gate_calls']==0 and failure['construction_calls']==18
    a=load(ROOT/'train_C/PARAMETER_WHITELIST.json');b=load(ROOT/'train_D/PARAMETER_WHITELIST.json')
    for key in ('initial_encoder_sha256','initial_decoder_sha256','graph_signature'):assert a[key]==b[key]
    receipt=load(ROOT/'train_C/TRAINING_RECEIPT.json');assert sha(ROOT/'C_decoder_final.safetensors')==receipt['final_model_sha256']
    target=ROOT/'train_D_attempt_001_failed';assert not target.exists()
    (ROOT/'train_D').rename(target)
    write(ROOT/'INITIAL_CONTROLLER_STATUS_FAILED.json',load(ROOT/'CONTROLLER_STATUS.json'))
    c=load(ROOT/'CONTROLLER_STATUS.json');c['calls_completed']+=failure['calls']
    c.update(status='engineering_recovery_running',failed_construction_calls=failure['calls'],recovery_lock_sha256=sha(REPO/'src/research_results/phase120_20261015/ENGINEERING_RECOVERY_LOCK.json'))
    write(ROOT/'CONTROLLER_STATUS.json',c);(ROOT/'RESUME_PID').write_text(str(os.getpid()))
    def run(script,args,label,status):
        with (ROOT/(label+'.console.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools'/script),*args],stdout=log,stderr=subprocess.STDOUT)
            assert p.wait()==0,label
        s=load(ROOT/status/'RUN_STATUS.json');c['calls_completed']+=s['all_calls'];c['completed'].append(label)
        c['elapsed_seconds']=time.time()-c['started_epoch'];assert c['calls_completed']<=8000 and c['elapsed_seconds']<=14400
        write(ROOT/'CONTROLLER_STATUS.json',c)
    run('train_phase120.py',['D'],'train_D_recovery_001','train_D')
    write(ROOT/'TRAINING_SEAL.json',{'models':{p:sha(ROOT/(p+'_decoder_final.safetensors')) for p in ('C','D')},
        'receipts':{p:sha(ROOT/('train_'+p)/'TRAINING_RECEIPT.json') for p in ('C','D')},
        'source_lock_sha256':sha(LOCK),'engineering_recovery_lock_sha256':c['recovery_lock_sha256'],
        'both_fixed1000_endpoints_before_evaluation':True,'no_checkpoint_selection':True,'C_reused_without_retraining':True})
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
    write(ROOT/'CONTROLLER_STATUS.json',c);(ROOT/'EXIT_CODE').write_text('0\n');(ROOT/'RECOVERY_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'RECOVERY_FAILURE.json',{'failure':traceback.format_exc()});(ROOT/'RECOVERY_EXIT_CODE').write_text('1\n')
        traceback.print_exc();sys.exit(1)
