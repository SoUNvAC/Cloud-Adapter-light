"""Resume evaluation only; both trained endpoints remain untouched."""
import os,sys,time,subprocess,traceback
from pathlib import Path
from phase120_common import *
def main():
    setup();assert (ROOT/'RECOVERY_EXIT_CODE').read_text().strip()=='1'
    pid=int((ROOT/'RESUME_PID').read_text());proc=Path('/proc')/str(pid)
    assert not proc.exists() or (proc/'stat').read_text().split()[2]=='Z'
    for mode,h in load(ROOT/'TRAINING_SEAL.json')['models'].items():
        assert sha(ROOT/(mode+'_decoder_final.safetensors'))==h
        assert (ROOT/('train_'+mode)/'EXIT_CODE').read_text().strip()=='0'
        assert load(ROOT/('train_'+mode)/'RUN_STATUS.json')['step']==1000
    # Import acceptance before creating any model; catches unresolved shared exports.
    import infer_phase120,evaluate_phase120_catalogue
    assert not (ROOT/'C_Catalogue').exists() and not (ROOT/'D_Catalogue').exists()
    c=load(ROOT/'CONTROLLER_STATUS.json');assert c['calls_completed']==4049
    write(ROOT/'RECOVERY_001_CONTROLLER_STATUS.json',c)
    (ROOT/'RESUME_002_PID').write_text(str(os.getpid()));c['evaluation_recovery_002_calls']=0
    def run(mode,pool):
        label=mode+'_'+pool
        with (ROOT/(label+'_recovery002.console.log')).open('xb') as log:
            p=subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools/infer_phase120.py'),mode,pool],stdout=log,stderr=subprocess.STDOUT)
            assert p.wait()==0,label
        s=load(ROOT/label/'RUN_STATUS.json');c['calls_completed']+=s['all_calls'];c['completed'].append(label)
        c['elapsed_seconds']=time.time()-c['started_epoch'];assert c['calls_completed']<=8000 and c['elapsed_seconds']<=14400
        write(ROOT/'CONTROLLER_STATUS.json',c)
    for mode in ('C','D'):run(mode,'Catalogue')
    with (ROOT/'Catalogue_CPU.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-B',str(REPO/'src/tools/evaluate_phase120_catalogue.py')],stdout=log,stderr=subprocess.STDOUT)
        assert p.wait()==0,'Catalogue CPU acceptance'
    c['Catalogue_evaluation_complete_before_PixBox']=sha(ROOT/'Catalogue_CPU_ACCEPTANCE.json');write(ROOT/'CONTROLLER_STATUS.json',c)
    for mode in ('C','D','R'):run(mode,'PixBox')
    write(ROOT/'ALL_SCORES_SEALED.json',{'training':sha(ROOT/'TRAINING_SEAL.json'),
        'settings':{p+'_'+pool:sha(ROOT/(p+'_'+pool)/'SCORES_SEALED.json') for p in ('C','D') for pool in ('Catalogue','PixBox')},
        'PixBox_labels_joined':False,'actual_calls':c['calls_completed']})
    c.update(status='completed_pending_local_PixBox_reference_join',elapsed_seconds=time.time()-c['started_epoch'])
    write(ROOT/'CONTROLLER_STATUS.json',c);(ROOT/'EXIT_CODE').write_text('0\n');(ROOT/'RECOVERY_002_EXIT_CODE').write_text('0\n')
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'RECOVERY_002_FAILURE.json',{'failure':traceback.format_exc()});(ROOT/'RECOVERY_002_EXIT_CODE').write_text('1\n')
        traceback.print_exc();sys.exit(1)
