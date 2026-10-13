"""Resume only evaluations after missing aggregate report; never call fit."""
from phase107_common import *
from run_phase107 import development,pixbox

if __name__=='__main__':
    lock=read(RESULT/'evaluation_resume_lock.json');model=read(ROOT/'model_lock.json')
    assert sha(REPO/'src/tools/run_phase107.py')==lock['evaluation_resume_code_sha256']
    assert model['fit_attempts']==1 and sha(ROOT/'R_model.json')==model['R_model_sha256']
    assert sha(ROOT/'phase67_reference_report.json')==lock['reference_report_sha256']
    assert not (ROOT/'development_aggregate.json').exists() and not (ROOT/'PixBox_score_seal.json').exists()
    logs=SHARED/'phase107_20261013_logs'
    write(ROOT/'EVALUATION_RESUME_STARTED.json',dict(utc=now(),R_model_sha256=sha(ROOT/'R_model.json'),
        resume_code_sha256=sha(Path(__file__)),reference_sha256=sha(ROOT/'phase67_reference_report.json'),
        resume_lock_sha256=sha(RESULT/'evaluation_resume_lock.json'),additional_fits=0,additional_forwards=0),True)
    try:
        development(ROOT/'phase67_reference_report.json');pixbox()
        write(logs/'RESUME_EXIT_CODE.json',dict(code=0,utc=now()),True)
        (logs/'RESUME_EXIT_CODE').write_text('0\n',encoding='ascii')
    except BaseException:
        (logs/'RESUME_EXIT_CODE').write_text('1\n',encoding='ascii')
        raise
