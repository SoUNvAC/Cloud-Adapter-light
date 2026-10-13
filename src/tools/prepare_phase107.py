"""Local metadata lock before the sole new readout fit."""
import ast
from phase107_common import *

def main():
    assert not RESULT.exists()
    base=REPO/'outputs/phase67/phase67_20261010/delivery';control=base/'control_lock.json'
    assert sha(control)==CONTROL_SHA
    model=read(control)['models']['Source_spectral'];audit=base/'fit_input_audit.json'
    assert sha(audit)==read(control)['fit_input_audit_sha256']
    assert len(read(audit)['rows'])==153
    RESULT.mkdir(parents=True);(RESULT/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n',encoding='ascii')
    for p in (REPO/'src/tools').glob('*phase107*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    write(RESULT/'pre_fit_lock.json',dict(utc=now(),status='frozen_before_sole_fit',
        protocol_sha256=sha(REPO/'src/research_plans/PHASE107_MATCHED_RGB_CONTROL_20261013.md'),
        code_sha256={str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in (REPO/'src/tools').glob('*phase107*') if p.is_file()},
        original_control_sha256=CONTROL_SHA,fit_input_audit_sha256=sha(audit),original_S=model,
        historical_numeric_code_sha256=read(control)['code_sha256'],
        new_model='Source_RGB',columns=[1,5,6,7],fit_groups=153,fit_attempt_limit=1,
        rule='original fit_model; four inputs; positive_features1; penalty.001; original fixed153 sample indices/labels',
        weights='1/(153*n_i), normalized by original fit_model; not tree weights',threshold_fits=0,
        PixBox_input_sha256=read(REPO/'src/research_results/phase105_20261013/CPU_tree_acceptance.json')['point_inputs_sha256'],
        PixBox_score_sha256=read(REPO/'src/research_results/phase106_20261013/SCORES_SEALED.json')['score_sha256'],
        phase106_group_sha256=sha(REPO/'src/research_results/phase106_20261013/per_group_results.csv'),
        phase106_summary_sha256=sha(REPO/'src/research_results/phase106_20261013/two_score_summary.csv'),
        reference_sha256=read(REPO/'src/research_results/phase105_20261013/task_lock.json')['phase103_reference_sha256'],
        primary='PixBox S-R mean related-group AP',auxiliary=['development S-R','PixBox R-A','development R-A'],
        bootstrap=dict(seed=107013,resamples=10000,paired_related_groups=True),
        domains=dict(development=32,PixBox=10),new_forwards=0,new_downloads=0,exploratory=True),True)
    print('Phase107 locked before sole R fit')

if __name__=='__main__':main()
