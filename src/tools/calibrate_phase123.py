"""One new full53 calibration; R/seed12201 thresholds strictly reused."""
import gc,time,resource
import numpy as np
from phase123_common import *
from phase121_metrics import oracle
from phase122_cpu_guard import start_guard,finish_guard
def main():
    setup();assert not (ROOT/'THRESHOLD_LOCK.json').exists();start_guard(ROOT);start=time.time()
    seal=load(ROOT/'TRAINING_SEAL.json');assert sha(ROOT/'C_decoder_final.safetensors')==seal['models']['C']
    rows=load(REPO/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json')['calibration']
    old=load(PREVIOUS/'THRESHOLD_LOCK.json');binding={r['group_id']:r for r in old['bindings']}
    scores=load(ROOT/'C_Calibration/SCORES_SEALED.json')['scores_sha256'];groups={};bindings=[]
    for g in rows:
        pid=g['product'];path=ROOT/'C_Calibration/scores/Calibration'/(pid+'.npz')
        assert sha(path)==scores['Calibration/'+pid]
        with np.load(path,allow_pickle=False) as z:
            idx=z['flat_indices'].copy();s=z['scores'].copy();assert z['valid'].all()
        prior=PREVIOUS/'R_Calibration/scores/Calibration'/(pid+'.npz')
        assert sha(prior)==binding[g['group_id']]['R_sha256']
        with np.load(prior,allow_pickle=False) as z:assert z['valid'].all() and np.array_equal(idx,z['flat_indices'])
        _,valid,y=catalog_image(g,read_truth=True);assert np.array_equal(idx,np.flatnonzero(valid))
        n=int((y!=2).sum());assert n==binding[g['group_id']]['negative']
        groups[g['group_id']]=dict(products=pid,truth=y==2,C=s)
        bindings.append(dict(group_id=g['group_id'],product=pid,negative=n,valid=len(s),C_sha256=sha(path),R_reused_sha256=sha(prior)))
        del valid,y;gc.collect()
    assert len(groups)==53
    point,detail=oracle(groups,'C','Calibration')
    point.update(status='heldout_calibration_frozen_before_evaluation',not_evaluation_pool_oracle=True)
    write(ROOT/'THRESHOLD_LOCK.json',dict(thresholds={'R':old['thresholds']['R'],'C122':old['thresholds']['C'],'C':point['threshold']},
        calibration={'C':point},boundary_details={'C':detail},bindings=bindings,previous_threshold_lock_sha256=sha(PREVIOUS/'THRESHOLD_LOCK.json'),
        training_seal_sha256=sha(ROOT/'TRAINING_SEAL.json'),member_lock_sha256=sha(REPO/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json'),
        models_and_thresholds_before_evaluation=True,no_evaluation_reference_used=True,CPU_model_calls=0,CPU_seconds=time.time()-start,
        CPU_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    finish_guard(ROOT)
if __name__=='__main__':main()
