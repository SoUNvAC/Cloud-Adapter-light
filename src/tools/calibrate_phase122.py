"""Calibrate only post-seal held-out groups; no evaluation labels or models."""
import resource,time,gc
import numpy as np
from phase122_common import *
from phase121_metrics import oracle
from phase122_cpu_guard import start_guard,finish_guard
def main():
    assert not (ROOT/'THRESHOLD_LOCK.json').exists()
    start=time.time();setup();start_guard(ROOT);seal=load(ROOT/'TRAINING_SEAL.json')
    assert sha(ROOT/'C_decoder_final.safetensors')==seal['models']['C']
    rows=[g for g in load(ROOT/'INPUT_PATCH_PLAN.json')['rows'] if g['pool']=='Calibration']
    assert len(rows)==53;groups={};bindings=[]
    for row in rows:
        pid=row['product_id'];record=dict(products=pid);common=None;indices=None
        for arm in ('R','C'):
            base=ROOT/(arm+'_Calibration');f=base/'scores/Calibration'/(pid+'.npz')
            assert sha(f)==load(base/'SCORES_SEALED.json')['scores_sha256']['Calibration/'+pid]
            with np.load(f,allow_pickle=False) as z:
                if common is None:common=z['valid'].copy();indices=z['flat_indices'].copy()
                else:assert np.array_equal(indices,z['flat_indices']);common &= z['valid']
                record[arm]=z['scores'].copy()
        assert common.all(),'complete calibrated support required, no approximate exclusion'
        _,valid,truth=catalog_image(row['group'],read_truth=True);assert np.array_equal(indices,np.flatnonzero(valid))
        record['truth']=truth==2;groups[row['group']['group_id']]=record
        bindings.append(dict(group_id=row['group']['group_id'],product=pid,valid=int(common.sum()),negative=int((truth!=2).sum()),
            R_sha256=sha(ROOT/'R_Calibration/scores/Calibration'/(pid+'.npz')),C_sha256=sha(ROOT/'C_Calibration/scores/Calibration'/(pid+'.npz'))))
        del valid,truth;gc.collect()
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=16*1024**3
    out={};details={}
    for arm in ('R','C'):
        point,detail=oracle(groups,arm,'Calibration')
        point['status']='heldout_calibration_frozen_before_evaluation'
        point['not_evaluation_pool_oracle']=True;out[arm]=point;details[arm]=detail
    write(ROOT/'THRESHOLD_LOCK.json',dict(thresholds={p:v['threshold'] for p,v in out.items()},calibration=out,
        boundary_details=details,bindings=bindings,training_seal_sha256=sha(ROOT/'TRAINING_SEAL.json'),member_lock_sha256=sha(REPO/'src/research_results/phase122_20261015/MEMBER_INPUT_LOCK.json'),
        models_and_thresholds_before_evaluation=True,no_evaluation_reference_used=True,no_confidence_upper_bound_threshold_search=True,
        CPU_model_calls=0,CPU_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CPU_seconds=time.time()-start))
    finish_guard(ROOT);print({p:v['threshold'] for p,v in out.items()},flush=True)
if __name__=='__main__':main()
