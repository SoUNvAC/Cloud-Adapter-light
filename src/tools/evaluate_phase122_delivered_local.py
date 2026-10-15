"""Same frozen local evaluation; calibration checked by a separate full CPU pass."""
from evaluate_phase122_local import *
def main():
    from phase122_local_resources import start as resource_guard
    start=time.time();finish=resource_guard(R,[U,D,P],lambda name,v:save(O/name,v))
    for n,h in load(O/'PRE_EXECUTION_LOCK.json')['source_sha256'].items():assert sha(R/n)==h,n
    for n,h in load(O/'POSTPROCESS_LOCK.json')['source_sha256'].items():assert sha(R/n)==h,n
    unpack();seal=load(U/'delivery/ALL_SCORES_SEALED.json');assert not seal['evaluation_labels_joined']
    assert sha(U/'delivery/THRESHOLD_LOCK.json')==seal['thresholds'] and sha(U/'delivery/TRAINING_SEAL.json')==seal['training']
    for stage,h in seal['settings'].items():assert sha(U/'delivery'/stage/'SCORES_SEALED.json')==h
    check=load(U/'delivery/CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json')
    assert check['status']=='passed' and check['model_calls']==0 and check['full53_groups_no_sampling']
    assert check['threshold_lock_sha256']==seal['thresholds']
    source=R/'src/tools/collect_phase122_delivery.py';assert sha(source)==check['source_sha256']
    thresholds=load(U/'delivery/THRESHOLD_LOCK.json')['thresholds'];plan=load(U/'delivery/INPUT_PATCH_PLAN.json')['rows']
    save(O/'CALIBRATION_INDEPENDENT_CPU_ACCEPTANCE.json',check)
    save(O/'CALIBRATION_SCORE_LOCATIONS.json',load(U/'delivery/CALIBRATION_SCORE_LOCATIONS.json'))
    data={'Catalogue':catalogue([r for r in plan if r['pool']=='Catalogue'],thresholds),'PixBox':pixbox([r for r in plan if r['pool']=='PixBox'],thresholds)}
    save(O/'LOCAL_CPU_ACCEPTANCE.json',dict(status='passed',elapsed_seconds=time.time()-start,model_calls=0,labels_uploaded=0,
        thresholds_sha256=seal['thresholds'],no_evaluation_threshold_search=True,AP_threshold_invariance=True,
        calibration_full_boundary_replay='separate shared CPU process with exact source/score/threshold SHA verification'))
    finish();print(json.dumps(data,indent=2))
if __name__=='__main__':main()
