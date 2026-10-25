"""Portable fixed-protocol delivery validation; no local scientific recomputation."""
from phase311_common import *
def main():
    lock=load(S/'SUMMARY_SHA256.json');assert lock['status']=='completed'
    for n,h in lock['files'].items():assert sha(S/n)==h,n
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    a=load(S/'MODEL_ACCEPTANCE.json');e=load(S/'ENDPOINT_LOCK.json');c=load(S/'CPU_ACCEPTANCE.json');s=load(S/'STATUS.json');b=load(S/'BACKUP_RECEIPT.json')
    assert a['trainable_parameters']==294912 and a['teacher_is_separate_Source'] and a['initial_equals_Phase304_tensorwise']
    assert max(a[k] for k in ('lambda0_loss_error','lambda0_gradient_max_error','lambda0_update_max_error'))<=1e-6
    assert abs(a['initial_L_CM'])<=1e-10 and max(a['zero_lora_errors'].values())<=1e-6
    assert e['updates']==955 and e['skipped']==45 and e['all_base_tensors_unchanged'] and e['teacher_all_tensors_unchanged']
    assert c['groups']==32 and c['strategies']==4 and c['original_D']==506798 and c['original_Cloud_correction']==784478 and c['nonG_D']==450058
    assert c['all27_cells_and_marginals_checked'] and c['P_A_other_partition_checked'] and c['baselines_metrics_reproduced'] and not c['torch_imported']
    assert c['teacher_calls_formal']==c['student_calls_formal']==955 and c['empty_A_steps_no_forwards']
    assert s['neural_calls']<=2400 and sum(r['seconds'] for k,r in s['stages'].items() if k in ('train','evaluate'))<=14400
    assert all(r['peak_cpu_bytes']<=16*1024**3 and r['minimum_available_memory']>=4000000000 for r in s['stages'].values())
    assert b['all_archive_members_verified'] and b['archive_SHA_verified_before_extraction'] and 2*b['raw_bytes']+2*b['archive_bytes']+64000000<=4000000000
    assert len(read(S/'PER_GROUP_METRICS.csv'))==128 and len(read(S/'METRICS.csv'))==4 and len(read(S/'PROBES.csv'))==192
    replay=load(S/'BASELINE_REPLAY_ACCEPTANCE.json');assert replay['total_windows']==18 and len(replay['rows'])==2
    assert all(r['maximum_error']<=1e-6 and r['pred_identical'] for r in replay['rows'])
    assert load(S/'PRIOR_MASK_REPRODUCTION.json')['all_100_product_and_1000_crop_array_SHA_identical']
    assert len(read(S/'TRANSITIONS27.csv'))==33*4*3*27 and len(read(S/'CONSTRAINT_ACTIVITY.csv'))==1000
    assert [int(r['attempt']) for r in read(S/'GRADIENT_AUDITS.csv')]==[1,101,500,900]
    print(f"Phase311 verified: {len(lock['files'])} summary files; {len(lock['code'])} scripts; {s['neural_calls']} calls; 955 unique training updates")
if __name__=='__main__':main()
