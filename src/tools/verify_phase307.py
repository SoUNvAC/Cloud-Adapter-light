"""Portable validation of fixed LOGO probe delivery; numerical and convergence status remain separate."""
from phase307_common import *

def main():
    lock=load(S/'SUMMARY_SHA256.json')
    for n,h in lock['files'].items():assert sha(S/n)==h,n
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    resource=load(S/'RESOURCE.json');assert resource['neural_calls']==resource['training_updates']==0 and resource['classifier_fits']<=96
    assert resource['seconds']<=3600 and resource['peak_CPU_bytes']<=16*1024**3 and resource['minimum_available_bytes']>=4000000000
    receipt=load(S/'BACKUP_RECEIPT.json');assert receipt['all_members_SHA_verified'] and receipt['raw_bytes']*5+10000000<1000000000
    if lock['status']=='completed':
        a=load(S/'CPU_ACCEPTANCE.json');assert not a['torch_imported'] and a['neural_calls']==a['segmentation_updates']==a['downloads']==0
        assert a['fit_attempts']==resource['classifier_fits'] and a['supervised_target_reference_information_diagnosis']
        for k in ('all32fold_rows_preserved','training_groups_exclude_heldout','shared_samples_labels_weights_support_all_models',
            'train_only_scaler_independently_checked','weights_equal_groups_sum_one','parameters_saved_before_probability_and_metrics',
            'covered_OOF_keys_exactly_once','fixed_half_threshold_counts_independent','Shadow_masks_and_scores_unchanged'):assert a[k],k
        assert a['all_C_coverage'] or a['fit_failures']>0
        assert a['objective_tests']['objective_gradient_finite_difference_max_error']<2e-7 and all(a['AUC_tests'].values())
        assert all(int(v)<=4 for v in a['BLAS_thread_limits'].values())
        assert len(read(S/'FOLD_METRICS.csv'))==96 and len(read(S/'FOLD_SUPPORT.csv'))==32
        assert len(read(S/'GROUP_SAMPLING.csv'))==32 and len(read(S/'FIXED_RESULTS.csv'))==3
        assert len(read(S/'PAIRED_FOLDS.csv'))==96 and len(read(S/'PAIRED_COMPARISONS.csv'))==3
        assert len(read(S/'STRATIFIED_RESULTS.csv'))==27
        assert len(load(S/'FOLD_LOCK.json')['rows'])==32
        if a['all_C_coverage']:
            for r in read(S/'FIXED_RESULTS.csv'):assert (int(r['C']),int(r['D']),int(r['K']),int(r['C_Shadow']))==(546400,450058,84005,12337)
    print('Phase307 verified: '+lock['status']+'; '+str(len(lock['files']))+' summary files; '+str(len(lock['code']))+' scripts')

if __name__=='__main__':main()
