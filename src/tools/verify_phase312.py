"""Portable SHA/protocol validation; no local scientific recomputation."""
from phase312_common import *

def main():
    summary=load(S/'SUMMARY_SHA256.json');assert summary['status']=='completed'
    for n,h in summary['files'].items():assert sha(S/n)==h,n
    for n,h in summary['code'].items():assert sha(R/n)==h,n
    fit=load(S/'FIT_LOCK.json');prediction=load(S/'PREDICTION_LOCK.json');seal=load(S/'READOUT_LOCK.json');cpu=load(S/'CPU_ACCEPTANCE.json');state=load(S/'STATUS.json');b=load(S/'BACKUP_RECEIPT.json')
    assert fit['fit_attempts']<=3 and fit['no_test_scaler_or_fit'] and fit['no_reference_reads'] and fit['positive_class']=='Source-Surface'
    assert fit['numerical_tests']['gradient_finite_difference_max_error']<=2e-7
    assert fit['status']==prediction['status']==seal['status']=='sealed' and fit['sealed_epoch']<=prediction['sealed_epoch']<=seal['sealed_epoch']
    assert len(prediction['rows'])==len(seal['rows'])==32 and all(v<=1e-12 for v in prediction['probability_reproduction_max_error'].values())
    assert cpu['groups']==32 and cpu['original_D']==506798 and cpu['original_B']==784478
    assert cpu['zero_neural_calls'] and cpu['zero_neural_training'] and cpu['all_Shadow_binary_and_scores_identical_to_PL'] and cpu['all_equal_count_controls_exact'] and cpu['only_C_image_Cloud_to_Surface'] and cpu['Source_PL_metrics_reproduced'] and cpu['original_B_disjoint_P_and_fully_retained'] and not cpu['torch_imported']
    assert state['neural_forwards']==0 and state['logistic_fit_attempts']<=3 and state['total_seconds']<=7200
    assert all(r['peak_cpu_bytes']<=16*1024**3 and r['minimum_available_memory']>=4000000000 for r in state['stages'].values())
    assert b['all_archive_members_verified'] and b['archive_SHA_verified_before_extraction'] and b['projected_two_hosts_including_archives_and_summary']<=2000000000
    assert len(read(S/'FIT_SUPPORT.csv'))==100 and len(read(S/'MODEL_FIT.csv'))==3 and len(read(S/'SIX_BAND_VALIDITY.csv'))==132
    assert len(read(S/'PER_GROUP_METRICS.csv'))==cpu['strategies']*32
    if summary['scientific_status']=='complete':assert len(fit['successful_models'])==3 and fit['fit_attempts']==3 and cpu['strategies']==8
    print('Phase312 verified:',summary['scientific_status'],len(summary['files']),'files;',fit['fit_attempts'],'logistic fits; zero neural forwards')
if __name__=='__main__':main()
