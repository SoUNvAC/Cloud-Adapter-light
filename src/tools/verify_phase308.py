"""Portable delivery and immutability verification; no local scientific computation."""
from phase308_common import *
def main():
    summary=load(S/'SUMMARY_SHA256.json');assert summary['status']=='completed'
    for n,h in summary['files'].items():assert sha(S/n)==h,n
    for n,h in summary['code'].items():assert sha(R/n)==h,n
    a=load(S/'CPU_ACCEPTANCE.json');r=load(S/'RESOURCE.json');receipt=load(S/'BACKUP_RECEIPT.json')
    assert a['folds']==32 and a['rows']==96 and len(read(S/'REPRODUCTION.csv'))==96
    assert a['probability_max_error']<=1e-12 and a['contribution_sum_max_error']<=1e-12
    for k in ('all_original_counts_Brier_AUC_reproduced','samples_weights_labels_SHA_restored','train_only_scaler_checked','bounds_boolean_complement_checked'):assert a[k],k
    assert all(a['tests'].values()) and not a['torch_imported']
    assert a['classifier_fits']==a['training_updates']==a['neural_calls']==a['downloads']==r['classifier_fits']==0
    assert r['seconds']<=3600 and r['peak_CPU_bytes']<=16*1024**3 and r['minimum_available_bytes']>=4000000000
    assert receipt['all_members_SHA_verified'] and receipt['raw_bytes']*5+10000000<1000000000
    coords=load(S/'PREVIEW_COORDINATES.json');assert len(coords)==12 and len(read(S/'IMAGE_REFERENCE_AUDIT.csv'))==3
    for row in coords:assert sha(S/row['path'])==row['sha256']
    for row in read(S/'REPRODUCTION.csv'):
        assert row['original_repaired_D']==row['recomputed_repaired_D'] and row['original_revoked_K']==row['recomputed_revoked_K']
        if 'T25XEG' in row['product']:assert int(row['D'])==197082 and int(row['K'])==int(row['recomputed_repaired_D'])==0 and row['AUC']=='NA'
    assert load(S/'FAILURES.json')==[]
    print(f"Phase308 verified: {len(summary['files'])} summary files; {len(summary['code'])} scripts; all96 reproduced; zero fits/forwards")
if __name__=='__main__':main()
