"""Portable verification of actual Phase306 artifacts, budgets and fixed diagnostic tables."""
from phase306_common import *

def main():
    lock=load(S/'SUMMARY_SHA256.json')
    for n,h in lock['files'].items():assert sha(S/n)==h,n
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    resource=load(S/'RESOURCE.json');assert resource['neural_calls']==resource['training_updates']==0
    assert resource['seconds']<=3600 and resource['peak_CPU_bytes']<=16*1024**3 and resource['minimum_available_bytes']>=4000000000
    receipt=load(S/'BACKUP_RECEIPT.json');assert receipt['all_members_SHA_verified'] and receipt['raw_bytes']*5+10000000<1000000000
    if lock['status']=='completed':
        a=load(S/'CPU_ACCEPTANCE.json');assert a['neural_calls']==a['classifier_fits']==a['downloads']==0 and not a['torch_imported']
        assert all(v for k,v in a.items() if isinstance(v,bool) and k!='torch_imported')
        assert all(a['auc_numerical_tests'].values()) and a['independent_auc_checks']>0
        t=read(S/'EVENTS_POOLED.csv')[0];assert (int(t['D_correct']),int(t['G_damage']),int(t['nonG_damage']),int(t['nonG_correct_Surface']))==(506798,56740,450058,2812248)
        assert int(t['D'])+int(t['tiny_D'])==450058
        for name,n in (('EVENTS_PER_GROUP.csv',32),('SCENE_EVENT_STRATA.csv',33),('SCENE_MAPPING.csv',352),('SPECTRAL_SUPPORT_LOSS.csv',224),
            ('SUPPORT_AND_SAMPLE_SHA.csv',32),('BOUNDARY_EVENTS.csv',96),('CONFIDENCE_EVENT_PARTITIONS.csv',96),('BAND_QUANTILES.csv',864),
            ('AUC_RESULTS.csv',888),('EVIDENCE_COVERAGE.csv',6),('TOP_1_3_5_CONCENTRATION.csv',3)):
            assert len(read(S/name))==n,name
        assert set(r['band'] for r in read(S/'AUC_RESULTS.csv'))==set(BANDS)
    print('Phase306 verified: '+lock['status']+'; '+str(len(lock['files']))+' summary files; '+str(len(lock['code']))+' scripts')

if __name__=='__main__':main()
