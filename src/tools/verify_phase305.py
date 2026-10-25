"""Portable verification of Phase305 diagnostic delivery and all fixed partitions."""
from phase305_common import *
def main():
    lock=load(S/'SUMMARY_SHA256.json')
    for n,h in lock['files'].items():assert sha(S/n)==h,n
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    r=load(S/'RESOURCE.json');assert r['neural_calls']==r['training_updates']==0
    assert r['seconds']<=3600 and r['peak_CPU_bytes']<=16*1024**3 and r['minimum_available_bytes']>=4000000000
    backup=load(S/'BACKUP_RECEIPT.json');assert backup['all_members_SHA_verified']
    assert backup['raw_bytes']*5+10000000<1000000000
    if lock['status']=='completed':
        a=load(S/'CPU_ACCEPTANCE.json');assert all(v for k,v in a.items() if isinstance(v,bool) and k!='torch_imported') and not a['torch_imported']
        assert len(read(S/'SCENE_STRATA.csv'))==33 and len(read(S/'SCENE_MAPPING.csv'))==352
        assert len(read(S/'G_IMAGE_ONLY_LOCK.csv'))==32 and len(read(S/'TRANSITIONS27.csv'))==27
        assert len(read(S/'FIXED_READOUT_RESULTS.csv'))==3 and len(read(S/'READOUT_CHANGES.csv'))==33
        g=read(S/'G_RELIABILITY.csv')[0];assert int(g['D_correct'])==506798 and int(g['D_other'])==19553
    print('Phase305 verified: '+lock['status']+'; '+str(len(lock['files']))+' summary files; '+str(len(lock['code']))+' scripts')
if __name__=='__main__':main()
