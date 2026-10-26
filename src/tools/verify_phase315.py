"""Portable published files/protocol/backup SHA verification; no local analysis or model loads."""
from phase315_common import *
def main():
    summary=load(S/'SUMMARY_SHA256.json');assert summary['status']=='completed'
    for n,h in summary['files'].items():assert sha(S/n)==h,n
    for n,h in summary['code'].items():assert sha(R/n)==h,n
    a=load(S/'CPU_ACCEPTANCE.json');ind=load(S/'INDEPENDENT_ACCEPTANCE.json');state=load(S/'STATUS.json');b=load(S/'BACKUP_RECEIPT.json');ident=read(S/'SEED_IDENTITY.csv')
    assert a['status']==ind['status']=='passed' and state['status']=='completed';assert a['both_endpoints_sealed_before_references'] and a['historical64_confusions_exact'] and a['no_torch']
    assert len(ident)==3 and len({r['initial_A_sha256'] for r in ident})==3 and all(r['B_zero']=='True' and int(r['parameters'])==294912 and int(r['updates'])==955 and int(r['skipped'])==45 for r in ident)
    assert ind['all96_group_tensors_and3pooled_exact'] and ind['all_original_input_cache_SHA_reverified'] and state['neural_forwards']<=2800 and state['gpu_seconds']<=28800
    assert a['historical_D0']==506798 and a['historical_B0']==784478 and len(read(S/'PER_GROUP_METRICS.csv'))==128 and len(read(S/'TRANSITIONS27.csv'))==3*33*27
    assert load(S/'CONFIRMATION_ADMISSION_LOCK.json')['ready_independent_pools']==0 and state['independent_confirmation_pools_ready']==0
    assert b['all_archive_members_verified'] and b['archive_SHA_verified_before_extraction'] and b['projected_two_hosts_including_archives_and_summary']<=6000000000
    if ROOT.exists():
        for f,h in load(S/'CPU_INPUT_SHA256.json').items():assert sha(f)==h,f
        for f,h in load(S/'EXECUTION_LOCK.json')['inputs'].items():assert sha(f)==h,f
        assert sha(ROOT/b['archive'])==b['archive_sha256'] and sha(ROOT/'BACKUP_FILES.json')==b['manifest_sha256']
        for n,v in load(ROOT/'BACKUP_FILES.json').items():assert sha(ROOT/n)==v['sha256'] and (ROOT/n).stat().st_size==v['bytes'],n
    print('Phase315 verified:3distinctinitialA,955updates each,all32groups,originalD0/B0,27cell margins,full backup SHA;0newpools evaluated')
if __name__=='__main__':main()
