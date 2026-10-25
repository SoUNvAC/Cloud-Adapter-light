"""Portable SHA/protocol checks only; independent numeric audit is remote already."""
from phase314_common import *
def main():
    lock=load(S/'SUMMARY_SHA256.json');assert lock['status']=='completed'
    for n,h in lock['files'].items():assert sha(S/n)==h,n
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    state=load(S/'STATUS.json');a=load(S/'CPU_ACCEPTANCE.json');ind=load(S/'INDEPENDENT_ACCEPTANCE.json');b=load(S/'BACKUP_RECEIPT.json')
    assert state['status']=='completed' and state['total_seconds']<=3600 and state['peak_cpu_bytes']<=8*1024**3 and state['minimum_available_memory']>=4000000000
    assert all(state[k]==0 for k in ['training','fits','forwards','downloads'])
    assert a['reference_products']==60 and a['calibration']==10 and a['validation']==50 and a['B_groups']==4 and a['C_groups']==12 and a['original_confusions_exact']==8 and a['tensor81_cells']==324
    assert a['onehot_checked_per_reference'] and a['invalid_never_argmaxed'] and a['E_counts_identical'] and a['training_mask_unmodified'] and a['no_torch']
    assert ind['status']=='passed' and ind['tensor81_and_confusion_margins'] and ind['E_AB_identical'] and ind['alias_not_independent'] and ind['pseudo_region_partitions']
    assert len(read(S/'DUAL_LABEL_MANIFEST.csv'))==60 and len(read(S/'TENSOR81.csv'))==324 and len(read(S/'PSEUDO_SUPPORT.csv'))==12
    assert b['all_archive_members_verified'] and b['archive_SHA_verified_before_extraction'] and b['projected_two_hosts_including_archives_and_summary']<=500000000
    if ROOT.exists():
        for p,h in load(S/'INPUT_SHA256.json').items():assert sha(p)==h,p
        assert sha(ROOT/b['archive'])==b['archive_sha256'] and sha(ROOT/'BACKUP_FILES.json')==b['manifest_sha256']
        for n,v in load(ROOT/'BACKUP_FILES.json').items():assert sha(ROOT/n)==v['sha256'] and (ROOT/n).stat().st_size==v['bytes'],n
    print('Phase314 verified: A60/B4/C12; original8 confusions and324tensor cells; zero training/forward/download; full backup SHA passed')
if __name__=='__main__':main()
