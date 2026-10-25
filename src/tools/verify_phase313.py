"""Portable metadata/protocol/byte validation; no local scientific calculations."""
from phase313_common import *
def main():
    v=load(S/'SUMMARY_SHA256.json')
    for n,h in v['files'].items():assert sha(S/n)==h,n
    for n,h in v['code'].items():assert sha(R/n)==h,n
    a=load(S/'INVENTORY_ACCEPTANCE.json');state=load(S/'STATUS.json');b=load(S/'BACKUP_RECEIPT.json');network=load(S/'NETWORK.json')
    rows=read(S/'INPUT_READINESS.csv');assert len(rows)==132 and len({r['product'] for r in rows})==132
    assert sum(r['pool']=='fit' for r in rows)==100 and sum(r['pool']=='development' for r in rows)==32
    assert a['ready']==0 and a['approximate_only']==3 and a['missing']==129 and a['B11_verified']==3
    assert len(read(S/'OLD75_INTERSECTION.csv'))==3 and all(r['pool']=='fit' for r in read(S/'OLD75_INTERSECTION.csv'))
    assert a['source_scores_existing']==132 and a['student_caches_existing']==32
    assert all(r['Source_checkpoint_sha256']==SOURCE for r in rows)
    assert all(state[k]==0 for k in ['train','fit','forwards','projection','raster_downloads','DEM_downloads'])
    assert state['status']=='completed' and state['total_seconds']<3600 and state['peak_cpu_bytes']<=8*1024**3 and state['minimum_available_memory']>=4000000000
    assert network['response_body_bytes']<=20000000 and network['objects']<=8
    assert sum(r['response_body_bytes'] for r in network['requests'])==network['response_body_bytes']
    assert b['all_archive_members_verified'] and b['archive_SHA_verified_before_extraction'] and b['projected_two_hosts_including_archives_and_summary']<=300000000
    assert load(S/'SOURCE_VISUAL_QA.json')['status']=='passed'
    assert 'geometry experiment not executable'==v['decision']
    if ROOT.exists():
        for p,h in load(S/'INPUT_SHA256.json').items():assert sha(p)==h,p
        assert sha(ROOT/b['archive'])==b['archive_sha256']
        assert sha(ROOT/'BACKUP_FILES.json')==b['manifest_sha256']
        for name,item in load(ROOT/'BACKUP_FILES.json').items():
            assert sha(ROOT/name)==item['sha256'] and (ROOT/name).stat().st_size==item['bytes'],name
    print('Phase313 verified:132products, ready0/approximate3/missing129; zero training/forward/projection; full backup SHA passed')
if __name__=='__main__':main()
