"""Portable provenance/units/unknown-state checks, without re-running scientific computation."""
from phase309_common import *
def main():
    lock=load(S/'SUMMARY_SHA256.json');assert lock['status']=='completed'
    for n,h in lock['files'].items():assert sha(S/n)==h,n
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    a=load(S/'CPU_ACCEPTANCE.json');r=load(S/'RESOURCE.json');b=load(S/'BACKUP_RECEIPT.json');net=load(S/'NETWORK_EVIDENCE.json')
    assert a['products']==3 and a['pages']==9 and a['semantic_questions']==15 and a['issues']==7
    for k in ('all_reference_rows_including_missing','all_space_dimensions_in_metres','unknown_not_negation','no_hypothesis_written_as_affine','four_corner_center_residuals_NA_no_complete_transform','main_alt_fixed3_missing','historical_SHA_unchanged','production_packages_unchanged'):assert a[k],k
    assert a['fits']==a['training_updates']==a['neural_calls']==a['performance_recalculations']==0
    assert r['total_seconds']<=3600 and r['peak_CPU_tree_bytes']<=8*1024**3 and r['minimum_available_bytes']>=4000000000
    assert net['documents_attempted']<=10 and net['total_download_bytes']<=20000000
    assert b['all_members_SHA_verified'] and 6*b['raw_bytes']+20000000<500000000
    assert (S/'PRODUCTION_PACKAGES_BEFORE.json').read_bytes()==(S/'PRODUCTION_PACKAGES_AFTER.json').read_bytes()
    for p in load(S/'PDF_EXTRACTION.json')['pages']:
        assert sha(S/p['text'])==p['text_sha256'] and sha(S/p['png'])==p['png_sha256']
    for row in read(S/'SEMANTIC_EVIDENCE.csv'):
        assert row['status'] in ('有直接证据已澄清','证据有冲突','缺少证据仍未知') and (S/row['evidence_path']).exists()
    for name in ('SPATIAL_PROVENANCE','REFERENCE_RELATION','REFERENCE_CROSS_COUNTS'):assert len(read(S/(name+'.csv')))==3
    for row in read(S/'SPATIAL_PROVENANCE.csv'):assert row['units']=='metre' and row['origin']==row['affine']==row['four_corner_center_residuals']=='NA'
    for row in read(S/'REFERENCE_RELATION.csv'):assert row['alt_member']==row['cross_3x3_path']=='NA' and row['dataset_stage']=='MAIN'
    print(f"Phase309 verified: {len(lock['files'])} evidence/summary files; {len(lock['code'])} scripts; nine-page README; history unchanged")
if __name__=='__main__':main()
