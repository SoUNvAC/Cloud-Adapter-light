"""Portable verification of actual Phase304 delivery and budget receipts."""
from phase304_common import R,sha,load,read

def main():
    s=R/'src/research_results/phase304_20261024';lock=load(s/'SUMMARY_SHA256.json')
    for n,h in lock['files'].items():assert sha(s/n)==h,n
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    d=load(s/'DELIVERY_STATUS.json');assert d['complete_backup'] and d['actual_calls']<=3000
    receipt=load(s/'BACKUP_RECEIPT.json');assert receipt['all_stream_members_verified']
    stages=load(s/'STAGES.json')
    assert sum(v['seconds'] for k,v in stages.items() if k!='analyze')<=14400
    assert all(v['peak_cpu_bytes']<=16*1024**3 and v['minimum_available_memory']>=4000000000 for v in stages.values())
    assert receipt['raw_bytes']*2+sum(p.stat().st_size for p in s.iterdir() if p.is_file())*2+64000000<=6000000000
    if d['status']=='completed':
        end=load(s/'ENDPOINT_LOCK.json');assert (end['updates'],end['skipped'])==(955,45) and end['all_base_tensors_unchanged']
        pseudo=load(s/'PSEUDO_LOCK.json');assert pseudo['classes']==[40622307,33554491,1292360] and pseudo['accepted']==75469158
        assert len(read(s/'CROP_REPRODUCTION.csv'))==1000
        cpu=load(s/'CPU_ACCEPTANCE.json');assert cpu['groups']==32 and cpu['source_student_marginals_exact'] and cpu['AP_independent_threshold_group_checked']
        audit=load(s/'PSEUDO_AUDIT_ACCEPTANCE.json');assert audit['fit_products']==100 and audit['student_audit_probes_only']==4 and audit['training_mask_unmodified']
        assert len(read(s/'TRANSITIONS27.csv'))==27 and len(read(s/'SAME512_TRAINING.csv'))==48
        assert len(read(s/'PER_GROUP_METRICS.csv'))==64 and len(read(s/'GROUP_DAMAGE_AND_ABSENCE.csv'))==32
    print('Phase304 verified: '+d['status']+'; '+str(len(lock['files']))+' summary files; '+str(len(lock['code']))+' scripts')

if __name__=='__main__':main()
