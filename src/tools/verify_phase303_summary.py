"""Portable acceptance of Phase303 evidence and fixed support arithmetic."""
import csv
from pathlib import Path
from transfer_phase303 import R,load,sha


def main():
    root=R/'src/research_results/phase303_20261024';lock=load(root/'SUMMARY_SHA256.json')
    for name,h in lock['files'].items():assert Path(name).name==name and sha(root/name)==h,name
    for name,h in lock['code'].items():assert sha(R/name)==h,name
    cached=load(root/'CACHED_ACCEPTANCE.json')
    assert cached['status']=='passed' and cached['valid']==104448397 and cached['classes']==[101857,0,0]
    assert cached['accepted_products']==5 and cached['updates']==6 and cached['gradient_products']==2
    with (root/'ORIGINAL_SIX_UPDATES.csv').open(encoding='utf-8-sig') as f:updates=list(csv.DictReader(f))
    assert len(updates)==6 and len({r['product'] for r in updates})==2
    assert sum(int(r['accepted']) for r in updates)==152948
    state=load(root/'DELIVERY_STATUS.json')
    if state['status']=='completed':
        recovery=load(root/'RECOVERY_ACCEPTANCE.json');assert 873<=recovery['calls']<=900
        import json
        calls=[json.loads(s) for s in (root/'CALLS.jsonl').read_text().splitlines()]
        assert len(calls)==recovery['calls']
        assert sum(r['kind']=='source_reconstruction_window' for r in calls)==873
        assert load(root/'CPU_ACCEPTANCE.json')['status']=='passed'
        assert load(root/'DIAGNOSTIC_ACCEPTANCE.json')['replayed_rows']==8000
        stages=load(root/'STAGES.json');assert stages['recover']['seconds']<=3600
        assert sum(v['seconds'] for k,v in stages.items() if k!='recover')<=3600
        assert all(v['peak_cpu_bytes']<=16*1024**3 and v['minimum_available_memory']>=4000000000 for v in stages.values())
        with (root/'THRESHOLD_SUMMARY.csv').open(encoding='utf-8-sig') as f:summary=list(csv.DictReader(f))
        assert len(summary)==8
        for r in summary:assert sum(int(r[c]) for c in ('Surface','Cloud','Shadow'))==int(r['accepted'])
        assert load(root/'BACKUP_FINAL.json')['all_archive_members_verified']
    else:
        assert state['status']=='cached_complete_waiting_gpu' and state['actual_calls']==0
        assert load(root/'BACKUP_CACHED.json')['all_archive_members_verified']
    print(__import__('json').dumps(dict(status='passed',experiment_status=state['status'],summary_files=len(lock['files']),
        source_files=len(lock['code']),actual_calls=state['actual_calls'],training_updates=0)))


if __name__=='__main__':main()
