"""Verify the portable Phase302 summary, source seals, and execution accounting."""
import json
from pathlib import Path
from transfer_phase302 import R,sha,load


def main():
    root=R/'src/research_results/phase302_20261020'
    lock=load(root/'SUMMARY_SHA256.json')
    for n,h in lock['files'].items():
        assert Path(n).name==n and n not in ('.','..')
        assert sha(root/n)==h,n
    for n,h in lock['code'].items():assert sha(R/n)==h,n
    end=load(root/'ENDPOINT_LOCK.json');execution=load(root/'EXECUTION_SUMMARY.json')
    assert end['updates']+end['skipped']==1000 and end['all_base_tensors_unchanged']
    calls=[json.loads(s) for s in (root/'CALLS.jsonl').read_text().splitlines()]
    kind={k:sum(r['kind']==k for r in calls) for k in {r['kind'] for r in calls}}
    assert kind['teacher_window']==900
    assert kind['Source_evaluation_window']==kind['LoRA_evaluation_window']==288
    assert all(kind['probe_'+str(n)]==4 for n in (0,100,500,1000))
    assert kind.get('train',0)==end['updates']
    assert len(calls)==execution['calls']<=4000 and execution['gpu_pipeline_seconds']<=14400
    assert execution['peak_gpu_reserved']<=20*1024**3 and execution['peak_cpu_bytes']<=16*1024**3
    assert load(root/'BACKUP_RECEIPT.json')['all_archive_members_verified']
    if lock['status']=='completed':
        assert load(root/'CPU_ACCEPTANCE.json')['status']=='passed'
        resource=load(root/('CPU_ANALYSIS_RESOURCE.json' if (root/'CPU_ANALYSIS_RESOURCE.json').exists() else 'LOCAL_ANALYSIS_RESOURCE.json'))
        assert resource['status']=='completed' and resource['minimum_available_physical_bytes']>=4000000000
        if (root/'CPU_BACKUP_RECEIPT.json').exists():assert load(root/'CPU_BACKUP_RECEIPT.json')['all_archive_members_verified']
    else:
        assert lock['status']=='gpu_completed_local_cpu_waiting'
        assert load(root/'LOCAL_ANALYSIS_WAIT.json')['remote_analysis_allowed'] is False
        assert not (root/'RESULT.md').exists(), 'No final statistics before local CPU analysis'
    print(json.dumps(dict(status='passed',experiment_status=lock['status'],summary_files=len(lock['files']),
        code_files=len(lock['code']),neural_calls=len(calls),actual_updates=end['updates'],
        independent_cpu_analysis_complete=lock['status']=='completed')))


if __name__=='__main__':main()
