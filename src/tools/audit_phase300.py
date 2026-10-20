"""Independent, non-neural remote receipt/cache/resource audit after exit."""
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path('/home/scv/shared/phase300_20261020')
REPO = Path('/home/scv/Cloud-Adapter-light')
OLD = Path('/home/scv/shared/phase208_20261020')


def load(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''): h.update(part)
    return h.hexdigest()


def main():
    assert (ROOT/'EXIT_CODE').read_text().strip() == '0'
    controller = load(ROOT/'CONTROLLER_STATUS.json')
    assert controller['status']=='completed_pending_local_evaluation'
    events = [json.loads(line) for line in (ROOT/'CALL_LEDGER.jsonl').read_text().splitlines()]
    assert [e['global_call'] for e in events]==list(range(1,len(events)+1))
    assert len(events)==controller['total_calls']<=7000
    assert controller['stage_seconds_total']<=14400
    kinds = dict(collections.Counter(e['kind'] for e in events))
    assert kinds.get('fixed_teacher_full_mosaic')==400
    assert sum(n for k,n in kinds.items() if k.startswith('probe_'))==32
    locked_code = load(ROOT/'EXECUTION_LOCK.json')['code']
    for name,expected in locked_code.items(): assert sha(REPO/name)==expected
    historical_sources = load(OLD/'INPUT_LOCK.json')['source_sha256']
    for name,expected in historical_sources.items(): assert sha(REPO/name)==expected
    for name,expected in load(OLD/'CODE_LOCK.json')['files'].items(): assert sha(OLD/'code'/name)==expected
    source_weights = load(ROOT/'V4_WEIGHT_LOCK.json')['weights']
    for row in source_weights:
        assert sha(Path('/home/scv/shared/phase200_20261017/assets')/row['filename'])==row['sha256']
    teacher = load(ROOT/'teacher/CACHE_LOCK.json')
    assert len(teacher['rows'])==100 and teacher['windows']==400 and teacher['base_unchanged']
    for row in teacher['rows']: assert sha(ROOT/'teacher'/(row['product']+'.npz'))==row['sha256']
    gate = load(ROOT/'gate/GATE_ACCEPTANCE.json')
    assert gate['status']=='passed' and gate['rng_restored']
    assert teacher['frozen_base_digest']==gate['fixed_base_digest']
    seal = load(ROOT/'TRAINING_SEAL.json')
    stages = []
    access = {}
    receipts = {}
    for stage in ('teacher','gate','train_B','train_M','infer'):
        status = load(ROOT/stage/'RUN_STATUS.json')
        assert status['status']=='completed'
        assert status['cpu_peak_bytes']<=16*1024**3 and status['gpu_peak_bytes']<=20*1024**3
        assert status['minimum_host_free_bytes']>=4000000000
        stages.append(status)
        file = ROOT/stage/'FILE_ACCESS.jsonl'
        if file.exists():
            records = [json.loads(line) for line in file.read_text().splitlines()]
            forbidden = [r for r in records if any(t in r['path'].lower() for t in ('masks.zip','alt_masks.zip','classification_tags','coordinate_metadata','pixbox','ground_truth'))]
            assert not forbidden, (stage,forbidden)
            access[stage] = dict(recorded_python_opens=len(records),forbidden_label_opens=0,
                                 limitation='Python audit only, not native syscall tracing')
        if stage.startswith('train_'):
            arm = stage[-1]
            receipt = load(ROOT/stage/'TRAINING_RECEIPT.json')
            assert receipt['attempts']==1000 and receipt['updates']+receipt['skipped']==1000
            assert kinds['training_'+arm]==receipt['updates']
            assert receipt['original_parameters_buffers_unchanged'] and receipt['buffers_unchanged'] and receipt['base_all_eval']
            assert receipt['base_before']==receipt['base_after']==gate['fixed_base_digest']
            assert sha(ROOT/stage/'module_final.safetensors')==seal[arm]==receipt['final_sha256']
            assert sha(ROOT/'gate'/(arm+'_initial.safetensors'))==receipt['initial_sha256']
            receipts[arm] = receipt
    assert receipts['B']['pseudo_cache_lock_sha256']==receipts['M']['pseudo_cache_lock_sha256']==sha(ROOT/'teacher/CACHE_LOCK.json')
    assert receipts['B']['crop_schedule_sha256']==receipts['M']['crop_schedule_sha256']==sha(ROOT/'CROP_SCHEDULE.json')
    assert [r['attempt'] for r in receipts['B']['losses'] if r['accepted']==0]==[r['attempt'] for r in receipts['M']['losses'] if r['accepted']==0]
    assert set(load(ROOT/'infer/SCORES_SEALED.json')['scores'])=={'B','M'}
    baseline = Path('/home/scv/shared/phase205_20261019')
    baseline_seal = load(baseline/'infer/SCORES_SEALED.json')['scores']
    count = 0
    for strategy in ('R4','E4'):
        for key,expected in baseline_seal[strategy].items():
            pool,product = key.split('/',1)
            suffix = product+'.npz' if pool=='Catalogue' else product+'/UNLABELLED_POINT_SCORES.json'
            assert sha(baseline/'infer'/strategy/'scores'/pool/suffix)==expected
            count += 1
    size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    # Include the prospective archive plus local archive/extracted copies.
    assert 4*size < 6000000000, 'Combined complete backup projection exceeds persistent cap'
    report = dict(status='passed',calls=len(events),by_kind=kinds,
      GPU_stage_seconds=controller['stage_seconds_total'],GPU_peak_bytes=max(r['gpu_peak_bytes'] for r in stages),
      CPU_peak_bytes=max(r['cpu_peak_bytes'] for r in stages),host_min_free_bytes=min(r['minimum_host_free_bytes'] for r in stages),
      current_remote_root_bytes=size,combined_backup_upper_bytes=4*size,
      baseline_caches_SHA_verified=count,teacher_members_SHA_verified=100,
      fixed_source_weights=source_weights,locked_code_sha256=locked_code,
      historical_source_sha256=historical_sources,image_only_access=access,
      both_endpoints_and_pseudo_crop_hashes_verified=True,reference_metrics_evaluated=False,
      runtime_distributions={n:importlib.metadata.version(n) for n in ('torch','segmentation-models-pytorch','timm','numpy','safetensors')})
    (ROOT/'FINAL_REMOTE_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
