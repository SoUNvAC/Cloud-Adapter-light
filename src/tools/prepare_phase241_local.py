"""Prepare Source/C diagnostic identities and support from sealed metadata only."""
from pathlib import Path
import collections
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
P236 = ROOT / 'outputs/phase236/phase236_20261027'
OUT = ROOT / 'outputs/phase241/phase241_preparation_20261030'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8')


def csvwrite(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sealpath = P236 / 'delivery/VAL_SELECTION_LOCK.json'
    lock = read(sealpath)
    assert sha(sealpath) == read(P236 / 'delivery/VAL_SELECTION_SEAL.json')['SHA']
    inputs = read(P236 / 'EVAL_INPUTS.json')
    identities = []
    for chosen in lock['selections']:
        if chosen['strategy'] != 'E':
            continue
        assert chosen['lr'] == 1e-4 and not chosen['source']
        relative = f"fits/{chosen['domain']}_{chosen['seed']}_1e-04/head_{chosen['step']}.pth"
        expected = lock['model_SHA'][relative]
        receipt = P236 / 'delivery' / Path(relative).parent / f"SAVE_{chosen['step']}.json"
        assert read(receipt)['head_SHA'] == expected
        local = P236 / 'delivery' / relative
        if local.exists():
            assert sha(local) == expected
        identities.append(dict(domain=chosen['domain'], model='C', seed=chosen['seed'],
                               learning_rate=chosen['lr'], step=chosen['step'],
                               head_SHA256=expected, remote_path='/home/scv/shared/phase236_20261027/' + relative,
                               local_checkpoint_exists=local.exists(),
                               local_status='hash_verified' if local.exists() else 'receipt_only_remote_not_checked'))
    assert len(identities) == 4
    write('MODEL_IDENTITIES.json', dict(Source_SHA256=read(P236 / 'LOCK.json')['Source_SHA256'],
          Source_remote_path='/home/scv/Cloud-Adapter-light/src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',
          C=identities, selection_seal_SHA256=sha(sealpath), remote_access=False))

    # Source rows contain the same reference support as C; verify duplicate references.
    refs = {}
    with (P236 / 'PER_SCENE.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['domain'], row['split'], row['scene']
            counts = tuple(int(row[k]) for k in ('Surface_pixels', 'Cloud_pixels', 'Shadow_pixels'))
            if key in refs:
                assert refs[key] == counts, key
            refs[key] = counts
    support, populations = [], []
    for domain, splits in inputs.items():
        for split, rows in splits.items():
            grouped = collections.defaultdict(list)
            for row in rows:
                grouped[row['scene']].append(row)
            assert len({row['name'] for row in rows}) == len(rows)
            for scene, items in sorted(grouped.items()):
                counts = refs[domain, split, scene]
                total = sum(counts)
                status = {v.get('usgs_shadows') for v in items}
                assert len(status) == 1
                flag = next(iter(status))
                eligible = domain == 'SPARCS' or flag == 'yes'
                detail = dict(domain=domain, split=split, scene=scene, inputs=len(items),
                              reference_shadow_status=flag if domain == 'L8' else 'published_SP_mapping',
                              primary_reference_support=eligible,
                              quality='published_reference_not_independent_truth' if eligible else 'known_uncertain_shadow_coverage',
                              Surface=counts[0], Cloud=counts[1], Shadow=counts[2])
                for i, name in enumerate(('Surface', 'Cloud', 'Shadow')):
                    detail[name + '_OvR_evaluable'] = 0 < counts[i] < total
                for i, j in ((2, 0), (1, 0), (1, 2)):
                    detail[f'{("Surface", "Cloud", "Shadow")[i]}_vs_{("Surface", "Cloud", "Shadow")[j]}_evaluable'] = counts[i] > 0 and counts[j] > 0
                support.append(detail)
            pool = [r for r in support if r['domain'] == domain and r['split'] == split]
            populations.append(dict(domain=domain, split=split, inputs=len(rows), scenes=len(grouped),
                primary_scenes=sum(r['primary_reference_support'] for r in pool),
                nominal_OvR_support={c: sum(r[c + '_OvR_evaluable'] for r in pool) for c in ('Surface', 'Cloud', 'Shadow')},
                primary_OvR_support={c: sum(r['primary_reference_support'] and r[c + '_OvR_evaluable'] for r in pool) for c in ('Surface', 'Cloud', 'Shadow')}))
        assert not ({v['scene'] for v in splits['val']} & {v['scene'] for v in splits['test']})
    csvwrite('SCENE_SUPPORT.csv', support)
    write('POPULATIONS.json', populations)

    cache = []
    for stage in ('val', 'test'):
        complete = read(P236 / 'delivery' / f'{stage.upper()}_COMPLETE.json')
        assert complete['no_saved_scores'] is True
        cache.append(dict(split=stage, official_receipt_no_saved_scores=True,
                          receipt=str((P236 / 'delivery' / f'{stage.upper()}_COMPLETE.json').relative_to(ROOT))))
    local_arrays = [p.relative_to(ROOT).as_posix() for p in P236.rglob('*')
                    if p.is_file() and p.suffix.lower() in ('.npy', '.npz')]
    # No archive extraction; presence checks cannot establish remote cache availability.
    write('CACHE_READINESS.json', dict(phase236_completion=cache, local_phase236_array_files=local_arrays,
          evaluation_code_explicitly_discards_scores=True,
          legacy_phase231_d_cache='binary_score_difference_for_older_heads_not_three_class_scores_for_selected_C',
          remote_cache_inventory='not_checked', archive_payload_inventory='not_extracted',
          conclusion='No verified reusable full three-class Source/C cache locally; executor must check remote before inference.'))
    forwards = sum(3 * r['inputs'] for r in populations)
    write('COST_ESTIMATE.json', dict(full_val_test_model_input_forwards=forwards,
          three_FP32_score_channels_uncompressed_GiB=forwards * 3 * 512 * 512 * 4 / 1024**3,
          calibration_extra_model_forwards=0, network_training_updates=0,
          inference_not_run=True, excludes_archive_headers_masks_CPU_sort_scratch_and_failures=True))
    evidence = [Path(__file__).resolve(),
                ROOT / 'src/research_plans/PHASE241_SOURCE_C_RANKING_ALIGNMENT_AUDIT.md',
                P236 / 'LOCK.json', sealpath, P236 / 'delivery/VAL_SELECTION_SEAL.json',
                P236 / 'EVAL_INPUTS.json', P236 / 'PER_SCENE.csv',
                P236 / 'code/evaluate_phase236.py',
                P236 / 'delivery/VAL_COMPLETE.json', P236 / 'delivery/TEST_COMPLETE.json']
    write('INPUT_SHA256.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in evidence})
    write('LOCAL_ACCEPTANCE.json', dict(status='passed', selected_C_models=4,
          scene_support_rows=len(support), selection_seal_verified=True,
          reference_support_identical_across_legacy_candidates=True,
          within_target_val_test_scene_disjoint=True,
          local_checkpoint_payloads_verified=sum(i['local_checkpoint_exists'] for i in identities),
          no_image_label_or_weight_payloads_read=True, no_remote_access=True,
          no_new_predictions_fits_or_uncertainty_computation=True))
    write('SHA256SUMS.json', {p.name: sha(p) for p in sorted(OUT.iterdir())
                            if p.is_file() and p.name != 'SHA256SUMS.json'})
    print(json.dumps(dict(output=str(OUT), populations=populations,
                          minimum_full_inference_calls=forwards), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
