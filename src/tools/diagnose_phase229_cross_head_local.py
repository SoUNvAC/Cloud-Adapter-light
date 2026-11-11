"""Decompose frozen confusion counts. No model, pixel access, or new CI."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/phase229/phase229_20261026'
OUT = ROOT / 'outputs/research_audits/phase229_cross_head_diagnosis_20261026'
NAMES = ['Surface', 'Cloud', 'Shadow']
HASHES = {}


def read(path):
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def add(matrices):
    matrices = list(matrices)
    return [[sum(m[i][j] for m in matrices) for j in range(3)] for i in range(3)]


def scores(cm):
    support = [sum(r) for r in cm]
    unions = [support[i] + sum(cm[j][i] for j in range(3)) - cm[i][i] for i in range(3)]
    ious = [100 * cm[i][i] / unions[i] if unions[i] else None for i in range(3)]
    defined = [x for x in ious if x is not None]
    return {'mIoU': mean(defined), 'IoU': ious, 'classes': len(defined),
            'errors': sum(cm[i][j] for i in range(3) for j in range(3) if i != j)}


def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    scene_rows, class_rows, flows, leave_rows, summaries = [], [], [], [], []
    for domain, other in [('L8', 'SPARCS'), ('SPARCS', 'L8')]:
        source = read(BASE / f'delivery/evaluation/{domain}_Source.json')
        groups = sorted(source['per_group'])
        for seed in [21701, 21702]:
            native = read(BASE / f'delivery/evaluation/{domain}_{domain}_{seed}.json')
            cross = read(BASE / f'delivery/evaluation/{domain}_{other}_{seed}.json')
            for data in [source, native, cross]:
                assert sorted(data['per_group']) == groups
                assert add(x['confusion'] for x in data['per_input']) == data['confusion']
                assert add(data['per_group'].values()) == data['confusion']
                for group in groups:
                    assert add(x['confusion'] for x in data['per_input'] if x['scene']==group) == data['per_group'][group]
            reference = {x['name']: (x['scene'], x['image_SHA'], x['mask_SHA'], x['reference_support']) for x in source['per_input']}
            for data in [native, cross]:
                assert {x['name']: (x['scene'], x['image_SHA'], x['mask_SHA'], x['reference_support']) for x in data['per_input']} == reference
            for group in groups:
                cm_s, cm_n, cm_c = [x['per_group'][group] for x in [source, native, cross]]
                assert [sum(x) for x in cm_s] == [sum(x) for x in cm_n] == [sum(x) for x in cm_c]
                ms, mn, mc = [scores(x) for x in [cm_s, cm_n, cm_c]]
                row = dict(domain=domain, seed=seed, scene=group,
                           Source_mIoU=ms['mIoU'], native_mIoU=mn['mIoU'], cross_mIoU=mc['mIoU'],
                           cross_minus_native_pp=mc['mIoU']-mn['mIoU'],
                           native_minus_Source_pp=mn['mIoU']-ms['mIoU'],
                           cross_minus_Source_pp=mc['mIoU']-ms['mIoU'],
                           Source_valid_classes=ms['classes'], native_valid_classes=mn['classes'], cross_valid_classes=mc['classes'],
                           reference_Surface=sum(cm_s[0]), reference_Cloud=sum(cm_s[1]), reference_Shadow=sum(cm_s[2]))
                scene_rows.append(row)
                for label, base_cm in [('native', cm_n), ('Source', cm_s)]:
                    for i in range(3):
                        for j in range(3):
                            if i != j:
                                flows.append(dict(domain=domain, seed=seed, scene=group, baseline=label,
                                                  flow=f'{NAMES[i]}_to_{NAMES[j]}',
                                                  error_count_delta=cm_c[i][j]-base_cm[i][j], reference_class_pixels=sum(base_cm[i])))
            s, n, c = [scores(x['confusion']) for x in [source, native, cross]]
            for i, label in enumerate(NAMES):
                class_rows.append(dict(domain=domain, seed=seed, class_name=label,
                                       Source_IoU=s['IoU'][i], native_IoU=n['IoU'][i], cross_IoU=c['IoU'][i],
                                       cross_minus_native_pp=c['IoU'][i]-n['IoU'][i],
                                       cross_minus_Source_pp=c['IoU'][i]-s['IoU'][i],
                                       contribution_to_pooled_mIoU_gap_pp=(c['IoU'][i]-n['IoU'][i])/3))
            for group in groups:
                n_without = scores(add(native['per_group'][g] for g in groups if g != group))
                c_without = scores(add(cross['per_group'][g] for g in groups if g != group))
                leave_rows.append(dict(domain=domain, seed=seed, excluded_scene=group,
                                       cross_minus_native_pp=c_without['mIoU']-n_without['mIoU']))
            summaries.append(dict(domain=domain, seed=seed, Source_mIoU=s['mIoU'], native_mIoU=n['mIoU'], cross_mIoU=c['mIoU'],
                                  cross_minus_native_pp=c['mIoU']-n['mIoU'],
                                  native_minus_Source_pp=n['mIoU']-s['mIoU'], cross_minus_Source_pp=c['mIoU']-s['mIoU'],
                                  cross_minus_native_errors=c['errors']-n['errors'], cross_minus_Source_errors=c['errors']-s['errors']))
    save('SCENE_DECOMPOSITION.csv', scene_rows)
    save('CLASS_DECOMPOSITION.csv', class_rows)
    save('NET_ERROR_FLOWS.csv', flows)
    save('LEAVE_ONE_SCENE_OUT.csv', leave_rows)
    save('POOLED_SUMMARY.csv', summaries)
    audit = dict(inputs=HASHES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 checks='All 10 file input/scene sums; matched image/mask SHA and reference supports; original confusion unchanged.',
                 new_model_calls=0, training_updates=0, new_confidence_intervals=0,
                 interpretation='Net count differences, not direct per-pixel transitions. Leave-one-out is descriptive influence analysis, not selection.')
    (OUT / 'AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'pooled': summaries, 'classes': class_rows}, indent=2))


if __name__ == '__main__':
    main()
