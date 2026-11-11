"""Local aggregate audit; never infer confidence/student joint counts from margins."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/phase219/phase219_20261023'
OUT = ROOT / 'outputs/research_audits/phase219_class_damage_20261024'
locks = {}


def read(name):
    p = BASE / name
    locks[p] = hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


tensors = {(r['seed'], r['arm'], r['ROI']): json.loads(r['reference_teacher_student_counts'])
           for r in read('ENDPOINT_TEACHER_CORRECTNESS.csv')}
original = read('ALL_ROI_CORRECTNESS_DIFFERENCES.csv')
conf = read('TEACHER_CONFIDENCE_DIAGNOSTIC.csv')
rows = []
for seed in ['21701', '21702']:
    rois = sorted(roi for s, a, roi in tensors if s == seed and a == 'D')
    assert len(rois) == 107
    for roi in rois:
        g, d = tensors[seed, 'G', roi], tensors[seed, 'D', roi]
        for i, label in enumerate(['Surface', 'Cloud', 'Shadow']):
            n = sum(map(sum, g[i]))
            assert n == sum(map(sum, d[i]))
            assert [sum(g[i][j]) for j in range(3)] == [sum(d[i][j]) for j in range(3)]
            preservation = sum(d[i][i][k]-g[i][i][k] for k in range(3) if k != i)
            correction_cost = sum(d[i][j][i]-g[i][j][i] for j in range(3) if j != i)
            delta = sum(g[i][j][k]-d[i][j][k] for j in range(3) for k in range(3) if k != i)
            assert delta == correction_cost-preservation
            rows.append(dict(seed=seed,ROI=roi,reference_class=label,support=n,
                             teacher_correct_preservation_gain=preservation,
                             teacher_error_corrections_lost=correction_cost,
                             G_minus_D_reference_errors=delta))
        expected = next(r for r in original if r['seed']==seed and r['ROI']==roi and r['comparison']=='G-D')
        assert sum(r['G_minus_D_reference_errors'] for r in rows[-3:]) == int(expected['total_error_delta_pixels'])

summary = []
for seed in ['21701','21702']:
    for label in ['Surface','Cloud','Shadow']:
        rr = [r for r in rows if r['seed']==seed and r['reference_class']==label]
        fields = ['support','teacher_correct_preservation_gain','teacher_error_corrections_lost','G_minus_D_reference_errors']
        summary.append(dict(seed=seed,reference_class=label,**{k:sum(r[k] for r in rr) for k in fields}))
roirows = [r for r in original if r['comparison']=='G-D']
sets = [{r['ROI'] for r in roirows if r['seed']==s and float(r['source_mIoU_diff_pp']) < -1} for s in ['21701','21702']]
both = sorted(sets[0] & sets[1])
focus = {'ROI_0050','ROI_1013','ROI_1404','ROI_4813'}
newsummary=[]
for seed in ['21701','21702']:
    rr=[r for r in roirows if r['seed']==seed]
    for scope in ['all107','old_focus4','other103']:
        ss=[r for r in rr if scope=='all107' or ((r['ROI'] in focus) == (scope=='old_focus4'))]
        newsummary.append(dict(seed=seed,scope=scope,ROI_count=len(ss),
                              errors_added=sum(int(r['total_error_delta_pixels']) for r in ss),
                              mIoU_declined_ROIs=sum(float(r['source_mIoU_diff_pp'])<0 for r in ss)))
confidence_rois=collections.defaultdict(dict)
for r in conf:
    if r['ROI']!='__pooled__': confidence_rois[r['ROI']][r['band']]=r
assert len(confidence_rois)==107
assert all(float(v['high']['reference_error_rate'])<float(v['low']['reference_error_rate']) for v in confidence_rois.values())
exposure=[]
for label in ['Surface','Cloud','Shadow']:
    support={band:sum(int(v[band][label+'_support']) for v in confidence_rois.values()) for band in ['high','low']}
    n=sum(support.values())
    exposure.append(dict(reference_class=label,high_support=support['high'],low_support=support['low'],
                         fraction_in_high=support['high']/n,
                         diagnostic_G_mean_weight=(1.5*support['high']+.5*support['low'])/n,
                         uniform_D_weight=1,
                         scope='Source val only; not measured training exposure or gradient strength'))
OUT.mkdir(parents=True,exist_ok=True)
save('PER_ROI_REFERENCE_CLASS.csv',rows)
save('CLASS_SUMMARY.csv',summary)
save('FOCUS_VS_OTHER.csv',newsummary)
save('VAL_CLASS_WEIGHT_EXPOSURE.csv',exposure)
save('REPEATED_GT1_MIOU_DAMAGE.csv',[r for r in roirows if r['ROI'] in both])
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in locks.items())
audit=dict(inputs=[dict(path=str(p.relative_to(ROOT)),sha256=h) for p,h in locks.items()],
           input_hashes_unchanged=True,source_ROIs=107,confidence_ordering_holds_in_all_ROIs=True,
           missing_joint='Confidence band x reference x teacher prediction x student prediction is not saved.',
           cached_training_masks='Replay256 only; not the source-val535 masks. SHA alone cannot reconstruct a mask.',
           new_training=0,new_inference=0,
           limitation='Class/ROI results are exact marginal decompositions; they do not localize damage to confidence bands.')
(OUT/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json'}
(OUT/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(class_summary=summary,focus_summary=newsummary,class_weight_exposure=exposure,repeated_gt1_ROIs=both,output=str(OUT)),indent=2))
