"""Local count decomposition of Phase218; no pixel reads or model execution."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/phase218/phase218_20261023'
OUT = ROOT / 'outputs/research_audits/teacher_correctness_tradeoff_20261023'
FOCUS = {'ROI_0050', 'ROI_1013', 'ROI_1404', 'ROI_4813'}


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


locks = {}
states = {}
for r in read('PER_ROI_CORRECTNESS.csv'):
    assert r['pool'] == 'Source_val535'
    key = (r['seed'], r['arm'], r['ROI'])
    state = (r['teacher_correct'] == 'True', r['student_correct'] == 'True')
    assert state not in states.setdefault(key, {})
    states[key][state] = int(r['pixels'])
metrics = {(r['seed'], r['arm'], r['ROI']): r
           for r in read('PER_ROI.csv') if r['pool'] == 'Source_val535'}
both = {}
for r in read('PER_ROI_BOTH_WRONG.csv'):
    key = (r['seed'], r['arm'], r['ROI'])
    both.setdefault(key, []).append(int(r['pixels']))

rows = []
checks = 0
for seed in ['21701', '21702']:
    rois = sorted(roi for s, arm, roi in states if s == seed and arm == 'C')
    assert len(rois) == 107
    for roi in rois:
        c, d = states[(seed, 'C', roi)], states[(seed, 'D', roi)]
        cm, dm = metrics[(seed, 'C', roi)], metrics[(seed, 'D', roi)]
        for arm, st, m in [('C', c, cm), ('D', d, dm)]:
            assert len(st) == 4
            assert sum(st.values()) == int(m['reference_valid_pixels'])
            assert st[(True, False)] + st[(False, False)] == int(m['reference_student_errors'])
            assert st[(False, True)] + st[(False, False)] == int(m['reference_teacher_errors'])
            assert sum(both[(seed, arm, roi)]) == st[(False, False)]
            checks += 5
        assert sum(c.values()) == sum(d.values())
        assert c[(True, True)] + c[(True, False)] == d[(True, True)] + d[(True, False)]
        saved = c[(True, False)] - d[(True, False)]
        lost = c[(False, True)] - d[(False, True)]
        delta_error = int(dm['reference_student_errors']) - int(cm['reference_student_errors'])
        assert delta_error == lost - saved
        checks += 3
        n = sum(c.values())
        rows.append(dict(seed=seed,ROI=roi,focus=roi in FOCUS,valid_pixels=n,
                         C_teacher_correct_student_wrong=c[(True, False)],
                         D_teacher_correct_student_wrong=d[(True, False)],
                         net_teacher_correct_preservation_gain=saved,
                         C_teacher_wrong_student_correct=c[(False, True)],
                         D_teacher_wrong_student_correct=d[(False, True)],
                         net_teacher_error_corrections_lost=lost,
                         D_minus_C_reference_errors=delta_error,
                         D_minus_C_error_rate_pp=100*delta_error/n,
                         D_minus_C_mIoU_pp=100*(float(dm['mIoU'])-float(cm['mIoU'])) if abs(float(cm['mIoU'])) <= 1 else float(dm['mIoU'])-float(cm['mIoU']),
                         C_KL=float(cm['KL_mean_reference']),D_KL=float(dm['KL_mean_reference']),
                         KL_lower_errors_higher=float(dm['KL_mean_reference'])<float(cm['KL_mean_reference']) and delta_error>0))

summaries = []
for seed in ['21701', '21702']:
    for scope in ['all107', 'focus4', 'other103']:
        rr = [r for r in rows if r['seed']==seed and
              (scope=='all107' or (r['focus'] if scope=='focus4' else not r['focus']))]
        fields = ['valid_pixels', 'net_teacher_correct_preservation_gain',
                  'net_teacher_error_corrections_lost', 'D_minus_C_reference_errors']
        sums = {k: sum(r[k] for r in rr) for k in fields}
        assert sums['D_minus_C_reference_errors'] == sums['net_teacher_error_corrections_lost']-sums['net_teacher_correct_preservation_gain']
        checks += 1
        summaries.append(dict(seed=seed,scope=scope,ROIs=len(rr),**sums,
                              error_increased_ROIs=sum(r['D_minus_C_reference_errors']>0 for r in rr),
                              KL_lower_errors_higher_ROIs=sum(r['KL_lower_errors_higher'] for r in rr),
                              preservation_declined_ROIs=sum(r['net_teacher_correct_preservation_gain']<0 for r in rr),
                              corrections_lost_ROIs=sum(r['net_teacher_error_corrections_lost']>0 for r in rr)))
OUT.mkdir(parents=True, exist_ok=True)
save('PER_ROI_DECOMPOSITION.csv', rows)
save('FOCUS_ROI_DECOMPOSITION.csv', [r for r in rows if r['focus']])
save('SUMMARY.csv', summaries)
assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in locks.items())
audit = dict(checks=checks,inputs=[dict(path=str(p.relative_to(ROOT)),sha256=h) for p,h in locks.items()],
             input_hashes_unchanged=True,training=0,inference=0,
             definition='D-C errors = net teacher-error corrections lost - net teacher-correct preservation gain',
             limitations=['Net counts within fixed teacher-correct/wrong strata, not direct C-to-D pixel transition counts.',
                          'Pixel-error decomposition does not decompose nonlinear mIoU.',
                          'Reference correctness is diagnostic only; no validation labels used for training.',
                          'Observed count identity does not establish training-gradient causality.'])
(OUT/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256.json'}
(OUT/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'checks':checks,'summaries':summaries,'focus':[r for r in rows if r['focus']]},ensure_ascii=False,indent=2))
