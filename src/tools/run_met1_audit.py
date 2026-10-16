"""MET-1: sealed aggregate matrices only; no imagery, models or resampling."""
import argparse
import csv
import datetime
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

CLASSES = ('Surface', 'Cloud', 'Shadow')
POLICIES = {'81': ('MM', 'LL', 'LM', 'ML'), '84': ('LL', 'R4', 'F_real', 'F_shuffle')}
DISPLAY = {'F_real': 'F真实', 'F_shuffle': 'F打乱'}
PAIRS = {'81': (('LM', 'LL'), ('LM', 'MM'), ('ML', 'MM')),
         '84': (('F_real', 'R4'), ('F_real', 'F_shuffle'), ('F_real', 'LL'))}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def csvrows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def near(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b) <= 1e-10), (a, b)


def matrix_stats(m, reference=None):
    assert isinstance(m, list) and len(m) == 3
    assert all(isinstance(r, list) and len(r) == 3 for r in m)
    assert all(type(x) is int and x >= 0 for r in m for x in r)
    t = [sum(r) for r in m]
    if reference is not None:
        assert t == reference, ('matrix orientation/reference mismatch', t, reference)
    p = [sum(m[i][j] for i in range(3)) for j in range(3)]
    tp = [m[i][i] for i in range(3)]
    union = [t[i]+p[i]-tp[i] for i in range(3)]
    u = [i for i in range(3) if union[i] > 0]
    ref = [i for i in range(3) if t[i] > 0]
    assert ref, 'empty reference'
    iou = [tp[i]/union[i] if union[i] else None for i in range(3)]
    missing = sorted(set(u)-set(ref))
    errors = sum(t)-sum(tp)
    return dict(t=t, p=p, TP=tp, union=union, IoU=iou, U=u, T=ref,
                old_mIoU=sum(iou[i] for i in u)/len(u),
                reference_classes_mIoU=sum(iou[i] for i in ref)/len(ref),
                missing_reference_predicted_classes=missing,
                missing_class_false_positives={CLASSES[i]: p[i] for i in missing},
                errors=errors, error_rate=errors/sum(t), valid=sum(t),
                surface_to_cloud=m[0][1], cloud_to_surface=m[1][0],
                surface_cloud_errors=m[0][1]+m[1][0],
                surface_cloud_error_rate=(m[0][1]+m[1][0])/sum(t),
                shadow_to_surface=m[2][0], shadow_to_cloud=m[2][1],
                shadow_recall=tp[2]/t[2] if t[2] else None,
                nonshadow_to_shadow=m[0][2]+m[1][2], nonshadow_denominator=t[0]+t[1],
                nonshadow_FPR=(m[0][2]+m[1][2])/(t[0]+t[1]) if t[0]+t[1] else None)


def independent(m):
    """Second implementation: direct set unions and exact rational averaging."""
    truth = [sum(m[c][j] for j in range(3)) for c in range(3)]
    fractions = []
    reference = []
    for c in range(3):
        # Intersection plus every off-diagonal cell incident to this class.
        denominator = m[c][c] + sum(m[c][j]+m[j][c] for j in range(3) if j != c)
        if denominator:
            q = Fraction(m[c][c], denominator)
            fractions.append(q)
            if truth[c]:
                reference.append(q)
    return dict(old=float(sum(fractions)/len(fractions)),
                reference=float(sum(reference)/len(reference)),
                errors=sum(m[i][j] for i in range(3) for j in range(3) if i != j))


def summed(matrices):
    return [[sum(m[i][j] for m in matrices) for j in range(3)] for i in range(3)]


def reproduce(m, record, csvrow=None):
    s = matrix_stats(m, record['reference_support'])
    assert s['valid'] == record['valid_pixels'] == 1044484
    assert record['confusion'] == m
    assert len(s['U']) == record['defined_iou_classes']
    near(s['old_mIoU'], record['miou_percent']/100)
    for a, b in zip(s['IoU'], record['iou_percent']):
        near(a, b/100 if b is not None else None)
    for a, b in [('shadow_to_surface', 'shadow_to_surface'), ('shadow_to_cloud', 'shadow_to_cloud'),
                 ('surface_to_cloud', 'surface_to_cloud'), ('cloud_to_surface', 'cloud_to_surface')]:
        assert s[a] == record[b]
    assert s['t'][2] == record['positive'] and s['nonshadow_denominator'] == record['negative']
    assert s['TP'][2] == record['true_positive'] and s['nonshadow_to_shadow'] == record['false_positive']
    near(s['shadow_recall'], record['recall'])
    near(s['nonshadow_FPR'], record['fpr'])
    if csvrow:
        assert csvrow['product'] == record['product'] and csvrow['group_id'] == record['group_id']
        near(s['old_mIoU'], float(csvrow['miou_percent'])/100)
        assert int(csvrow['valid_pixels']) == s['valid']
        for i, a in enumerate(CLASSES):
            assert int(csvrow[a+'_reference_pixels']) == s['t'][i]
            raw = csvrow[a+'_IoU_percent']
            near(s['IoU'][i], float(raw)/100 if raw else None)
            for j, b in enumerate(CLASSES):
                assert int(csvrow[a+'_to_'+b]) == m[i][j], ('CSV axis mismatch', a, b)
    alt = independent(m)
    near(s['old_mIoU'], alt['old'])
    near(s['reference_classes_mIoU'], alt['reference'])
    assert s['errors'] == alt['errors']
    return s


def sign(v):
    return (v > 0)-(v < 0)


def difference(a, b):
    assert a['t'] == b['t'] and a['valid'] == b['valid']
    old = a['old_mIoU']-b['old_mIoU']
    ref = a['reference_classes_mIoU']-b['reference_classes_mIoU']
    return dict(old_delta_pp=old*100, reference_delta_pp=ref*100,
                gap_pp=(ref-old)*100, old_sign=sign(old), reference_sign=sign(ref),
                absolute_old_delta=abs(old), absolute_reference_delta=abs(ref),
                sign_changed=sign(old) != sign(ref),
                reference_class_count=len(a['T']),
                U_count_changed=len(a['U']) != len(b['U']),
                rule_class_count_changed_A=len(a['U']) != len(a['T']),
                rule_class_count_changed_B=len(b['U']) != len(b['T']),
                error_delta=a['errors']-b['errors'],
                error_rate_delta=(a['errors']-b['errors'])/a['valid'],
                surface_cloud_delta=a['surface_cloud_errors']-b['surface_cloud_errors'])


class Budget:
    def __init__(self, out, input_root):
        self.start = time.monotonic()
        self.out = out
        self.input_root = input_root
        self.min_free = None
        self.peak = 0
        self.max_bytes = 0
        if os.name != 'nt':
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    def check(self):
        if os.name == 'nt':
            from phase122_local_resources import peak_rss
            rss = peak_rss()
            free = shutil.disk_usage(self.out).free
        else:
            import resource
            from install_phase111_runtime import host_free
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            free = host_free()  # WSL host physical disk; never substitute virtual df.
        self.peak = max(self.peak, rss)
        self.min_free = min(self.min_free or free, free)
        self.max_bytes = max(self.max_bytes, sum(p.stat().st_size for root in (self.out, self.input_root) for p in root.rglob('*') if p.is_file()))
        assert self.peak <= 2*1024**3 and free >= 4000000000
        assert self.max_bytes <= 128*1024**2 and time.monotonic()-self.start <= 1800
        assert datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))) < datetime.datetime(2026,10,20,tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
    def receipt(self):
        self.check()
        return dict(seconds=time.monotonic()-self.start, RSS_peak_bytes=self.peak,
                    host_min_free_bytes=self.min_free, new_bytes=self.max_bytes,
                    CPU_threads_max=4, GPU_calls=0, image_reads=0, training=0, downloads=0)


def run(manifest, input_root, out, budget):
    config = load(manifest)
    paths = {}
    for rel, h in config['files'].items():
        p = (input_root/rel).resolve()
        assert p.is_relative_to(input_root.resolve()) and sha(p) == h, rel
        paths[rel] = p
    assert config['axis'] == dict(rows='truth', columns='prediction', classes=list(CLASSES))
    for phase, field in [('81', 'code_sha256'), ('84', 'code_files')]:
        artifacts = load(paths[phase+'/artifact_hashes.json'])
        lock = load(paths[phase+'/input_configuration_lock.json'])
        for name in ('confusion_matrices.json','development_groups.csv','summary.csv',f'phase{phase}_report.json','input_configuration_lock.json'):
            assert sha(paths[phase+'/'+name]) == artifacts[name]
        assert load(paths[phase+'/LOCAL_VERIFIED.json'])['status'] == 'passed'
        for code in ('src/tools/phase65d_metrics.py','src/tools/run_phase81.py'):
            assert sha(paths['evidence/'+Path(code).name]) == lock[field][code]
        if phase == '84':
            assert sha(paths['evidence/run_phase84.py']) == lock[field]['src/tools/run_phase84.py']
    metric_source = paths['evidence/phase65d_metrics.py'].read_text(encoding='utf-8')
    assert 'truth[valid]*3+pred[valid]' in metric_source and '.reshape(3,3)' in metric_source
    assert "('Surface','Cloud','Shadow')" in paths['evidence/run_phase81.py'].read_text(encoding='utf-8')
    write(out/'INPUT_MANIFEST.json', config)
    write(out/'EXECUTION.json', dict(python=sys.executable,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        manifest_sha256=sha(manifest), code_sha256={name:sha(Path(__file__).with_name(name))
            for name in ('run_met1_audit.py','prepare_met1_inputs.py','test_met1_audit.py')}))
    details, comparisons, rankings, data = [], {}, {}, {}
    for phase, names in POLICIES.items():
        matrices = load(paths[phase+'/confusion_matrices.json'])
        report = load(paths[phase+f'/phase{phase}_report.json'])['policies']
        assert set(names) <= set(matrices) and set(names) <= set(report)
        fkeys = {k for k in matrices if k.startswith('F_')}
        if phase == '84':
            assert fkeys == {'F_real','F_shuffle'}
            assert "NAMES=('F_real','F_shuffle')" in paths['evidence/run_phase84.py'].read_text(encoding='utf-8')
        csv_all = csvrows(paths[phase+'/development_groups.csv'])
        summary = {r['policy']: r for r in csvrows(paths[phase+'/summary.csv'])}
        data[phase] = {}
        identity = None
        for name in names:
            raw = matrices[name]['groups' if phase == '81' else 'rows']
            rows = report[name]['rows']
            ids = {r['product']: r['group_id'] for r in rows}
            assert len(rows) == len(ids) == len(set(ids.values())) == len(raw) == 32
            assert {r['product'] for r in raw} == set(ids)
            if identity is None: identity = ids
            assert ids == identity
            csv_by = {r['product']: r for r in csv_all if r['policy'] == name}
            assert len(csv_by) == sum(r['policy'] == name for r in csv_all) == 32
            by = {r['product']: r for r in rows}
            data[phase][name] = {}
            for r in raw:
                pid = r['product']
                s = reproduce(r['confusion'], by[pid], csv_by[pid])
                if name != names[0]:
                    assert s['t'] == data[phase][names[0]][pid]['t']
                s.update(product=pid, group_id=ids[pid], policy=name, phase=phase)
                data[phase][name][pid] = s
                details.append(s)
            total = summed([r['confusion'] for r in raw])
            assert total == matrices[name]['pooled'] == report[name]['confusion']
            pooled = matrix_stats(total)
            near(pooled['old_mIoU'], report[name]['miou_percent']/100)
            macro = sum(s['old_mIoU'] for s in data[phase][name].values())/32
            near(macro, report[name]['macro_miou_percent']['mean']/100)
            near(macro, float(summary[name]['macro_mIoU_percent'])/100)
            near(pooled['old_mIoU'], float(summary[name]['pooled_mIoU_percent'])/100)
            for i, cl in enumerate(CLASSES):
                near(pooled['IoU'][i], report[name]['iou_percent'][i]/100)
                k = cl+'_IoU_percent' if phase == '84' else 'pooled_'+cl+'_IoU'
                near(pooled['IoU'][i], float(summary[name][k])/100)
            write(out/f'{phase}_{name}_pooled.json', pooled)
            budget.check()
        rankings[phase] = {}
        for metric in ('old_mIoU','reference_classes_mIoU'):
            values = {n: sum(s[metric] for s in data[phase][n].values())/32 for n in names}
            rankings[phase][metric] = [dict(policy=n, value=values[n], rank=1+sum(v>values[n] for v in values.values())) for n in sorted(names,key=lambda n:(-values[n],n))]
        for a, b in PAIRS[phase]:
            deltas = []
            for pid, s in data[phase][a].items():
                delta = difference(s, data[phase][b][pid])
                delta.update(product=pid, group_id=s['group_id'])
                deltas.append(delta)
            strata = []
            for count in (1,2,3):
                rr = [r for r in deltas if r['reference_class_count'] == count]
                present = [r['product'] for r in rr]
                strata.append(dict(reference_classes=count, groups=len(rr),
                    old_delta_pp=sum(r['old_delta_pp'] for r in rr)/len(rr) if rr else None,
                    reference_delta_pp=sum(r['reference_delta_pp'] for r in rr)/len(rr) if rr else None,
                    sign_changed=sum(r['sign_changed'] for r in rr),
                    missing_predicted={n:[dict(product=p, classes=data[phase][n][p]['missing_reference_predicted_classes'],
                        false_positives=data[phase][n][p]['missing_class_false_positives']) for p in present] for n in (a,b)}))
            comparisons[phase+':'+a+'-'+b] = dict(rows=deltas, strata=strata,
                old_delta_pp=sum(r['old_delta_pp'] for r in deltas)/32,
                reference_delta_pp=sum(r['reference_delta_pp'] for r in deltas)/32,
                sign_changed=sum(r['sign_changed'] for r in deltas),
                error_delta=sum(r['error_delta'] for r in deltas),
                error_rate_delta=sum(r['error_delta'] for r in deltas)/(32*1044484))
    assert data['81']['LL'] == {p:dict(s, phase='81') for p,s in data['84']['LL'].items()}, 'cross-phase LL reproduction'
    write(out/'GROUP_DETAILS.json', details)
    write(out/'COMPARISONS.json', comparisons)
    write(out/'RANKINGS.json', rankings)
    write(out/'CHECKS.json', dict(status='passed', old_metrics_reproduced=256, independent_rational_recompute=True,
          class_axes='rows=truth,columns=prediction', class_order=CLASSES, strategy_display=DISPLAY,
          cross_phase_LL_equal=True, no_new_bootstrap=True, old_decisions_unchanged=True))
    cases = [('81','LM','LL','20WNV'),('84','F_real','R4','51TUM'),('84','F_real','R4','50PLR'),('84','F_real','R4','44UNA')]
    picked = []
    for phase,a,b,tile in cases:
        rr = [r for r in comparisons[phase+':'+a+'-'+b]['rows'] if '_T'+tile+'_' in r['product']]
        assert len(rr) == 1
        picked.append((phase,a,b,tile,rr[0]))
    assert next(s for s in details if s['phase']=='81' and s['policy']=='LM' and '_T20WNV_' in s['product'])['cloud_to_surface'] == 254
    assert next(s for s in details if s['phase']=='84' and s['policy']=='F_real' and '_T51TUM_' in s['product'])['surface_to_cloud'] == 1292
    assert next(s for s in details if s['phase']=='84' and s['policy']=='F_real' and '_T50PLR_' in s['product'])['cloud_to_surface'] == 74
    text = ['|来源／对象|比较或案例|旧定义mIoU差 pp|参考类固定定义mIoU差 pp|差值符号改变组数／总组数|实际误分变化|备注|', '|---|---|---|---|---|---|---|']
    def label(a,b): return DISPLAY.get(a,a)+'−'+DISPLAY.get(b,b)
    def row(phase,title,v,n,note):
        return f"|{phase}|{title}|{v['old_delta_pp']:+.6f}|{v['reference_delta_pp']:+.6f}|{v.get('sign_changed',False)}/{n}|{v['error_delta']:+d}像元；{v['error_rate_delta']*100:+.6f} pp|{note}|"
    for phase, pairs in PAIRS.items():
        for a,b in pairs:
            v = comparisons[phase+':'+a+'-'+b]
            ss = '；'.join(f"参考{x['reference_classes']}类n={x['groups']}：旧/新差"+(f"{x['old_delta_pp']:+.4f}/{x['reference_delta_pp']:+.4f}pp，变号{x['sign_changed']}" if x['groups'] else 'NA') for x in v['strata'])
            nk = next(r for r in v['rows'] if '_T32NKF_' in r['product'])
            text.append(row(phase,label(a,b),v,32,ss+f"；T32NKF真实Surface/Cloud错分变化{nk['surface_cloud_delta']:+d}像元"))
    for phase,a,b,tile,v in picked:
        aa = next(s for s in details if s['phase']==phase and s['policy']==a and s['product']==v['product'])
        bb = next(s for s in details if s['phase']==phase and s['policy']==b and s['product']==v['product'])
        v = dict(v, sign_changed=int(v['sign_changed']))
        note = f"参考类{len(aa['T'])}；旧有效类{len(bb['U'])}→{len(aa['U'])}；Surface→Cloud {bb['surface_to_cloud']}→{aa['surface_to_cloud']}，Cloud→Surface {bb['cloud_to_surface']}→{aa['cloud_to_surface']}"
        text.append(row(phase,'T'+tile+'：'+label(a,b),v,1,note))
    receipt = budget.receipt()
    write(out/'RESOURCE_RECEIPT.json', receipt)
    text.append(f"|运行状态|完成|—|—|—|—|{receipt['seconds']:.3f}秒；原指标复现、双实现、256组矩阵与pooled检查通过；无异常；GPU/影像/训练/下载=0；执行端{out}|" )
    (out/'phasemet-1-result.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    budget.check()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.output, args.input_root)
    try:
        budget.check()
        run(args.manifest, args.input_root, args.output, budget)
        (args.output/'EXIT_CODE').write_text('0\n')
    except Exception:
        write(args.output/'FAILURE.json', dict(status='paused_input_or_engineering', error=traceback.format_exc()))
        (args.output/'phasemet-1-result.md').write_text('|来源／对象|比较或案例|旧定义mIoU差 pp|参考类固定定义mIoU差 pp|差值符号改变组数／总组数|实际误分变化|备注|\n|---|---|---|---|---|---|---|\n|运行状态|阻塞|NA|NA|NA|NA|校验失败，详见执行端FAILURE.json；未修改旧文件|\n', encoding='utf-8')
        (args.output/'EXIT_CODE').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
