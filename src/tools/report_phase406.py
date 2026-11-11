"""Paired scientific comparison against sealed Phase405."""
from pathlib import Path
import json, csv, statistics, hashlib
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'outputs/phase406/phase406_20261101'

def js(p):
    return json.loads(p.read_text(encoding='utf-8'))

def rows(p):
    with p.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def table(p, values):
    with p.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)

def report(complete, combined, checks):
    previous = ROOT / 'outputs/phase405/phase405_20261101'
    assert js(R / 'PENALTY_SELF_TEST.json')['passed']
    paired, penalties = [], []
    oldrows = rows(previous / 'summary.csv')
    baseline_manifest = js(previous/'LOCAL_MANIFEST.json')
    for name in ['summary.csv'] + [f'{seed}/{file}' for seed in [40501,40502] for file in ['formal_ACTIVE_SCOPE.json','formal_STEP_TRACE.jsonl']]:
        assert hashlib.sha256((previous/name).read_bytes()).hexdigest() == baseline_manifest[name]
    for seed in [40501, 40502]:
        p, old = R / str(seed), previous / str(seed)
        assert js(p/'formal_ACTIVE_SCOPE.json')['head_initial_SHA'] == js(old/'formal_ACTIVE_SCOPE.json')['head_initial_SHA']
        now = [json.loads(x) for x in (p/'formal_STEP_TRACE.jsonl').read_text().splitlines()]
        before = [json.loads(x) for x in (old/'formal_STEP_TRACE.jsonl').read_text().splitlines()]
        assert len(now) == len(before) == 4000
        keys = ['step', 'draw', 'product', 'image_SHA', 'label_SHA', 'augmentation_seed', 'LR']
        assert all(all(a[k] == b[k] for k in keys) for a, b in zip(now, before))
        assert all(abs(now[0]['losses'][k] - value) < 1e-6 for k, value in before[0]['losses'].items())
        vals = [x['losses']['loss_residual_energy'] for x in now]
        assert vals[0] == 0 and all(x > 0 for x in vals[1:])
        penalties.append({'seed': seed, 'first': vals[0], 'last': vals[-1], 'mean': statistics.mean(vals), 'maximum': max(vals)})
        for v in combined:
            if v['seed'] != seed:
                continue
            oldv = next(x for x in oldrows if int(x['seed']) == seed and x['case'] == v['case'])
            paired.append({'seed': seed, 'case': v['case'], **{k: v[k]-float(oldv[k]) for k in ['target_mIoU', 'source_mIoU', 'Shadow_IoU', 'macro_AP']}})
    table(R/'versus_phase405.csv', paired)
    table(R/'penalty_summary.csv', penalties)
    primary = [x for x in paired if x['case'] == 'weak_5']
    passed = all(x['target_mIoU'] >= 0 and x['source_mIoU'] > 0 for x in primary)
    lines = ['# Phase406：训练期残差能量约束', '', '## 做了什么', '',
             '仅增加四层H残差相对原特征能量的平均惩罚，系数1。复用Phase405两个seed及逐步输入；最终4000步固定评价。源val未进入训练，无额外前向。', '',
             '## 结果', '', '|seed|配置|目标mIoU|源mIoU|目标Δ405|源Δ405|', '|---|---|---:|---:|---:|---:|']
    for v, d in zip(combined, paired):
        assert (v['seed'], v['case']) == (d['seed'], d['case'])
        lines.append(f"|{v['seed']}|{v['case']}|{v['target_mIoU']:.4f}|{v['source_mIoU']:.4f}|{d['target_mIoU']:+.4f}|{d['source_mIoU']:+.4f}|")
    lines += ['', '差值均相对同seed、同推理缩放的Phase405。', '', '## 判断', '',
              '主要weak_5配置两个seed均达到目标不降、源域改善。' if passed else '主要weak_5配置未在两个seed均达到目标不降、源域改善；本轮双域改善判据未通过。', '',
              f"主要配置平均目标差{statistics.mean(x['target_mIoU'] for x in primary):+.4f}、源差{statistics.mean(x['source_mIoU'] for x in primary):+.4f}个百分点。", '',
              f"共{complete['calls']}次前向、{complete['updates']}次更新，墙钟{complete['seconds']:.1f}秒。", '', '## 验收与限制', '',
              '哈希与调用账本、逐图/汇总混淆指标、宏AP算术、冻结回执、Source与最终H端点通过。两个seed初始化及每步图像/标签/增强/学习率与405一致；正则自测梯度通过，训练正则非负且更新后非零。新点池化AP未独立像元复算。相同已暴露数据上的两seed探索，不是独立泛化或显著性证明。未启动407。']
    full = [x for x in paired if x['case']=='1.0']
    lines += ['', '## 次要观察与决策', '',
              f"全强度H相对405平均目标差{statistics.mean(x['target_mIoU'] for x in full):+.4f}、源差{statistics.mean(x['source_mIoU'] for x in full):+.4f}个百分点。"+
              ('两个seed方向均为双域改善；这是预设检查配置中的次要观察，不替代weak_5主要判据。' if all(x['target_mIoU']>0 and x['source_mIoU']>0 for x in full) else '未在两个seed均复现双域改善。'), '',
              '保留残差能量约束作为训练改进候选；当前结果不支持直接将它叠加到原weak_5配方。训练约束与推理缩放可能共同压低适配幅度，这只是解释假设。全强度源域仍有明显损伤，尚不能称抗遗忘成功。下一轮若继续，先用固定检查点比较目标相近时的源域表现，明确比较协议后再评价，避免继续叠加训练改动。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps({'passed': True, 'checks': checks+['paired inputs and LR with405', 'penalty gradient selftest and loss trajectory'], 'scientific_dual_domain_criterion_passed': passed, 'new_point_pooled_AP_independent_pixel_recompute': False}, indent=2))
    print('\n'.join(lines))
