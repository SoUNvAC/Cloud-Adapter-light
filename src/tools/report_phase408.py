"""Paired scientific comparison against sealed Phase405."""
from pathlib import Path
import json, csv, statistics, hashlib
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'outputs/phase408/phase408_20261101'

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
    assert js(R / 'PENALTY_SELF_TEST.json')['penalty_gradient_only_block5_head']
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
    lines = ['# Phase408：仅block5头的训练期残差约束', '', '## 做了什么', '',
             '仅惩罚block5对应H头的归一化残差能量，系数0.25；其他三个头不加惩罚，保持Phase406该头原有系数。复用Phase405两个seed及逐步输入；最终4000步固定评价。源val未进入训练，无额外前向。', '',
             '## 结果', '', '|seed|配置|目标mIoU|源mIoU|目标Δ405|源Δ405|', '|---|---|---:|---:|---:|---:|']
    for v, d in zip(combined, paired):
        assert (v['seed'], v['case']) == (d['seed'], d['case'])
        lines.append(f"|{v['seed']}|{v['case']}|{v['target_mIoU']:.4f}|{v['source_mIoU']:.4f}|{d['target_mIoU']:+.4f}|{d['source_mIoU']:+.4f}|")
    lines += ['', '差值均相对同seed、同推理缩放的Phase405。', '', '## 判断', '',
              '主要weak_5配置两个seed均达到目标不降、源域改善。' if passed else '主要weak_5配置未在两个seed均达到目标不降、源域改善；本轮双域改善判据未通过。', '',
              f"主要配置平均目标差{statistics.mean(x['target_mIoU'] for x in primary):+.4f}、源差{statistics.mean(x['source_mIoU'] for x in primary):+.4f}个百分点。", '',
              f"共{complete['calls']}次前向、{complete['updates']}次更新，墙钟{complete['seconds']:.1f}秒。", '', '## 验收与限制', '',
              '哈希与调用账本、逐图/汇总混淆指标、宏AP算术、冻结回执、Source与最终H端点通过。两个seed初始化及每步图像/标签/增强/学习率与405一致；正则自测梯度通过，训练正则非负且更新后非零。新点池化AP未独立像元复算。相同已暴露数据上的两seed探索，不是独立泛化或显著性证明。未启动409。']
    full = [v for v in paired if v['case']=='1.0']
    lines += ['', '次要观察：全强度H平均目标差'+f"{statistics.mean(v['target_mIoU'] for v in full):+.4f}"+'、源差'+f"{statistics.mean(v['source_mIoU'] for v in full):+.4f}"+'个百分点。这是同推理强度比较，本轮未进行同目标精度匹配，不能据此宣布超越原weak_5。']
    if not passed:
        lines += ['', '决策：保留405 weak_5基线，当前单头系数0.25方案不升级为主方案。结合406全层与本轮单头结果，暂止残差能量惩罚的系数/层位细搜；后续优先核实CloudSEN训练集可用性，再设计少量源训练样本重放的配对实验。源val继续只评价。该建议尚未执行，也不预设重放一定有效。']
    (R/'RESULT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps({'passed': True, 'checks': checks+['paired inputs and LR with405', 'penalty gradient selftest and loss trajectory'], 'scientific_dual_domain_criterion_passed': passed, 'new_point_pooled_AP_independent_pixel_recompute': False}, indent=2))
    print('\n'.join(lines))
