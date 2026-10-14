"""Render aggregate Phase118 results and preserve actual engineering costs."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
O = R / 'src/research_results/phase118_20261014'
U = R / 'outputs/phase118/phase118_20261014'


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    with p.open('xb') as f:
        f.write(data)


def obj(p, data):
    save(p, (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def rows(name):
    with (O / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def f(x, digits=3):
    return f'{float(x):.{digits}f}' if x not in ('', None) else 'NA'


def main():
    summary = rows('three_setting_summary.csv')
    groups = rows('per_group_results.csv')
    comparisons = load(O / 'FIXED_COMPARISONS.json')['rows']
    support = load(O / 'COMMON_SUPPORT_LOCK.json')
    metric = load(O / 'CPU_METRIC_ACCEPTANCE.json')
    state = load(U / 'RUN_STATUS.json')
    receipts = load(U / 'SCENE_RECEIPTS.json')
    plan = load(U / 'INPUT_PATCH_PLAN.json')
    assert len(groups) == 10 and len(summary) == 3 and len(receipts) == 32
    assert support['before_reference_join'] and support['common_valid'] == 9500
    assert metric['evaluated_positive'] == 379 and metric['evaluated_negative'] == 8190
    assert metric['ignore_on_common_support'] == 931
    costs = collections.defaultdict(lambda: {'scene_calls': 0, 'seconds': 0., 'scenes': 0})
    for r in receipts:
        c = costs[r['setting']]
        c['scene_calls'] += r['calls']; c['seconds'] += r['seconds']; c['scenes'] += 1
    assert all(c['scenes'] == 16 for c in costs.values())
    successful_calls = sum(c['scene_calls'] for c in costs.values())
    clean_plan_calls = plan['planned_calls_including_11_construction_and_8_repeats']
    assert clean_plan_calls == successful_calls + 19
    for name in ('FIRST_ATTEMPT_STATUS.json', 'SECOND_ATTEMPT_STATUS.json', 'FAILED_WORKER_TERMINATION.json'):
        save(O / name, (U / name).read_bytes())
    first = load(U / 'FIRST_ATTEMPT_STATUS.json'); second = load(U / 'SECOND_ATTEMPT_STATUS.json')
    assert first['all_calls'] == 15 and second['all_calls'] == 3072
    assert state['all_calls'] == state['main_calls'] + state['repeat_calls'] + state['construction_calls'] + second['all_calls'] == 16771
    assert state['status'] == 'all_scores_sealed_before_local_reference_join'
    ledger = {
        'actual_cumulative_network_calls': state['all_calls'],
        'final_successful_scene_calls': successful_calls,
        'corrected_clean_plan_total_including_checks': clean_plan_calls,
        'engineering_overhead_vs_clean_plan_calls': state['all_calls'] - clean_plan_calls,
        'all_attempt_construction_calls': 33, 'all_attempt_repeat_calls': 14,
        'all_attempt_main_forward_calls': first['main_calls'] + second['main_calls'] + state['main_calls'],
        'attempts': [
            {'attempt': 'first', 'calls': 15, 'outcome': 'keyword capture API failure; 0 sealed scenes',
             'conservative_wall_seconds': first['elapsed_seconds']},
            {'attempt': 'recovery_001', 'own_calls': 3057, 'outcome': 'planner omitted author deduplication; 5 sealed scenes reused',
             'own_wall_seconds': second['elapsed_seconds'] - first['elapsed_seconds']},
            {'attempt': 'recovery_002', 'own_calls': state['all_calls'] - second['all_calls'],
             'outcome': '32 sealed scenes, exit0; original settings unchanged',
             'own_wall_seconds': state['elapsed_seconds'] - second['elapsed_seconds']}
        ],
        'successful_scene_costs': dict(costs),
        'conservative_cumulative_wall_seconds': state['elapsed_seconds'],
        'wall_is_not_active_GPU_kernel_timer': True,
        'CPU_RSS_peak_bytes': max(s['rss_peak_bytes'] for s in (first, second, state)),
        'GPU_observed_peak_bytes': max(s.get('gpu_observed_peak_bytes', 0) for s in (first, second, state)),
        'host_physical_free_bytes': load(U / 'COST_SNAPSHOT.json')['host_free_after_bytes'],
        'original_O20_new_calls': 0, 'fitting_training_download_threshold_search': 0,
        'new_file_size_before_package_bytes': load(U / 'COST_SNAPSHOT.json')['new_intermediate_bytes_before_package']
    }
    assert ledger['all_attempt_main_forward_calls'] + 33 + 14 == 16771
    obj(O / 'ACTUAL_COST_LEDGER.json', ledger)
    text = ['# Phase118：冻结Omni的分辨率／上下文诊断', '', '## 1．数据结果', '',
            '执行完成。零训练、拟合、下载、校准或阈值搜索；原V1双模型和影像身份不变。O20复用20m／20km上下文；C20为20m／10km；H10为10m／10km。', '',
            '16产品、10相关组、9500原native10m固定点全部保留共同支持；其中379参考云影、8190参考非云影、931 ignore。8组有正负支持计算AP，48NUG／19GBQ无参考云影为NA，但进入误报描述。无组低于5正／20负；54TWN仍仅7个正点，小样本局限保留。先封存全部分数和标签盲共同支持，再本地连接原Phase103参考。', '',
            '本轮Omni只需原三通道，不额外套用Phase112四模型共同支持的Source六波段筛选，因此恢复那两条Omni有效负点；三个设置全部重算，未直接拼接历史AP。', '',
            '| 设置 | 逐组平均AP (%) | 池化AP (%) | 逐组平均召回 (%) | 逐组平均负类误报 (%) | 最坏组负类误报 (%) |',
            '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append('| ' + r['setting'] + ' | ' + ' | '.join(f(r[k]) for k in (
            'macro_AP_pp', 'pooled_AP_pp', 'macro_recall_pp', 'macro_negative_FP_pp', 'worst_group_negative_FP_pp')) + ' |')
    text += ['', '召回／误报固定confidence≥0.5；这是选择性采样点统计，不是全景FPR或安全工作点。AP每组先合并产品，再相关组等权；池化AP单独列出。', '',
             '| 固定比较 | 平均ΔAP (百分点) | 配对组bootstrap 95%区间 | 上升／不变／下降组 |',
             '|---|---:|---|---|']
    for r in comparisons:
        label = r['comparison'] + ('（主要）' if r['primary'] else '（辅助）')
        text.append(f"| {label} | {f(r['mean_delta_AP_pp'])} | [{f(r['CI95_pp'][0])}, {f(r['CI95_pp'][1])}] | {r['positive_direction_groups']}/{r['exact_zero_groups']}/{r['negative_direction_groups']} |")
    text += ['', '固定10000次配对相关组bootstrap、seed118014。8个组是统计单位，9500点与多个设置不增加独立组数。方向按实际差的正／零／负描述，没有追加显著性或1pp通过门。', '',
             '| Tile相关组 | 正／负点 | O20 AP (%) | C20 AP (%) | H10 AP (%) | H10−C20 (pp) | C20−O20 (pp) | H10−O20 (pp) |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in groups:
        text.append(f"| {r['tiles']} | {r['positive']}/{r['negative']} | " + ' | '.join(f(float(r[k])*100) if r[k] else 'NA' for k in ('O20_AP', 'C20_AP', 'H10_AP')) + ' | ' + ' | '.join(f(r[k]) for k in ('H10_minus_C20_AP_pp', 'C20_minus_O20_AP_pp', 'H10_minus_O20_AP_pp')) + ' |')
    text += ['', '完整逐组召回、精确率、误报和计数见per_group_results.csv。H10在22WES召回为0（C20／O20均12.5%）；33VVE误报最高，O20／C20／H10分别3.622%／4.720%／5.818%。无影48NUG误报分别0.711%／1.659%／1.185%，19GBQ分别0／0／0.086%。', '',
             '| 设置 | 池化召回 (%) | 池化精确率 (%) | 池化负类误报 (%) | TP／FP |', '|---|---:|---:|---:|---|']
    for r in summary:
        text.append(f"| {r['setting']} | {f(float(r['pooled_recall'])*100)} | {f(float(r['pooled_precision'])*100)} | {f(float(r['pooled_negative_FP_fraction'])*100)} | {r['pooled_TP']}/{r['pooled_FP']} |")
    ratio = costs['H10']['seconds'] / costs['C20']['seconds']
    text += ['', '## 2．结论探讨', '',
             '主要H10−C20均值−0.032pp，描述区间跨零；没有显示总体排序增益，也不能由此证明两设置等效或处理尺度无作用。3组上升、5组下降；54TWN上升12.286pp但仅7正点，19FDV下降8.924pp、22WES下降4.744pp。收益和损伤抵消，比总体近零更值得保留为局部尺度线索。', '',
             '将20m上下文从20km缩为10km时，C20−O20为−0.892pp，区间也跨零；虽然5组小幅或明显上升，44SKJ的−13.613pp和22WES的−5.287pp拉低均值。H10−O20为−0.924pp；54TWN／33VVE／48XWG上升不能替代主要比较。', '',
             '本批O20的组平均AP与采样误报取舍较好；C20池化AP略高0.217pp，但不是预定主要终点。H10没有呈现总体收益，并增加采样误报和全景执行成本；暂不把H10替换为默认设置。依据宽松探索规则，区间跨零仅记不确定，保留尺度方向，不机械关闭，也不继续搜索窗口。', '',
             '若启动下一小轮，建议仅做两项零训练、无新前向的固定诊断：①按固定10m点在所属20m单元内的位置（行列奇偶）分层，检查H10−C20变化是否与点位采样尺度有关；②利用已保存的原patch坐标／CPU回执，按拼接覆盖次数及距patch边缘的固定分层比较，检查44SKJ／22WES上下文损伤及54TWN收益是否伴随调度差异。仍以相关组为单位，不把分层点当独立场景，不搜索阈值或窗口。本轮不启动上述工作、不改变参考或阈值。', '',
             '这些只是可检验的解释，不是本轮已证实机制。两种新设置同时涉及原采样尺度、红／绿波段降采样／NIR插值、逐块归一化和拼接调度；名义物理上下文相同不构成严格单因素因果识别。B8A由20m插值为10m不会创造新的NIR观测。PixBox已暴露且选择性采样，训练地点交集未知、原参考语义限制保留，不能称独立外部确认、全场景安全性或所有分辨率策略无效。Phase117及旧主检验的原失败不改写。', '',
             '## 3．验收、成本与交付', '',
             f"实际累计16771次网络调用，32景有效前向{successful_calls}次；纠正后的无故障计划含构建和重复为{clean_plan_calls}次，工程额外{state['all_calls']-clean_plan_calls}次。三次尝试共33构建、14重复；全部重复差0。两次工程失败分别为捕获接口关键字、规划器遗漏作者去重，均保留原回执，在既定恢复预算内修复，复用前5景分数，不修改模型／窗口／数值容差。累计保守wall {state['elapsed_seconds']:.3f}秒（{state['elapsed_seconds']/60:.2f}分钟，含故障等待），不是GPU核函数计时。",
             '', f"成功场景调用C20／H10分别{costs['C20']['scene_calls']}／{costs['H10']['scene_calls']}次；逐景耗时和分别{costs['C20']['seconds']:.3f}／{costs['H10']['seconds']:.3f}秒，H10约{ratio:.2f}倍。这不是独立基准计时：同机顺序运行、包含影像SHA／读取／CPU检查，构建与失败另记。O20本轮新增调用0，不能将复用的0成本当作其实际推理耗时。", '',
             f"CPU RSS峰{ledger['CPU_RSS_peak_bytes']/1e9:.3f}GB，GPU观测峰{ledger['GPU_observed_peak_bytes']/1e9:.3f}GB；宿主物理盘剩余{ledger['host_physical_free_bytes']/1e9:.3f}GB。打包前新增文件{ledger['new_file_size_before_package_bytes']}bytes，私有点分数压缩包1.42MB；没有保存／传输密集四通道数组或work_dirs。均在预算与物理空间保留线内。", '',
             '32景独立CPU拼接／点位confidence误差均≤1e−5，格网／点位覆盖和gradient检查通过；独立tied AP与原CPU AP一致，配对组bootstrap向量与标量复算通过。48原波段SHA、96格网输入绑定、冻结runtime／V1双权重核验通过。', '',
             '交付：输入／patch计划、分数SHA与远端小回执在remote_delivery/；共同支持、损失表、三行总表、全部逐组表、固定比较及CPU验收在本目录。无标签逐点分数／raw logits保留在远端shared及本地data受控目录，参考连接仅在本地data；Git只含小聚合数据、元数据和代码。工程失败代码快照和日志保留。最迟2026-10-20北京时间停止线不变。', '']
    report = '\n'.join(text).encode()
    save(O / 'README.md', report); save(U / 'FINAL_RESULTS.md', report)
    completion = {'phase': 118, 'status': 'completed_exploratory_uncertainty_reported',
                  'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'primary': comparisons[0], 'all32_scenes_sealed': True,
                  'CPU_acceptance_passed': True, 'common_support': 9500,
                  'all_original10_groups_retained': True, 'work_dirs_transferred': False,
                  'all_models_settings_reference_thresholds_unchanged': True,
                  'no_followup_experiment_started': True,
                  'historical_phase117_verdict_unchanged': True}
    obj(O / 'COMPLETION.json', completion)
    for name in ('ACTUAL_COST_LEDGER.json', 'COMPLETION.json'):
        save(U / name, (O / name).read_bytes())
    paths = sorted(p for p in O.rglob('*') if p.is_file())
    paths += sorted(R.glob('src/tools/*phase118*.py'))
    paths += [R / 'src/research_plans/PHASE118_RESOLUTION_CONTEXT_DIAGNOSTIC_20261014.md',
              R / 'src/research_plans/EXPLORATION_POLICY_20261014.md']
    manifest = {p.relative_to(R).as_posix(): sha(p) for p in paths}
    obj(O / 'REPOSITORY_DELIVERY_SHA256.json', manifest)
    save(U / 'REPOSITORY_DELIVERY_SHA256.json', (O / 'REPOSITORY_DELIVERY_SHA256.json').read_bytes())
    print(json.dumps({'status': 'completed', 'repository_files': len(manifest), 'costs': dict(costs)}))


if __name__ == '__main__':
    main()
