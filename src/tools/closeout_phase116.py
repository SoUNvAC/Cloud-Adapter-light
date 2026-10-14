"""Copy accepted aggregate facts into a bounded closeout; never recompute metrics."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'src/research_results/phase116_20261014'
LOCAL = ROOT / 'outputs/phase116/phase116_20261014'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads((ROOT / p).read_text(encoding='utf-8'))


def main():
    assert not DEST.exists() and not LOCAL.exists()
    sources = [
        'src/research_plans/PHASE116_RESEARCH_ROUTE_CLOSEOUT_20261014.md',
        'src/research_plans/PHASE66_20261010.md',
        'src/research_results/phase101_20261013/sources/66/delivery/phase66_report.json',
        'outputs/phase66/phase66_20261010/delivery_verification.json',
        'outputs/phase66/phase66_20261010/RESULTS.md',
    ]
    names = {
        105: ['README.md', 'primary_endpoint.json', 'task_lock.json', 'CPU_statistics_acceptance.json'],
        106: ['README.md', 'delta_interval.json', 'model_lock.json', 'CPU_acceptance.json'],
        107: ['README.md', 'fixed_comparisons.json', 'model_lock.json', 'CPU_acceptance.json'],
        110: ['READINESS_DECISION.md'],
        112: ['README.md', 'primary_endpoint.json', 'four_score_summary.csv', 'two_operating_points.csv', 'CPU_STATISTICS_ACCEPTANCE.json'],
        113: ['README.md', 'PRIMARY_DIAGNOSTIC.json', 'fixed_error_flow_pooled.csv', 'CPU_ACCEPTANCE.json'],
        114: ['FINAL_REVIEW_AGGREGATES.json', 'REVIEW_CPU_ACCEPTANCE.json'],
        115: ['FINAL_REVIEW_REPORT.md', 'FINAL_POSTHOC_REVIEW_AGGREGATES.json', 'FINAL_REVIEW_ACCEPTANCE.json', 'FINAL_REVIEW_DELIVERY_SHA256.json'],
    }
    for phase, files in names.items():
        date = '20261013' if phase in (105, 106, 107) else '20261014'
        sources.extend(f'src/research_results/phase{phase}_{date}/{n}' for n in files)
    hashes = {p: sha(ROOT / p) for p in sources}
    p66 = load(sources[2])
    assert hashes[sources[2]] == load(sources[3])['files']['delivery/phase66_report.json']
    for phase in (112, 113, 114):
        manifest = load(f'src/research_results/phase{phase}_20261014/DELIVERY_SHA256.json')
        assert all(sha(ROOT / p) == h for p, h in manifest.items())
    manifest = load('src/research_results/phase115_20261014/FINAL_REVIEW_DELIVERY_SHA256.json')
    assert all(sha(ROOT / p) == h for p, h in manifest.items())
    p105 = load('src/research_results/phase105_20261013/primary_endpoint.json')
    p106 = load('src/research_results/phase106_20261013/delta_interval.json')
    p107 = load('src/research_results/phase107_20261013/fixed_comparisons.json')['rows'][0]
    p112 = load('src/research_results/phase112_20261014/primary_endpoint.json')
    p113 = load('src/research_results/phase113_20261014/PRIMARY_DIAGNOSTIC.json')
    p115 = load('src/research_results/phase115_20261014/FINAL_POSTHOC_REVIEW_AGGREGATES.json')
    ap66 = p66['comparisons']['L3_minus_L2']['ap']
    assert f"{ap66['mean']:.4f}" == '10.8610'
    assert [f'{x:.4f}' for x in ap66['ci95']] == ['5.1980', '17.6094']
    for value, expected in ((p105['mean_delta_AP_pp'], '8.745'), (p106['mean_delta_AP_pp'], '8.307'),
                            (p107['mean_delta_AP_pp'], '2.272'), (p112['mean_delta_AP_pp'], '-36.123'),
                            (p113['mean_delta_conditional_AUC_pp'], '-11.132')):
        assert f'{value:.3f}' == expected
    assert p105['passed'] and p106['post_hoc'] and p107['exploratory']
    assert not p112['passed'] and p112['declined'] == 8 and not p113['passed']
    assert p115['complete_fields_cases'] == 2 and p115['complete_fields_groups'] == 1
    assert p115['eligible_independent_cases'] == 0 and not p115['independent_support_established']
    with (ROOT / 'src/research_results/phase112_20261014/four_score_summary.csv').open(encoding='utf-8') as f:
        score_rows = list(csv.DictReader(f))
    table = '\n'.join(f"|{r['score']}|{float(r['macro_sampled_AP_pp']):.3f}|{float(r['pooled_sampled_AP_pp']):.3f}|" for r in score_rows)
    draft = f'''# 当前研究分枝结果稿（两页以内篇幅）

## 1．数据结果

**有限正证据。**Phase66在Catalogue4172871原65a_confirmation的64个固定相关组代表上一次独立评价，15组有参考云影。L2为MsRE＋Source分数，L3另加B08/B11/B12的冻结逻辑读出；原153组fit拟合、32组development选定，Source固定seed64、MsRE固定2000步。预定L3−L2平均AP增量10.8610个百分点，原配对组95%区间[5.1980,17.6094]，改善／稳定／下降13/0/2。L3平均FPR0.4813%，单侧95%上界0.8200%，满足本批组平均1%约束；8/64组超1%，最大8.6731%，4个有影组召回为0。只确认固定候选在该池的条件增量，不改原H1支持不足结论。[1]

Phase105固定原153组拟合G6六波段树及G7加Source分数树，模型不重拟合；PixBox16产品／10组原选择性点样本、379正／8188负、8个有效AP组。预定G7−G6平均+8.745个百分点，描述95%区间[2.213,16.319]，5/2/1，原支持和区间规则通过。它是Source条件增量的外部采样点复现，不是对所有预训练位置已排除交集的独立保证。[2]

Phase106复用同池原Source概率及Phase67原153组拟合的Source_spectral四输入逻辑读出（Source＋B08/B11/B12），平均AP差+8.307个百分点，描述区间[0.636,17.016]，6/0/2；这是已暴露池的事后反向增量，不能与前两项合成联合确认。[3]

**限制性对照。**Phase107用相同四输入容量、组权、正则和非负Source系数约束比较Source_spectral(S)与Source_RGB(R)。PixBox S−R为+2.272个百分点，区间[-3.143,8.808]，4/2/2；光谱特异性不确定，区间跨零不表示等效。[4]

**强基线。**Phase112在同一PixBox支持复用G7_153、Source和Source_spectral，与作者固定代码f68cf49b／V1双权重、默认B8A的Omni兼容探索比较；原无影两组AP为NA。

|固定评分|8组等权AP(%)|池化AP(%)|
|---|---:|---:|
{table}

预定G7−Omni平均−36.123个百分点，原描述95%区间[-56.140,−17.892]，8组全部下降，最差48XWG−85.647；G7增量规则失败。G7原阈值0.206662789与Omni Shadow confidence≥0.5下，宏召回51.984%／71.969%，精确率21.610%／59.753%，所选负类宏误报比例10.148%／1.125%；池化召回63.061%／80.211%、精确率21.113%／76.574%、负类误报10.906%／1.136%。工作点预算不同，Omni≥0.5不是作者argmax；不能写成同误报率优势。严格V1训练地点交集仍未知，这不是严格论文复现或新独立确认。[5]

**补充路线与判读。**Phase113固定Omni五箱、同组同箱点对，6组平均条件AUC谱76.273%／Omni87.405%，差−11.132个百分点，原区间[-23.556,−1.276]，未满足补充判据。Omni漏75参考影点，谱检出31但新增241负类误报，且漏113个Omni已检出的影点；未构造融合或搜索阈值。[6] Phase114／115固定七候选来自五组；Phase115最终意见全部自报落地云影／20m同质，字段完整两例仅一组，五条不确定性缺失，原2 reviewed／5 draft保留。揭示后事后复核，合格独立判读0例／0组，跨三组门未达；不能写为七个真实漏检已确认或已否定。[7]

## 2．结论探讨

现有证据保留部分固定读出的条件增量，但加入公平强对照后，G7竞争力主张不成立，既有Source_spectral补充Omni路线及残余语义修复线索收口。相关组平均、池化排序和工作点代价必须并报，不能选择较好的指标替代原失败主要比较。

这些结果不支持新适配机制、NIR/SWIR独有优势、优于成熟云影方法或漏检修复机制；也不推断所有基础模型、光谱或未来融合无效。选择性PixBox点样本不能推断全景FPR、尾部安全或密集mIoU。若以后研究方法型TGRS问题，尚缺有实际文献差异的可证伪问题、公平强对照和适当新验证证据；本稿不承诺投稿可行或录用，不提出新模型／追加判读任务，不启动下一实验。引用路径和原字节SHA见SOURCE_SHA256.json；所有区间原样摘录，未重算。
'''
    boundaries = '''# 贡献边界

|证据层次|可以保留的表述|边界|
|---|---|---|
|现有结果支持|Phase66冻结L3相对L2在一次确认池的AP增量及组平均FPR约束；Phase105 Source在六波段树上的条件增量；Phase106固定谱读出的事后正增量|三项不是同一联合终点；模型、池和独立性不同；原H1失败保持|
|现有结果支持|加入Omni后G7主要比较失败；平均／逐组／池化／工作点指标提供不同信息|Omni默认B8A旧池兼容探索，V1训练交集未知，非严格论文复现|
|不能支持|新适配机制、光谱独有优势、优于成熟方法、可靠全景误报预算、已证实漏影修复机制|107特异性不确定、112/113失败、115独立裁决未成立；点采样不代表密集全景|
|尚缺证据|方法型研究的文献差异、可证伪问题、公平强对照、适当新验证和可靠语义支持|本轮只收口，不用更多浅层组合代替缺口，不承诺TGRS可投／录用|

不改原标签、最终意见、缺项、阈值、抽样或历史失败；七候选既未独立确认，也未被独立否定。无独立支持时关闭残余线索，不自动立项。
'''
    branches = '''# 当前研究分枝状态（一页）

|分枝／比较|最终状态|保存的事实与处理|
|---|---|---|
|原Phase65 H1|支持不足，原失败保留|原确认15/16，Phase66不改写|
|Phase66 L3−L2|预定AP及组平均预算支持|64组／15 AP组，小样本及单组损伤保留|
|Phase105 G7−G6|预定有限条件增量通过|选择性PixBox样本，不外推全景|
|Phase106 谱−Source|事后正增量|不是新确认，不联合其他结果|
|Phase107 同容量S−R|特异性不确定|区间跨零；不称等效、不继续搜索|
|Phase112 G7−Omni|主要比较失败，G7路线关闭|8/8 AP下降；旧池探索和交集未知|
|Phase113 既有谱补充Omni|固定判据失败，补充分枝关闭|局部检出伴额外误报与损失；不自动融合|
|Phase114／115 七候选|事后复核完成，语义未独立裁决；残余修复线索关闭|0合格独立组；最终意见／2 reviewed与5 draft／缺项原样保留|
|Phase116|结果整理完成，当前研究分枝收口|零新实验；不自动启动下一研究；2026-10-20北京时间预算线保持|
'''
    DEST.mkdir(parents=True)
    LOCAL.mkdir(parents=True)
    texts = {'RESULTS_DRAFT.md': draft, 'CLAIM_BOUNDARIES.md': boundaries, 'BRANCH_STATUS.md': branches,
             'README.md': '# Phase116已完成\n\n本轮仅整理既有最终验收结果。结果稿篇幅控制为两页以内的短稿，数据结果与结论分开；贡献边界及一页分枝状态另列。没有新实验、指标／区间重算、判读或补样。Phase115当前状态取FINAL文件，早期awaiting为历史快照。源SHA、CPU摘录核验和交付SHA随附；受控原图／逐点标签／私有意见不进入git，work_dirs未同步。\n\nRESULTS_DRAFT引用：[1]Phase66报告及验收；[2]105；[3]106；[4]107；[5]112及110严格复现限制；[6]113；[7]114与115最终文件。准确路径见SOURCE_SHA256.json。\n'}
    for n, text in texts.items():
        (DEST / n).write_text(text, encoding='utf-8', newline='\n')
    documents = {
        'SOURCE_SHA256.json': hashes,
        'CPU_EXCERPT_ACCEPTANCE.json': {
            'status': 'passed', 'mode': 'accepted aggregate lookup and formatting only',
            'historical_source_hashes_unchanged': True,
            'phase66_original_delivery_sha_matches': True,
            'primary_comparisons_exact_rounded': True,
            'all_four_AP_rows_read_directly_from_accepted_csv': True,
            'model_and_support_scope_checked_against_final_READMEs_and_locks': True,
            'phase115_final_complete_cases_groups': [2, 1], 'independent_review_cases_groups': [0, 0],
            'bootstrap_AP_or_AUC_recomputed': False, 'new_pixel_or_private_opinion_read': False,
            'draft_format': 'short Markdown manuscript, no paginated PDF generated',
        },
        'EXECUTION_STATUS.json': {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'completed_route_closed',
                                  'new_experiments_fit_forward_download_review_sampling_thresholds': 0,
                                  'interval_recomputations': 0, 'work_dirs_synced': False,
                                  'next_experiment_authorized': False, 'budget_deadline_beijing': '2026-10-20'},
    }
    for n, obj in documents.items():
        (DEST / n).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    assert all(sha(ROOT / p) == h for p, h in hashes.items())
    delivery = {p.relative_to(ROOT).as_posix(): sha(p) for p in DEST.iterdir() if p.is_file()}
    delivery['src/tools/closeout_phase116.py'] = sha(Path(__file__))
    (DEST / 'DELIVERY_SHA256.json').write_text(json.dumps(delivery, indent=2) + '\n', encoding='utf-8', newline='\n')
    for p in DEST.iterdir():
        (LOCAL / p.name).write_bytes(p.read_bytes())
    print('Phase116 closeout completed; source hashes verified; no new statistics or experiments')


if __name__ == '__main__':
    main()
