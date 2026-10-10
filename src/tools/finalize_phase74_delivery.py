"""Render a concise result/interpretation split after independent local verification."""
import argparse,json
from pathlib import Path
from verify_phase74 import verify


def main(root):
    verification=verify(root)
    r=json.loads((root/'phase74_report.json').read_text(encoding='utf-8'));p=r['policies']
    text=['# Phase74：预测云邻居排除检验','','## 1 数据结果','',
        '仅原153个fit相关组拟合、32个development相关组评价，其中7组具有有效参考云影。没有新网络前向、下载或确认池评价。原L3/B的32组AP与混淆矩阵精确复现。',
        '', '固定61×61窗口、B08/B11/B12原生TOA；排除中心/无效观测，剩余邻居不足16时回退原B均值。E由冻结Source三类argmax的Cloud预测生成；R保留全景排除数量，但不匹配每个窗口的数量和空间结构。',
        '', '两个拟合版仅用原fit样本、组等权BCE、固定正则与优化器；全部fit负类按原规则冻结宏FPR≤1%阈值。固定参数版沿用原B全部参数及阈值。六行均使用原MsRE的非云影类别胜者。',
        '', '| 模型 | 宏AP % | 池化AP % | 宏召回 % | 全组宏FPR % | FPR单侧95%上界 % | 池化mIoU % | 零召回组 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['L3','B','B_E_fixed','B_R_fixed','B_E_fit','B_R_fit']:
        q=p[name];text.append(f"| {name} | {q['macro_ap']*100:.3f} | {q['ranking']['ap']*100:.3f} | {q['macro_recall']*100:.3f} | {q['fpr_strata']['all']['mean']*100:.3f} | {q['fpr_strata']['all']['upper95']*100:.3f} | {q['miou_percent']:.3f} | {q['zero_recall_groups']} |")
    text+=['','主要探索比较及相对B/L3的配对差，单位为百分点：','',
        '| 比较 | 宏AP差 [95%区间] | 宏召回差 [95%区间] | 全组宏FPR差 [95%区间] | AP改善/稳定/下降 |',
        '|---|---:|---:|---:|---:|']
    def fmt(a):return f"{a['mean']*100:+.3f} [{a['ci95'][0]*100:+.3f}, {a['ci95'][1]*100:+.3f}]"
    for name,c in r['comparisons'].items():text.append(f"| {name} | {fmt(c['ap'])} | {fmt(c['recall'])} | {fmt(c['fpr'])} | {c['improved']}/{c['stable']}/{c['declined']} |")
    text+=['','AP/召回只对7个有云影组计算；无云影组AP留空，继续参与FPR。改善/稳定/下降沿用±1 AP百分点。区间为10000次配对相关组bootstrap（seed74010），仅描述已有development探索，不是独立确认。FPR上界约束全组平均，不表示每组均≤1%。',
        '', '| 模型 | 有影组宏FPR % | 无影组宏FPR % | FPR>1%组数 | 新增FP / 消除FP / 救回TP / 丢失TP（相对B） |',
        '|---|---:|---:|---:|---|']
    for name,q in p.items():
        f=q['flows_vs_B'];text.append(f"| {name} | {q['fpr_strata']['shadow']['mean']*100:.3f} | {q['fpr_strata']['no_shadow']['mean']*100:.3f} | {q['groups_fpr_above_1pct']} | {f['new_fp']} / {f['removed_fp']} / {f['rescued_tp']} / {f['lost_tp']} |")
    text+=['','按原B相对L3定义的固定像元层，报告分数变化，不据此删组或改标签：','',
        '| 队列 | 排除 | 原B−L3层 | 支持组数 | 像元数 | 组等权平均logit变化 |','|---|---|---|---:|---:|---:|']
    for a in r['stratum_summary']:
        value='NA' if a['equal_group_mean_logit_change'] is None else f"{a['equal_group_mean_logit_change']:+.5f}"
        text.append(f"| {a['cohort']} | {a['variant']} | {a['stratum']} | {a['support_groups']} | {a['pixels']} | {value} |")
    audits=json.loads((root/'neighbor_audit.json').read_text(encoding='utf-8'))
    text+=['','| 队列 | 排除 | 组平均回退率 % | 排除的参考Surface / Cloud / Shadow / 无效像元 |',
        '|---|---|---:|---|']
    for cohort in ['fit','development']:
        for variant in ['E','R']:
            a=[x for x in audits if x['cohort']==cohort and x['variant']==variant]
            counts=[sum(x['excluded_gt_crosscount_offline_only'][str(c)] for x in a) for c in [0,1,2,255]]
            fallback=sum(x['fallback_eval_fraction'] for x in a)/len(a)*100
            text.append(f"| {cohort} | {variant} | {fallback:.4f} | {' / '.join(map(str,counts))} |")
    text+=['','主要E拟合版−R拟合版比较的逐组AP损伤（最差三组，非额外抽样）：','']
    main=r['comparisons']['B_E_fit_minus_B_R_fit']
    for a in sorted([x for x in main['pairs'] if x['delta_ap'] is not None],key=lambda x:x['delta_ap'])[:3]:
        text.append(f"- {a['product']}：AP差{a['delta_ap']*100:+.3f}个百分点，召回差{a['delta_recall']*100:+.3f}个百分点。")
    text+=['','逐组剩余邻居/排除比例/回退比例及E/R与参考类别交叉计数见 `neighbor_audit.json`；局部均值、特征、logit变化与四种转移见 `fixed_interventions.json`。',
        '', '## 2 结论探讨','', '执行前锁定的继续研究条件：','']
    descriptions={'E_FPR_below_B':'E拟合版全组宏FPR低于B','E_recall_loss_le_1pp':'E拟合版相对B宏召回损失不超过1个百分点',
        'E_AP_ge_B':'E拟合版宏AP不低于B','E_AP_gt_R':'E拟合版宏AP高于R拟合版','E_FPR_le_R':'E拟合版全组宏FPR不高于R拟合版'}
    for key,value in r['continue_conditions'].items():text.append(f"- {descriptions[key]}：{'满足' if value else '未满足'}。")
    if r['continue_branch']:
        text+=['','五项条件全部满足，仅支持继续研究该排除方案；不证明云污染是成因，不代表独立确认通过。']
    else:
        text+=['','预设联合条件未全部满足，停止本轮预测云邻居排除分枝；不从固定参数版中挑一个好结果替代失败拟合版，不继续搜索窗口、排除阈值或回退数量。此结论不表示全部空间信息无用。']
    text+=['','E是Source预测，可能误识别真实云影；R仅匹配全景数量，空间分布和局部邻居数不同，因此E−R差不能单独证明光谱云污染因果。只有7个有云影development组，区间及逐组损伤须完整保留。旧H1支持门、Phase72失败结论与地形分枝not_ready均保持。',
        '', '交付：`summary.csv` 六行汇总；`development_groups.csv` 192行原始指标/错误流；`comparisons_groups.csv` 全部逐组差；`phase74_report.json` 完整池化/逐组指标与区间；`policy_lock.json` 模型/标准化/阈值；`fit_audit.json` 抽样哈希；`input_lock.json` 输入与代码锁；`console.log`、`error.log`、`EXIT_CODE` 实际日志。只从shared数据目录获取，没有同步work_dirs。','']
    (root/'README.md').write_text('\n'.join(text),encoding='utf-8')
    (root/'LOCAL_VERIFIED.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');a=parser.parse_args();main(Path(a.root))
