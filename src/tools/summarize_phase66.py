"""Render all prespecified metrics without selecting alternative endpoints."""
import argparse,csv,json
from pathlib import Path


def num(x,scale=1):
    return '未定义' if x is None else f'{x*scale:.4f}'


def ci(d,scale=1):
    return '未定义' if d['ci95'] is None else '['+', '.join(num(v,scale) for v in d['ci95'])+']'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.root/'delivery/phase66_report.json').read_text())
    policies=report['policies'];cmp=report['comparisons']['L3_minus_L2'];ap=cmp['ap']
    lines=['# Phase66：冻结光谱候选的独立验证','', '## 1. 数据结果','',
        f"固定64相关组原代表；实际有参考云影{report['actual_shadow_support']}组（公开元数据{report['public_shadow_support']}组）。有效参考像元{sum(r['positive']+r['negative'] for r in policies['L3']['rows'])}个，不能充当独立样本数。五个策略均为既有冻结版本，无拟合或调阈值。",
        '',f"唯一主要比较 L3−L2：平均ΔAP **{num(ap['mean'])} AP百分点**，配对相关组bootstrap95%区间 **{ci(ap)}**，区间宽度{num(ap['ci95'][1]-ap['ci95'][0]) if ap['ci95'] else '未定义'}百分点。改善/稳定/下降（±1AP百分点）：**{ap['improved']}/{ap['stable']}/{ap['declined']}**。bootstrap10000次，seed66010。",'',
        '| 策略 | 逐组AP均值(%) | 池化AP(%) | 逐组召回(%) | 逐组精确率(%) | 逐组FPR(%) [双侧95%CI] | 单侧95%FPR上界(%) | FPR>1%组数 | 池化mIoU(%) |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for name,p in policies.items():
        f=p['fpr_constraint'];m=p['macro'];pool=p['pooled']
        lines.append(f"| {name} | {num(p['macro_ap']['mean'],100)} | {num(pool['ranking']['ap'],100)} | {num(m['recall']['mean'],100)} | {num(m['precision']['mean'],100)} | {num(f['mean'],100)} {ci(f,100)} | {num(f['upper95_one_sided'],100)} | {f['groups_above_budget']}/{f['n_groups']} | {num(pool['miou_percent'])} |")
    lines += ['', '全部召回及误报均为冻结阈值或原模型原argmax的实际操作结果。召回平均仅有Shadow组；精确率平均所有有预测Shadow的组，无预测时未定义，不能填0。', '',
        '| 策略 | 精确率有效组数 | 池化召回/精确率/FPR(%) | 池化Surface/Cloud/Shadow IoU(%) | Shadow→Surface/Cloud像元 | 无Shadow组平均FPR(%) |',
        '|---|---:|---|---|---|---:|']
    for name,p in policies.items():
        q=p['pooled'];lines.append(f"| {name} | {p['macro']['precision']['n_groups']} | {' / '.join(num(q[k],100) for k in ['recall','precision','fpr'])} | {' / '.join(num(v) for v in q['iou_percent'])} | {q['shadow_to_surface']} / {q['shadow_to_cloud']} | {num(p['no_shadow_fpr']['mean'],100)} ({p['no_shadow_groups']}组) |")
    lines += ['', '| 策略 | 池化1%FPR扫描召回(%) | 池化5%FPR扫描召回(%) | 逐组mIoU均值(%) |',
        '|---|---:|---:|---:|']
    for name,p in policies.items():
        c=p['pooled']['ranking']['matched_fpr']
        lines.append(f"| {name} | {num(c.get('0.01',{}).get('recall'),100)} | {num(c.get('0.05',{}).get('recall'),100)} | {num(p['macro_miou_percent']['mean'])} |")
    lines += ['', '扫描召回仅是有真值条件下的oracle排序诊断，扫描阈值不作为部署阈值。', '',
        '| 产品（原相关组代表） | L3−L2 ΔAP(百分点) | L3−L2实际召回差(百分点) |',
        '|---|---:|---:|']
    for r in cmp['pairs']:
        lines.append(f"| {r['product']} | {num(r['delta_ap'],100)} | {num(r['delta_recall'],100)} |")
    worst=min(cmp['pairs'],key=lambda r:r['delta_ap']) if cmp['pairs'] else None
    if worst:lines+=['',f"最差ΔAP：{worst['product']}，{num(worst['delta_ap'],100)}百分点。"]
    worst_r=cmp['max_recall_decline']
    if worst_r:lines+=[f"最小召回差：{worst_r['product']}，{num(worst_r['delta_recall'],100)}百分点（下降幅度{num(max(0,-worst_r['delta_recall']),100)}百分点）。"]
    maximum_fpr=max(policies['L3']['rows'],key=lambda r:r['fpr'])
    lines += ['', f"L3最大逐组FPR：{maximum_fpr['product']}，{num(maximum_fpr['fpr'],100)}%；15个有Shadow组中{sum(r['positive']>0 and r['recall']==0 for r in policies['L3']['rows'])}组在冻结阈值下召回为0。"]
    for name in ['Source','MsRE']:
        pair=report['comparisons']['L3_minus_'+name]
        wa=min(pair['pairs'],key=lambda r:r['delta_ap'])
        wr=pair['max_recall_decline']
        lines += [f"L3相对{name}：最差AP差{num(wa['delta_ap'],100)}百分点（{wa['product']}）；最大召回下降{num(max(0,-wr['delta_recall']),100)}百分点，最小原始差{num(wr['delta_recall'],100)}（{wr['product']}）。"]
    auxiliary=report['comparisons']['L2_minus_L1']['ap']
    lines+=['',f"辅助L2−L1平均ΔAP {num(auxiliary['mean'])}百分点，描述95%区间{ci(auxiliary)}；不得取代主要比较。",
        '', '逐组全部策略的AP/FPR/召回/精确率/三类IoU/mIoU/漏影流向及扫描召回详见delivery/groups.csv；完整分母、置信区间、无Shadow逐组值、Source和原MsRE配对损伤在delivery/phase66_report.json。', '', '## 2. 结论探讨','']
    if report['primary_decision']=='positive_increment_independently_supported':
        lines+=['按启封前固定规则，本批独立数据支持L3相对L2的平均光谱AP增量。']
    elif report['primary_decision']=='insufficient_support':
        lines+=['有效AP支持不足，不能作独立增量确认。']
    else:lines+=['主要AP增量未获独立确认；不以任何变好的辅助指标替代该结论。']
    f=policies['L3']['fpr_constraint']
    if f['upper_bound_meets_budget']:
        lines+=['L3相关组平均误报率的单侧95%上界也满足1%预算；这不是每景误报保证。']
    elif f['point_estimate_meets_budget']:
        lines+=['L3平均误报点估计满足1%预算，但单侧上界超过预算，稳定满足仍不确定。']
    else:lines+=['L3平均误报点估计未满足1%预算，不能宣称可稳定部署；冻结阈值不因此改变。']
    lines+=['AP增量得到支持不等于全部云影恢复：逐组实际召回、零召回组、严重单组损伤和最大FPR均需同时保留。无Shadow组的误报变化也不能被全体组的平均掩盖。']
    lines+=['仅有约15个有效云影相关组，bootstrap区间条件于固定候选，不包含fit/先前模型选择不确定性，也不证明参考标签无误、跨区域泛化或每景均改善。池化排序与逐组收益必须同时解释。',
        '保留原H1支持门失败，不自动授权65c。不围绕此批确认集继续添加特征、调正则或改阈值；任何后续机制须另立协议和独立数据，L3保留为必须超过的核心基线。','']
    target=a.root/'RESULTS.md'
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
    with (a.root/'primary_group_deltas.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['group_id','product','delta_ap_pp','delta_recall_pp']);w.writeheader()
        for r in cmp['pairs']:w.writerow(dict(group_id=r['group_id'],product=r['product'],delta_ap_pp=100*r['delta_ap'],delta_recall_pp=100*r['delta_recall']))
    print(target)


if __name__=='__main__':main()
