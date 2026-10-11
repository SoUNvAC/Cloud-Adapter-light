"""Render already-verified Phase89 reports; no raster/model access."""
import argparse,json
from pathlib import Path

def main(root):
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    r=read('phase89_report.json');v=read('CPU_VERIFIED.json');assert v['statistic_rows']>0
    def f(x,scale=1):return 'NA' if x is None else f'{x*scale:.4f}'
    def ci(x):return 'NA' if x['ci95'] is None else '['+', '.join(f(z) for z in x['ci95'])+']'
    lines=['# Phase89：九组ALCD外部参考一致性评价','',
           '## 1．数据结果','',
           '固定Source、原MsRE 2000步、原恢复L2、L3。零训练、零拟合、零调阈值、零新下载。9个固定相关组代表；原版为主分析，6组精确匹配修订版，另外3组不补样。',
           '原10m四波段面积平均到20m；B11/B12保持20m。每景36个不重叠块，每块1020×1020个20m格（20.4km），按原流程缩放512输入并后处理恢复；边缘仅右/下补零，padding排除。原三类概率及校准概率在原60m格聚合。', '',
           '| 原版主分析 | 宏AP % | 池化AP % | 宏召回 % | 宏FPR % | FPR单侧95%上界 % | 宏mIoU % | 池化mIoU % |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    main=r['versions']['original_main']
    for n,p in main['policies'].items():
        vals=[f(p['macro_ap']['mean'],100),f(p['pooled']['ap'],100),f(p['macro_recall']['mean'],100),f(p['macro_fpr']['mean'],100),f(p['macro_fpr']['upper95_one_sided'],100),f(p['macro_miou_percent']['mean']),f(p['pooled']['miou_percent'])]
        lines.append('|'+n+'|'+'|'.join(vals)+'|')
    lines+=['','主要比较固定L3−L2，逐组等权，10000次配对相关组bootstrap，seed89011；以下区间均为描述区间。±1 AP百分点用于改善／稳定／下降。','',
            '| 参考与支持 | AP组数 | 平均ΔAP pp | 95%描述区间 pp | 改善/稳定/下降 | 最差ΔAP pp | 最大召回下降 pp |',
            '|---|---:|---:|---|---|---:|---:|']
    for ver,lab in [('original_main','原版全部9组'),('original_common','原版共同支持'),('revised_common','修订版同一共同支持')]:
        p=r['versions'][ver]['primary_L3_minus_L2']
        lines.append(f"|{lab}|{p['n_groups']}|{f(p['mean'])}|{ci(p)}|{p['improved']}/{p['stable']}/{p['declined']}|{f(p['worst_delta_ap_pp'])}|{f(p['worst_delta_recall_pp'])}|")
    lines+=['','| 代表tile | 原版全部支持ΔAP pp | 原版共同支持ΔAP pp | 修订版共同支持ΔAP pp |','|---|---:|---:|---:|']
    pairs={ver:{p['product']:p for p in x['pairs']} for ver,x in r['versions'].items()}
    for s in r['support']:
        values=[f(pairs[ver].get(s['product'],{}).get('delta_ap_pp')) for ver in pairs]
        lines.append('|'+s['tile']+'|'+'|'.join(values)+'|')
    lines+=['','完整四策略×三支持表、逐组混淆矩阵、三类IoU、Shadow→Surface／Cloud、分层FPR、零召回及实际支持在CSV和phase89_report.json中保留。mIoU只平均union>0类别，每组同时记录有效类别数。无参考Shadow的AP为NA，不填0。', '',
            '| 修订版支持 | 原版影像+标签有效格 | 共同有效格 | 混合/无效排除格 | 保留比例 % |','|---|---:|---:|---:|---:|']
    for s in r['support']:
        q=s['common_support']
        lines.append(f"|{s['tile']}|{q['image_label_valid'] if q else 'NA'}|{q['common_valid'] if q else 'NA'}|{q['revised_invalid_or_mixed_excluded'] if q else 'NA'}|{f(q['retention'],100) if q else '无匹配版本'}|")
    lines+=['','57点为原人工A判断的自动参考对照，原话及空字段保留；人工B仍未执行。','',
            '| 分层 | 参考 | 一致 | 不一致 | 混合 | 无法判断/参考不可用 |','|---|---|---:|---:|---:|---:|']
    for st,x in r['human_automatic_comparison'].items():
        for ver,c in x.items():lines.append('|'+st+'|'+ver+'|'+'|'.join(str(c[k]) for k in ['一致','不一致','混合','无法判断'])+'|')
    lines+=['','| 推理 | 前向次数 | 用时秒（含产物写入） | 峰值GPU分配bytes |','|---|---:|---:|---:|']
    for n,x in r['runtime'].items():lines.append(f"|{n}|{x['forward_count']}|{f(x['elapsed_seconds'])}|{x['peak_gpu_memory_bytes']}|")
    lines+=['',f"CPU独立复核{v['statistic_rows']}行混淆矩阵与sklearn AP；全部原20m概率块重聚合、校准logit/sigmoid/60m均值复算和无重叠覆盖检查通过。Catalogue32个已暴露development代表的L2/L3像元分数及混淆矩阵事先精确复现。",'',
            '## 2．结论分析','']
    p=main['primary_L3_minus_L2'];a=r['versions']['original_common']['primary_L3_minus_L2'];b=r['versions']['revised_common']['primary_L3_minus_L2']
    if p['ci95'] is None or p['ci95'][0]<=0<=p['ci95'][1]:
        lines+=['原版主要增量区间跨零或支持不足，光谱增益证据不确定；不能只凭点估计宣布通过。']
    elif p['ci95'][0]>0:lines+=['原版主要增量的描述区间在零以上，支持与该公开参考的一致性增益；不能升级为标签验真或正式确认通过。']
    else:lines+=['原版主要增量的描述区间在零以下，当前外部参考一致性未支持L3优于L2。']
    if a['mean'] is not None and b['mean'] is not None:
        if a['mean']>0 and b['mean']>0:lines+=['同一支持下两版平均增量均正，只能描述方向一致；不据此确定哪版是真值。']
        elif a['mean']*b['mean']<0:lines+=['同一支持下两版平均增量方向相反，结论对参考版本敏感，证据不确定。']
        else:lines+=['共同支持的两版结果不能作为稳定正向光谱增益的证据；完整列出损伤，不选择有利版本。']
    q=main['policies']['L3']['macro_fpr']
    if q['mean']>.01:lines+=['L3在本池原版主分析的实际平均误报率超过1%预算；阈值保持冻结，不在外部池重校准。']
    elif q['upper95_one_sided'] is not None and q['upper95_one_sided']<=.01:lines+=['L3在本批九组的平均FPR点估计及单侧95%描述上界满足1%；九组均含参考Shadow，不能据此推及一般场景组合。']
    else:lines+=['L3平均FPR点估计满足1%，但单侧上界未支持预算稳定满足；不改阈值。']
    lines+=['','九组是统计单位，像元和块数不会增加独立样本量。人工与参考不一致不等于标签错误，也不估计全景正确率。此前质量抽查暴露及标签未完全裁决的限制仍在；项目内空间排除不代表已知基础模型全部预训练地理谱系。',
            '不修改参考、成员、mask、候选或阈值，不恢复旧主检验，不宣称TGRS方法成立。','',
            '交付：four_policy_tables.csv、per_group.csv、paired_AP_changes.csv、reference_support.csv、human57_automatic_comparison.csv、phase89_report.json、输入锁、Catalogue复现与CPU_VERIFIED。大概率块保留shared，不同步work_dirs。']
    with (root/'README.md').open('x',encoding='utf-8') as f0:f0.write('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
