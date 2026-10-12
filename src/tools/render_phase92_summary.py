"""Local Chinese delivery summary from sealed Phase92 numbers; no model/data access."""
import argparse
import hashlib
import json
from pathlib import Path
from verify_phase92_delivery import verify


def main(root):
    acceptance=verify(root)
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    report=read('phase92_report.json');lock=read('two_model_lock.json')
    labels=dict(development='Catalogue development（32组／7组有云影）',original_main='ALCD原版九组',
                original_common='ALCD原版六组共同支持',revised_common='ALCD修订版六组共同支持')
    names=dict(S3='S3：三波段',R6='R6：六波段',Source_spectral='Source＋光谱',L3='L3')
    f=lambda x:'NA' if x is None else f'{100*x:.4f}'
    lines=['# Phase92：原始光谱基线结果与讨论','','## 1．数据结果','',
        '只新增两个逻辑回归：S3使用B08/B11/B12；R6使用B04/B03/B02/B08/B11/B12。均复用原153个fit代表、1,253,376个抽样像元、原索引与组等权；不平衡类别权重，不加入局部特征。全部fit非云影像元冻结1%全局误报阈值后，才读取评价结果。',
        'S3、R6分别18、25次迭代收敛。优化耗时3.38、4.51秒；完整fit阈值计算49.58、53.51秒；输入核验与拟合阶段178.09秒。development评分及复现31.58秒，ALCD评分及复现258.88秒；含统计／oracle／独立核验的评价阶段859.08秒。这些是本次CPU壁钟时间，含相应I/O和核验，并非纯单像元推理延迟。',
        '使用固定Source＋光谱、原L3作为对照。没有分割网络训练、网络前向或下载。S3/R6只输出Shadow概率，不输出Surface／Cloud分类，故不报告其三类mIoU。',
        '', '### 冻结工作点','',
        'AP／召回仅在存在有效参考云影的相关组上等权平均；FPR包括全部有有效非云影像元的组。表中均为百分数，FPR上界为单侧95%组bootstrap上界。精确率未定义的组不补0，参与组数见JSON中各宏指标的n_groups。',
        '', '| 评价池 | 策略 | 宏AP | 池化AP | 宏召回 | 宏精确率 | 宏FPR | FPR上界 | 宏Shadow IoU | 零召回组 | FPR>1%组 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d,v in report['domains'].items():
        for n,q in v['policies'].items():
            m=q['macro']
            cells=[labels[d],names[n],f(m['ap']['mean']),f(q['pooled']['ap']),f(m['recall']['mean']),f(m['precision']['mean']),
                   f(m['fpr']['mean']),f(m['fpr']['upper95_one_sided']),f(m['shadow_iou']['mean']),str(len(q['zero_recall_groups'])),str(q['groups_above_1percent'])]
            lines.append('| '+' | '.join(cells)+' |')
    lines+=['','### 固定比较','',
        '主要比较始终为Source＋光谱−S3。下表单位为AP百分点；相关组配对bootstrap10000次、seed92012，区间仅描述，不是新的独立确认。改善／稳定／下降沿用±1 AP百分点。',
        '', '| 评价池 | 比较 | 平均ΔAP | 95%区间 | 改善／稳定／下降 | 最差逐组ΔAP |',
        '|---|---|---:|---|---|---|']
    for d,v in report['domains'].items():
        for comparison,q in v['comparisons'].items():
            a,b=comparison.split('_minus_');ci=q['delta_ap_pp']['ci95'];w=q['worst_ap'];tile=w['tile'];tile=tile if tile.startswith('T') else 'T'+tile
            lines.append(f"| {labels[d]} | {names[a]}−{names[b]} | {q['delta_ap_pp']['mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {q['improved']}/{q['stable']}/{q['declined']} | {tile} {w['delta_ap_pp']:+.4f} |")
    lines+=['','### 1%宏FPR统一阈值oracle','',
        '以下阈值利用当前评价参考事后扫描，只作排序／低误报表现诊断，不能用于部署或修改冻结阈值；每组FPR并不都≤1%。',
        '', '| 评价池 | S3宏召回% | R6宏召回% | Source＋光谱宏召回% | L3宏召回% |',
        '|---|---:|---:|---:|---:|']
    for d,v in report['domains'].items():
        lines.append('| '+' | '.join([labels[d],*[f(v['oracle_1percent'][n]['macro']['recall']['mean']) for n in ['S3','R6','Source_spectral','L3']]])+' |')
    lines+=['','### 全部逐组差值','',
        '下表同时保留全部32个development组，包括25个无参考云影组（AP与召回记NA），以及三个ALCD分析版本。全部四项固定比较的逐组原值见paired_changes.csv；分层FPR、池化指标、零召回产品及Shadow IoU见per_group.csv和four_strategy_totals.csv。',
        '', '| 评价池 | Tile | Source＋光谱−S3 ΔAP pp | Δ召回 pp | ΔFPR pp |',
        '|---|---|---:|---:|---:|']
    fp=lambda x:'NA' if x is None else f'{x:+.4f}'
    for d,v in report['domains'].items():
        for row in v['comparisons']['Source_spectral_minus_S3']['pairs']:
            tile=row['tile'];tile=tile if tile.startswith('T') else 'T'+tile
            lines.append('| '+' | '.join([labels[d],tile,fp(row['delta_ap_pp']),fp(row['delta_recall_pp']),fp(row['delta_fpr_pp'])])+' |')
    lines+=['','## 2．结论探讨','',
        '本轮ALCD结果支持保留Source网络分数的实用价值：原版九组相对S3宏AP增加24.23个百分点，九组全部超过+1个百分点；六组原／修订共同支持的增量方向也一致。相对R6的原版九组增量为25.20个百分点，加入原始RGB波段没有消除这次差距。这里比较的是冻结Source分数与两种固定线性光谱方案，不是证明任何任务都必须使用网络，也不是对所有光谱模型的结论；R6与Source＋光谱维数不同。',
        'development证据较弱且异质：Source＋光谱−S3平均+6.06个百分点，95%区间[-9.91,19.47]跨零，T32NKF反而下降33.83个百分点。因此不能把ALCD的正向结果外推成逐景保证；区间跨零也不表示等效。原始光谱在七个有影development组的宏AP较高，却有很低池化AP和冻结召回，说明单组排序分数不能代替跨组工作点评价。',
        'S3/R6在ALCD低误报区域表现尤其弱：冻结召回仅0.21%／3.93%，宏FPR仍为1.67%／1.71%；连利用评价标签扫描1%宏FPR的全局oracle召回也仅0.0132%／1.1447%。这说明仅为这两个固定线性方案调整一个全局阈值，无法得到Source＋光谱在当前参考池的低误报召回；并未检验逐景阈值或更复杂光谱模型。',
        '本轮没有为MsRE适配的必要性增加证据：ALCD原版九组Source＋光谱与L3宏AP仍为63.88%／63.70%，二者冻结召回优势不能脱离FPR比较；Phase91的等预算限制仍适用。现阶段更有根据把“冻结RGB Source＋额外光谱”保留为核心基线，而不是把所有收益归因于MsRE。',
        '误报约束尚不能普遍宣称稳定满足：ALCD原版九组Source＋光谱点估计0.5358%≤1%，但单侧95%上界1.0618%>1%；L3点估计1.4586%已超过预算。两个光谱基线也未满足本批外部点估计预算。共同支持六组的结果单列，不能替代原版九组主分析。',
        '所有评价池均已暴露，只提供探索性参考一致性证据；ALCD标签质量、场景相关性和Phase89故障后54个复用概率缺少故障前输出SHA的限制保留。原人工意见、旧失败结论及1%预算不变；没有选择赢家部署，也不新增机制、特征或模型。',
        '', '## 3．交付与数值口径','',
        '14个原生小文件通过SHA校验；106条旧基线AP／二分类混淆计数精确复现，972块20m→60m概率均值／决策独立核验，456行指标与16个oracle扫描点通过验收。本地、远端均通过。模型参数、标准化、原抽样SHA、完整fit阈值计数已在两模型锁及fit_input_lock.json中保存。',
        'oracle_threshold_receipts.json复用Phase91帮助函数，内嵌通用score_precision文字沿用“60m概率”。本轮development扫描实际使用float32 logit（扫描内部转float64保留原ties）；ALCD才是float64的60m平均概率。以ORACLE_SCORE_UNITS.json明确的域口径为准。阈值原值及扫描结果未修改，原生密封文件保留。']
    output={'RESULTS_AND_INTERPRETATION.md':'\n'.join(lines)+'\n',
        'LOCAL_ACCEPTANCE.json':json.dumps(acceptance,indent=2)+'\n',
        'ORACLE_SCORE_UNITS.json':json.dumps(dict(
            native_reports_preserved=True,no_threshold_or_numeric_changes=True,
            development='20m float32 logit, converted to float64 by scan without altering values or ties',
            original_main='60m float64 sigmoid20 probability mean9',
            original_common='60m float64 sigmoid20 probability mean9',
            revised_common='60m float64 sigmoid20 probability mean9',
            reason='Inherited Phase91 scan receipt generic score_precision text; explicit domain clarification'),indent=2)+'\n'}
    for name,content in output.items():
        with (root/name).open('x',encoding='utf-8',newline='\n') as f:f.write(content)
    supplement={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in output}
    with (root/'LOCAL_SUPPLEMENT_SHA256.json').open('x',encoding='utf-8') as f:json.dump(supplement,f,indent=2)
    print(json.dumps(acceptance,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
