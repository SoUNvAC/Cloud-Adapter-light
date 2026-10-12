"""Chinese data/interpretation-separated summary from verified small reports."""
import argparse
import hashlib
import json
from pathlib import Path
from verify_phase93_delivery import verify


def main(root):
    acceptance=verify(root)
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8-sig'))
    report=read('phase93_report.json');lock=read('two_model_lock.json')
    index_names=['INPUT_INDEX_BEFORE.json','INPUT_INDEX_AFTER.json']
    before,after=[read(n) for n in index_names]
    assert before['all_original_indices_unchanged'] and after['all_original_indices_unchanged']
    assert before['index_sha256']==after['index_sha256']
    labels=dict(development='Catalogue development：32组／7组有影',original_main='ALCD原版9组',
                original_common='ALCD原版6组共同支持',revised_common='ALCD修订版6组共同支持')
    names=['S3','R6','Source_spectral','L3','G6','G7']
    f=lambda x:'NA' if x is None else f'{100*x:.4f}'
    pp=lambda x:'NA' if x is None else f'{x:+.4f}'
    lines=['# Phase93：固定非线性光谱基线','','## 1．数据结果','',
        '只新增两个scikit-learn1.5.2直方图梯度提升树：G6输入B04/B03/B02/B08/B11/B12，G7在同六波段后增加原Source云影logit。均使用原153个fit代表、1,253,376抽样像元、原索引／标签／有效掩膜；不标准化、不加类平衡、不加局部特征。组等权样本权重和为N。固定200轮，4 CPU线程上限，无早停或模型选择。',
        '先完整复现旧四策略的Phase92统计，再拟合两个新模型；序列化重载在全部拟合样本上逐值一致。每模型使用全部153组fit非影像元锁定1%宏FPR的概率域阈值，然后才评价新模型。fit阈值为训练内校准，不是独立误报保证。',
        'G6/G7只输出二分类云影概率，不伪造三类mIoU。ALCD按20m逐像元预测float32概率，再以float64作3×3均值到原生60m参考格网。六策略均报告；旧S3/R6/Source_spectral/L3点估计不变，描述性区间重新使用seed93012。development旧四策略阈值为logit，新G6/G7为概率；ALCD全部为概率，不能对G6/G7再sigmoid。口径见SCORE_UNITS.json。',
        '', '### 冻结阈值工作点','',
        '下表百分数；AP／召回只在有参考云影的组等权平均，FPR包括全部有有效非影像元的组。精确率未定义组不补零。FPR上界为单侧95%相关组bootstrap上界。',
        '', '| 评价池 | 策略 | 宏AP | 池化AP | 宏召回 | 宏精确率 | 宏FPR | FPR上界 | 宏Shadow IoU | 零召回组 | 超1%组 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d,v in report['domains'].items():
        for n in names:
            s=v['policies'][n];m=s['macro']
            lines.append('| '+' | '.join([labels[d],n,f(m['ap']['mean']),f(s['pooled']['ap']),
                *[f(m[k]['mean']) for k in ['recall','precision','fpr']],f(m['fpr']['upper95_one_sided']),
                f(m['shadow_iou']['mean']),str(len(s['zero_recall_groups'])),str(s['groups_above_1percent'])])+' |')
    lines+=['','### 三项固定比较','',
        '主要G7−G6，次要Source_spectral−G6、G6−R6。AP单位百分点，配对相关组bootstrap10000次、seed93012；区间仅描述。±1 AP百分点划分改善／稳定／下降。',
        '', '| 评价池 | 比较 | 平均ΔAP | 95%区间 | 改善／稳定／下降 | 最差组及ΔAP | 最小逐组召回差 |',
        '|---|---|---:|---|---|---|---|']
    for d,v in report['domains'].items():
        for n,c in v['comparisons'].items():
            t=c['delta_ap_pp'];w=c['worst_ap'];r=c['worst_recall'];ci=t['ci95']
            lines.append('| '+' | '.join([labels[d],n,pp(t['mean']),f'[{ci[0]:+.4f}, {ci[1]:+.4f}]',
                f"{c['improved']}/{c['stable']}/{c['declined']}",f"{w['tile']} {pp(w['delta_ap_pp'])}",
                f"{r['tile']} {pp(r['delta_recall_pp'])}"])+' |')
    lines+=['','### 参考扫描oracle，不能部署','',
        '同一评价池内使用参考标签扫描统一阈值满足1%相关组平均FPR。该结果与冻结阈值工作点分开，不能据此改模型阈值；不表示每景FPR≤1%。',
        '', '| 评价池 | S3召回% | R6召回% | Source_spectral召回% | L3召回% | G6召回% | G7召回% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d,v in report['domains'].items():lines.append('| '+' | '.join([labels[d],*[f(v['oracle_1percent'][n]['macro']['recall']['mean']) for n in names]])+' |')
    lines+=['','### 全部逐组主要增量','',
        '无参考云影组AP／召回为NA，但仍参与FPR。全部三项对照、产品ID及分层误报原值见paired_changes.csv、per_group.csv。',
        '', '| 评价池 | Tile | G7−G6 ΔAP pp | Δ召回 pp | ΔFPR pp |',
        '|---|---|---:|---:|---:|']
    for d,v in report['domains'].items():
        for r in v['comparisons']['G7_minus_G6']['pairs']:
            lines.append('| '+' | '.join([labels[d],r['tile'],*[pp(r['delta_'+k+'_pp']) for k in ['ap','recall','fpr']]])+' |')
    lines+=['','### 成本与阈值','', '| 模型 | 轮数 | 拟合秒 | 完整fit阈值秒 | 概率阈值 | fit宏FPR% | 累计进程峰值GiB |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for n,m in lock['models'].items():
        c=m['cost'];lines.append(f"| {n} | {m['rounds']} | {c['fit_seconds']:.2f} | {c['full_fit_threshold_seconds']:.2f} | {m['threshold']:.9g} | {100*m['fit_macro_fpr']:.7f} | {c['process_peak_rss_bytes']/2**30:.3f} |")
    lines.append(f"\n评价阶段总壁钟{report['costs']['evaluation_seconds']:.2f}秒；模型predict_proba计时：{report['costs']['predict_seconds']}秒。独立探针／AP／bootstrap／I/O等另计，非纯部署延迟。峰值是进程累计高水位，不能解释为各模型独占增量内存。preflight、fit和evaluate由三个独立Python进程执行；原生cost.peak_scope中的including preflight为描述笔误，fit峰值实际覆盖输入核验、两模型拟合／阈值阶段，不含前一进程；见COST_SCOPE_CLARIFICATION.json，原始数值及密封文件未修改。")
    lines+=['','## 2．结论探讨','']
    for d,v in report['domains'].items():
        c=v['comparisons']['G6_minus_R6'];t=c['delta_ap_pp'];ci=t['ci95']
        lines.append(f"{labels[d]}：G6−R6平均{t['mean']:+.2f} AP百分点，描述区间[{ci[0]:+.2f},{ci[1]:+.2f}]。"
                     +('这支持承认此前线性光谱基线表达能力偏弱，不能把Source相对它的差距都归因于网络信息。' if ci[0]>0 else
                       '本域非线性基线反而下降，不能视为能力越强收益必然越好。' if ci[1]<0 else
                       '本域未获得清晰的非线性改进证据；区间跨零不能解释为等效。'))
        c=v['comparisons']['G7_minus_G6'];t=c['delta_ap_pp'];ci=t['ci95']
        lines.append(f"同域主要G7−G6平均{t['mean']:+.2f} AP百分点，描述区间[{ci[0]:+.2f},{ci[1]:+.2f}]，改善／稳定／下降{c['improved']}/{c['stable']}/{c['declined']}。"
                     +('结果支持Source分数在这一固定树学习器下提供额外排序信息。' if ci[0]>0 else
                       '该域加入Source分数反而下降，不支持其在这一固定学习器下有排序增量。' if ci[1]<0 else
                       '该域未支持清晰的Source额外排序收益，应降低网络分数必要性的主张。')
                     +'无论方向，均不能自动归因为预训练、空间上下文或物理机制。')
    original=report['domains']['original_main'];g6=original['policies']['G6'];g7=original['policies']['G7']
    assert all(v['policies'][n]['macro']['fpr']['mean']>.01 for v in report['domains'].values() for n in ['G6','G7'])
    lines += ['',f"冻结决策的关键问题是误报：四个评价域中G6和G7的宏FPR点估计全部超过1%。ALCD原版九组G6／G7为{f(g6['macro']['fpr']['mean'])}%／{f(g7['macro']['fpr']['mean'])}%，单侧95%上界{f(g6['macro']['fpr']['upper95_one_sided'])}%／{f(g7['macro']['fpr']['upper95_one_sided'])}%；G7有{g7['groups_above_1percent']}/9组超过1%。G7的{f(g7['macro']['recall']['mean'])}%冻结召回不能脱离这一误报代价解释；本轮不支持直接部署这个冻结工作点。",
        f"在利用评价参考扫描的1%宏FPR oracle下，ALCD原版九组G6／G7召回为{f(original['oracle_1percent']['G6']['macro']['recall']['mean'])}%／{f(original['oracle_1percent']['G7']['macro']['recall']['mean'])}%，Source_spectral／L3为{f(original['oracle_1percent']['Source_spectral']['macro']['recall']['mean'])}%／{f(original['oracle_1percent']['L3']['macro']['recall']['mean'])}%。development上G7 oracle召回{f(report['domains']['development']['oracle_1percent']['G7']['macro']['recall']['mean'])}%，低于L3的{f(report['domains']['development']['oracle_1percent']['L3']['macro']['recall']['mean'])}%；不能宣称G7对所有数据域或基线都占优。这些阈值使用评价标签，只作诊断。",
        '排序与实际工作点必须分开：主要AP改善不能替代冻结召回／误报约束。各域FPR及上界、逐组超预算数、零召回和最差损伤均已列出；阈值训练内满足1%，不保证外部数据稳定满足。Source_spectral与G6的特征及学习器均不同，其差值只是固定基线比较；G7−G6才是本轮的同配置Source分数增量。',
        '没有选择赢家部署，不增加第三个树模型、参数搜索、网络前向或分割训练。两个ALCD参考版本分别报告，不能挑有利版本。所有评价池反复暴露，仅为探索；标签版本差异不是像素错误率，区间跨零不是等效。',
        '保留原支持门失败、人工B未执行及原空字段；Phase89故障后54个复用概率缺少故障前输出SHA，故不能声称缓存谱系完整。CPU复算及模型重载通过也不构成独立数据确认。',
        '', '## 3．交付与验收','',
        f"原生{acceptance['sealed_files']}个小文件SHA通过；{acceptance['independent_metric_rows']}行指标与24个oracle点通过独立计数／组重采样验收。两个模型全部拟合样本重载概率一致；{acceptance['tree_probe_blocks']}个组／块执行独立树遍历探针，ALCD均值及阈值判定另行核验。探针只覆盖每块固定最多257点，不冒充全部像元的双实现推理。",
        '补充前后只读索引核验通过：development Source／MsRE索引、ALCD概率块索引及prepared清单均与此前封存输入一致，前后SHA未变化。原运行代码锁保持不变；本地补充文件另行SHA密封。',
        '模型pickle仅由本项目自有冻结模型生成，模型SHA、依赖wheelSHA、参数、样本／输入代码SHA及阈值均保存在锁文件。无需传回任何work_dirs或大数组。']
    output={'RESULTS_AND_INTERPRETATION.md':'\n'.join(lines)+'\n','LOCAL_ACCEPTANCE.json':json.dumps(acceptance,indent=2)+'\n',
        'COST_SCOPE_CLARIFICATION.json':json.dumps(dict(native_reports_preserved=True,no_numeric_changes=True,
            process_model='preflight, fit, evaluate are three separate Python processes',
            fit_peak_rss='Linux ru_maxrss*1024; cumulative fit-process high-water including fit input verification, fitting and threshold scoring; G7 can include prior G6 high-water',
            evaluation_peak_rss='separate evaluation-process cumulative high-water',
            correction='Native model cost.peak_scope including preflight is a prose error; the preceding preflight process is excluded'),indent=2)+'\n',
        'SCORE_UNITS.json':json.dumps(dict(native_reports_preserved=True,no_numeric_changes=True,
            development_legacy='S3/R6/Source_spectral/L3: float32 logit; direct >= original logit threshold',
            development_G6_G7='class1 probability float32; direct >= frozen probability threshold; never apply sigmoid again',
            ALCD_legacy='float32 logit20m -> sigmoid float64 -> mean9 probability60m float64; original sigmoid(logit threshold)',
            ALCD_G6_G7='predict probability20m -> float32 -> float64 mean9 probability60m; direct >= same frozen probability threshold',
            AP='All scores quantized to float32 for original noninterpolated average precision with atomic ties',
            oracle='scan uses actual score values cast float64 without changing ties; evaluation-reference-derived, never deploy',
            legacy_receipt_note='Copied old points retain inherited generic score_precision wording; top-level score_units provides the explicit correction'),indent=2)+'\n'}
    for n,value in output.items():
        with (root/n).open('x',encoding='utf-8',newline='\n') as f:f.write(value)
    with (root/'LOCAL_SUPPLEMENT_SHA256.json').open('x',encoding='utf-8') as f:
        json.dump({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in [*output,*index_names]},f,indent=2)
    print(json.dumps(acceptance,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);a=parser.parse_args();main(a.root)
