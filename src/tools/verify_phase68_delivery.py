"""Independent arithmetic from delivered CSV, no image/cache or heldout reads."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def boot(values):
    a=np.array(values,float);rng=np.random.default_rng(68010)
    means=np.mean(a[rng.integers(0,len(a),(10000,len(a)))],axis=1)
    return float(a.mean()),np.quantile(means,[.025,.975]),float(np.quantile(means,.95))


def main(root):
    delivery=root/'delivery';report=read(delivery/'phase68_report.json')
    with (delivery/'development_groups.csv').open(newline='',encoding='utf-8') as f:raw=list(csv.DictReader(f))
    assert len(raw)==128 and set(r['policy'] for r in raw)==set('ABCD')
    groups={n:[r for r in raw if r['policy']==n] for n in 'ABCD'}
    for name,rows in groups.items():
        p=report['policies'][name];assert len(rows)==len({r['group_id'] for r in rows})==32
        pos=[r for r in rows if int(r['positive'])>0];assert len(pos)==7
        assert {r['group_id'] for r in rows}=={r['group_id'] for r in groups['A']}
        np.testing.assert_allclose(np.mean([float(r['ap']) for r in pos]),p['macro_ap'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(np.mean([float(r['recall']) for r in pos]),p['macro_recall'],atol=1e-14,rtol=0)
        defined=[r for r in rows if r['precision']!='']
        assert len(defined)==p['macro_precision_groups']
        np.testing.assert_allclose(np.mean([float(r['precision']) for r in defined]),p['macro_precision'],atol=1e-14,rtol=0)
        for k,rr in [('all',rows),('shadow',pos),('no_shadow',[r for r in rows if not int(r['positive'])])]:
            mean,ci,upper=boot([float(r['fpr']) for r in rr]);reference=p['fpr_strata'][k]
            np.testing.assert_allclose([mean,*ci,upper],[reference['mean'],*reference['ci95'],reference['upper95']],atol=1e-14,rtol=0)
        for key in ['positive','negative','true_positive','false_positive','shadow_to_surface','shadow_to_cloud']:
            assert sum(int(r[key]) for r in rows)==p[key]
        tp=p['true_positive'];fp=p['false_positive']
        np.testing.assert_allclose([tp/p['positive'],fp/p['negative'],tp/(tp+fp)],
            [p['recall'],p['fpr'],p['precision']],atol=1e-14,rtol=0)
        # Confusion arrays are delivered in JSON; independently recover their pooled IoU.
        cm=np.sum([r['confusion'] for r in p['rows']],axis=0)
        iou=np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm))*100
        np.testing.assert_allclose(iou,p['iou_percent'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(iou.mean(),p['miou_percent'],atol=1e-12,rtol=0)
        assert sum(int(r['true_positive'])==0 for r in pos)==p['zero_recall_groups']
    for comparison,c in report['comparisons'].items():
        left,right=comparison.split('_minus_')
        pairs=[(l,r) for l,r in zip(groups[left],groups[right]) if int(l['positive'])>0]
        for key,column in [('recall','recall'),('ap','ap')]:
            mean,ci,upper=boot([float(l[column])-float(r[column]) for l,r in pairs])
            np.testing.assert_allclose([mean,*ci,upper],[c[key]['mean'],*c[key]['ci95'],c[key]['upper95']],atol=1e-14,rtol=0)
    a,b,d=[report['policies'][n] for n in 'ABD']
    conditions=[d['macro_recall']>b['macro_recall'],d['macro_recall']>a['macro_recall'],
        d['fpr_strata']['shadow']['mean']<=.01,d['fpr_strata']['no_shadow']['mean']<=.01,
        d['macro_ap']>=a['macro_ap'],d['miou_percent']>=a['miou_percent'],d['zero_recall_groups']<=a['zero_recall_groups']]
    assert conditions==list(report['continue_gate']['conditions'].values())
    assert all(conditions)==report['continue_gate']['all_met']
    paired=[dict(comparison=name,**r) for name,c in report['comparisons'].items() for r in c['pairs']]
    with (root/'paired_group_deltas.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in delivery.iterdir() if p.is_file()}
    (root/'metric_recomputation.json').write_text(json.dumps(dict(csv_rows=128,group_bootstrap_recomputed=True,
        all_macro_and_pooled_confusion_arithmetic_verified=True,gate_recomputed=True,file_sha256=sha),indent=2),encoding='utf-8')
    write_results(root,report,read(delivery/'policy_lock.json'),read(delivery/'fit_input_audit.json'))
    print(json.dumps(dict(verified=True,gate=report['continue_gate'],policies={n:{k:p[k] for k in
        ['macro_ap','macro_recall','miou_percent','zero_recall_groups','fpr_strata']} for n,p in report['policies'].items()}),indent=2))


def write_results(root,report,lock,audit):
    p=report['policies'];f=lambda x:'未定义' if x is None else f'{x*100:.4f}'
    lines=['# Phase68 固定损失与校准对照','', '## 1. 数据结果','',
        '153原fit相关组代表；32原development代表，其中7组有有效Shadow、25组无Shadow。只新增一次五输入浅层拟合；无分割网络前向、无确认集评价。A的全部32组分数、confusion及AP精确复现。', '',
        'A=原L3/原校准，B=原L3/双分层校准，C=新条件组均衡损失/原校准，D=同C模型/双分层校准。四策略均用相同MsRE非Shadow输出。', '',
        '|策略|宏AP %|池化AP %|宏召回 %|池化召回 %|宏精确率 %（定义组数）|池化精确率 %|池化mIoU %|零召回/7|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in p.items():lines.append(f"|{n}|{f(r['macro_ap'])}|{f(r['ranking']['ap'])}|{f(r['macro_recall'])}|{f(r['recall'])}|{f(r['macro_precision'])} ({r['macro_precision_groups']}/32)|{f(r['precision'])}|{r['miou_percent']:.4f}|{r['zero_recall_groups']}|")
    lines+=['','宏AP/召回支持7组；宏精确率只对有预测Shadow的组定义，分母列明；无Shadow组AP留空，不填零。池化指标由全部有效参考像元汇总，不能作为独立样本数。','',
        '|策略|全部组宏FPR % [双侧95%CI] / 单侧95%上界|Shadow宏FPR % / 上界|无Shadow宏FPR % / 上界|池化FPR %|超过1%组数/32|',
        '|---|---:|---:|---:|---:|---:|']
    for n,r in p.items():
        q=r['fpr_strata'];allq=q['all']
        lines.append(f"|{n}|{f(allq['mean'])} [{f(allq['ci95'][0])}, {f(allq['ci95'][1])}] / {f(allq['upper95'])}|{f(q['shadow']['mean'])} / {f(q['shadow']['upper95'])}|{f(q['no_shadow']['mean'])} / {f(q['no_shadow']['upper95'])}|{f(r['fpr'])}|{r['groups_fpr_above_1pct']}|")
    lines+=['','完整两层双侧区间、逐组FPR见JSON及development_groups.csv；约束针对相关组均值，不保证每景≤1%。','',
        '|对比|宏实际召回差 pp [95%CI]|宏AP差 pp [95%CI]|AP改善/稳定/下降（±1pp）|',
        '|---|---:|---:|---:|']
    for name,c in report['comparisons'].items():
        r=c['recall'];a=c['ap'];lines.append(f"|{name}|{f(r['mean'])} [{f(r['ci95'][0])}, {f(r['ci95'][1])}]|{f(a['mean'])} [{f(a['ci95'][0])}, {f(a['ci95'][1])}]|{c['improved_ap']}/{c['stable_ap']}/{c['declined_ap']}|")
    lines+=['','D−B为唯一主要开发比较，D−A同时必报。其余三项用于拆分因素，不能替代主要比较。10000次配对相关组bootstrap，seed68010，条件于固定模型；7组为多轮查看的开发数据，不能称独立确认。','',
        '|策略|Surface / Cloud / Shadow池化IoU %|宏mIoU %|宏预测Shadow %|池化预测Shadow %|Shadow→Surface / Cloud像元数|池化oracle召回@1% / @5%FPR %|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,r in p.items():
        q=r['ranking']['matched_fpr'];ious=' / '.join(f'{v:.4f}' for v in r['iou_percent'])
        lines.append(f"|{n}|{ious}|{r['macro_miou_percent']:.4f}|{f(r['macro_predicted_shadow_fraction'])}|{f(r['pooled_predicted_shadow_fraction'])}|{r['shadow_to_surface']} / {r['shadow_to_cloud']}|{f(q['0.01']['recall'])} / {f(q['0.05']['recall'])}|")
    lines+=['','oracle扫描只作排序诊断；实际识别全部使用已在fit冻结的阈值。宏各类IoU、支持分母及逐组oracle值完整保存在JSON/CSV。','',
        '|主要对比损伤|最差逐组AP差 pp（产品）|最大逐组召回下降 pp（产品）|','|---|---|---|']
    for name in ['D_minus_B','D_minus_A']:
        c=report['comparisons'][name];a=c['worst_ap'];r=c['worst_recall']
        lines.append(f"|{name}|{f(a['delta_ap'])} ({a['product']})|{f(r['delta_recall'])} ({r['product']})|")
    lines+=['','### 拟合与校准审计','',f"固定样本{audit['sample_count']}个，其中Shadow {audit['sample_positive']}；支持：`{json.dumps(audit['support'],ensure_ascii=False)}`。每组完整/抽样正负数、遗漏支持差异见fit_input_audit.json。新模型优化器收敛{lock['models']['conditional']['iterations']}步；均值/尺度与原L3完全相同。",'',
        '|策略|冻结logit阈值|fit全部组FPR %|fit Shadow组FPR %|fit无Shadow组FPR %|','|---|---:|---:|---:|---:|']
    for n,r in lock['fit_calibration'].items():lines.append(f"|{n}|{r['threshold']:.15g}|{f(r['all_group_mean_fpr'])}|{f(r['shadow_group_mean_fpr'])}|{f(r['no_shadow_group_mean_fpr'])}|")
    lines+=['','## 2. 结论探讨','', '冻结继续条件逐项结果：','']
    descriptions=['D宏召回严格高于B','D宏召回严格高于A','D Shadow层宏FPR≤1%','D无Shadow层宏FPR≤1%',
        'D宏AP不低于A','D池化mIoU不低于A','D零召回组数不高于A']
    for desc,ok in zip(descriptions,report['continue_gate']['conditions'].values()):lines.append(f"- {desc}：{'满足' if ok else '不满足'}。")
    lines+=['',('全部点估计条件满足：仅值得另拟新的独立确认协议，尚未获得独立确认或普遍安全支持。' if report['continue_gate']['all_met'] else
        '未满足全部冻结继续条件：本轮停止该假说，不追加候选、权重网格、正则化或阈值调试。保留原L3及既有简化候选。'), '',
        'A/B与C/D分别共享完全相同的分数，因此各自AP相同；阈值收紧不会提高同分数的实际召回。D−B用于检验新损失能否补回分层校准代价，D−A衡量相对现有方案的取舍；不能将单一指标改善当成整体恢复。','',
        '置信区间跨零的比较不支持确定增益；任何分层FPR上界超过1%，均需标明平均预算稳定满足仍不确定。fit达标不保证development达标，development点估计达标也不是独立安全保证。标签版本质量与7组小样本限制仍在。','',
        f"本轮D−B宏召回仅{f(report['comparisons']['D_minus_B']['recall']['mean'])}pp且95%区间跨零，尚未显示明确的补偿收益。D的Shadow层宏FPR为{f(p['D']['fpr_strata']['shadow']['mean'])}%，单侧95%上界{f(p['D']['fpr_strata']['shadow']['upper95'])}%；点估计已超过1%，不能以全部32组平均达标替代分层约束。无Shadow层单侧上界为{f(p['D']['fpr_strata']['no_shadow']['upper95'])}%。",'',
        f"原L3在fit全部组平均FPR≤1%时，Shadow层FPR为{f(lock['fit_calibration']['A']['shadow_group_mean_fpr'])}%，说明总体均值确实掩盖了分层差异；收紧校准降低误报同时显著损失召回。条件组均衡损失相对原模型的池化AP变化为{f(p['C']['ranking']['ap']-p['A']['ranking']['ap'])}pp，但宏AP变化为{f(p['C']['macro_ap']-p['A']['macro_ap'])}pp，不能称排序全面恢复，也不能用池化或oracle改善替代冻结继续条件。",'',
        '原H1失败结论、原split及Phase66对原L3/L2的既有结论不变；C/D不能借用Phase66确认结果。未访问65b/65c/旧sealed，不自动启动新确认。本轮规定的拟合、评价和交付完成；新独立验证未获授权且未实施。','',
        '交付：development_groups.csv（128行）、paired_group_deltas.csv（35行）、phase68_report.json、input_lock.json、policy_lock.json、fit_input_audit.json、A_reproduction.json、run_logs与双机SHA/独立重算证据。所有产物源于shared新目录，不同步work_dirs。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();main(args.root)
