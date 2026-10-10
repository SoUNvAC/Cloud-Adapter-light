"""Independent arithmetic from delivered CSV, no image/cache or heldout reads."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def boot(values):
    a=np.array(values,float);rng=np.random.default_rng(69010)
    means=np.mean(a[rng.integers(0,len(a),(10000,len(a)))],axis=1)
    return float(a.mean()),np.quantile(means,[.025,.975]),float(np.quantile(means,.95))


def main(root):
    delivery=root/'delivery';report=read(delivery/'phase69_report.json')
    with (delivery/'development_groups.csv').open(newline='',encoding='utf-8') as f:raw=list(csv.DictReader(f))
    assert len(raw)==128 and set(r['policy'] for r in raw)==set('ABDS')
    groups={n:[r for r in raw if r['policy']==n] for n in 'ABDS'}
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
    a,d=[report['policies'][n] for n in 'AD']
    c=report['comparisons']
    conditions=[c['D_minus_B']['ap']['ci95'][0]>0,c['D_minus_S']['ap']['ci95'][0]>0,
        d['macro_recall']>=a['macro_recall'],d['fpr_strata']['all']['mean']<=.01,
        d['miou_percent']>=a['miou_percent'],d['zero_recall_groups']<=a['zero_recall_groups']]
    features=read(delivery/'feature_index.json')
    for cohort,count in [('fit',153),('development',32)]:
        assert len(features[cohort])==count
        for row in features[cohort]:
            seed=69010+int(hashlib.sha256(row['product'].encode()).hexdigest()[:8],16)
            assert seed==row['permutation_seed']
            permutation=np.random.default_rng(seed).permutation(row['permutation_count'])
            assert hashlib.sha256(permutation.tobytes()).hexdigest()==row['permutation_sha256']
    assert conditions==list(report['continue_gate']['conditions'].values())
    assert all(conditions)==report['continue_gate']['all_met']
    paired=[dict(comparison=name,**r) for name,c in report['comparisons'].items() for r in c['pairs']]
    with (root/'paired_group_deltas.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in delivery.iterdir() if p.is_file()}
    (root/'metric_recomputation.json').write_text(json.dumps(dict(csv_rows=128,group_bootstrap_recomputed=True,
        all_macro_and_pooled_confusion_arithmetic_verified=True,gate_recomputed=True,all_185_permutation_hashes_recomputed=True,file_sha256=sha),indent=2),encoding='utf-8')
    write_results(root,report,read(delivery/'policy_lock.json'),read(delivery/'fit_input_audit.json'))
    print(json.dumps(dict(verified=True,gate=report['continue_gate'],policies={n:{k:p[k] for k in
        ['macro_ap','macro_recall','miou_percent','zero_recall_groups','fpr_strata']} for n,p in report['policies'].items()}),indent=2))


def write_results(root,report,lock,audit):
    p=report['policies'];f=lambda x:'未定义' if x is None else f'{x*100:.4f}'
    lines=['# Phase69 云预测邻域信息筛查','', '## 1. 数据结果','',
        '原153个fit相关组代表、32个development代表，其中7组有有效Shadow、25组无Shadow。原L3全部32组像元分数、逐组confusion/AP精确复现；只新增B/D/S三个浅层逻辑回归，无新分割网络前向、无确认集评价。','',
        'A=原L3五输入；B=加中心云预测指示（六输入）；D=B加31×31、127×127云邻域（八输入）；S=B加共同置换的两个邻域（八输入），S单独拟合。所有策略共用MsRE非Shadow分类输出。','',
        '|策略|宏AP %|池化AP %|宏实际召回 %|池化召回 %|宏精确率 %（定义组数）|池化精确率 %|池化mIoU %|零召回/7|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in p.items():lines.append(f"|{n}|{f(r['macro_ap'])}|{f(r['ranking']['ap'])}|{f(r['macro_recall'])}|{f(r['recall'])}|{f(r['macro_precision'])} ({r['macro_precision_groups']}/32)|{f(r['precision'])}|{r['miou_percent']:.4f}|{r['zero_recall_groups']}|")
    lines+=['','宏AP与召回支持7组，AP无支持留null/CSV空值；精确率只在有预测Shadow时定义，分母列明。无Shadow组继续参与FPR，全部组不按结果删除。','',
        '|策略|全部组宏FPR % [95%CI] / 单侧95%上界|Shadow宏FPR % / 上界|无Shadow宏FPR % / 上界|池化FPR %|超过1%组数/32|',
        '|---|---:|---:|---:|---:|---:|']
    for n,r in p.items():
        q=r['fpr_strata'];a=q['all']
        lines.append(f"|{n}|{f(a['mean'])} [{f(a['ci95'][0])}, {f(a['ci95'][1])}] / {f(a['upper95'])}|{f(q['shadow']['mean'])} / {f(q['shadow']['upper95'])}|{f(q['no_shadow']['mean'])} / {f(q['no_shadow']['upper95'])}|{f(r['fpr'])}|{r['groups_fpr_above_1pct']}|")
    lines+=['','各层完整双侧区间、逐组FPR及最差损伤见JSON/CSV。全部组平均1%预算不表示每景或每层≤1%，fit达标不保证development达标。','',
        '|固定比较|宏AP差 pp [95%CI]|宏召回差 pp [95%CI]|AP改善/稳定/下降（±1pp）|',
        '|---|---:|---:|---:|']
    for name,c in report['comparisons'].items():
        a=c['ap'];r=c['recall']
        lines.append(f"|{name}|{f(a['mean'])} [{f(a['ci95'][0])}, {f(a['ci95'][1])}]|{f(r['mean'])} [{f(r['ci95'][0])}, {f(r['ci95'][1])}]|{c['improved_ap']}/{c['stable_ap']}/{c['declined_ap']}|")
    lines+=['','D−B检验超过中心云指示的邻域增量，D−S检验超过匹配维数的打乱对照，D−A检验相对既有方案的实际收益。10000次配对相关组bootstrap，seed69010；只有7个多轮查看的development组，区间条件于固定拟合模型，不能称独立确认或像元iid推断。','',
        '|策略|Surface / Cloud / Shadow池化IoU %|宏mIoU %|宏预测Shadow %|池化预测Shadow %|Shadow→Surface / Cloud像元数|',
        '|---|---:|---:|---:|---:|---:|']
    for n,r in p.items():
        ious=' / '.join(f'{v:.4f}' for v in r['iou_percent'])
        lines.append(f"|{n}|{ious}|{r['macro_miou_percent']:.4f}|{f(r['macro_predicted_shadow_fraction'])}|{f(r['pooled_predicted_shadow_fraction'])}|{r['shadow_to_surface']} / {r['shadow_to_cloud']}|")
    lines+=['','宏各类IoU及完整confusion保存在JSON。CSV中额外保留1%/5%扫描召回仅供诊断，实际评价只使用fit冻结阈值。','',
        '|比较|最差逐组AP差 pp（产品）|最大逐组召回下降 pp（产品）|','|---|---|---|']
    for name,c in report['comparisons'].items():
        a=c['worst_ap'];r=c['worst_recall']
        lines.append(f"|{name}|{f(a['delta_ap'])} ({a['product']})|{f(r['delta_recall'])} ({r['product']})|")
    lines+=['','### 特征、拟合与校准锁','',
        f"固定原fit抽样{audit['sample_count']}像元，其中Shadow {audit['sample_positive']}；抽样/五输入/标签SHA与原恢复实验一致。邻域只用原MsRE argmax云指示与image_valid生成：窗口排除中心、边缘只计实际范围、无效邻居不计分母。共同置换覆盖每产品全部image-valid像元，先置换后选择标签有效像元；标签不参与特征构造。",'',
        '每产品预测/掩膜/邻域文件SHA、置换种子/索引SHA在feature_index.json；本地独立重算185个置换SHA。B/D/S均使用原逐组等权BCE、0.001正则、fit标准化及原优化器；前两分数系数非负，其他自由；未用Phase68损失或分层校准。模型/阈值先落锁再评价新策略。','',
        '|策略|输入数|优化迭代数|冻结logit阈值|fit全组平均FPR %|','|---|---:|---:|---:|---:|']
    for n,m in lock['models'].items():lines.append(f"|{n}|{5 if n=='A' else m['n_features']}|{m['iterations']}|{m['threshold']:.15g}|{f(m['fit_macro_fpr'])}|")
    lines+=['','## 2. 结论探讨','', '冻结继续条件：','']
    desc=['D−B宏AP区间下界>0','D−S宏AP区间下界>0','D宏召回不低于A','D全组宏FPR点估计≤1%',
        'D池化mIoU不低于A','D零召回组数不增加']
    for label,ok in zip(desc,report['continue_gate']['conditions'].values()):lines.append(f"- {label}：{'满足' if ok else '不满足'}。")
    lines+=['',('全部继续条件满足：本轮仅支持现有云预测邻域在这些开发组上提供可用信息，不能证明太阳几何、真实云影对应关系、新算法成立或独立泛化。' if report['continue_gate']['all_met'] else
        '至少一个冻结条件未满足：按协议停止本轮，不追加窗口、方向、特征、权重或拟合。'),'',
        f"D相对原A的宏AP变化{f(report['comparisons']['D_minus_A']['ap']['mean'])}pp、宏实际召回变化{f(report['comparisons']['D_minus_A']['recall']['mean'])}pp；需同时评估D−B、D−S，而不能仅凭超过A就归因于空间邻域。打乱对照保留每产品邻域分布及成对相关性，但破坏像元位置对应；本轮推断范围限于此固定对照。",'',
        f"D全部组宏FPR {f(p['D']['fpr_strata']['all']['mean'])}%，单侧95%上界{f(p['D']['fpr_strata']['all']['upper95'])}%；Shadow层{f(p['D']['fpr_strata']['shadow']['mean'])}%（上界{f(p['D']['fpr_strata']['shadow']['upper95'])}%），无Shadow层{f(p['D']['fpr_strata']['no_shadow']['mean'])}%（上界{f(p['D']['fpr_strata']['no_shadow']['upper95'])}%）。点估计与上界分别解读，平均预算不能称每景保证。",'',
        ('两个邻域AP比较的区间均跨零，本次固定窗口线性探针未检出超过中心云信息或打乱对照的稳定开发增益。这不证明空间信息不存在，也不是D/S严格等效证据。' if all(report['comparisons'][n]['ap']['ci95'][0]<=0<=report['comparisons'][n]['ap']['ci95'][1] for n in ['D_minus_B','D_minus_S']) else
         '邻域信息的判断只依冻结的D−B与D−S比较，不以其他变好的指标替代；结果限于本轮两个窗口与线性容量。'),'',
        ('D全组平均FPR点估计满足1%，但单侧95%上界超过1%，稳定满足预算仍不确定。' if p['D']['fpr_strata']['all']['mean']<=.01<p['D']['fpr_strata']['all']['upper95'] else
         'FPR是否满足预算以表中点估计与预设单侧上界分别判断，不改变预算。'),'',
        '已有7个开发组反复探索、小样本及参考标签质量限制仍在；bootstrap不覆盖拟合/先前选择不确定性。原H1失败、原split与Phase66原L3/L2结论不变；未读Phase66/65b/65c/旧sealed像元或分数，不加入ALCD，不自动启封65c。未检索太阳几何或扩展数据。','',
        '交付：README.md、summary.csv（四行）、development_groups.csv（128行）、paired_group_deltas.csv（21行）、完整报告、输入/特征/抽样/阈值锁、shared新执行日志及SHA/独立重算证据；不自动同步work_dirs。本轮规定的三个探针与评价已完成，新独立确认未获授权且未实施。']
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary=[]
    for n,r in p.items():summary.append(dict(policy=n,macro_ap=r['macro_ap'],pooled_ap=r['ranking']['ap'],macro_recall=r['macro_recall'],
        all_macro_fpr=r['fpr_strata']['all']['mean'],all_fpr_upper95=r['fpr_strata']['all']['upper95'],
        shadow_macro_fpr=r['fpr_strata']['shadow']['mean'],no_shadow_macro_fpr=r['fpr_strata']['no_shadow']['mean'],
        pooled_miou_percent=r['miou_percent'],zero_recall_groups=r['zero_recall_groups']))
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();main(args.root)

