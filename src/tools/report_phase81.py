"""Human readable fixed crossed-readout report; no data analysis choices."""
from run_phase81 import ROOT,csvwrite
from phase80_common import read,write,digest

def main():
    r=read(ROOT/'phase81_report.json');policies=r['policies'];table=[]
    for n,p in policies.items():
        row=dict(policy=n,macro_AP_percent=p['macro_ap']['mean']*100,pooled_AP_percent=p['pooled_ap']*100,
             macro_recall_percent=p['macro_recall']['mean']*100,macro_FPR_percent=p['macro_fpr']['mean']*100,
             macro_FPR_upper95_percent=p['macro_fpr']['upper95_one_sided']*100,
             shadow_FPR_percent=p['shadow_fpr']['mean']*100,no_shadow_FPR_percent=p['no_shadow_fpr']['mean']*100,
             macro_mIoU_percent=p['macro_miou_percent']['mean'],pooled_mIoU_percent=p['miou_percent'],
             pooled_Surface_IoU=p['iou_percent'][0],pooled_Cloud_IoU=p['iou_percent'][1],pooled_Shadow_IoU=p['iou_percent'][2],
             groups_FPR_above_one_percent=p['fpr_above_one_percent_groups'])
        table.append(row)
    csvwrite('summary.csv',table)
    lines=['# Phase81：冻结云影决策与非云影读出交叉诊断','', '## 1 数据结果','',
       '32个development相关组（7组有参考云影）。仅使用既有缓存：零训练、零拟合、零阈值搜索、零新增网络推理；未读取fit像元或任何确认池。', '',
       'M=冻结MsRE＋光谱云影评分器，L=冻结LoRA＋光谱评分器。组合首字母是评分器，第二字母是Surface／Cloud二类胜者。每个评分器沿用自己的float32 logit分数、原阈值与>=运算。','',
       '|组合|宏AP %|池化AP %|宏召回 %|宏FPR %|FPR单侧95%上界 %|宏mIoU %|池化mIoU %|',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in table:lines.append('|'+x['policy']+'|'+'|'.join(f'{x[k]:.4f}' for k in ['macro_AP_percent','pooled_AP_percent','macro_recall_percent','macro_FPR_percent','macro_FPR_upper95_percent','macro_mIoU_percent','pooled_mIoU_percent'])+'|')
    lines+=['','固定比较（单位：百分点；32组配对bootstrap 10000次，seed81011，仅描述）：','']
    for k,v in r['comparisons'].items():
        m=v['macro_miou_pp'];lines.append(f"- {k}：宏mIoU {m['mean']:+.4f}，95%区间[{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]；池化mIoU {v['pooled_miou_pp']:+.4f}；下降{v['groups_declining']}/32组，最差{v['worst_group_delta_miou_pp']:.4f}。")
    lines+=['','三类池化IoU与误差流向：','', '|组合|Surface IoU %|Cloud IoU %|Shadow IoU %|Surface→Cloud|Cloud→Surface|Shadow→Surface|Shadow→Cloud|有效IoU类别数1/2/3的组数|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for n,p in policies.items():
        c=p['confusion'];lines.append(f"|{n}|{p['iou_percent'][0]:.4f}|{p['iou_percent'][1]:.4f}|{p['iou_percent'][2]:.4f}|{c[0][1]}|{c[1][0]}|{c[2][0]}|{c[2][1]}|"+'/'.join(str(p['defined_iou_class_counts'][str(i)]) for i in (1,2,3))+'|')
    lines+=['','二值云影决策组合（L/M位顺序；同一有效参考像元）：','','|决策|Surface参考数|Cloud参考数|Shadow参考数|','|---|---:|---:|---:|']
    d=read(ROOT/'decision_combinations.json')
    for n,c in zip(d['order'],d['pooled_counts']):lines.append('|'+n+'|'+'|'.join(map(str,c))+'|')
    lines+=['','原MM/LL全部64条逐组混淆矩阵、AP与决策指标及宏汇总精确复现。交叉后LL/LM与MM/ML的云影二值掩膜逐像元一致，AP、召回、FPR、精确率、Shadow IoU均一致。AP与混淆矩阵同时使用独立实现检查。','',
       '## 2 结论探讨','']
    descriptions={'pooled_gap_eliminated':'此次固定交叉读出消除了池化mIoU差距。','pooled_gap_partly_recovered':'此次固定交叉读出仅恢复部分池化mIoU差距。','pooled_gap_not_recovered':'该固定交叉读出未恢复池化mIoU差距。'}
    lines.append(descriptions[r['outcome']])
    lines+=['', '输出干预描述两套现有网络的组合行为，不能证明网络内部因果机制；不据此搜索新读出或调整阈值。云影AP优势仍只来自7个已探索development组，计算复核不是独立数据验证。','',
       '宏mIoU对相关组等权；池化mIoU先汇总混淆矩阵再计算。各组有效像元数、参考类支持、各类IoU及参与平均的类别数均在development_groups.csv中。沿用union=0记NA且只平均有定义类的旧实现；类别缺失导致的分母变化不能解释成所有类别全面改善。','',
       '同评分器的FPR与Shadow决策不受非云影读出影响。平均FPR点估计≤1%不能替代单侧95%上界≤1%；相关组平均预算也不表示每组都满足1%。有／无云影分层与超预算组数完整见summary.csv。','',
       '交叉读出需要同时运行及保存MsRE、LoRA两套网络；本轮因复用缓存没有额外网络推理，但将来部署不属于免费提升。全部旧失败结论保留，所有旧确认池均已使用，不能称新方法的未见确认。','',
       '交付：summary.csv四行；development_groups.csv 128行（32组×4组合）；paired_differences.csv 96行；confusion_matrices.json；decision_combinations.json；输入／配置SHA锁及复现／不变量检查。缓存留在shared，未同步work_dirs。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    names=['README.md','summary.csv','development_groups.csv','paired_differences.csv','confusion_matrices.json',
           'decision_combinations.json','phase81_report.json','input_configuration_lock.json','input_audit.json',
           'reproduction_checks.json','invariant_checks.json','COMPLETED.json']
    write(ROOT/'artifact_hashes.json',{n:digest(ROOT/n) for n in names})

if __name__=='__main__':main()
