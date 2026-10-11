"""Small shared Phase80 report delivery; never copies checkpoints/work_dirs."""
import argparse,csv,hashlib,json,re
from pathlib import Path
import numpy as np
from phase80_common import interval,digest,read,write,ROOT

def cell(v,scale=1):return 'NA' if v is None else f'{v*scale:.4f}'
def stat(s,scale=1):
    if s['ci95'] is None:return cell(s['mean'],scale)+' [NA]'
    return cell(s['mean'],scale)+' ['+', '.join(cell(v,scale) for v in s['ci95'])+']'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();root=a.root
    report=read(root/'phase80_report.json');pre=read(root/'preflight.json');train=read(root/'training_completed.json')
    lock=read(root/'calibration_lock.json');capacity=read(root/'capacity_audit.json');pol=report['policies'];names=['M','M+S','L','L+S','Source','MsRE','LoRA_argmax','L3']
    assert read(root/'COMPLETED.json')['report_sha256']==digest(root/'phase80_report.json')
    log_path=root/'console.log'
    if not log_path.exists():log_path=Path('/home/scv/shared/phase80_control_20261011/console.log')
    memory_rows=[dict(iteration=int(step),allocated_MiB_reported=int(mem)) for step,mem in
        re.findall(r'Iter\(train\)\s+\[\s*(\d+)/4000\].*?memory:\s*(\d+)',log_path.read_text())]
    assert [r['iteration'] for r in memory_rows]==list(range(50,4001,50))
    logged_peak=max(r['allocated_MiB_reported'] for r in memory_rows)
    write(root/'memory_audit.json',dict(training_logger_peak_allocated_MiB=logged_peak,training_logger_samples=memory_rows,
        console_sha256=digest(log_path),original_end_of_run_allocated_bytes=train['peak_cuda_allocated_bytes'],
        original_end_of_run_reserved_bytes=train['peak_cuda_reserved_bytes'],
        note='MMEngine resets peak counters at logging intervals; original end-of-run counters are not global training peaks. Use maximum of all 80 logged interval peaks, integer MiB. No retraining.'))
    table=[]
    definition_rows=[];definition_counts={}
    for n in names:
        counts=[]
        for row in pol[n]['rows']:
            cm=np.asarray(row['confusion']);union=cm.sum(0)+cm.sum(1)-np.diag(cm)
            defined=np.flatnonzero(union>0).tolist();counts.append(len(defined))
            definition_rows.append(dict(policy=n,product=row['product'],group_id=row['group_id'],defined_IoU_classes=defined))
        definition_counts[n]={str(k):counts.count(k) for k in [1,2,3]}
    write(root/'miou_definition_audit.json',dict(rows=definition_rows,counts=definition_counts,
        rule='Unchanged existing measures(): each group mIoU is nanmean over class IoUs with positive union; absent truth AND prediction class has null IoU, not zero. Pooled mIoU computed after pooling all confusion counts.'))
    for n in names:
        r=pol[n];table.append(dict(policy=n,macro_ap=r['macro_ap']['mean'],pooled_ap=r['ranking']['ap'],
            macro_recall=r['macro_recall']['mean'],pooled_recall=r['recall'],
            macro_precision=r['macro_precision']['mean'],pooled_precision=r['precision'],
            macro_fpr=r['macro_fpr']['mean'],macro_fpr_upper95=r['macro_fpr']['upper95_one_sided'],pooled_fpr=r['fpr'],
            supported_macro_fpr=r['shadow_supported_fpr']['mean'],no_shadow_macro_fpr=r['no_shadow_fpr']['mean'],
            above_budget_groups=r['fpr_above_one_percent_groups'],zero_recall_groups=len(r['zero_recall_groups']),
            macro_miou_percent=r['macro_miou_percent']['mean'],pooled_miou_percent=r['miou_percent'],
            surface_iou_percent=r['iou_percent'][0],cloud_iou_percent=r['iou_percent'][1],shadow_iou_percent=r['iou_percent'][2],
            shadow_to_surface=r['shadow_to_surface'],shadow_to_cloud=r['shadow_to_cloud']))
    with (root/'summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (root/'training_curve.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['iteration','development_mIoU','selected']);w.writeheader()
        for r in train['curve']:w.writerow(dict(iteration=r['iteration'],development_mIoU=r['metrics']['mIoU'],selected=r['iteration']==train['best_iteration']))
    lines=['# Phase80：适配方法 × 光谱补充交叉探索','',
        '## 1 数据结果','',
        f"固定LoRA完成4000步；seed65，rank4/alpha4/dropout0，FP32。155训练产品属于153相关组；153fit代表用于两个浅层模型与全负类阈值，32development代表评价，其中7组有有效Shadow、25组无Shadow。最佳checkpoint仅按development mIoU选择：第{train['best_iteration']}步，平手最早。未读取任何确认池或ALCD。",'',
        f"LoRA可训练参数{pre['trainable_parameters']:,}，总参数{pre['total_parameters']:,}；训练耗时{train['seconds']/60:.2f}分钟，训练日志峰值分配显存{logged_peak}MiB（{logged_peak/1024:.3f}GiB，50步区间、MiB整数记录）。MMEngine按日志间隔重置CUDA峰值计数，training_completed.json最初保存的结束时allocated/reserved计数不代表全程峰值；原文件保留，memory_audit.json给出全部80区间复核，未为测量重训。有效标签预算为155训练产品全图有效像元计数{pre['valid_label_pixels_by_training_product']:,}，不是独立样本数，也不是4000次裁块中实际暴露像元数。",'',
        f"严格Source加载和全部基础张量一致性通过；LoRA关闭/开启零残差与Source logits最大绝对差分别为{pre['zero_residual_max_absolute_logit_error']['False']}/{pre['zero_residual_max_absolute_logit_error']['True']}，要求≤1e-6。仅96个LoRA A/B张量可训练，梯度有限；训练后全部Source参数/缓冲仍逐张量相同。8个迭代checkpoint及最佳文件SHA见training_completed.json，权重保留远端新work_dir，未自动同步。M/M+S共64条逐组原AP与混淆矩阵先精确复现后开训，最终再复现。",'',
        'M=原MsRE单分数校准；M+S=原MsRE分数+B08/B11/B12；L=LoRA单分数校准；L+S=LoRA分数+B08/B11/B12。网络Shadow输入为原流程的softmax分数再logit变换，不称校准正确概率。两个新增拟合均使用固定旧fit抽样索引/组等权BCE、0.001正则、L-BFGS-B、非负网络系数；光谱系数自由。标准化及exact-ties全局阈值仅fit冻结，各满足fit组平均FPR≤1%。', '',
        'AP、召回、精确率、FPR以下均以百分数列示，mIoU本身为百分数；AP/召回宏平均的支持是7组，FPR/mIoU是全部32组。精确率仅有预测Shadow的可定义组参与，分母见报告。池化指标按全有效像元汇总，不将像元当独立样本。','',
        '| 策略 | 宏AP | 池化AP | 宏召回 | 宏FPR | FPR单侧95%上界 | 宏mIoU | 池化mIoU |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in names:
        r=pol[n];lines.append('| '+' | '.join([n,cell(r['macro_ap']['mean'],100),cell(r['ranking']['ap'],100),cell(r['macro_recall']['mean'],100),
            cell(r['macro_fpr']['mean'],100),cell(r['macro_fpr']['upper95_one_sided'],100),cell(r['macro_miou_percent']['mean']),cell(r['miou_percent'])])+' |')
    lines+=['','| 策略 | 宏精确率 | 池化召回 | 池化精确率 | 池化FPR | 有影组FPR | 无影组FPR | 超1%组数/32 | 零召回组/7 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in names:
        r=pol[n];lines.append('| '+' | '.join([n,cell(r['macro_precision']['mean'],100),cell(r['recall'],100),cell(r['precision'],100),cell(r['fpr'],100),
            cell(r['shadow_supported_fpr']['mean'],100),cell(r['no_shadow_fpr']['mean'],100),str(r['fpr_above_one_percent_groups']),str(len(r['zero_recall_groups']))])+' |')
    lines+=['','固定比较：单位百分点；配对相关组bootstrap10000次、seed80011。AP/召回7组，FPR/mIoU32组；区间均条件于单次训练及既有development选择，仅描述。', '',
            '| 比较 | 宏ΔAP [95%区间] | 改善/稳定/下降 | 宏Δ召回 [95%区间] | 宏ΔFPR [95%区间] | 宏ΔmIoU [95%区间] | 最差ΔAP | 最大召回损伤 |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
    for n,r in report['comparisons'].items():
        lines.append('| '+' | '.join([n,stat(r['ap_pp']),f"{r['improved']}/{r['stable']}/{r['declined']}",stat(r['recall_pp']),stat(r['fpr_pp']),stat(r['miou_pp']),cell(r['worst_ap_pp']),cell(r['worst_recall_pp'])])+' |')
    lines+=['', '两方案光谱AP增量之差 `(L+S−L)−(M+S−M)`：'+stat(report['spectral_increment_difference'])+'百分点。改善/稳定/下降沿用±1AP百分点边界，全部7组原始值在下表及spectral_increment_differences.csv。','',
            '| Tile | M AP | M+S AP | L AP | L+S AP | L−M | L+S−L | M+S−M | L+S−M+S |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i,rr in enumerate(pol['M']['rows']):
        if not rr['positive']:continue
        vals=[pol[n]['rows'][i]['ranking']['ap']*100 for n in ['M','M+S','L','L+S']]
        m,ms,l,ls=vals
        lines.append('| '+rr['product'].split('_')[5]+' | '+' | '.join(cell(v) for v in vals+[l-m,ls-l,ms-m,ls-ms])+' |')
    lines+=['','## 2 结论探讨','',
        f"本轮L+S相对M+S逐组平均AP增加{report['comparisons']['L+S minus M+S']['ap_pp']['mean']:.4f}百分点（描述95%区间{report['comparisons']['L+S minus M+S']['ap_pp']['ci95']}），7组差值全部为正，按±1百分点边界5改善/2稳定/0下降；平均召回增加{report['comparisons']['L+S minus M+S']['recall_pp']['mean']:.4f}百分点，全部32组平均FPR下降{-report['comparisons']['L+S minus M+S']['fpr_pp']['mean']:.4f}百分点。本批排序和工作点支持保留固定LoRA分枝作为下一轮必要对照候选，不启动新增训练，也不宣称已独立胜出。", '',
        f"但L+S的平均FPR点估计{pol['L+S']['macro_fpr']['mean']*100:.4f}%虽≤1%，单侧95%上界{pol['L+S']['macro_fpr']['upper95_one_sided']*100:.4f}%仍>1%；预算稳定满足未获支持，且{pol['L+S']['fpr_above_one_percent_groups']}/32组实际FPR>1%。L+S对M+S最大逐组召回下降{report['comparisons']['L+S minus M+S']['worst_recall_pp']:.4f}百分点，平均改善不保证每景无损伤。", '',
        f"光谱在两种方案内均提高组平均AP：LoRA增量{report['comparisons']['L+S minus L']['ap_pp']['mean']:.4f}百分点，MsRE增量{report['comparisons']['M+S minus M']['ap_pp']['mean']:.4f}百分点。这支持光谱补充可能不专属于MsRE的探索解释；两增量之差{report['spectral_increment_difference']['mean']:.4f}百分点，不能说增益已证明等效。光谱加入LoRA后池化AP变化{100*(pol['L+S']['ranking']['ap']-pol['L']['ranking']['ap']):.4f}百分点，加入MsRE后变化{100*(pol['M+S']['ranking']['ap']-pol['M']['ranking']['ap']):.4f}百分点；场景内排序改善并不等于跨场景统一分数标尺改善。", '',
        f"L+S对M+S宏mIoU增加{report['comparisons']['L+S minus M+S']['miou_pp']['mean']:.4f}百分点，但池化mIoU变化{pol['L+S']['miou_percent']-pol['M+S']['miou_percent']:.4f}百分点，存在整体面积汇总代价。沿用旧定义：每组各类IoU的union=0时记NA并从该组nanmean排除；预测是否引入本景不存在的类会改变mIoU参与类数，宏差不能直接解释为同样三个有效类的均匀提升。M+S的一/二/三有效类组数为{definition_counts['M+S']}，L+S为{definition_counts['L+S']}；完整256条类支持审计见miou_definition_audit.json，不改变指标补救结果。", '',
        '本轮提供适配方案及其光谱增量的固定探索对照。不能据本轮development区间宣布独立确认、不能改写旧主检验/Phase72/74及Phase78/79的停止结论。没有新增阈值、checkpoint、rank、学习率或方向搜索。', '',
        f"LoRA可训练参数{capacity['lora_trainable_parameters']:,}，原MsRE可训练参数{capacity['msre_trainable_parameters']:,}，容量不匹配；MsRE含target_head_delta而此LoRA无。二者Surface/Cloud胜者不同，因此整体mIoU差异也不全来自Shadow排序。两套完整方案的差异不能归因为单一结构或纯光谱机制。", '',
        '随机种子固定，但继承原流程的CUDA确定性warn-only设置；日志记录了部分CUDA算子缺少确定性实现的警告，不承诺再次训练逐位重现。', '',
        '是否保留LoRA分枝必须同时看L+S−M+S排序、冻结阈值召回与FPR代价。若只改变工作点或未改善，按协议关闭本轮固定LoRA方案；若两者均有光谱收益，只支持光谱补充可能不依赖MsRE的探索解释。这里没有新增独立确认晋级检验，也不授权另一种网络。', '',
        '全部32组×8策略原始AP/混淆派生指标见development_groups.csv与phase80_report.json；无参考Shadow的25组AP保持NA并继续参与FPR。报告保留各类IoU、两种漏影流向、零召回名单、支持分母；扫描FPR预算的召回只在ranking.matched_fpr作诊断，不替代冻结阈值运行结果。', '',
        'training_curve.csv/.jsonl记录8次development mIoU；训练日志来自shared控制目录，不自动传work_dirs日志或权重。calibration_lock.json/fit_input_audit.json保存两个模型、标准化、阈值、153组抽样索引SHA与输入锁；training_lock.json锁代码、来源、配置。Local verification记录实际复核范围，不把哈希核验说成对全部缓存的本地像元复算。','']
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    files=[p for p in root.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.jsonl','.py','.md']]
    files += [root/c/'index.json' for c in ['fit_representatives','development_val']]
    write(root/'artifact_hashes.json',{str(p.relative_to(root)):digest(p) for p in sorted(files)})

if __name__=='__main__':main()
