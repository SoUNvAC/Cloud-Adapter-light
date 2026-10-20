"""Write concise real-data Phase300 results and copy reviewable summary artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/phase300/phase300_20261020'
D = ROOT/'remote_backup/phase300_20261020'
SUMMARY = REPO/'src/research_results/phase300_20261020'


def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(value): return 'NA' if value is None else f'{value:.3f}'


def main():
    assert load(ROOT/'CPU_ACCEPTANCE.json')['status']=='passed'
    assert load(ROOT/'SUBGROUP_ACCEPTANCE.json')['status']=='passed'
    assert load(D/'FINAL_REMOTE_ACCEPTANCE.json')['status']=='passed'
    assert load(ROOT/'LOCAL_STATUS.json')['status']=='completed'
    data = load(ROOT/'FIXED_RESULTS.json')
    summary = {(r['pool'],r['strategy']):r for r in data['summary']}
    contrasts = {(r['pool'],r['comparison']):r for r in data['comparisons']}
    audit = load(D/'FINAL_REMOTE_ACCEPTANCE.json')
    backup = load(ROOT/'BACKUP_RECEIPT.json')
    with (ROOT/'PROBE_FINAL_TRUTH_AUDIT.csv').open(encoding='utf-8-sig') as stream:
        probe = list(csv.DictReader(stream))
    lines = ['# Phase300：固定伪标签适配','', '## 数据结果','',
      '固定100个fit组生成一次R4伪标签；单训练seed30001，初始化seed20801。每臂1000尝试步，只评价末步。无源样本适配（SFDA）；使用用户明确允许的历史PyTorch 2.11固定runtime。', '',
      'Catalogue32组（AP有效7）；PixBox16产品/10组（AP有效8）。下表mIoU/AP/召回/误报均为%；误报为逐组等权负类云影判定比例。完整各类IoU、最大组误报见 TABLE1_FIXED_RESULTS.csv。','',
      '|池|策略|组均mIoU|池化mIoU|Shadow宏AP|宏召回|宏误报|',
      '|---|---|---:|---:|---:|---:|---:|']
    for row in data['summary']:
        lines.append('|'+ '|'.join([row['pool'],row['strategy']]+[fmt(row[k]) for k in
           ('group_mean_mIoU_pp','pooled_mIoU_pp','macro_Shadow_AP_pp','macro_Shadow_recall_pp','macro_negative_FPR_pp')])+'|')
    diagnostics = []
    with (ROOT/'Catalogue_PER_GROUP.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            matrix = json.loads(row['confusion'])
            truth_counts = [sum(v) for v in matrix]
            pred_counts = [sum(matrix[i][j] for i in range(3)) for j in range(3)]
            diagnostics.append(dict(group_id=row['group_id'],product=row['products'],strategy=row['strategy'],
                truth_class_counts=truth_counts,prediction_class_counts=pred_counts,
                empty_truth_classes=[i for i,n in enumerate(truth_counts) if not n],
                predictions_in_empty_truth_classes=[pred_counts[i] if not truth_counts[i] else 0 for i in range(3)],
                nonzero_union_classes=sum(truth_counts[i]+pred_counts[i]-matrix[i][i]>0 for i in range(3)),
                original_group_mIoU_pct=float(row['mIoU_pct'])))
    with (ROOT/'CLASS_ABSENCE_DIAGNOSTIC.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(diagnostics[0]))
        writer.writeheader()
        for row in diagnostics:
            writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
    lines += ['', 'Catalogue三项固定比较（差值为百分点；括号为32相关组配对bootstrap10000次、seed300020的95%描述区间）：','']
    for name in ('M-PL−R4','B-PL−R4','M-PL−B-PL'):
        row = contrasts['Catalogue',name]
        lines.append(f"- {name}：组均mIoU {fmt(row['delta_group_mean_mIoU_pp'])} [{fmt(row['group_mean_CI95_low_pp'])}, {fmt(row['group_mean_CI95_high_pp'])}]；池化mIoU {fmt(row['delta_pooled_mIoU_pp'])} [{fmt(row['pooled_CI95_low_pp'])}, {fmt(row['pooled_CI95_high_pp'])}]。")
    coverage = data['training'][0]
    lines += ['', f"图像有效支持 {coverage['image_valid']:,} 像元；接受 {100*coverage['accepted_fraction']:.2f}%（Surface/Cloud/Shadow分别 {100*coverage['Surface_fraction']:.2f}/{100*coverage['Cloud_fraction']:.2f}/{100*coverage['Shadow_fraction']:.2f}%，同一分母）。"]
    for row in data['training']:
        lines.append(f"{row['strategy']}：{row['updates']}次更新、{row['all_ignore_skips']}次全ignore跳过；首/末非空步CE {row['first_nonempty_CE']:.6f}/{row['last_nonempty_CE']:.6f}。")
    lines += [f"新增神经前向 {audit['calls']} 次，GPU阶段 {audit['GPU_stage_seconds']:.1f} 秒；GPU/远程CPU峰值 {audit['GPU_peak_bytes']/1024**3:.3f}/{audit['CPU_peak_bytes']/1024**3:.3f} GiB。原参数、BN/buffer初末一致；两臂缓存/裁剪SHA相同。完整远程目录已备份，归档SHA {backup['archive_sha256']}。",'', '## 结论分析','']
    for pool in ('Catalogue','PixBox'):
        for arm in ('B-PL','M-PL'):
            row = contrasts[pool,arm+'−R4']
            phrase = '存在漏影与低误报的代价交换' if row['delta_macro_Shadow_recall_pp']<0 and row['delta_macro_negative_FPR_pp']<0 else '不能概括为用漏影换低误报'
            lines.append(f"- {pool} {arm}：宏AP/召回/误报变化 {fmt(row['delta_macro_Shadow_AP_pp'])}/{fmt(row['delta_macro_Shadow_recall_pp'])}/{fmt(row['delta_macro_negative_FPR_pp'])} pp；{phrase}。")
    for row in data['oracle']:
        statement = '变化主要与工作点移动相容' if row['student_minus_R4_macro_AP_pp']<=0 and row['student_minus_R4_oracle_recall_pp']<=0 else '匹配误报召回有点估计收益，但AP下降；机制与稳定性未证实'
        lines.append(f"- {row['pool']} {row['student']}：匹配误报预算的R4 oracle召回差 {fmt(row['student_minus_R4_oracle_recall_pp'])} pp，实际预算差 {row['actual_minus_budget_pp']:.6g} pp；{statement}。oracle使用评价标签，仅作CPU诊断，不产生部署阈值。")
    difference = contrasts['Catalogue','M-PL−B-PL']
    lines += [f"- M比B多25728参数；组均mIoU增量 {fmt(difference['delta_group_mean_mIoU_pp'])} pp，区间 [{fmt(difference['group_mean_CI95_low_pp'])}, {fmt(difference['group_mean_CI95_high_pp'])}]，未显示整体增益。B/M相对R4的组均区间均跨零，不能认定稳健改善，也不据此判整个方向失败。"]
    worst = next(r for r in diagnostics if r['group_id']==difference['worst_group'] and r['strategy']=='M-PL')
    counterpart = next(r for r in diagnostics if r['group_id']==difference['worst_group'] and r['strategy']=='B-PL')
    lines += [f"- 缺类敏感性：{worst['product'].split('_T')[-1][:5]}全Surface组，M在原本缺失的Shadow类预测 {worst['predictions_in_empty_truth_classes'][2]} 像元；计分有效类从 {counterpart['nonzero_union_classes']} 变 {worst['nonzero_union_classes']}，mIoU {counterpart['original_group_mIoU_pct']:.6f}→{worst['original_group_mIoU_pct']:.6f}%。该组与规则保留，未删组或重选指标；全部组结构诊断另交。"]
    for arm in ('B-PL','M-PL'):
        selected = [r for r in probe if r['strategy']==arm]
        corrected = sum(int(r['teacher_wrong_student_correct']) for r in selected)
        harmed = sum(int(r['teacher_correct_student_wrong']) for r in selected)
        lines.append(f"- 四个固定中心裁剪最终审计：{arm} 教师错→学生对 {corrected:,}、教师对→学生错 {harmed:,} 像元。教师使用完整上下文、学生用512裁剪；该计数含上下文差异，不单独证明训练纠错。")
    lines += ['', '地表混淆未获修复证据：PixBox原始雪冰地表1062点的Surface→Cloud为R4/B/M=38/58/56；地形/树/建筑影331点的Surface→Shadow均为5。原始Dark water子类零支持，不能推断暗水体改善。两池均Sentinel-2，没有跨传感器验证或B08替换实验；池间差异不能直接归因于标注偏移。完整定义及全部分层见表五。',
      '', '局限：单训练seed；历史development/PixBox已用于开发；V4完整训练覆盖未知；不作为新盲测或严格源域保持证据。正式末步封存前曾人工预览少量本地PixBox参考记录，属于阶段读取隔离偏差，未连接学生结果或改变已锁训练。训练进程的Python文件审计通过，但不等于原生系统调用审计。',
      'PixBox全部指标仅来自固定参考点；mIoU/各类IoU为NA，不报告全景FPR。',
      '', '下一项具体实验：在VFM骨干上实施B/M固定伪标签、相同预算和裁剪序列的三seed配对验证，并另行准入独立评价支持；本轮Omni机制预实验不替代VFM验证。本次不启动下一phase。',
      '', '完整表：TABLE1_FIXED_RESULTS.csv、TABLE2_COMPARISONS.csv、E4_COMPARISONS.csv、TABLE3_TRAINING.csv、TABLE4_PROBES.csv、TABLE5_*.csv、TABLE6_ORACLE_WORKPOINT.csv；逐组/产品、探针真值、锁及CPU验收另存同目录。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    notes = '''# Phase300 表注与定义

- 单训练seed30001，初始化seed20801；历史development/PixBox已用于开发；V4完整训练覆盖未知。结果不是新盲测或严格源域保持证据。Omni机制预实验不替代VFM骨干验证。
- 所有原相关组保留：Catalogue32；PixBox16产品/10相关组。Catalogue组均三类mIoU对相关组等权；池化值先相加混淆矩阵再计算。无影组仍进入mIoU与负类误报比例。
- 混淆矩阵：行真值、列预测；0=Surface，1=Cloud，2=Shadow。每类IoU的union=0时为NA；mIoU在union非零类上取均值。AP要求至少一正一负，否则NA；召回无正类为NA；负类误报比例无负类为NA；宏指标对定义有效组取均值，并报告有效组数。精确度无Shadow预测为NA。
- 绝对AP、IoU和率以%报告；差值为百分点。训练覆盖/探针以fraction报告。PixBox只计算参考点指标和负参考点的Shadow误判比例，不计算全景mIoU、IoU或FPR；相应mIoU/IoU列填NA。
- 默认决策为四类raw argmax，再按[Surface,Thick,Thin,Shadow]→[0,1,1,2]合并Cloud。AP分数沿用官方Shadow口径：clip(softmax(raw)+0.001,0.001,0.999)，并用官方输入有效掩膜。没有53组阈值校准，也未按学生评价结果改阈值。
- 训练伪标签使用完整1000/300教师raw-logit拼接，稳定softmax后相加Thick/Thin概率；三父类最大概率>=0.90且三输入有效才接受。训练概率不加0.001、不clip。父类Cloud logit为logsumexp(Thick,Thin)。CE只对接受像元求均值；全ignore不前向、不反传、不optimizer.step，也不补步。
- 纯影像支持仅B04/B03/B8A均有限、非零、非float32(65535/10000)。历史监督/评价支持包含标签ignore和更多波段规则；FIT_SUPPORT_DIFFERENCES.csv在双末步封存后的本地审计才连接历史支持计数。
- 32次探针=4产品×4时点×2臂。预测占比不是正确率。相对教师变化率在共同图像有效支持计算；教师来自完整上下文，学生来自中心512裁剪。最终真值审计另交三类分解，使用image-valid与唯一真值交集，不回流训练。
- Catalogue95%区间：按32相关组配对bootstrap10000次、seed300020；每次重算组均及池化混淆指标，不把像元视为独立样本。区间仅描述固定单seed模型；不增加辅助指标的显著性声明。PixBox mIoU及其区间为NA。
- 表五原始地表字段按原始definitions.txt全代码锁定，含零支持及缺失类别。SURFACE_TYPE_ID是辅助数据字段，其他列按原始说明记录。Bright turbid water与Dark water直接来自WATER BODY CHARACTERISTICS_ID；雪/盐湖/沙地名称不代表实测亮度。不同字段的分层重叠，不能相加。无模型分数推断地表类别。
- 两池均Sentinel-2，本轮没有跨传感器验证。PixBox B04/B03从10m按历史bilinear uint16规则到20m，B8A为20m；未做B08替换试验。Catalogue原subscene更早的原始重采样细节本轮未重建。
- 标签：Catalogue固定三通道Surface/Cloud/Shadow，未恢复原Thin/Thick细类。PixBox沿用固定C/S/U映射；S2地形/树/建筑影为地表负类，S3地表云影为正类，S4云上阴影等冲突进入ignore。两个标签源并不保证地形阴影或边界定义完全等价，池间性能差异不能单独证明标注偏移。
- 表六为CPU oracle：每池按学生默认全组等权负类误报比例设预算，在R4连续分数上取最低可行全局阈值；并列分数整批处理，无逐景阈值或插值。召回仅平均有正参考组。记录实际预算差；阈值来自评价标签，不可作为部署结果，不回流训练或checkpoint选择。
- 训练/教师Python文件打开审计没有成功读取标签；该审计不覆盖原生syscall。操作代理在末步封存前人工预览少量本地PixBox参考记录的偏差另见phase300.md；不能声称整个操作过程标签不可见。
'''
    (ROOT/'TABLE_NOTES.md').write_text(notes,encoding='utf-8')
    SUMMARY.mkdir(parents=True,exist_ok=True)
    for path in list(ROOT.glob('*.csv'))+[ROOT/n for n in ('RESULT.md','TABLE_NOTES.md','CPU_ACCEPTANCE.json','COMMON_SUPPORT_ACCEPTANCE.json',
                 'FIXED_RESULTS.json','LOCAL_EVALUATION_LOCK.json','SUBGROUP_MAPPING_LOCK.json','SUBGROUP_ACCEPTANCE.json','LOCAL_STATUS.json','BACKUP_RECEIPT.json',
                 'EXPERIMENT_EVENT_LEDGER.json','LOCAL_PREFLIGHT_ACCEPTANCE.json')]:
        shutil.copyfile(path,SUMMARY/path.name)
    for name in ('FINAL_REMOTE_ACCEPTANCE.json','EXECUTION_LOCK.json','CONTROLLER_STATUS.json','TRAINING_SEAL.json','V4_WEIGHT_LOCK.json'):
        shutil.copyfile(D/name,SUMMARY/name)
    manifest = {str(p.relative_to(SUMMARY)):sha(p) for p in SUMMARY.iterdir() if p.is_file() and p.name!='FINAL_SUMMARY_LOCK.json'}
    (SUMMARY/'FINAL_SUMMARY_LOCK.json').write_text(json.dumps(dict(status='completed',files=manifest,
             full_backup_archive_sha256=backup['archive_sha256']),indent=2)+'\n')
    print(json.dumps(dict(status='completed',summary_files=len(manifest),calls=audit['calls'])))


if __name__=='__main__': main()
