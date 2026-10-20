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
      '下表mIoU/AP/召回/误报均为%；误报为逐组等权负类云影判定比例。完整八行及各类IoU、最大组误报见 TABLE1_FIXED_RESULTS.csv。','',
      '|池|策略|组均mIoU|池化mIoU|Shadow宏AP|宏召回|宏误报|',
      '|---|---|---:|---:|---:|---:|---:|']
    for row in data['summary']:
        lines.append('|'+ '|'.join([row['pool'],row['strategy']]+[fmt(row[k]) for k in
           ('group_mean_mIoU_pp','pooled_mIoU_pp','macro_Shadow_AP_pp','macro_Shadow_recall_pp','macro_negative_FPR_pp')])+'|')
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
        statement = '变化主要与工作点移动相容' if row['student_minus_R4_macro_AP_pp']<=0 and row['student_minus_R4_oracle_recall_pp']<=0 else '尚有排序或匹配误报召回的点估计变化，稳定性与机制未证实'
        lines.append(f"- {row['pool']} {row['student']}：匹配误报预算的R4 oracle召回差 {fmt(row['student_minus_R4_oracle_recall_pp'])} pp，实际预算差 {fmt(row['actual_minus_budget_pp'])} pp；{statement}。oracle使用评价标签，仅作CPU诊断，不产生部署阈值。")
    difference = contrasts['Catalogue','M-PL−B-PL']
    lines += [f"- M比B多25728参数；组均mIoU增量 {fmt(difference['delta_group_mean_mIoU_pp'])} pp，区间 [{fmt(difference['group_mean_CI95_low_pp'])}, {fmt(difference['group_mean_CI95_high_pp'])}]。不能仅凭该单seed差值认定模块机制。"]
    for arm in ('B-PL','M-PL'):
        selected = [r for r in probe if r['strategy']==arm]
        corrected = sum(int(r['teacher_wrong_student_correct']) for r in selected)
        harmed = sum(int(r['teacher_correct_student_wrong']) for r in selected)
        lines.append(f"- 四个固定中心裁剪最终审计：{arm} 教师错→学生对 {corrected:,}、教师对→学生错 {harmed:,} 像元。教师使用完整上下文、学生用512裁剪；该计数含上下文差异，不单独证明训练纠错。")
    lines += ['', '云/影与地表混淆的方向计数及全部原始地表子类见表五附件；亮/暗水体按原始字段定义，雪/盐湖/沙地名称不等于测得亮度。两池均为Sentinel-2，本轮没有跨传感器验证，也没有B08替换实验。标签共同父类映射可比范围有限，不能将池间差异直接归因于标注偏移。',
      '', '局限：单训练seed；历史development/PixBox已用于开发；V4完整训练覆盖未知；不作为新盲测或严格源域保持证据。正式末步封存前曾人工预览少量本地PixBox参考记录，属于阶段读取隔离偏差，未连接学生结果或改变已锁训练。训练进程的Python文件审计通过，但不等于原生系统调用审计。',
      '', '下一项具体实验：在VFM骨干上实施B/M固定伪标签、相同预算和裁剪序列的三seed配对验证，并另行准入独立评价支持；本轮Omni机制预实验不替代VFM验证。本次不启动下一phase。',
      '', '完整表：TABLE1_FIXED_RESULTS.csv、TABLE2_COMPARISONS.csv、E4_COMPARISONS.csv、TABLE3_TRAINING.csv、TABLE4_PROBES.csv、TABLE5_*.csv、TABLE6_ORACLE_WORKPOINT.csv；逐组/产品、探针真值、锁及CPU验收另存同目录。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    SUMMARY.mkdir(parents=True,exist_ok=True)
    for path in list(ROOT.glob('*.csv'))+[ROOT/n for n in ('RESULT.md','CPU_ACCEPTANCE.json','COMMON_SUPPORT_ACCEPTANCE.json',
                 'FIXED_RESULTS.json','LOCAL_EVALUATION_LOCK.json','SUBGROUP_MAPPING_LOCK.json','SUBGROUP_ACCEPTANCE.json','LOCAL_STATUS.json','BACKUP_RECEIPT.json')]:
        shutil.copyfile(path,SUMMARY/path.name)
    for name in ('FINAL_REMOTE_ACCEPTANCE.json','EXECUTION_LOCK.json','CONTROLLER_STATUS.json','TRAINING_SEAL.json','V4_WEIGHT_LOCK.json'):
        shutil.copyfile(D/name,SUMMARY/name)
    manifest = {str(p.relative_to(SUMMARY)):sha(p) for p in SUMMARY.iterdir() if p.is_file() and p.name!='FINAL_SUMMARY_LOCK.json'}
    (SUMMARY/'FINAL_SUMMARY_LOCK.json').write_text(json.dumps(dict(status='completed',files=manifest,
             full_backup_archive_sha256=backup['archive_sha256']),indent=2)+'\n')
    print(json.dumps(dict(status='completed',summary_files=len(manifest),calls=audit['calls'])))


if __name__=='__main__': main()
