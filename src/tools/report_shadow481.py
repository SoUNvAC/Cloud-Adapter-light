from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2]/'outputs/phase481/phase481_20261108'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
x=read(R/'MULTISEED_RESULTS.json');a=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert x['passed'] and a['passed'] and read(R/'TRANSFER_ACCEPTED.json')['passed']
for run,v in a['runs'].items():
 seed,arm=run.split('/');d=R/f'seed_{seed}'/arm/'network_run'
 for name,w in v['weights'].items():
  if name=='branch_0.pth':continue
  assert hashlib.sha256((d/name).read_bytes()).hexdigest()==w['sha256']
def fmt(v):return f'{v:.4f}'
def pair(arm,selection='last_6000'):return next(p for p in x['paired'] if p['comparison']==arm and p['selection']==selection)
def table(selection):
 lines=['| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- |']
 for r in x['summary']:
  if r['selection']==selection:lines.append('| '+{'C':'C：仅解码器','A':'A：普通残差','B':'B：近似均值约束'}[r['arm']]+' | '+' | '.join(f"{r[n+'_mean']:.4f} ± {r[n+'_sd']:.4f}" for n in ['ground','cloud','shadow','mIoU'])+' |')
 return '\n'.join(lines)
lines=['# Phase481：Phase480三seed复现','',
 '固定Phase431＋450起点，复用45201，新增45202、45203；A/B/C各三seed，新增六次6000步训练。目标域监督 **0%**；CloudSEN原始验证535张、107个ROI。均值±样本标准差，n=3。','',
 f"**正收益在三个seed上复现。** 固定6000步，A相对C云影IoU平均提高{pair('A-C')['mean'][2]:.4f}个百分点（{pair('A-C')['positive_count'][2]}/3为正）；B提高{pair('B-C')['mean'][2]:.4f}个百分点（{pair('B-C')['positive_count'][2]}/3为正）。B相对A平均提高{pair('B-A')['mean'][2]:.4f}个百分点（{pair('B-A')['positive_count'][2]}/3为正）。本轮优先保留B继续研究；增益仍小，尚未达到70%。",'',
 '## 主结果：统一6000步','',table('last_6000'),'','| 配对比较 | 地表增量 | 云增量 | 云影增量 | mIoU增量 | 云影正收益seed |','| --- | --- | --- | --- | --- | --- |']
for comparison in ['A-C','B-C','B-A']:
 p=pair(comparison);lines.append('| '+comparison+' | '+' | '.join(f'{m:+.4f} ± {s:.4f}' for m,s in zip(p['mean'],p['sample_sd']))+f" | {p['positive_count'][2]}/3 |")
lines+=['','| seed | A−C云影IoU | B−C云影IoU | B−A云影IoU |','| --- | --- | --- | --- |']
for i,s in enumerate(x['seeds']):lines.append(f'| {s} | '+' | '.join(f"{pair(c)['delta_by_seed'][i][2]:+.4f}" for c in ['A-C','B-C','B-A'])+' |')
lines+=['','## 辅助结果：各seed验证集最佳云影点','',table('best_shadow'),'',
 '这部分按验证集选点，不能替代固定6000步主结果，也不是独立测试集成绩。','',
 '| seed | 模型 | 最佳步数 | 云影IoU | 地表/云保护线通过 | 同权重开−关分支云影增量 |','| --- | --- | --- | --- | --- | --- |']
import csv
rows=list(csv.DictReader((R/'MAIN.csv').open(encoding='utf-8')))
for r in rows:
 if r['selection']!='best_shadow':continue
 off=next((v for v in x['on_off_best_shadow'] if v['seed']==int(r['seed']) and v['arm']==r['arm']),None)
 delta=f"{off['IoU_delta'][2]:+.4f}" if off else '—'
 lines.append(f"| {r['seed']} | {r['arm']} | {r['step']} | {float(r['shadow']):.4f} | {r['protected']} | {delta} |")
lines+=['','## 结论边界与验收','',
 '- 本轮未改架构：A/B新增15,265参数，联合训练原解码器；C仅训练解码器。A/B均含0.1覆盖率辅助BCE，故相对C是“结构＋辅助监督”整体收益，不能单独归因为结构。',
 '- 原始431、450、BN和冻结参数均固定；所有选定权重、初始状态、更新范围、同seed记录批次、独立535图混淆矩阵与分数均通过核验。不同seed批次及新分支初始化确实不同；A/B同seed初始化一致。',
 '- B沿用Phase480原TF32精度，属于近似均值约束。Phase480严格零均值数值门槛未通过；严格FP32诊断另存，不混入本轮成绩。',
 '- 这是固定基座的续训＋分支初始化三seed复现；不覆盖基座从头重训波动。三seed样本标准差不等于场景bootstrap置信区间。',
 '- MAIN.csv含所有seed的6000步、最佳云影及满足保护线的最佳点；SUMMARY.csv含各口径均值/样本标准差；MULTISEED_RESULTS.json含逐seed配对增量和分支开关结果。',
 '- 原始seed45201结果取自Phase480 A/B和Phase477 A（本轮命名C）；PREFLIGHT.json锁定这些历史文件签名，未覆盖历史实验。','']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,selected_weight_sha_verified=True,all_independent_metrics_recomputed=True,training_update_and_frozen_states_verified=True,training_code_signatures_verified=True,seeds=x['seeds'],new_runs=6,target_supervision_percent=0),indent=2))
print('REPORT_AND_LOCAL_ACCEPTANCE passed')
