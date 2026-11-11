from pathlib import Path
import json,hashlib,csv,statistics
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase483/phase483_20261108';read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=read(R/'MULTISEED_RESULTS.json');a=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert x['passed'] and a['passed'] and read(R/'TRANSFER_ACCEPTED.json')['passed']
for seed,v in a['runs'].items():
 for n,w in v['weights'].items():
  if n!='branch_0.pth':assert sha(R/f'seed_{seed}'/'B/network_run'/n)==w['sha256']
rows=list(csv.DictReader((R/'MAIN.csv').open(encoding='utf-8')))
for s in x['summary']:
 rr=[r for r in rows if r['method']==s['method'] and r['selection']==s['selection']];assert len(rr)==3
 for n in ['ground','cloud','shadow','mIoU']:
  v=[float(r[n]) for r in rr];assert abs(statistics.mean(v)-s[n+'_mean'])<1e-10 and abs(statistics.stdev(v)-s[n+'_sd'])<1e-10
names={'C':'仅解码器','S':'有覆盖率辅助损失','N':'无覆盖率辅助损失'}
def pair(c,selection='last_6000'):return next(v for v in x['paired'] if v['comparison']==c and v['selection']==selection)
def table(selection):
 lines=['| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- |']
 for s in x['summary']:
  if s['selection']==selection:lines.append('| '+names[s['method']]+' | '+' | '.join(f"{s[n+'_mean']:.4f} ± {s[n+'_sd']:.4f}" for n in ['ground','cloud','shadow','mIoU'])+' |')
 return lines
lines=['# Phase483：无覆盖率辅助损失版的三seed复现','',
 '固定431＋450起点，CloudSEN训练8490/原始验证535张、107ROI，目标域监督0%。复用45201的482B，新增45202/45203两次6000步训练，分支初始化48002/48003；与同seed的480B/481B（有辅助损失）、477/481C（仅解码器）比较。网络与482B一致，无新结构。','',
 '主结果统一6000步，均值±样本标准差（n=3，ddof=1）：','',*table('last_6000'),'','| 配对比较 | 云影平均增量±标准差 | 云影正收益 | 三类同时提升 |','| --- | --- | --- | --- |']
for c in ['N-S','N-C','S-C']:
 p=pair(c);lines.append(f"| {c} | {p['mean'][2]:+.4f} ± {p['sample_sd'][2]:.4f} | {p['positive_count'][2]}/3 | {p['all_classes_improve_count']}/3 |")
lines+=['','N=无辅助损失，S=有辅助损失，C=仅解码器。','', '| seed | N−S地表 | N−S云 | N−S云影 | N−C云影 |','| --- | --- | --- | --- | --- |']
for i,seed in enumerate(x['seeds']):lines.append(f'| {seed} | '+' | '.join(f'{z:+.4f}' for z in [*pair('N-S')['delta_by_seed'][i][:3],pair('N-C')['delta_by_seed'][i][2]])+' |')
lines+=['','辅助结果：各seed验证集最佳云影点（验证选点，不替代统一6000步主结果）。','',*table('best_shadow'),'','| seed | 无辅助损失版最佳步数 | 云影IoU | 同权重开−关分支云影增量 |','| --- | --- | --- | --- |']
for v in x['on_off_best_shadow']:
 r=next(r for r in rows if int(r['seed'])==v['seed'] and r['method']=='N' and r['selection']=='best_shadow')
 lines.append(f"| {v['seed']} | {v['step']} | {float(r['shadow']):.6f} | {v['IoU_delta'][2]:+.6f} |")
lines+=['','验收：模块与482逐字节相同；同seed初始可学习参数及记录批次与有辅助损失对照一致；新增分支13个张量和原解码器更新，原骨干/450/BN固定；独立535图混淆矩阵、指标、全部选定权重和签名核验通过。','',
 '本轮仅覆盖固定基座的续训与分支初始化随机性。三seed标准差与场景置信区间不同。沿用默认TF32精度，块均值约束仍按近似实现解释。无辅助监督的门控值不直接解释为校准覆盖率。', '',
 'MAIN.csv保留逐seed、固定6000步及验证最佳/保护线最佳成绩；SUMMARY.csv给出所有均值和标准差；MULTISEED_RESULTS.json给出逐seed配对增量。', '']
if (R/'INTERPRETATION.md').exists():lines.insert(2,(R/'INTERPRETATION.md').read_text(encoding='utf-8')+'\n')
for seed in [45202,45203]:
 d=R/f'seed_{seed}'/'B/network_run';lines.append(f"新增seed{seed}：并行训练含阶段验证墙钟{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟，峰值分配显存{read(d/'UPDATE_CHECK.json')['peak_memory_bytes']}字节。")
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,selected_weights_verified=True,summary_independently_recomputed=True,seeds=x['seeds'],target_supervision_percent=0,new_full_training_runs=2,training_and_metric_acceptance=True),indent=2))
files=list((ROOT/'src/tools').glob('*483.py'))+list((ROOT/'src/tools').glob('*483.sh'))+[ROOT/'src/research_plans/phase483.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2));print('REPORT_AND_LOCAL_ACCEPTANCE passed')
