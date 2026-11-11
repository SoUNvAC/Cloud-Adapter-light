from pathlib import Path
import json,hashlib,csv,statistics
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase489/phase489_20261108'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=read(R/'MULTISEED_RESULTS.json');a=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json')
assert x['passed'] and a['passed'] and read(R/'TRANSFER_ACCEPTED.json')['passed']
for seed,v in a['runs'].items():
 for n,w in v['weights'].items():
  if n!='branch_0.pth':assert sha(R/f'seed_{seed}'/'B/network_run'/n)==w['sha256']
rows=list(csv.DictReader((R/'MAIN.csv').open(encoding='utf-8')))
for s in x['summary']:
 rr=[r for r in rows if r['method']==s['method'] and r['selection']==s['selection']];assert len(rr)==3
 for n in ['ground','cloud','shadow','mIoU']:
  v=[float(r[n]) for r in rr];assert abs(statistics.mean(v)-s[n+'_mean'])<1e-10 and abs(statistics.stdev(v)-s[n+'_sd'])<1e-10
names={'S':'482B/483基线','N':'488B/489纹理分支'}
def table(selection):
 lines=['| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- |']
 for s in x['summary']:
  if s['selection']==selection:lines.append('| '+names[s['method']]+' | '+' | '.join(f"{s[n+'_mean']:.4f} ± {s[n+'_sd']:.4f}" for n in ['ground','cloud','shadow','mIoU'])+' |')
 return lines
p=next(z for z in x['paired'] if z['selection']=='last_6000')
lines=['# Phase489：Phase488 B三seed复现','',
 '网络与Phase488逐字节一致，固定使用B原始梯度纹理证据，不采用亮度归一化。复用45201/48001，新增45202/48002、45203/48003；与482B/483相同seed基线配对。固定431＋450起点，6000步、batch4，骨干/450/BN冻结，原始CloudSEN训练8490/验证535、107ROI，目标监督0%，固定argmax。','']
if (R/'INTERPRETATION.md').exists():lines += [(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'']
lines += ['主结果：固定6000步，均值±样本标准差（n=3，ddof=1）。','',*table('last_6000'),'','| seed | 基线云影IoU | 纹理分支云影IoU | 地表增量 | 云增量 | 云影增量 | mIoU增量 |','| --- | --- | --- | --- | --- | --- | --- |']
for i,seed in enumerate(x['seeds']):
 get=lambda method:next(r for r in rows if int(r['seed'])==seed and r['method']==method and r['selection']=='last_6000')
 lines.append(f"| {seed} | {float(get('S')['shadow']):.6f} | {float(get('N')['shadow']):.6f} | "+' | '.join(f'{v:+.6f}' for v in p['delta_by_seed'][i])+' |')
lines += ['',f"三类及mIoU配对平均增量：{p['mean']}；增量样本标准差：{p['sample_sd']}；逐指标正收益次数：{p['positive_count']}；三类同时提升：{p['all_classes_improve_count']}/3。",'',
 '辅助结果：各seed训练验证预选最佳云影点，不代替固定6000步主结果。','',*table('best_shadow'),'','| seed | 预选步数 | 云影IoU | 新纹理支路开−关：地表/云/影 | 整体细节模块开−关：云影 |','| --- | --- | --- | --- | --- |']
for v in x['texture_on_off_best_shadow']:
 r=next(r for r in rows if int(r['seed'])==v['seed'] and r['method']=='N' and r['selection']=='best_shadow')
 full=next(z for z in x['on_off_best_shadow'] if z['seed']==v['seed'])
 lines.append(f"| {v['seed']} | {v['step']} | {float(r['shadow']):.6f} | "+'/'.join(f'{z:+.6f}' for z in v['IoU_delta'])+f" | {full['IoU_delta'][2]:+.6f} |")
lines += ['', '验收：网络模块逐字节相同；同seed公共初始参数和记录批次与基线一致；17个新增模块参数张量与原解码器更新，原骨干/450/BN固定；535图独立重载、混淆矩阵复算、历史引用及权重签名通过。','',
 '三seed仅覆盖固定基座下的续训、采样和分支初始化随机性，不覆盖源模型重训或新数据划分。标准差是seed间波动，不是场景置信区间。支路开关为同权重诊断，不能替代独立训练消融。最佳点来自验证集选点，连续多轮探索后的泛化仍需独立测试集。','']
for seed in [45202,45203]:
 d=R/f'seed_{seed}'/'B/network_run';lines.append(f"seed{seed}并行训练含阶段验证{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟，峰值分配显存{read(d/'UPDATE_CHECK.json')['peak_memory_bytes']}字节。")
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,selected_weights_verified=True,summary_independently_recomputed=True,seeds=x['seeds'],target_supervision_percent=0,new_full_training_runs=2,training_and_metric_acceptance=True),indent=2))
files=list((ROOT/'src/tools').glob('*489.py'))+list((ROOT/'src/tools').glob('*489.sh'))+[ROOT/'src/research_plans/phase489.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2));print('REPORT_AND_LOCAL_ACCEPTANCE passed')
