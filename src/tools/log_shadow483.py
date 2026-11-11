from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase483/phase483_20261108';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed'];x=read(R/'MULTISEED_RESULTS.json')
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase483' not in s and '## 后续维护规则' in s
lines=['## Phase483：无覆盖率辅助损失版三seed复现（2026-11-08，完成）','','- 网络与482B一致，新增分支15265参数＋原解码器联合训练，辅助BCE为0。固定431＋450起点；目标监督0%，CloudSEN训练8490/原始验证535、107ROI。复用45201，新增45202/45203，分支初始化48002/48003；两次6000步、batch4、warmup200、分支lr1e-3/解码器lr1e-5，原骨干/450/BN冻结，沿用默认TF32，固定argmax。','- 对照：S为480B/481B有辅助损失，C为477/481C仅解码器，N为482B/483B无辅助损失；同seed初始可学习参数和记录批次匹配。','- 固定6000步均值±样本标准差（n=3、ddof=1）：']
for r in x['summary']:
 if r['selection']=='last_6000':lines.append('- '+r['method']+'：地表/云/影/mIoU '+ '/'.join(f"{r[n+'_mean']:.6f}±{r[n+'_sd']:.6f}" for n in ['ground','cloud','shadow','mIoU'])+'。')
for v in x['paired']:
 if v['selection']=='last_6000':lines.append(f"- {v['comparison']}云影增量{v['mean'][2]:+.6f}±{v['sample_sd'][2]:.6f}点，正收益{v['positive_count'][2]}/3，三类同时提升{v['all_classes_improve_count']}/3。")
for seed in [45202,45203]:
 d=R/f'seed_{seed}'/'B/network_run';lines.append(f"- 新增seed{seed}：并行训练含阶段验证{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟，峰值分配显存{read(d/'UPDATE_CHECK.json')['peak_memory_bytes']}字节。")
lines+=['',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'- 模块字节一致、初始参数/记录批次、更新/冻结、独立535图复算、选定权重及签名验收通过。三seed仅覆盖固定基座的续训与分支初始化波动。','- 结果：`../outputs/phase483/phase483_20261108/RESULT.md`；计划：`research_plans/phase483.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8');print('HISTORY written')
