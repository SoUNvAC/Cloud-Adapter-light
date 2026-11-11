from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase481/phase481_20261108'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
x=read(R/'MULTISEED_RESULTS.json')
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8')
assert '## Phase481' not in s and '## 后续维护规则' in s
lines=['## Phase481：Phase480三seed复现（2026-11-08，完成）','','- 固定431+450起点；CloudSEN训练8490/原始验证535、107ROI、目标监督0%。复用45201的Phase480 A/B、Phase477 A（C），新增45202/45203，分支初始化48002/48003；新增六次完整6000步训练。','- 原结构、损失、学习率及默认TF32执行不变；A/B新增15265参数并联合训练解码器，C仅训练解码器。原参数/450/BN冻结，同seed A/B初始化和A/B/C记录批次配对。','- 主口径统一6000步，均值±样本标准差（ddof=1，n=3）：']
for r in x['summary']:
 if r['selection']=='last_6000':lines.append('- '+r['arm']+'：地表/云/影/mIoU '+ '/'.join(f"{r[n+'_mean']:.6f}±{r[n+'_sd']:.6f}" for n in ['ground','cloud','shadow','mIoU'])+'。')
for v in x['paired']:
 if v['selection']=='last_6000':lines.append(f"- {v['comparison']}：云影增量{v['mean'][2]:+.6f}±{v['sample_sd'][2]:.6f}个百分点，正收益{v['positive_count'][2]}/3；逐seed云影增量"+str([z[2] for z in v['delta_by_seed']])+'。')
lines+=['- 结论：A/B对仅解码器C的正收益在三seed复现，B云影均值更高且波动更小，本轮优先保留B；各seed地表/云保护线均通过，70%未达到。最佳点与同权重关分支详见RESULT/MAIN/MULTISEED_RESULTS。','- A/B对C包含新增结构及0.1覆盖率辅助BCE整体效应；仅覆盖固定基座下续训/分支初始化随机性。B仍为Phase480原精度下近似均值约束，不混入严格FP32诊断。','- 训练代码未变、初始权重、更新范围、冻结状态、批次配对、独立535图复算、选定权重及传输签名均通过。','- 结果：`../outputs/phase481/phase481_20261108/RESULT.md`；计划：`research_plans/phase481.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8')
files=list((ROOT/'src/tools').glob('*481.py'))+list((ROOT/'src/tools').glob('*481.sh'))+[ROOT/'src/research_plans/phase481.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('HISTORY_AND_CODE_SHA written')
