from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase489/phase489_20261108';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed'];x=read(R/'MULTISEED_RESULTS.json')
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase489' not in s and '## 后续维护规则' in s
lines=['## Phase489：Phase488 B三seed复现（2026-11-08，完成）','','- 原始梯度纹理支路与488逐字节一致；固定431＋450起点，复用45201/48001，新增45202/48002、45203/48003，与482B/483同seed配对。每组6000步、batch4、分支lr1e-3/解码器1e-5；骨干/450/BN固定；CloudSEN训练8490/原始验证535、目标监督0%，固定argmax。','- 固定6000步均值±样本标准差：']
for r in x['summary']:
 if r['selection']=='last_6000':lines.append('- '+r['method']+'：地表/云/影/mIoU '+ '/'.join(f"{r[n+'_mean']:.6f}±{r[n+'_sd']:.6f}" for n in ['ground','cloud','shadow','mIoU'])+'。')
for v in x['paired']:
 if v['selection']=='last_6000':lines.append(f"- N−S云影增量{v['mean'][2]:+.6f}±{v['sample_sd'][2]:.6f}点，正收益{v['positive_count'][2]}/3，三类同时提升{v['all_classes_improve_count']}/3。")
lines+=['',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'- 代码、公共初始化/记录批次、更新冻结、独立535图复算、权重签名验收通过。三seed仅覆盖固定基座续训随机性。','- 结果：`../outputs/phase489/phase489_20261108/RESULT.md`；计划：`research_plans/phase489.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8');print('HISTORY written')
