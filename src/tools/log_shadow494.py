from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase494/phase494_20261109'
assert json.loads((R/'LOCAL_ACCEPTANCE.json').read_text())['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8')
assert '## Phase494' not in s and '## 后续维护规则' in s
x=json.loads((R/'MULTISEED_RESULTS.json').read_text());pair=next(v for v in x['paired'] if v['selection']=='last_6000')
lines=['## Phase494：493B三seed复现（2026-11-09，完成）','',
'- 不改结构、损失或决策；复用493B的45201/48001，新增45202/48002、45203/48003，与488B/489对应seed逐一配对。',
'- 固定431＋450起点，6000步、batch4；CloudSEN训练8490/原始验证535，目标监督0%；原骨干/450/BN冻结，固定argmax。',
(R/'INTERPRETATION.md').read_text(encoding='utf-8'),
'- 逐seed固定6000步增量（地表/云/影/mIoU）：']
for seed,delta in zip(pair['seeds'],pair['delta_by_seed']):lines.append(f"  - {seed}: "+'/'.join(f'{v:+.6f}' for v in delta))
lines += ['- 模块逐字节相同、历史引用签名、公共初始化、记录批次、更新冻结、独立评测混淆矩阵及远端权重验收通过。',
'- 结果：`../outputs/phase494/phase494_20261109/RESULT.md`；计划：`research_plans/phase494.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8');print('HISTORY written')
