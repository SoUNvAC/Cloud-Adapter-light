from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase491/phase491_20261108'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8')
assert '## Phase491' not in s and '## 后续维护规则' in s
lines=['## Phase491：浅深层关系交互（2026-11-08，完成）','',
'- 保留488B，读取第1及第4交互位置输出的同网格特征，分别LayerNorm并投影到16维。A拼接浅深差分及逐通道乘积，B直接拼接两层投影作同参数对照，经卷积与零投影注入第一层特征。新增19088参数，总模块42625参数，不增加骨干前向。没有将乘积直接解释成真实材质保持量。',
 '- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。',
 '- 直接对照为同seed、同6000步的488B；完整535图独立评测、107ROI配对区间、整体支路关闭及新跨层支路单独关闭、边界/内部错误统计。']
for arm in ['A','B']:
 d=R/arm/'network_run';f=read(d/'FINAL_RESULTS.json')
 for v in f['results']:lines.append(f"- {arm}/{v['step']}：地表/云/影 "+'/'.join(f'{z:.6f}' for z in v['IoU'])+f"，mIoU{v['mIoU']:.6f}；角色{v['roles']}。")
 z=read(R/'MATCHED_COMPARISON.json')[f'491{arm}_6000_minus_488B_6000']
 lines.append(f"- {arm}6000步相对488B云影{z['IoU_delta'][2]:+.6f}点，107ROI区间{z['CI95'][2]}；训练{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟。")
lines+=['',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。','- 结果：`../outputs/phase491/phase491_20261108/RESULT.md`；模块：`tools/shadow_redistribution491.py`；计划：`research_plans/phase491.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8');print('HISTORY written')
