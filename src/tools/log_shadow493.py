from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase493/phase493_20261109'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8')
assert '## Phase493' not in s and '## 后续维护规则' in s
lines=['## Phase493：下采样前亮暗次序纹理（2026-11-09，完成）','',
'- 保留488B，新增下采样前亮暗次序纹理支路：原图灰度在1/2/4像素半径比较八邻域，先编码再4倍池化，并与周围12–28像素环形邻域比较。A取亮度差符号，B取tanh(20×亮度差)，同参数对照。48通道经卷积和零投影注入第一层，新增13456参数，总模块36993参数，不增加骨干前向。次序编码参考[Census局部变换](https://woodfill.com/Papers/Census94.pdf)，本轮检验其在云影网络中的作用。',
 '- 两组从431＋450共同起点重训6000步，batch4、seed45201/分支seed48001；分支lr1e-3/解码器1e-5，warmup200，原骨干、450及BN固定；CloudSEN训练8490/原始验证535、107ROI，目标监督0%，固定argmax。',
 '- 直接对照为同seed、同6000步的488B；完整535图独立评测、107ROI配对区间、整体支路关闭及新次序纹理支路单独关闭、边界/内部错误统计。']
for arm in ['A','B']:
 d=R/arm/'network_run';f=read(d/'FINAL_RESULTS.json')
 for v in f['results']:lines.append(f"- {arm}/{v['step']}：地表/云/影 "+'/'.join(f'{z:.6f}' for z in v['IoU'])+f"，mIoU{v['mIoU']:.6f}；角色{v['roles']}。")
 z=read(R/'MATCHED_COMPARISON.json')[f'493{arm}_6000_minus_488B_6000']
 lines.append(f"- {arm}6000步相对488B云影{z['IoU_delta'][2]:+.6f}点，107ROI区间{z['CI95'][2]}；训练{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟。")
lines+=['',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'- 原有参数初始化/记录批次、更新/冻结、独立混淆矩阵、权重和传输签名通过。单seed验证集探索；TF32均值约束为近似实现。','- 结果：`../outputs/phase493/phase493_20261109/RESULT.md`；模块：`tools/shadow_redistribution493.py`；计划：`research_plans/phase493.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8');print('HISTORY written')
