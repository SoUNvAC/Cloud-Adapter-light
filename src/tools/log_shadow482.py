from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase482/phase482_20261108';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase482' not in s and '## 后续维护规则' in s
lines=['## Phase482：云影占比对比重分配（2026-11-08，完成）','','- A在480B基础上将细节权重p换为p−本patch平均p，再保留块均值约束，辅助覆盖率BCE为0.1；B保留原480B结构，将辅助覆盖率BCE设为0。两组15265分支参数，原解码器同时训练。','- 原431+450共同起点，CloudSEN训练8490/原始验证535、107ROI、目标监督0%；6000步、batch4、seed45201/分支seed48001、warmup200，分支lr1e-3/解码器lr1e-5，原骨干/450/BN冻结，FP32张量与原TF32默认执行不变，固定argmax。','- 以同seed同一步480B为直接对照；升级要求地表/云不降且影提升。完整535图独立评测、107ROI配对区间、最佳点关分支、A只关占比对比、边界/内部漏检误报诊断。']
for arm in ['A','B']:
 d=R/arm/'network_run';f=read(d/'FINAL_RESULTS.json')
 for v in f['results']:lines.append(f"- {arm}/{v['step']}：地表/云/影 "+'/'.join(f'{z:.6f}' for z in v['IoU'])+f"，mIoU{v['mIoU']:.6f}；角色{v['roles']}。")
 z=read(R/'MATCHED_COMPARISON.json')[f'482{arm}_6000_minus_480B_6000'];lines.append(f"- {arm}6000步相对480B影{z['IoU_delta'][2]:+.6f}点，107ROI区间{z['CI95'][2]}；训练{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟，峰值分配显存{read(d/'UPDATE_CHECK.json')['peak_memory_bytes']}字节。")
lines+=['',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'- 共同初始化/记录批次、更新/冻结、独立混淆矩阵、权重与传输签名通过。单seed验证集探索，不代替多seed结论；默认TF32下仍按近似块均值约束解释。','- 结果：`../outputs/phase482/phase482_20261108/RESULT.md`；模块：`tools/shadow_redistribution482.py`；计划：`research_plans/phase482.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8');print('HISTORY written')
