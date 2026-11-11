from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase474/phase474_20261107'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase474' not in s
lines=['## Phase474：云内容—地表遮光条件解码（2026-11-07，完成）','','- 动机：检验类别层级能否减少影分支对云的竞争，门槛保持450地表/云IoU且影70%。目标监督0%，CloudSEN训练8490/原始验证535，输入512²、FP32、RTX4090D，两组并行6000步、batch4、seed45201、lr1e-4、warmup200。','- 原450完整网络与BN全部冻结。两条独立CNN/语义投影，云内容路读取场景，地表遮光路读取三尺度局部亮度/色度/纹理。每组31106新参数。A普通三类残差组合，B云/非云与非云内影/地表条件组合。两组同初始化/损失/记录批次，无辅助标签损失，固定argmax。','- B的影分支不直接改变云概率；最终argmax及云IoU仍需实测。冻结参考同次前向缓存与单独推理完全一致。']
for arm in ['A','B']:
 d=R/arm/'network_run';f=read(d/'FINAL_RESULTS.json');x=next(v for v in f['results'] if 'best_shadow' in v['roles']);done=read(d/'COMPLETE.json');err=read(d/'BOUNDARY_DIAG.json')['delta474minus450'];update=read(d/'UPDATE_CHECK.json')
 lines.append(f"- {arm}最优{x['step']}步：地表/云/影"+'/'.join(f'{v:.6f}' for v in x['IoU'])+f"，mIoU{x['mIoU']:.6f}；相对450三类变化"+'/'.join(f'{v:+.6f}' for v in x['IoU_delta'])+f"点。漏检净变化{sum(err['FN'])}、误报净变化{sum(err['FP'])}像素。训练{done['seconds']/60:.2f}分钟，峰值分配显存{update['peak_memory_bytes']}字节。")
c=read(R/'MATCHED_COMPARISON.json')
lines.append('- B−A云影'+f"{c['B_minus_A_IoU_delta'][2]:+.6f}"+'点，107ROI配对区间'+str(c['CI95'][2])+'。')
for k in ['B_decoder_on_minus_off','B_conditional_on_minus_off','B_spatial_minus_mean']:
 lines.append('- '+k+'：影'+f"{c[k]['IoU_delta'][2]:+.6f}"+'点，区间'+str(c[k]['CI95'][2])+'。')
decision=next(x for x in (R/'RESULT.md').read_text(encoding='utf-8').splitlines() if x.startswith('决策：'));lines+=['- '+decision,'- 独立535图评测、三项同权重干预、参数更新与冻结、初始化/记录批次/权重SHA通过。单seed验证集选点探索。','- 结果：`../outputs/phase474/phase474_20261107/RESULT.md`；代码：`tools/shadow_conditional474.py`；计划：`research_plans/phase474.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8')
