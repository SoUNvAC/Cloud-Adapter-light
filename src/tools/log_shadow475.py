from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase475/phase475_20261107'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase475' not in s
lines=['## Phase475：云影误报的损失权重对照（2026-11-07，完成）','','- 诊断训练目标，不是新架构。固定474A普通组合网络，原450与BN冻结，仅训练31106参数。目标监督0%，CloudSEN训练8490/验证535、512²、FP32、RTX4090D；新A/B各6000步、batch4、seed45201、模块seed47401、lr1e-4、warmup200，固定argmax。','- 旧474A：云影难例4、正确影1.5、其他1；新A：云影难例4、其他1；新B：全部1。其他KL、softIoU及正则相同。三组同初始权重与记录批次；对照单因素逐级移除，检验误报增长是否来自像素权重。']
for arm in ['A','B']:
 d=R/arm/'network_run';f=read(d/'FINAL_RESULTS.json');x=next(v for v in f['results'] if 'best_shadow' in v['roles']);done=read(d/'COMPLETE.json');err=read(d/'BOUNDARY_DIAG.json')['delta475minus450'];update=read(d/'UPDATE_CHECK.json');last=read(d/'HISTORY.json')[-1]
 lines.append(f"- {arm}最优{x['step']}步：地表/云/影"+'/'.join(f'{v:.6f}' for v in x['IoU'])+f"，mIoU{x['mIoU']:.6f}；相对450三类变化"+'/'.join(f'{v:+.6f}' for v in x['IoU_delta'])+f"点。漏检净变化{sum(err['FN'])}、误报净变化{sum(err['FP'])}像素。末点影IoU{last['IoU'][2]:.6f}；训练{done['seconds']/60:.2f}分钟，峰值分配显存{update['peak_memory_bytes']}字节。")
c=read(R/'MATCHED_COMPARISON.json');lines.append('- B−A影'+f"{c['B_minus_A_IoU_delta'][2]:+.6f}"+'点，107ROI区间'+str(c['CI95'][2])+'。')
for k in ['A_minus_474A','B_minus_474A','B_decoder_on_minus_off','B_spatial_minus_mean']:
 lines.append('- '+k+'：影'+f"{c[k]['IoU_delta'][2]:+.6f}"+'点，区间'+str(c[k]['CI95'][2])+'。')
decision=next(x for x in (R/'RESULT.md').read_text(encoding='utf-8').splitlines() if x.startswith('决策：'));lines+=['- '+decision,'- 独立535图、107ROI配对区间、B关闭新解码与空间均值干预、参数更新/冻结、三组初始化/记录批次及权重SHA通过。当前网络与单seed训练设置的诊断。','- 结果：`../outputs/phase475/phase475_20261107/RESULT.md`；训练：`tools/train_shadow475.py`；计划：`research_plans/phase475.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8')
