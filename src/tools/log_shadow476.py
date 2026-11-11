from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase476/phase476_20261107'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase476' not in s
lines=['## Phase476：材质引导空间卷积（2026-11-07，完成）','','- 动机：通过亮度弱敏感的邻域引导跨越阴影明暗边界，检验云影/暗地物的特征区分能力。目标保护450地表/云且影70%。目标监督0%，CloudSEN训练8490/验证535、512²、FP32、RTX4090D。','- 在空间CNN1/4细节上加入8邻居逐通道差分卷积及1×1混合，并池化修正1/16上下文，位于全部VFM交互之前。A按logRGB距离引导，B按去通道均值log色度距离引导。各4609新参数，零差分初始化。色度只是材质线索，不等于物理反射率。','- 两组各6000步、batch4、seed45201/模块seed47601，新模块lr1e-3、原解码器1e-5、warmup200；其他原参数、450分支、BN冻结。统一不加权CE及已有保持约束，固定argmax。']
for arm in ['A','B']:
 d=R/arm/'network_run';f=read(d/'FINAL_RESULTS.json');x=next(v for v in f['results'] if 'best_shadow' in v['roles']);done=read(d/'COMPLETE.json');err=read(d/'BOUNDARY_DIAG.json')['delta476minus450'];update=read(d/'UPDATE_CHECK.json');last=read(d/'HISTORY.json')[-1]
 lines.append(f"- {arm}最优{x['step']}步：地表/云/影"+'/'.join(f'{v:.6f}' for v in x['IoU'])+f"，mIoU{x['mIoU']:.6f}；相对450三类变化"+'/'.join(f'{v:+.6f}' for v in x['IoU_delta'])+f"点。漏检净变化{sum(err['FN'])}、误报净变化{sum(err['FP'])}像素；末点影IoU{last['IoU'][2]:.6f}。训练{done['seconds']/60:.2f}分钟，峰值分配显存{update['peak_memory_bytes']}字节。")
c=read(R/'MATCHED_COMPARISON.json');lines.append('- B−A影'+f"{c['B_minus_A_IoU_delta'][2]:+.6f}"+'点，107ROI区间'+str(c['CI95'][2])+'。')
for k in ['A_convolution_on_minus_off','B_contrast_on_minus_off','B_appearance_on_minus_off']:
 lines.append('- '+k+'：影'+f"{c[k]['IoU_delta'][2]:+.6f}"+'点，区间'+str(c[k]['CI95'][2])+'。分别为空间卷积开关、色度对logRGB引导。')
diag=read(R/'CONTRAST_DIAG.json');lines.append('- B细节特征修正相对RMS均值'+str(diag['mean_relative_detail_rms'])+'；预测复现与权重不变检查通过。')
ex=read(R/'CANDIDATE_EXPORT.json');lines.append('- A/B移除新层后云影IoU均更高，未证明新材质层带来云影收益。保留B关层候选：地表/云/影'+ '/'.join(f'{v:.6f}' for v in ex['metrics']['IoU'])+'；候选移除material参数后535图逐像素复现。下一步补仅解码器同设置训练。')
decision=next(x for x in (R/'RESULT.md').read_text(encoding='utf-8').splitlines() if x.startswith('决策：'));lines+=['- '+decision,'- 535图独立评测、107ROI配对区间、同权重干预、3个新参数张量及原解码器更新、其余原参数冻结、初始化/记录批次/权重SHA通过。旧branch_updated字段按组名生成，验收改用独立张量对比，原记录保留。单seed探索。','- 结果：`../outputs/phase476/phase476_20261107/RESULT.md`；网络：`tools/shadow_material476.py`；计划：`research_plans/phase476.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8')
