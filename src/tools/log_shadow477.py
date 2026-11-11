from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase477/phase477_20261107';D=R/'A/network_run';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase477' not in s
f=read(D/'FINAL_RESULTS.json');best=next(x for x in f['results'] if 'best_shadow' in x['roles']);done=read(D/'COMPLETE.json');update=read(D/'UPDATE_CHECK.json');accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');cmp=read(R/'MATCHED_COMPARISON.json')
lines=['## Phase477：仅解码器续训归因对照（2026-11-07，完成）','','- 从450同一起点，只开放原解码器，新增参数0；骨干、450分支、BN冻结。源训练8490/验证535、512²、FP32、RTX4090D、目标监督0%。6000步、batch4、seed45201，lr1e-5、warmup200；与476相同不加权CE及保持约束，固定argmax。','- 目标：核查476关层候选的收益能否仅靠解码器续训获得。共同初始权重与记录批次均与476A/B核验一致；1000/3000/6000及最佳点均独立复算535图。']
for x in f['results']:
 lines.append(f"- {x['step']}步：地表/云/影"+'/'.join(f'{v:.6f}' for v in x['IoU'])+f"，mIoU{x['mIoU']:.6f}；角色"+','.join(x['roles'])+'。')
for k in ['477best_minus_476B_off','477_1000_minus_476A_off','477_3000_minus_476B_off']:
 lines.append('- '+k+'：影'+f"{cmp[k]['IoU_delta'][2]:+.6f}"+'点，107ROI区间'+str(cmp[k]['CI95'][2])+'。')
lines.append(f"- 训练参数{accept['task_parameters']}，耗时{done['seconds']/60:.2f}分钟，峰值分配显存{update['peak_memory_bytes']}字节。")
lines.append('- 归因：仅解码器即获得三类提升；同3000步较476B关层影低0.054821点，区间跨0，地表/云/mIoU更高。尚未证明材质层训练过程有稳定额外云影收益。后续结构比较采用匹配的仅解码器续训对照，476关层候选继续保留。')
decision=next(x for x in (R/'RESULT.md').read_text(encoding='utf-8').splitlines() if x.startswith('决策：'));lines+=['- '+decision,'- 参数更新/冻结、535图独立混淆矩阵、初始化/记录批次和权重SHA通过；单seed探索。','- 结果：`../outputs/phase477/phase477_20261107/RESULT.md`；训练：`tools/train_shadow477.py`；计划：`research_plans/phase477.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8')
