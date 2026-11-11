from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase478/phase478_20261108';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase478' not in s and '## 后续维护规则' in s
lines=['## Phase478：原特征路径可塑性对照（2026-11-08，完成）','','- A开放原MsRE/refinement、local、output_residual及解码器；B开放原空间CNN、交叉注意力及解码器。新增参数0，VFM、450分支、BN固定；本轮是训练范围对照，不称结构创新。','- 源训练8490/原始验证535、512²、FP32、RTX4090D、目标监督0%；450共同起点，6000步、batch4、seed45201、lr1e-5、warmup200，与477相同不加权CE及保持约束、固定argmax。','- 目标：较477仅解码器获得额外云影收益；保护450地表/云，云影目标70%。1000/3000/6000及最佳点独立535图复算。']
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');cmp=read(R/'MATCHED_COMPARISON.json')
for arm in ['A','B']:
 d=R/arm/'network_run';f=read(d/'FINAL_RESULTS.json');done=read(d/'COMPLETE.json');u=read(d/'UPDATE_CHECK.json')
 for x in f['results']:lines.append(f"- {arm}/{x['step']}步：地表/云/影"+'/'.join(f'{v:.6f}' for v in x['IoU'])+f"，mIoU{x['mIoU']:.6f}；角色"+','.join(x['roles'])+'。')
 lines.append(f"- {arm}训练参数{accept['arms'][arm]['task_parameters']}，并行墙钟{done['seconds']/60:.2f}分钟，峰值分配显存{u['peak_memory_bytes']}字节。")
 for step in [1000,3000,6000]:
  k=f'478{arm}_{step}_minus_477_{step}';v=cmp[k];lines.append(f"- {k}：云影{v['IoU_delta'][2]:+.6f}点，107ROI配对95%区间{v['CI95'][2]}。")
lines += ['- '+x for x in (R/'RESULT.md').read_text(encoding='utf-8').splitlines() if x.startswith(('结论：','决策：'))]
lines+=['- 参数更新/冻结、共同初始化/记录批次、独立混淆矩阵、权重及传输SHA通过；单seed验证集探索。','- 结果：`../outputs/phase478/phase478_20261108/RESULT.md`；训练：`tools/train_shadow478.py`；计划：`research_plans/phase478.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8')
