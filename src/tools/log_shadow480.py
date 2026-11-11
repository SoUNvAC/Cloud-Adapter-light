from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase480/phase480_20261108';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase480' not in s and '## 后续维护规则' in s
lines=['## Phase480：云影亚patch重分配（2026-11-08，完成）', '', '- 第一层VFM金字塔前新增15265参数分支：1/16语义与1/4 CNN细节融合，覆盖率预测软门控细节并投影384维；A普通残差，B约束每4×4子格均值为0，末层无bias零初始化。', '- 原431+450起点；CloudSEN训练8490/验证535、512²、FP32、4090D、目标监督0%。6000步、batch4、seed45201、分支seed48001，warmup200；新分支lr1e-3/解码器lr1e-5，其余原参数、450分支与BN冻结。', '- A/B相同分割保持损失加0.1覆盖率BCE；477没有新分支及辅助损失。固定argmax，保护450地表/云、目标影70%。1000/3000/6000及最佳点独立535图，最佳点同权重关分支、107ROI配对区间及B均值保持复核。']
for arm in ['A','B']:
 d=R/arm/'network_run'
 for kind,file in [('on','FINAL_RESULTS.json'),('off','ABLATION_OFF.json')]:
  for x in read(d/file)['results']:lines.append(f"- {arm}/{kind}/{x['step']}：地表/云/影"+'/'.join(f'{v:.6f}' for v in x['IoU'])+f"，mIoU{x['mIoU']:.6f}。")
 done=read(d/'COMPLETE.json');u=read(d/'UPDATE_CHECK.json');lines.append(f"- {arm}并行墙钟{done['seconds']/60:.2f}分钟，峰值分配显存{u['peak_memory_bytes']}字节。")
lines+=['',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'- 新参数/解码器更新、其余原参数/BN冻结，共同初始化/记录批次与权重签名验收通过；单seed验证集探索。','- 细节引导上采样参考FADE，本轮测试云影覆盖率与patch均值保持组合。','- 结果：`../outputs/phase480/phase480_20261108/RESULT.md`；模块：`tools/shadow_redistribution480.py`；计划：`research_plans/phase480.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8')
