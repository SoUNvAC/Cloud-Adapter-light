from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase479/phase479_20261108';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
assert read(R/'LOCAL_ACCEPTANCE.json')['passed']
p=ROOT/'src/EXPERIMENT_LOG.md';s=p.read_text(encoding='utf-8');assert '## Phase479' not in s and '## 后续维护规则' in s
lines=['## Phase479：多尺度暗区包络分支（2026-11-08，完成）','','- 在空间CNN后、交互前新增4992参数分支：logRGB的5/13/29尺度暗区RGB差、平均差、通道离散度，与64维detail融合后写入detail/context。A均值参考，B闭运算参考；零末层初始化，A/B初始可学习参数相同。','- 源训练8490/原始验证535、512²、FP32、RTX4090D、目标监督0%；450起点、6000步、batch4、seed45201、分支seed47901、warmup200，新分支lr1e-3/解码器lr1e-5，其余原参数及450分支/BN冻结；损失与477相同，固定argmax。','- 目标：在450地表/云保护线下提高云影并向70%推进；区分477同设置续训、均值对照与形态学参考贡献。1000/3000/6000及最佳点独立535图，最佳点同权重关分支；107ROI配对区间。']
for arm in ['A','B']:
 d=R/arm/'network_run'
 for kind,file in [('on','FINAL_RESULTS.json'),('off','ABLATION_OFF.json')]:
  for x in read(d/file)['results']:lines.append(f"- {arm}/{kind}/{x['step']}：地表/云/影"+'/'.join(f'{v:.6f}' for v in x['IoU'])+f"，mIoU{x['mIoU']:.6f}。")
 done=read(d/'COMPLETE.json');u=read(d/'UPDATE_CHECK.json');lines.append(f"- {arm}并行墙钟{done['seconds']/60:.2f}分钟，峰值分配显存{u['peak_memory_bytes']}字节。")
lines+=['',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'- 新参数/解码器更新、其余原参数/BN冻结，共同初始化/记录批次与权重签名验收通过；单seed验证集探索。','- 形态学闭运算/黑顶帽为已有方法，研究本轮网络融合，不声称算子原创。','- 结果：`../outputs/phase479/phase479_20261108/RESULT.md`；模块：`tools/shadow_envelope479.py`；计划：`research_plans/phase479.md`。','']
p.write_text(s.replace('## 后续维护规则','\n'.join(lines)+'\n## 后续维护规则'),encoding='utf-8')
