"""Archive target-first architecture screening result."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase410/phase410_20261101'
def main():
 acceptance=json.loads((R/'INDEPENDENT_ACCEPTANCE.json').read_text());end=json.loads((R/'COMPLETE.json').read_text())
 assert acceptance['passed'] and end['passed']
 with (R/'summary.csv').open() as f:summary=list(csv.DictReader(f))
 with (R/'comparisons.csv').open() as f:deltas=list(csv.DictReader(f))
 status=f"完成：2026-11-01，9886次完整前向、8240次更新，{end['seconds']/60:.2f}分钟，独立验收通过。结果见 outputs/phase410/phase410_20261101/RESULT.md。未启动411。"
 p=ROOT/'src/research_plans/phase410.md';text=p.read_text(encoding='utf-8')
 if status not in text:p.write_text(text+'\n'+status+'\n',encoding='utf-8')
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 (R/'HANDOFF.md').write_text('# Phase410 已完成\n\n'+status+'\n权重留 /home/scv/shared/phase410_20261101。源域结果仅附表，不作筛选门槛。\n',encoding='utf-8')
 p=ROOT/'src/EXPERIMENT_LOG.md';raw=p.read_bytes();marker='## 后续维护规则'.encode();title='## Phase410：MsRE自适应尺度融合（2026-11-01）'
 assert raw.count(marker)==1
 if title.encode() not in raw:
  entry=title+'\n\n- 回到目标域性能优先。固定三尺度平均对比逐位置softmax尺度融合，block2/5/8/11，H关闭；新增1155参数，原版/改版训练参数354433/355588。seed40901，初始化/每步输入/增强/学习率匹配，短跑120后重置正式4000步；153 fit、32 development，每500步评价，best按目标mIoU选，final另报。\n'
  for v in summary:entry+=f"- {v['mode']} {v['role']} step{v['step']}：目标mIoU {float(v['target_mIoU']):.4f}，Shadow IoU {float(v['Shadow_IoU']):.4f}，宏AP {float(v['macro_AP']):.4f}。\n"
  entry+=f"- 自适应−固定平均best目标差{float(deltas[0]['target_mIoU']):+.4f}、final差{float(deltas[1]['target_mIoU']):+.4f}点；预设+0.30点晋级线{'达到' if acceptance['target_based_promotion'] else '未达到'}。历史400仅跨seed参考。本轮单seed、已暴露development，不证明稳定性/独立泛化/创新性。\n"
  entry+=f"- 9886次完整前向、8240次更新，控制器{end['seconds']:.1f}秒。等权初始化/梯度/恒等路径、配对身份与轨迹、冻结/参数梯度、调用/选模及逐组汇总指标验收通过。源域仅选模后535图附表；新点池化AP未独立像元复算。\n"
  entry+='- 训练后汇总存在旧目录名残留，已保留失败记录与原始执行代码，仅恢复CPU报告；新增训练/完整前向均为0。正式初始权重重建及训练后门控非零检查通过。\n'
  entry+='- 交付outputs/phase410/phase410_20261101/RESULT.md、summary.csv、完整TARGET_SUMMARY.csv/TARGET_GROUPS.csv、timing.csv及验收证据。未启动411。\n\n'
  if not acceptance['target_based_promotion']:
   entry+='决策：本轮门控未达到目标性能晋级线，保留原版；源域不参与决策。\n\n'
  p.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'phase':410,'calls':9886,'updates':8240,'target_based_promotion':acceptance['target_based_promotion'],'source_is_selection_gate':False,'next_phase_started':False},indent=2))
 code={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/tools').glob('*phase410*.py'))}
 (R/'LOCAL_CODE_LOCK.json').write_text(json.dumps(code,indent=2))
 manifest={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(R.rglob('*')) if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 print('Phase410 sealed:',len(manifest),'files')
if __name__=='__main__':main()
