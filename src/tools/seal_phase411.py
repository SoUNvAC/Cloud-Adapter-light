"""Seal Phase411 experiment and requested cloud-shadow-specific architecture proposals."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase411/phase411_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=json.loads((R/'INDEPENDENT_ACCEPTANCE.json').read_text());end=json.loads((R/'COMPLETE.json').read_text())
 assert a['passed'] and end['passed'] and (R/'CLOUD_SHADOW_ARCHITECTURE.md').exists()
 with (R/'summary.csv').open() as f:summary=list(csv.DictReader(f))
 with (R/'comparisons.csv').open() as f:deltas=list(csv.DictReader(f))
 status=f"完成：2026-11-01，新增4408次完整前向、4120次更新，{end['seconds']/60:.2f}分钟。独立验收通过；原版对照复用Phase410。结果与云影结构建议见outputs/phase411/phase411_20261101/。"
 p=ROOT/'src/research_plans/phase411.md';text=p.read_text(encoding='utf-8')
 if status not in text:p.write_text(text+'\n'+status+'\n',encoding='utf-8')
 (R/'README.md').write_bytes((R/'RESULT.md').read_bytes())
 (R/'HANDOFF.md').write_text('# Phase411 已完成\n\n'+status+'\n权重留/home/scv/shared/phase411_20261101/layerwise。未启动云影新结构训练。\n',encoding='utf-8')
 p=ROOT/'src/EXPERIMENT_LOG.md';raw=p.read_bytes();marker='## 后续维护规则'.encode();title='## Phase411：MsRE分层残差控制与云影结构建议（2026-11-01）'
 assert raw.count(marker)==1
 if title.encode() not in raw:
  entry=title+'\n\n- 共享残差系数改为block2/5/8/11各一个，初值0.001，净增3参数。原串联、H关闭，seed40901；复用Phase410原版，改版短跑120后重置正式4000步，每500步验证。共有权重、输入、增强和学习率配对验收；源域不作门槛，本轮不重复源域评价。\n'
  for v in summary:entry+=f"- {v['mode']} {v['role']} step{v['step']}：目标mIoU {float(v['target_mIoU']):.4f}，Shadow IoU {float(v['Shadow_IoU']):.4f}，宏AP {float(v['macro_AP']):.4f}。\n"
  entry+=f"- 分层−原版best差{float(deltas[0]['target_mIoU']):+.4f}，final差{float(deltas[1]['target_mIoU']):+.4f}点；预设+0.30点筛选线{'达到' if a['target_based_promotion'] else '未达到'}。单seed且development参与选模，不能称稳定或独立泛化。\n"
  entry+='- best总体增益来自Surface/Cloud；Shadow IoU 41.2633→41.2615，宏AP 69.1695→68.5388，不能称云影改善，后期回落仍在。\n'
  entry+=f"- 新增4408次前向、4120次更新，墙钟{end['seconds']:.1f}秒；历史对照不重复计费。初始输出等价、分层独立梯度、参数/输入身份、轨迹、调用及指标验收通过。\n"
  entry+='- 另交CLOUD_SHADOW_ARCHITECTURE.md：优先检验云遮挡与地表遮阴双解释，备选为未匹配可回退的云影位移关联、跨地物照明场。附目标错误构成、代码接入点、已有方法重叠及最小消融；均为待测方案，不宣称首创。本轮未启动这些结构训练。\n- 交付outputs/phase411/phase411_20261101/RESULT.md及完整曲线、参数系数、调用与验收。\n\n'
  p.write_bytes(raw.replace(marker,entry.replace('\n','\r\n').encode()+marker))
 (R/'DELIVERY_COMPLETE.json').write_text(json.dumps({'passed':True,'phase':411,'new_calls':4408,'new_updates':4120,'baseline_reused':'phase410/baseline','target_based_promotion':a['target_based_promotion'],'source_is_selection_gate':False,'architecture_proposals_delivered':True,'proposed_architectures_trained':False},indent=2))
 code={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'src/tools').glob('*phase411*.py'))}
 (R/'LOCAL_CODE_LOCK.json').write_text(json.dumps(code,indent=2))
 refs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [ROOT/'outputs/phase410/phase410_20261101/LOCAL_MANIFEST.json',ROOT/'src/research_results/phase319_20261027/RESULT.md',ROOT/'src/research_results/phase320_20261027/RESULT.md',ROOT/'outputs/phase74/phase74_20261010/README.md']}
 (R/'LOCAL_REFERENCE_LOCK.json').write_text(json.dumps(refs,indent=2))
 manifest={p.relative_to(R).as_posix():sha(p) for p in sorted(R.rglob('*')) if p.is_file() and p.name!='LOCAL_MANIFEST.json'}
 (R/'LOCAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
 print('Phase411 sealed:',len(manifest),'files')
if __name__=='__main__':main()
