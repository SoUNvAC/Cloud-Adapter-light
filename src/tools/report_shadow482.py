from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase482/phase482_20261108';P=ROOT/'outputs/phase480/phase480_20261108/B/network_run'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(R/'TRANSFER_ACCEPTED.json')['passed'] and read(R/'TRAINED_WEIGHT_ACCEPTANCE.json')['passed'] and read(R/'UNIT_ACCEPTANCE.json')['passed']
comparison=read(R/'MATCHED_COMPARISON.json');rows=[];allmetrics={}
for model,d in [('480B',P),('482A',R/'A/network_run'),('482B',R/'B/network_run')]:
 final=read(d/'FINAL_RESULTS.json');assert final['passed']
 for x in final['results']:
  v=read(d/f"INDEPENDENT_{x['step']}.json");assert len(v['per_image'])==535 and len({z['group'] for z in v['per_image']})==107
  cm=[[sum(z['confusion'][i][j] for z in v['per_image']) for j in range(3)] for i in range(3)]
  iou=[100*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(a-b) for a,b in zip(iou,x['IoU']))<1e-9 and v['result']['state_unchanged']
 for selection in ['last_6000','best_shadow','best_protected']:
  step=final['choices'][selection];x=next(z for z in final['results'] if z['step']==step);allmetrics[model,selection]=x
  rows.append(dict(model=model,selection=selection,step=step,ground=x['IoU'][0],cloud=x['IoU'][1],shadow=x['IoU'][2],mIoU=x['mIoU'],protect450=x['protected'],target_supervision_percent=0))
with (R/'MAIN.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for arm,v in read(R/'TRAINED_WEIGHT_ACCEPTANCE.json')['arms'].items():
 for n,s in v['weights'].items():
  if n!='branch_0.pth':assert sha(R/arm/'network_run'/n)==s['sha256']
lines=['# Phase482：云影占比对比重分配','',
 'A：把细节权重p改为p−patch平均p，再做原有块均值约束，保留0.1覆盖率辅助BCE。B：保留原480B结构，覆盖率辅助BCE改为0。均为15265分支参数；两组完整重训6000步，目标域监督0%，CloudSEN原始验证535张。共同431＋450起点、seed45201、分支seed48001，其余协议与480B一致。','',
 '主表统一6000步：','', '| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- |']
for model in ['480B','482A','482B']:
 x=allmetrics[model,'last_6000'];lines.append('| '+model+' | '+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
decisions={}
for arm in ['A','B']:
 z=comparison[f'482{arm}_6000_minus_480B_6000'];promote=all(v>=0 for v in z['IoU_delta'][:2]) and z['IoU_delta'][2]>0;decisions[arm]=promote
 lines+=['',f"{arm}相对480B三类增量："+'/'.join(f'{v:+.6f}' for v in z['IoU_delta'])+f"个百分点；云影107ROI配对95%区间{z['CI95'][2]}。同一步地表/云不降且影提升：{promote}。"]
lines+=['','辅助结果：验证集最佳云影点。','', '| 模型 | 步数 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- | --- |']
for model in ['480B','482A','482B']:
 x=allmetrics[model,'best_shadow'];lines.append(f"| {model} | {x['step']} | "+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
lines+=['','同权重干预（各自最佳点）：','']
for key in ['482Abest_minus_482Aoff','482Bbest_minus_482Boff','482Abest_minus_482Acontrastoff']:
 z=comparison[key];lines.append(f"- {key}：云影{z['IoU_delta'][2]:+.6f}点，107ROI区间{z['CI95'][2]}；漏检变化{z['FN_delta'][2]}、误报变化{z['FP_delta'][2]}像素。")
lines+=['','最佳点相对480B的错误变化（0–3/3–8/>8像素距云影边界；负数表示减少）：','']
for arm,v in comparison['boundary_best_vs_480B'].items():lines.append(f"- {arm}：漏检{v['FN']}；误报{v['FP']}。")
lines+=['','验收：初始参数和记录批次与480B一致；新分支13个参数张量、原解码器实际更新，450、原骨干及BN固定；独立535图混淆矩阵复算、权重签名和传输通过。','',
 '单seed验证集探索。场景区间不包含训练seed波动；最佳点属于验证选点。B无辅助损失与原480B构成单因素对照；本轮未包含“占比对比＋无辅助损失”，不能判断二者交互。', '',
 '默认TF32精度保持不变，块均值约束按近似实现解释；CPU单元检查通过不替代GPU严格均值数值验收。干预运行时enabled/contrast_enabled开关不保存在state_dict中，干预权重不能当作默认部署条件。','']
interpret=R/'INTERPRETATION.md'
if interpret.exists():lines.insert(2,interpret.read_text(encoding='utf-8')+'\n')
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,metric_recompute=True,weight_signatures=True,training_provenance=True,target_supervision_percent=0,matched_step_upgrade=decisions),indent=2))
files=list((ROOT/'src/tools').glob('*482.py'))+list((ROOT/'src/tools').glob('*482.sh'))+[ROOT/'src/research_plans/phase482.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2));print('REPORT_AND_ACCEPTANCE passed')
