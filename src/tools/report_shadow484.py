from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase484/phase484_20261108';P=ROOT/'outputs/phase482/phase482_20261108/B/network_run'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['TRANSFER_ACCEPTED','TRAINED_WEIGHT_ACCEPTANCE','UNIT_ACCEPTANCE']:assert read(R/f'{n}.json')['passed']
rows=[];metrics={};comparison=read(R/'MATCHED_COMPARISON.json')
for model,d in [('482B',P),('484A',R/'A/network_run'),('484B',R/'B/network_run')]:
 f=read(d/'FINAL_RESULTS.json');assert f['passed']
 for x in f['results']:
  v=read(d/f"INDEPENDENT_{x['step']}.json");assert len(v['per_image'])==535 and len({z['group'] for z in v['per_image']})==107 and v['result']['state_unchanged']
  cm=[[sum(z['confusion'][i][j] for z in v['per_image']) for j in range(3)] for i in range(3)]
  iou=[100*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)-cm[i][i]) for i in range(3)]
  assert cm==x['confusion'] and max(abs(a-b) for a,b in zip(iou,x['IoU']))<1e-9
 for selection in ['last_6000','best_shadow','best_protected']:
  x=next(z for z in f['results'] if z['step']==f['choices'][selection]);metrics[model,selection]=x
  rows.append(dict(model=model,selection=selection,step=x['step'],ground=x['IoU'][0],cloud=x['IoU'][1],shadow=x['IoU'][2],mIoU=x['mIoU'],target_supervision_percent=0))
with (R/'MAIN.csv').open('w',newline='',encoding='utf-8') as h:
 w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for arm,v in read(R/'TRAINED_WEIGHT_ACCEPTANCE.json')['arms'].items():
 for n,s in v['weights'].items():
  if n!='branch_0.pth':assert sha(R/arm/'network_run'/n)==s['sha256']
lines=['# Phase484：双尺度云影修正','',
 '保留482B细节支路，新增跨patch区域支路：细节编码池化至1/16网格，经5×5深度卷积、1×1卷积，插值回1/4网格，再以零初始化投影注入第一层金字塔。A约束64×64原图像素范围的区域修正均值，B允许区域整体偏移。新增6832参数，细节＋区域合计22097参数。','',
 '两组均无辅助BCE，与原解码器共同重训6000步。共同431＋450起点，seed45201、分支seed48001，原骨干、450及BN冻结。CloudSEN原始验证535张、107ROI；目标域监督0%，固定argmax。','']
if (R/'INTERPRETATION.md').exists():lines+=[(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'']
for selection,title in [('last_6000','主结果：统一6000步'),('best_shadow','辅助结果：验证集最佳云影点')]:
 lines += [title,'','| 模型 | 步数 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- | --- |']
 for model in ['482B','484A','484B']:
  x=metrics[model,selection];lines.append(f"| {model} | {x['step']} | "+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
 lines+=['']
decisions={}
for arm in ['A','B']:
 z=comparison[f'484{arm}_6000_minus_482B_6000'];decisions[arm]=all(v>=0 for v in z['IoU_delta'][:2]) and z['IoU_delta'][2]>0
 lines += [f"{arm}相对482B三类增量："+'/'.join(f'{v:+.6f}' for v in z['IoU_delta'])+f"个百分点；云影107ROI配对95%区间{z['CI95'][2]}；地表/云不降且影提升：{decisions[arm]}。",'']
 for suffix in ['off','regionoff']:
  z=comparison[f'484{arm}best_minus_484{arm}{suffix}'];lines += [f"{arm}最佳点相对{suffix}：影{z['IoU_delta'][2]:+.6f}点，区间{z['CI95'][2]}；漏检{z['FN_delta'][2]:+d}、误报{z['FP_delta'][2]:+d}像素。"]
 z=read(R/arm/'network_run/REGION_EFFECT_BOUNDARY_DIAG.json')['delta_on_minus_off'];lines += [f"区域支路开启−关闭的边界/内部错误（距边界0–3/3–8/>8像素）：漏检{z['FN']}，误报{z['FP']}。",'']
lines += ['验收：原有参数初始化与记录批次匹配482B；新增分支18个参数张量和解码器实际更新，原骨干、450与BN固定；独立混淆矩阵复算、权重签名与传输通过。','',
 '单seed验证集探索，场景区间不包含训练seed波动；最佳点属于验证选点。默认TF32下均值约束按近似实现解释。区域支路为潜在特征修正，不是物理遮光量估计。enabled、regional_enabled运行时开关不保存在state_dict，干预复算需显式设置。','']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,metric_recompute=True,weight_signatures=True,target_supervision_percent=0,matched_step_upgrade=decisions),indent=2))
files=list((ROOT/'src/tools').glob('*484.py'))+list((ROOT/'src/tools').glob('*484.sh'))+[ROOT/'src/research_plans/phase484.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2));print('REPORT_AND_ACCEPTANCE passed')
