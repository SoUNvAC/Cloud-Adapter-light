from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase486/phase486_20261108';P=ROOT/'outputs/phase482/phase482_20261108/B/network_run'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['TRANSFER_ACCEPTED','TRAINED_WEIGHT_ACCEPTANCE','UNIT_ACCEPTANCE']:assert read(R/f'{n}.json')['passed']
rows=[];metrics={};comparison=read(R/'MATCHED_COMPARISON.json')
for model,d in [('482B',P),('486A',R/'A/network_run'),('486B',R/'B/network_run')]:
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
lines=['# Phase486：场景遮光变化共识','',
 '保留482B细节支路，新增全图共享RGB变化方向：1/8网格比较8邻居，轴向距离32像素，以冻结VFM内容相似度、色度差和亮度差作软权重，从全图对数RGB差估计共同方向。局部沿方向变化、正交残差等6种量经邻居聚合形成12通道，轻量CNN后零投影注入第一层特征。A每图估计方向，B固定等比例RGB方向。新增8272参数，总模块23537参数。','',
 '两组均无辅助BCE，与原解码器共同重训6000步。共同431＋450起点，seed45201、分支seed48001，原骨干、450及BN冻结。CloudSEN原始验证535张、107ROI；目标域监督0%，固定argmax。','']
if (R/'INTERPRETATION.md').exists():lines+=[(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'']
for selection,title in [('last_6000','主结果：统一6000步'),('best_shadow','辅助结果：训练内验证预选点的独立重载结果')]:
 lines += [title,'','| 模型 | 步数 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- | --- |']
 for model in ['482B','486A','486B']:
  x=metrics[model,selection];lines.append(f"| {model} | {x['step']} | "+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
 lines+=['']
decisions={}
for arm in ['A','B']:
 z=comparison[f'486{arm}_6000_minus_482B_6000'];decisions[arm]=all(v>=0 for v in z['IoU_delta'][:2]) and z['IoU_delta'][2]>0
 lines += [f"{arm}相对482B三类增量："+'/'.join(f'{v:+.6f}' for v in z['IoU_delta'])+f"个百分点；云影107ROI配对95%区间{z['CI95'][2]}；地表/云不降且影提升：{decisions[arm]}。",'']
 for suffix in ['off','regionoff']:
  z=comparison[f'486{arm}best_minus_486{arm}{suffix}'];lines += [f"{arm}最佳点相对{suffix}：影{z['IoU_delta'][2]:+.6f}点，区间{z['CI95'][2]}；漏检{z['FN_delta'][2]:+d}、误报{z['FP_delta'][2]:+d}像素。"]
 z=read(R/arm/'network_run/REGION_EFFECT_BOUNDARY_DIAG.json')['delta_on_minus_off'];lines += [f"共识证据支路开启−关闭的边界/内部错误（距边界0–3/3–8/>8像素）：漏检{z['FN']}，误报{z['FP']}。",'']
z=comparison['486A_6000_minus_486B_6000'];lines += [f"A场景方向−B固定方向（6000步）：三类增量{z['IoU_delta']}，云影场景区间{z['CI95'][2]}。",'']
z=comparison['486Abest_minus_486Afixed'];lines += [f"A最佳点同权重，场景方向−替换固定方向：三类增量{z['IoU_delta']}，云影场景区间{z['CI95'][2]}。",'']
for arm,v in read(R/'DIRECTION_DIAGNOSTIC.json')['arms'].items():lines += [f"{arm}与等比例RGB方向夹角（度，最小/25%/中位/75%/最大）：{v['angle_to_equal_RGB_degrees_quantiles']}。",'']
lines += ['文献依据及与455/461/476/479/485的区别见LITERATURE.md；借鉴色度不变性及材质一致边缘思想，未复现原论文的阴影去除或测试时优化。','']
lines += ['验收：原有参数初始化与记录批次匹配482B；新增分支17个参数张量和解码器实际更新，原骨干、450与BN固定；独立混淆矩阵复算、权重签名与传输通过。','',
 '单seed验证集探索，场景区间不包含训练seed波动；最佳点属于验证选点。默认TF32下均值约束按近似实现解释。共识方向是图像统计代理，不是真实光照、反射率或太阳方向。REGION_OFF沿用通用文件名，实际关闭共识证据支路。方向替换干预将adaptive_mode设为False并保存在诊断权重中；该权重不是原A部署条件。enabled、evidence_enabled运行时开关不保存在state_dict，干预复算需显式设置。','']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,metric_recompute=True,weight_signatures=True,target_supervision_percent=0,matched_step_upgrade=decisions),indent=2))
files=list((ROOT/'src/tools').glob('*486.py'))+list((ROOT/'src/tools').glob('*486.sh'))+[ROOT/'src/research_plans/phase486.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2));print('REPORT_AND_ACCEPTANCE passed')
