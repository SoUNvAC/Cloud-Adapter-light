from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase492/phase492_20261109';P=ROOT/'outputs/phase488/phase488_20261108/B/network_run'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['TRANSFER_ACCEPTED','TRAINED_WEIGHT_ACCEPTANCE','UNIT_ACCEPTANCE']:assert read(R/f'{n}.json')['passed']
rows=[];metrics={};comparison=read(R/'MATCHED_COMPARISON.json')
for model,d in [('488B',P),('492A',R/'A/network_run'),('492B',R/'B/network_run')]:
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
lines=['# Phase492：颜色方向遮光边缘证据','',
'保留488B，新增颜色梯度分解支路：在12/28像素尺度上，把RGB空间梯度分成沿颜色方向的平行分量、垂直残差及能量占比。A采用局部RGB方向，B固定灰色方向，编码后零投影注入第一层。新增8272参数，总模块31809参数，不增加骨干前向；显示RGB仅作学习线索，不作严格反射率解释。','',
 '两组均无辅助BCE，与原解码器共同重训6000步。共同431＋450起点，seed45201、分支seed48001，原骨干、450及BN冻结。CloudSEN原始验证535张、107ROI；目标域监督0%，固定argmax。','']
if (R/'INTERPRETATION.md').exists():lines+=[(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'']
for selection,title in [('last_6000','主结果：统一6000步'),('best_shadow','辅助结果：训练内验证预选点的独立重载结果')]:
 lines += [title,'','| 模型 | 步数 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- | --- |']
 for model in ['488B','492A','492B']:
  x=metrics[model,selection];lines.append(f"| {model} | {x['step']} | "+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
 lines+=['']
decisions={}
for arm in ['A','B']:
 z=comparison[f'492{arm}_6000_minus_488B_6000'];decisions[arm]=all(v>=0 for v in z['IoU_delta'][:2]) and z['IoU_delta'][2]>0
 lines += [f"{arm}相对488B三类增量："+'/'.join(f'{v:+.6f}' for v in z['IoU_delta'])+f"个百分点；云影107ROI配对95%区间{z['CI95'][2]}；地表/云不降且影提升：{decisions[arm]}。",'']
 for suffix in ['off','regionoff']:
  z=comparison[f'492{arm}best_minus_492{arm}{suffix}'];lines += [f"{arm}最佳点相对{suffix}：影{z['IoU_delta'][2]:+.6f}点，区间{z['CI95'][2]}；漏检{z['FN_delta'][2]:+d}、误报{z['FP_delta'][2]:+d}像素。"]
 z=read(R/arm/'network_run/REGION_EFFECT_BOUNDARY_DIAG.json')['delta_on_minus_off'];lines += [f"颜色梯度支路开启−关闭的边界/内部错误（距边界0–3/3–8/>8像素）：漏检{z['FN']}，误报{z['FP']}。",'']
z=comparison['492A_6000_minus_492B_6000'];lines += [f"局部颜色方向A−固定灰色方向B（6000步）：三类增量{z['IoU_delta']}，云影场景区间{z['CI95'][2]}。",'']
cost=read(R/'INFERENCE_COST.json');assert cost['passed'] and cost['state_unchanged']
lines += [f"推理成本：同一张512图、batch1、各5次预热、交替顺序各30次CUDA event计时。中位毫秒{cost['CUDA_event_median_ms']}，A/488B={cost['ratio']:.3f}倍。仅GPU前向，不含加载、IO、预处理，不代表全数据吞吐。",'']
for arm in ['A','B']:
 d=R/arm/'network_run';lines += [f"{arm}并行训练墙钟{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟，峰值分配显存{read(d/'UPDATE_CHECK.json')['peak_memory_bytes']}字节。",'']
lines += ['验收：原有参数初始化与记录批次匹配488B；新增分支21个参数张量和解码器实际更新，原骨干、450与BN固定；独立混淆矩阵复算、权重签名与传输通过。','',
 '单seed验证集探索，场景区间不包含训练seed波动；最佳点属于验证选点。默认TF32下均值约束按近似实现解释。REGION_OFF沿用通用文件名，实际只关闭新增颜色梯度支路，保留488B纹理分支。颜色方向分解的有效性由A/B对照检验，不能视为严格光照不变量。enabled、evidence_enabled、chromatic_enabled运行时开关不保存在state_dict，干预复算需显式设置。','']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,metric_recompute=True,weight_signatures=True,target_supervision_percent=0,matched_step_upgrade=decisions),indent=2))
files=list((ROOT/'src/tools').glob('*492.py'))+list((ROOT/'src/tools').glob('*492.sh'))+[ROOT/'src/research_plans/phase492.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2));print('REPORT_AND_ACCEPTANCE passed')
