from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase495/phase495_20261109';P=ROOT/'outputs/phase493/phase493_20261109/B/network_run'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in ['TRANSFER_ACCEPTED','TRAINED_WEIGHT_ACCEPTANCE','UNIT_ACCEPTANCE']:assert read(R/f'{n}.json')['passed']
rows=[];metrics={};comparison=read(R/'MATCHED_COMPARISON.json')
for model,d in [('493B',P),('495A',R/'A/network_run'),('495B',R/'B/network_run')]:
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
  assert len(s['sha256'])==64 and v['passed']
lines=['# Phase495：细纹理写回MsRE主干token','',
'保留493B全部输出路径，共享已有16通道细纹理编码，池化到token网格并通过零初始化16→384投影写入MsRE输出。A在第1次交互写入并经过后续9个冻结VFM块，B在第4次交互写入；均按当前token的脱离梯度RMS匹配量纲。新增6144参数，总结构模块43137参数，无额外骨干前向，但A需要反向穿过后续冻结块。','',
 '两组均无辅助BCE，与原解码器共同重训6000步。共同431＋450起点，seed45201、分支seed48001，原骨干、450及BN冻结。CloudSEN原始验证535张、107ROI；目标域监督0%，固定argmax。','']
if (R/'INTERPRETATION.md').exists():lines+=[(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'']
for selection,title in [('last_6000','主结果：统一6000步'),('best_shadow','辅助结果：训练内验证预选点的独立重载结果')]:
 lines += [title,'','| 模型 | 步数 | 地表IoU | 云IoU | 云影IoU | mIoU |','| --- | --- | --- | --- | --- | --- |']
 for model in ['493B','495A','495B']:
  x=metrics[model,selection];lines.append(f"| {model} | {x['step']} | "+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
 lines+=['']
decisions={}
for arm in ['A','B']:
 z=comparison[f'495{arm}_6000_minus_493B_6000'];decisions[arm]=all(v>=0 for v in z['IoU_delta'][:2]) and z['IoU_delta'][2]>0
 lines += [f"{arm}相对493B三类增量："+'/'.join(f'{v:+.6f}' for v in z['IoU_delta'])+f"个百分点；云影107ROI配对95%区间{z['CI95'][2]}；地表/云不降且影提升：{decisions[arm]}。",'']
 for suffix in ['off','regionoff']:
  z=comparison[f'495{arm}best_minus_495{arm}{suffix}'];lines += [f"{arm}最佳点相对{suffix}：影{z['IoU_delta'][2]:+.6f}点，区间{z['CI95'][2]}；漏检{z['FN_delta'][2]:+d}、误报{z['FP_delta'][2]:+d}像素。"]
 z=read(R/arm/'network_run/REGION_EFFECT_BOUNDARY_DIAG.json')['delta_on_minus_off'];lines += [f"token写回支路开启−关闭的边界/内部错误（距边界0–3/3–8/>8像素）：漏检{z['FN']}，误报{z['FP']}。",'']
z=comparison['495A_6000_minus_495B_6000'];lines += [f"第1次写入A−第4次写入B（6000步）：三类增量{z['IoU_delta']}，云影场景区间{z['CI95'][2]}。",'']
cost=read(R/'INFERENCE_COST.json');assert cost['passed'] and cost['state_unchanged']
lines += [f"推理成本：同一张512图、batch1、各5次预热、交替顺序各30次CUDA event计时。中位毫秒{cost['CUDA_event_median_ms']}，A/493B={cost['ratio']:.3f}倍。仅GPU前向，不含加载、IO、预处理，不代表全数据吞吐。",'']
for arm in ['A','B']:
 d=R/arm/'network_run';lines += [f"{arm}并行训练墙钟{read(d/'COMPLETE.json')['seconds']/60:.2f}分钟，峰值分配显存{read(d/'UPDATE_CHECK.json')['peak_memory_bytes']}字节。",'']
lines += ['验收：原有参数初始化与记录批次匹配493B；结构模块22个参数张量（保留21个、新增1个）和解码器实际更新，原骨干、450与BN固定；独立混淆矩阵复算、远端权重验签及元数据传输通过；完整权重保留远端。','',
 '单seed验证集探索，场景区间不包含训练seed波动；最佳点属于验证选点。默认TF32下均值约束按近似实现解释。REGION_OFF沿用通用文件名，实际只关闭新增token写回支路，保留493B原有全部细节分支。A/B对照改变接入位置和后续传播路径；不能把差异归因于单一的语义传播机制。enabled、evidence_enabled、ordinal_enabled、feedback_enabled运行时开关不保存在state_dict，干预复算需显式设置。','']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,metric_recompute=True,remote_weight_signatures_verified=True,local_weights_complete=False,weight_storage="remote",target_supervision_percent=0,matched_step_upgrade=decisions),indent=2))
files=list((ROOT/'src/tools').glob('*495.py'))+list((ROOT/'src/tools').glob('*495.sh'))+[ROOT/'src/research_plans/phase495.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2));print('REPORT_AND_ACCEPTANCE passed')
