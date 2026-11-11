from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase471/phase471_20261107'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['identical_training_batches']
comparison=read(R/'MATCHED_COMPARISON.json');assert read(R/'MODULE_AUDIT.json')['passed']
assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
rows=[('450完整网络',base)];selections={};times={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json');assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles'])
 rows.append((arm+('自由二维位移' if arm=='A' else '轮廓法向位移'),best));selections[arm]=final['choices']
 complete=read(d/'COMPLETE.json');assert complete['steps']==6000;times[arm]=complete['seconds']
 update=read(d/'UPDATE_CHECK.json');assert update['step']==6000
 assert all(update[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert all(update['new_changed'].values()) and not any(update['changed'].values())
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','B关闭新增模块'),('ABLATION_NORMAL_OFF.json','NORMAL_OFF_','B同权重去掉法向约束')]:
   x=read(d/fn)['results'][0];rows.append((name,x));checks.append((prefix,x))
 for prefix,x in checks:
  per=read(d/f"{prefix}INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
  cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
  iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(u-v) for u,v in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
 assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in read(d/'DELIVERY_SHA.json').items())
 w=d/f"branch_{best['step']}.pth";assert hashlib.sha256(w.read_bytes()).hexdigest()==accept[arm]['sha'][w.name]
assert (R/'A/network_run/INPUT_TRACE.jsonl').read_bytes()==(R/'B/network_run/INPUT_TRACE.jsonl').read_bytes()
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase471：云影轮廓法向特征位移','','目标监督0%；CloudSEN训练8490/原始验证535；450起点，两组各6000步，batch4、seed45201，固定argmax。','','在解码器前1/4分辨率特征加入16348参数模块。融合空间CNN、四级语义和RGB描述，预测云影有符号距离、二维位移与混合门。A自由二维位移；B将同一向量投影到预测距离场的法向，最大位移2网格（8原图像素）。按预测距离衰减远离边界的修正。原解码器和450分支读取相同修正特征。','','原网络全部权重和BN冻结，仅新增模块lr1e-4。位移零初始化，使用同一采样网格差分保证初始精确零更新。两组辅助监督相同：0.1距离场损失，内含0.5梯度项；其余分割损失沿用450参考。','','训练距离场由源标签在1/4网格的云影占比≥50%生成，内正外负、截断±8网格；未知标签附近不监督。此二值化只生成辅助目标，不改最终分割阈值。推理不使用标签，也不额外运行参考网络。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','训练耗时/秒：'+json.dumps(times)]
lines+=['','6000步末点（训练期间batch4全535图评测；主表为选点后batch1独立复算）：']
for arm in ['A','B']:
 last=read(R/arm/'network_run/HISTORY.json')[-1];assert last['step']==6000
 lines.append(arm+': '+json.dumps(dict(IoU=last['IoU'],mIoU=last['mIoU'])))
for k in ['B_minus_A_IoU_delta','CI95','B_transport_on_minus_off','B_normal_on_minus_off']:
 lines+=['',k+': '+json.dumps(comparison[k])]
for a,row in zip(['A','B'],rows[1:3]):
 x=row[1];err=read(R/a/'network_run/BOUNDARY_DIAG.json')['delta471minus450']
 lines+=['',a+'较450三类IoU变化：'+json.dumps(x['IoU_delta'])+'；漏检净变化'+str(sum(err['FN']))+'、误报净变化'+str(sum(err['FP']))+'像素；边界距离0–3/3–8/8以上分解：'+json.dumps(err)+'。']
passed=[name for name,x in rows[1:3] if x['protected'] and x['IoU'][2]>base['IoU'][2]]
diag=read(R/'TRANSPORT_DIAG.json');assert diag['independent_predictions_identical'] and diag['state_unchanged']
lines+=['','B位移诊断（全部1/4网格）：平均预测位移'+str(diag['mean_flow_pixels'])+'原图像素；平均混合门'+str(diag['mean_gate'])+'；门加权位移代理量'+str(diag['blend_weighted_flow_proxy_pixels'])+'像素。代理量用于衡量修正幅度，不是最终预测边界移动距离。', '', '辅助距离头：归一化距离MAE='+str(diag['distance_mae'])+'；粗网格距离符号云影IoU='+str(diag['distance_head_shadow_IoU'])+'。该辅助头口径不同，不能替代主表的全分辨率分割IoU。']
lines+=['','决策：'+('候选获得保护其他两类条件下的云影提升：'+','.join(passed)+'；是否达到70%见主表。' if passed else '两组均未在保护其他两类的条件下超过450，保留450。'),'', '验收：实际共享初始化与批次一致、新增参数更新、原参数与BN冻结、535图独立混淆矩阵、权重和传输SHA通过。107ROI配对bootstrap2000次；单seed验证集选点的6000步探索，非独立测试集结论。','','参考：[Improving Semantic Segmentation via Decoupled Body and Edge Supervision](https://arxiv.org/abs/2007.10035)。借鉴特征位移思想，本轮检验云影距离场和法向约束，不将通用位移宣称为原创。','','B开关状态由ablate_shadow471.py和ablate_normal471.py设置，单独加载权重不会恢复Python开关。']
lines+=['','B的逐图特征修正RMS/原特征RMS平均值：'+str(diag['mean_relative_feature_rms'])+'。这些均为全图平均，不能单凭均值判定某个局部区域的作用。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*471.py'))+list((ROOT/'src/tools').glob('*471.sh'))+[ROOT/'src/research_plans/phase471.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
