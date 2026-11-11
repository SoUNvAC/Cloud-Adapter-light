from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase473/phase473_20261107'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['identical_training_batches']
comparison=read(R/'MATCHED_COMPARISON.json');assert read(R/'MODULE_AUDIT.json')['passed']
assert read(R/'AUX_AUDIT.json')['passed']
assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
paths={'A':ROOT/'outputs/phase472/phase472_20261107/A/network_run','B':R/'B/network_run'}
rows=[('450完整网络',base)];selections={};times={}
for arm,d in paths.items():
 final=read(d/'FINAL_RESULTS.json');assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles'])
 rows.append(('472A普通续训' if arm=='A' else '473类别引导遮光参照',best));selections[arm]=final['choices']
 complete=read(d/'COMPLETE.json');assert complete['steps']==6000;times[arm]=complete['seconds']
 update=read(d/'UPDATE_CHECK.json');assert update['step']==6000
 assert all(update[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert all(update['new_changed'].values())
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','473关闭新增模块'),('ABLATION_APPEARANCE_OFF.json','APPEARANCE_OFF_','473仅关闭类别路由'),('ABLATION_UNIFORM_ROUTING.json','UNIFORM_ROUTING_','473图内均匀类别路由')]:
   x=read(d/fn)['results'][0];rows.append((name,x));checks.append((prefix,x))
 for prefix,x in checks:
  per=read(d/f"{prefix}INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
  cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
  iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(u-v) for u,v in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
 assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in read(d/'DELIVERY_SHA.json').items())
 w=d/f"branch_{best['step']}.pth";assert hashlib.sha256(w.read_bytes()).hexdigest()==accept[arm]['sha'][w.name]
assert (paths['A']/'INPUT_TRACE.jsonl').read_bytes()==(paths['B']/'INPUT_TRACE.jsonl').read_bytes()
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase473：类别引导遮光参照','','CloudSEN源域训练8490/验证535，目标监督0%，固定argmax。完整6000步，batch4、seed45201。','','在472对比投影上增加三类辅助头：云影概率控制查询位置的修正幅度，邻居地表概率引导参照选择。8个间隔2-token邻居，保留语义/色度/亮度匹配；四阶段共享6198个新参数。原MsRE投影和解码器lr1e-5，新模块lr1e-4；DINO、BN、450分支及其余原参数固定。','','辅助监督为各token中有效三类标签的面积占比CE，四阶段平均，权重0.1；只用于源域训练，推理不读取标签。强度前2000步升至0.25，只从2000步后选训练最优；0步回退。','','对照复用472A已完成的相同普通续训；共同初始权重及已记录的批次签名一致。此比较衡量结构与辅助监督组成的完整方法，不单独归因辅助损失。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','训练耗时/秒：'+json.dumps(times)]
for k in ['B_minus_A_IoU_delta','CI95','B_contrast_on_minus_off','B_appearance_on_minus_off','B_spatial_minus_uniform_routing']:lines+=['',k+': '+json.dumps(comparison[k])]
lines+=['','字段B_appearance_on_minus_off在本轮表示类别路由开关，RGB匹配始终保留。关闭整个模块或类别路由均沿用相同已训练原参数，不能视作重新训练的消融。']
lines+=['','均匀路由对照：将每张图每阶段的辅助三类概率替换为空间平均值，保持平均云影门控强度，移除位置选择及邻居类别偏好。该对照仍是同权重干预。']
err=read(paths['B']/'BOUNDARY_DIAG.json')['delta473minus450'];best=rows[2][1]
diag=read(R/'CONTRAST_DIAG.json');assert diag['independent_predictions_identical'] and diag['state_unchanged']
lines+=['','四阶段输入修正相对RMS：'+json.dumps(diag['stage_mean_relative_input_rms']),'','四阶段真类条件下的辅助预测概率（每个3×3矩阵行=真实地表/云/影，列=预测地表/云/影）：'+json.dumps(diag['stage_true_class_mean_predicted_probability'])]
lines+=['','较450三类IoU变化：'+json.dumps(best['IoU_delta'])+'；漏检净变化'+str(sum(err['FN']))+'、误报净变化'+str(sum(err['FP']))+'像素。边界距离0–3/3–8/>8像素：'+json.dumps(err)]
protected=[x for x in read(paths['B']/'FINAL_RESULTS.json')['results'] if x['protected'] and x['IoU'][2]>base['IoU'][2]]
lines+=['','决策：'+('获得保护地表和云条件下的云影提升，是否达到70%见结果。' if protected else '未超过450的受保护云影性能，保留450。'),'','验收：535图独立混淆矩阵、107ROI场景配对bootstrap2000次、参数更新与冻结、初始权重、记录批次与传输SHA均通过。单seed验证集选点实验，不作为独立测试或稳定性结论。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*473.py'))+list((ROOT/'src/tools').glob('*473.sh'))+[ROOT/'src/research_plans/phase473.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
