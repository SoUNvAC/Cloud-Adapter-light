from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase477/phase477_20261107';D=R/'A/network_run'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
previous=read(ROOT/'outputs/phase476/phase476_20261107/B/network_run/ABLATION_OFF.json')['results'][0]
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['passed'] and accept['matches_476_common_initialization_and_recorded_batches']
assert all(sha(ROOT/n)==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
final=read(D/'FINAL_RESULTS.json');assert final['passed'];complete=read(D/'COMPLETE.json');assert complete['steps']==6000
update=read(D/'UPDATE_CHECK.json');assert update['step']==6000 and not update['new_changed']
assert all(update[n] for n in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
assert update['changed']['decode_head.']>0 and all(not v for n,v in update['changed'].items() if n!='decode_head.')
for x in final['results']:
 per=read(D/f"INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
 cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
 iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
 assert max(abs(a-b) for a,b in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
 if x['step']:
  p=D/f"branch_{x['step']}.pth";assert sha(p)==accept['weights'][p.name]['sha256']
assert all(sha(D/n)==h for n,h in read(D/'DELIVERY_SHA.json').items())
best=next(x for x in final['results'] if 'best_shadow' in x['roles']);protected=next(x for x in final['results'] if 'best_protected' in x['roles'])
rows=[('450完整网络',base),('476B联合训练后移除材质层',previous),('477仅解码器最佳影',best)]
if protected['step'] and protected['step']!=best['step']:rows.append(('477受保护选点',protected))
for step in [1000,3000,6000]:
 x=next(x for x in final['results'] if x['step']==step)
 if x['step']!=best['step'] and (x['step']!=protected['step'] or not protected['step']):rows.append((f'477固定{step}步',x))
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase477：仅解码器续训对照','','目标监督0%；CloudSEN训练8490/原始验证535；450起点，6000步、batch4、seed45201，原解码器lr1e-5、warmup200，固定argmax。原骨干、450分支与BN固定，没有新增参数。','','与476相同的不加权CE、KL/softIoU保持约束和修正正则。已核验共同初始权重、固定采样器及记录批次一致。保护线仍为450的地表/云IoU。','','1000、3000、6000步及最佳点均作535张batch1独立复算；同一步比较用于区分训练时材质层作用与单纯解码器续训。','','|模型|步数|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+str(x.get('step'))+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
comparison=read(R/'MATCHED_COMPARISON.json')
for k in ['477best_minus_476A','477best_minus_476B','477best_minus_476B_off','477_1000_minus_476A_off','477_3000_minus_476B_off']:lines+=['',k+': '+json.dumps(comparison[k])]
lines+=['','477−476的同一步区间是107ROI配对bootstrap2000次。最佳点比较还包含选点时长差异，不能代替同一步比较。']
lines+=['','归因结论：仅解码器续训已使地表、云和云影三类IoU同时超过450。3000步时，477较476B关层候选云影低0.054821点，107ROI区间[-0.117441,0.000883]跨0；地表、云和mIoU则更高。由此，上一轮整体提升的主要部分无需新增材质层即可获得，尚无证据认定材质层训练过程带来稳定的额外云影收益。后续结构实验以477式同设置解码器续训作为匹配训练对照；476关层权重保留为当前云影最高的单seed候选。']
err=read(D/'BOUNDARY_DIAG.json')['delta477minus450']
lines+=['','最佳点较450三类IoU变化：'+json.dumps(best['IoU_delta'])+'；漏检净变化'+str(sum(err['FN']))+'、误报净变化'+str(sum(err['FP']))+'像素；边界0–3/3–8/>8分解：'+json.dumps(err),'','训练耗时/秒：'+str(complete['seconds'])+'；训练参数：'+str(accept['task_parameters'])+'；峰值分配显存字节：'+str(update['peak_memory_bytes'])]
candidates=[x for x in final['results'] if x['protected'] and x['IoU'][2]>previous['IoU'][2]]
lines+=['','决策：'+('477存在同时保持450地表/云且云影超过476关层候选的检查点：'+','.join(str(x['step']) for x in candidates)+'；是否达到70%见主表。' if candidates else '477未在保护450地表/云条件下超过476关层候选，保留476关层候选及450原始基线。'),'','验收：仅原解码器更新，其他原参数/BN/450分支冻结；535图独立混淆矩阵、初始化/记录批次、权重和传输SHA通过。单seed验证集探索。权重使用原431基座加载student_patch，再以infer_shadow477.py加载state_dict，未使用任何隐藏关层开关。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,images_per_checkpoint=535,ROI_count=107,target_supervision_percent=0,choices=final['choices']),indent=2))
files=list((ROOT/'src/tools').glob('*477.py'))+list((ROOT/'src/tools').glob('*477.sh'))+[ROOT/'src/research_plans/phase477.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
