from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase478/phase478_20261108'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
previous=read(ROOT/'outputs/phase476/phase476_20261107/B/network_run/ABLATION_OFF.json')['results'][0]
control=next(x for x in read(ROOT/'outputs/phase477/phase477_20261107/A/network_run/FINAL_RESULTS.json')['results'] if 'best_shadow' in x['roles'])
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['passed'] and accept['matches_477_common_initialization_and_recorded_batches']
assert all(sha(ROOT/n)==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
rows=[('450完整网络',base),('476B移除材质层候选',previous),('477仅解码器',control)];details=[];choices={};candidates=[]
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json');assert final['passed'];complete=read(d/'COMPLETE.json');assert complete['steps']==6000
 assert all(sha(d/n)==h for n,h in read(d/'DELIVERY_SHA.json').items())
 choices[arm]=final['choices'];u=read(d/'UPDATE_CHECK.json');assert u['step']==6000 and not u['new_changed']
 assert all(u[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 for x in final['results']:
  per=read(d/f"INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
  cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
  iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(a-b) for a,b in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
  if x['step']:
   p=d/f"branch_{x['step']}.pth";assert sha(p)==accept['arms'][arm]['weights'][p.name]['sha256']
  rows.append((f"478{arm} {'/'.join(x['roles'])}",x))
  if x['protected'] and x['IoU'][2]>previous['IoU'][2]:candidates.append((arm,x['step'],x['IoU'][2]))
 best=next(x for x in final['results'] if 'best_shadow' in x['roles']);err=read(d/'BOUNDARY_DIAG.json')['delta478minus450']
 details.append(f"{arm}：训练参数{accept['arms'][arm]['task_parameters']}，耗时{complete['seconds']:.1f}秒，峰值分配显存{u['peak_memory_bytes']}字节。最佳点较450漏检净变化{sum(err['FN'])}、误报净变化{sum(err['FP'])}像素；边界分解：{json.dumps(err)}")
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase478：原特征路径可塑性对照','','A开放原MsRE/refinement、local读写、output_residual和解码器；B开放原空间CNN、交叉注意力和解码器。新增参数0。VFM、450分支、所有BN固定。本轮属于训练范围对照，不是新架构。','','CloudSEN源训练8490/原始验证535；目标监督0%；450起点，6000步、batch4、seed45201，开放参数lr1e-5；与477相同的不加权CE、KL/softIoU保持约束及修正正则，固定argmax。共同初始权重与记录批次签名核验通过。','','每组1000/3000/6000步和最佳点以batch1独立复算535张。保护线为450地表/云IoU，云影目标70%。','','|模型|步数|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+str(x.get('step'))+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
comparison=read(R/'MATCHED_COMPARISON.json')
lines+=['','与477同一步的107ROI配对bootstrap2000次；区间单位为百分点。最佳点比较含验证集选点，不替代同一步比较。','','|比较|云影增量|95%区间|','|---|---:|---|']
for k,x in comparison.items():
 if isinstance(x,dict) and 'IoU_delta' in x:lines.append(f"|{k}|{x['IoU_delta'][2]:.6f}|{x['CI95'][2]}|")
lines+=['',*details,'','超过476关层候选且保持450地表/云的检查点：'+json.dumps(candidates),'','结论：A最佳3000步云影65.686569%，较同一步477仅高0.015014点，107ROI区间[-0.047062,0.070123]跨0；地表/云各低0.042314/0.047961点。A云影TP较477减少13像素、FP减少3355像素，未体现新增云影找回能力。B最佳1000步云影65.589938%，同1000步较477低0.038799点；3000和6000步也低于对应477，全部云影差值区间跨0。两组最佳点仍保持450地表/云，但均未超过476关层候选65.726375%。','', '决策：本轮未发现扩大原路径训练范围带来可靠的额外云影收益，未达到70%。保留476关层候选和477匹配训练对照；本轮不替换最佳权重，不把解冻训练包装成新结构。后续应回到云影特征构造的结构假设，避免仅扩大解冻范围。','', '归因范围：每组开放整条路径并联合训练解码器，所有开放参数使用同一优化器组并共同做梯度裁剪。与477的差异反映这套联合训练方案，不能单独归因于某个子模块。两组并行训练，墙钟耗时不能直接当作与单独运行477的算力成本比。','', '验收：允许路径实际更新，其他原参数、450分支、BN与VFM固定；535图独立混淆矩阵、初始化/记录批次、权重及传输SHA通过。单seed验证集探索，未使用目标标签。权重依次加载原431、student_patch及450分支state_dict。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,images_per_checkpoint=535,ROI_count=107,target_supervision_percent=0,choices=choices),indent=2))
files=list((ROOT/'src/tools').glob('*478.py'))+list((ROOT/'src/tools').glob('*478.sh'))+[ROOT/'src/research_plans/phase478.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
