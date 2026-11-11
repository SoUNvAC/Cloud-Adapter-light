from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase467/phase467_20261107'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
accept=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text());assert accept['identical_training_batches']
comparison=json.loads((R/'MATCHED_COMPARISON.json').read_text());audit=json.loads((R/'MODULE_AUDIT.json').read_text());assert audit['passed']
rows=[('450完整网络',base)];selections={};times={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles']);rows.append((arm+('仅训练原投影' if arm=='A' else '遮光对比投影'),best));selections[arm]=final['choices']
 complete=json.loads((d/'COMPLETE.json').read_text());assert complete['steps']==6000;times[arm]=complete['seconds']
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','B关闭新增分支'),('ABLATION_RING_OFF.json','RING_OFF_','B关闭环形对比')]:
   x=json.loads((d/fn).read_text())['results'][0];rows.append((name,x));checks.append((prefix,x))
 for prefix,x in checks:
  per=json.loads((d/f"{prefix}INDEPENDENT_{x['step']}.json").read_text())['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
  cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
  iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(u-v) for u,v in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
 delivery=json.loads((d/'DELIVERY_SHA.json').read_text());assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in delivery.items())
 w=d/f"branch_{best['step']}.pth";assert hashlib.sha256(w.read_bytes()).hexdigest()==accept[arm]['sha'][w.name]
assert (R/'A/network_run/INPUT_TRACE.jsonl').read_bytes()==(R/'B/network_run/INPUT_TRACE.jsonl').read_bytes()
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase467：MsRE投影内的遮光对比','','目标监督0%；原始CloudSEN训练8490/验证535，450起点；A、B各6000步、batch4、seed45201，固定argmax。','','A只训练原MsRE 1×1投影，lr1e-5。B在相同设置中增加21712参数遮光对比投影，lr5e-5。原解码器、450分支、VFM、其他骨干及BN全部固定。正确非影保持及难例权重均使用固定450预测。','','新增结构：四个MsRE阶段共享此分支。MsRE投影输入归一化并压到16维；比较局部语义与5减3、9减5两圈邻域，拼接RGB log外观及环形差异，共89通道；89→32→384生成修正，加入原投影输出，再经原MsRE残差缩放。最后层零初始化。显示RGB不是线性反射率，此处不估计物理遮光系数。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','训练耗时/秒：'+json.dumps(times),'','A、B最佳训练点比较及同迭代比较见MATCHED_COMPARISON.json。完整分支关闭与环形对比关闭均使用B同一训练权重。关闭环形对比时环形语义和RGB用当前位置替代、差分置零，其余参数保持。','','验收：起点复现；环形边界计数、零初始化、所有新参数梯度通过；训练图序列一致；原投影更新、B新增6个参数张量全部更新，其余原参数和450分支固定；完整535张独立混淆矩阵、权重签名和传输哈希通过。107个ROI重抽样2000次。单seed、验证集选点的探索性续训，不是40K全量重训。']
for k in ['B_minus_A_IoU_delta','CI95','B_projection_on_minus_off','B_ring_on_minus_off']:
 lines+=['',k+': '+json.dumps(comparison[k])]
lines+=['','结论：本轮没有达到70%，保留450。B最佳云影65.522634%，较450下降0.036656点；A最佳65.535307%。B-A为-0.012673点，107ROI区间[-0.094410,0.066829]。B地表和云分别较450提高0.004356、0.035992点，mIoU只提高0.001231点，未满足优先提升云影的目标。','','B同权重：新增分支开启比关闭云影低0.010693点，区间[-0.028493,0.005813]；环形对比开启比局部替代高0.001229点，区间[-0.000063,0.002838]，未显示明确正收益。两组按保护规则的最佳点均保留初始450。','','错误构成：B相对450云影漏检净减少68789像素，误报净增加113169像素；其中距真实云影边界超过8像素的误报增加67530。没有证据将这些误报直接归为水体或特定地物。','','这轮确认了在MsRE主投影中加入局部—环形均值对比仍未产生可用收益。该判断限于本次结构和6000步继续训练；没有训练失败、错误权重或参数未更新的问题。','','复算关闭对照需运行ablate_shadow467.py或ablate_ring467.py中的开关逻辑；关闭标志不属于权重张量，单独加载combined_network.pth不会自动恢复关闭状态。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*467.py'))+list((ROOT/'src/tools').glob('*467.sh'))+[ROOT/'src/research_plans/phase467.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
