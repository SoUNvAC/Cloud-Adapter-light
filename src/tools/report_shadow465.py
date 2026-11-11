from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase465/phase465_20261107'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
control=ROOT/'outputs/phase459/phase459_20261106/A/network_run';c=next(x for x in json.loads((control/'FINAL_RESULTS.json').read_text())['results'] if 'best_shadow' in x['roles'])
original_off=json.loads((R/'original_interaction_off/ABLATION_OFF.json').read_text())['results'][0]
per=json.loads((R/'original_interaction_off/OFF_INDEPENDENT_6000.json').read_text())['per_image'];assert len(per)==535
assert [[sum(r['confusion'][i][j] for r in per) for j in range(3)] for i in range(3)]==original_off['confusion']
rows=[('450',base),('450关闭全部原交叉注意力',original_off),('459A仅解码器续训',c)];accept=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text());selections={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());assert final['passed'];best=next(x for x in final['results'] if 'best_shadow' in x['roles']);rows.append((arm+('尺度控制' if arm=='A' else '尺度控制加类别读取'),best));selections[arm]=final['choices']
 checks=[('',x) for x in final['results']]
 if arm=='B':
  off=json.loads((d/'ABLATION_OFF.json').read_text())['results'][0];rows.append(('B同权重关闭路由',off));checks.append(('OFF_',off));clsoff=json.loads((d/'CLASS_ABLATION_OFF.json').read_text())['results'][0];rows.append(('B仅关闭类别分区',clsoff));checks.append(('CLASS_OFF_',clsoff))
 for prefix,x in checks:
  per=json.loads((d/f"{prefix}INDEPENDENT_{x['step']}.json").read_text())['per_image'];assert len(per)==535
  cm=[[sum(r['confusion'][i][j] for r in per) for j in range(3)] for i in range(3)]
  iou=[100*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(u-v) for u,v in zip(iou,x['IoU']))<1e-9
 assert (d/'INPUT_TRACE.jsonl').read_bytes()==(control/'INPUT_TRACE.jsonl').read_bytes()
 locks=[json.loads(p.read_text()) for p in [d/'LOCK.json',control/'LOCK.json']]
 for key in ['origin_sha','steps','batch_size','seed','branch_lr','task_lr','target_label_supervision_percent']:assert locks[0][key]==locks[1][key],key
 weight=d/f"branch_{best['step']}.pth";assert hashlib.sha256(weight.read_bytes()).hexdigest()==accept[arm]['sha'][weight.name]
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,*x['IoU'],x['mIoU'],0])
lines=['# Phase465：尺度控制与分区上下文读取','','目标监督0%；CloudSEN源训练8490、原始验证535，450起点；两臂各6000步、batch4、seed45201，固定argmax。原交互/解码器lr1e-5、分支5e-5，其余原骨干/BN固定。','','诊断464：固定32张源训练图、零优化更新；每层抽取64个query、保留全部key。第四层原注意力分数std28.42，路由偏置std0.00106；新偏置几乎改不动原注意力。分割/保护梯度平均余弦为正，未见平均方向明显冲突。诊断采用数学SDPA支持重复求梯度，详情和抽样索引见ROUTING_DIAG.json。','','A：独立读取通道对每行注意力分数去均值，std超过1时缩放到1。B：在相同尺度控制上，按key软三类概率分别归一化聚合value，再由query类别及每头3×3关系矩阵混合。两组都用0.1软区域CE，B路由参数5688，A不使用的关系矩阵固定。','','新通道混合系数前1000步由0线性升到0.5，之后固定；不是阈值搜索，不按验证成绩调系数。B同权重测试关闭整条新读取、仅关闭类别分区，区分类别机制与尺度控制。','','| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
lines+=['','选点：'+json.dumps(selections),'','初始增益0复现450；新路径梯度、关闭复现、类别关闭等价、全ignore零损失、原交互/新增参数实际更新、其余骨干/BN固定、批次及权重签名、535图独立混淆矩阵验收通过。107ROI配对区间见MATCHED_COMPARISON.json。单seed、同验证集选点，6000步续训，不是40k全量重训。']
lines+=['','结论：保留450，未达到70%。A影65.323189，较450低0.236102点；B影65.308336，低0.250955点。两者地表/云均高于450，保护规则最佳均回退step0。B-A影-0.014853点，107ROI区间[-0.049387,0.016248]。','','同权重B整条新读取开启相对关闭影+0.052209点，区间[-0.024836,0.177496]；云IoU-0.022367点。仅类别分区开启影+0.000318点，区间[-0.026639,0.025061]。放大读取后点估计作用增加，但没有整体净收益，类别机制仍无明确贡献。B相对450漏检+16713、误报+30621像素。','','450原交互全关：地表/云/影变化-0.054874/-0.178256/+0.010546点，云影召回-0.967607点、非影误报率-0.144273点；漏检+119452、误报-184528像素。不满足其他两类保护条件，不能替换450，也不能说原交互完全无用。它对当前权重的云影IoU净贡献小，但确实改变了错误构成。该干预不证明没有交互的网络重新训练后必然等价。','','验收：A新增16个预测器张量更新、4个未使用关系矩阵固定；B新增20个张量全部更新；原交互更新、其余原骨干/BN固定、起点增益0/最佳增益0.5和类别模式、独立混淆矩阵与权重SHA通过。训练A2400.48秒、B2594.75秒。','','下一步决策：降低交叉注意力路由小改的优先级；优先核查MsRE与局部读写等仍保留路径的实际贡献，再选择结构改造位置。当前结论仅针对已验收权重及6000步续训设置。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,target_supervision_percent=0),indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
