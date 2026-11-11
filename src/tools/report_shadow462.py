from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase462/phase462_20261106'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
control=ROOT/'outputs/phase459/phase459_20261106/A/network_run';c=next(x for x in json.loads((control/'FINAL_RESULTS.json').read_text())['results'] if 'best_shadow' in x['roles'])
rows=[('450',base),('459A像素监督续训',c)];accept=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text());selections={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());assert final['passed'];best=next(x for x in final['results'] if 'best_shadow' in x['roles']);rows.append((arm+('区域查询监督' if arm=='A' else '云影内部/边缘监督'),best));selections[arm]=final['choices']
 for x in final['results']:
  per=json.loads((d/f"INDEPENDENT_{x['step']}.json").read_text())['per_image'];assert len(per)==535
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
lines=['# Phase462：区域查询监督与云影内部/边缘分解','','目标监督0%；CloudSEN训练8490、原始验证535。两臂各6000步，固定argmax，450起点。没有新增网络参数，推理结构不变。','','A加入有效区域加权的查询匈牙利匹配、分类CE、掩膜BCE/Dice；B进一步把云影内部和4像素边缘带作为两个同类区域分别匹配查询。几何边缘不是物理半影。查询损失权重0.1，每层监督；它不是原431随机点采样损失的逐项复现。','','| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
lines+=['', '选点：'+json.dumps(selections), '', '损失的标签守恒、有限梯度、255忽略梯度、全忽略零损失验收通过；两臂协议及抽查批次签名一致，骨干/BN固定，独立535张混淆矩阵已复算，最佳权重SHA通过。107ROI配对区间见MATCHED_COMPARISON.json。单seed、同验证集选点，属于训练方法探索。']
lines+=['', '结论：保留Phase450。A云影65.369448，比450低0.189843点；比普通续训459A仅高0.035420点，107ROI配对95%区间[-0.209642,0.290046]。B云影64.824357，比450低0.734934点，比A低0.545091点，区间[-0.984140,-0.126689]；所有相同步数B均低于A。两臂均未达到70%。', '', '错误来源：A相对450少漏23987个云影像素，但多误报79178个非云影像素；B多漏53914、多误报82679。B虽然减少距云影边界8像素以内的误报109068个，但更远处新增误报191747个，近边界漏检也增加61458个。几何边缘拆分没有带来整体改善。', '', '本轮否定的是当前损失权重、冻结范围与6000步续训协议下的这两种设置，不能推出所有查询监督或边缘建模无效。下一步不继续微调这项边缘拆分损失，保留450作为比较起点。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,target_supervision_percent=0),indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows},ensure_ascii=False))
