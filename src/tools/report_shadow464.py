from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase464/phase464_20261107'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
control=ROOT/'outputs/phase459/phase459_20261106/A/network_run';c=next(x for x in json.loads((control/'FINAL_RESULTS.json').read_text())['results'] if 'best_shadow' in x['roles'])
rows=[('450',base),('459A仅解码器续训',c)];accept=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text());selections={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());assert final['passed'];best=next(x for x in final['results'] if 'best_shadow' in x['roles']);rows.append((arm+('解冻交互' if arm=='A' else '交互内部类别路由'),best));selections[arm]=final['choices']
 checks=[('',x) for x in final['results']]
 if arm=='B':
  off=json.loads((d/'ABLATION_OFF.json').read_text())['results'][0];rows.append(('B同权重关闭路由',off));checks.append(('OFF_',off))
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
lines=['# Phase464：交互内部的云/地表/云影路由','','目标监督0%；CloudSEN源训练8490、原始验证535，450起点。两臂各6000步，固定argmax；原交互层和解码器lr1e-5、分支5e-5，VFM/其他原骨干参数/BN固定。','','A只解冻四次原交互；B相同解冻范围增加5688参数：query/key软三类预测、每头3×3类别关系矩阵，经4*tanh限幅形成注意力logit偏置，关系矩阵零初始化。四层query/key软标签CE平均权重0.1，忽略标签覆盖加权。B-A包含新路由和辅助监督的联合训练效应，不分离二者。','','| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
lines+=['','选点：'+json.dumps(selections),'','零初始化、梯度、关闭路由复现原交互、全ignore零损失、原交互和新增参数实际更新、其余骨干/BN固定、抽查批次及协议签名、独立混淆矩阵和最佳权重SHA验收通过。107ROI配对区间见MATCHED_COMPARISON.json。单seed、同验证集选点；6000步续训，不是40k全量重训。']
lines+=['','结论：保留450，未达到70%。A影65.270480，比450低0.288811点；B影65.303610，比450低0.255681点。B比A仅+0.033130点，107ROI区间[-0.004561,0.083217]；2000～6000步B均低于A，未形成持续优势。','','同权重关闭B路由，影65.303710；开启-0.000100点，区间[-0.001620,0.001358]。原交互32个张量、新增路由20个张量均更新，但当前路由推理作用几乎为零。不能把B与A的微小训练差异归因于路由机制。','','相对450，A漏检+11295、误报+47424；B漏检+11006、误报+40421。B近边界3像素漏检减少30860，但更深内部漏检增加，整体没有改善。','','两组训练用时A1943.44秒、B1964.26秒，并行约33分钟。最佳训练点均1000步，保护选择均回退step0。权重和可复算记录已本地及远端归档。','','本轮在相同训练预算下没有证实解冻交互或类别路由的收益。下一步不继续在此配置上堆路由参数，应先改变能让新结构获得实质作用的训练或连接方式；该结果不代表全量重训的结论。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,target_supervision_percent=0),indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
