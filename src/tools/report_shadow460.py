from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase460/phase460_20261106';D=R/'B/network_run'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
control=json.loads((ROOT/'outputs/phase459/phase459_20261106/A/network_run/FINAL_RESULTS.json').read_text())
a=next(x for x in control['results'] if 'best_shadow' in x['roles'])
final=json.loads((D/'FINAL_RESULTS.json').read_text());assert final['passed'];b=next(x for x in final['results'] if 'best_shadow' in x['roles'])
off=json.loads((D/'ABLATION_OFF.json').read_text())['results'][0];ref=json.loads((D/'REFERENCE_ABLATION_OFF.json').read_text())['results'][0]
rows=[('Phase450',base),('459A普通续训',a),('460新结构',b),('460整层关闭',off),('460仅参照关闭',ref)]
for prefix,x in [('',v) for v in final['results']]+[('OFF_',off),('REFERENCE_OFF_',ref)]:
 per=json.loads((D/f"{prefix}INDEPENDENT_{x['step']}.json").read_text())['per_image'];assert len(per)==535
 cm=[[sum(r['confusion'][i][j] for r in per) for j in range(3)] for i in range(3)]
 iou=[100*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)-cm[i][i]) for i in range(3)]
 assert max(abs(u-v) for u,v in zip(iou,x['IoU']))<1e-9
locks=[json.loads(p.read_text()) for p in [D/'LOCK.json',ROOT/'outputs/phase459/phase459_20261106/A/network_run/LOCK.json']]
for key in ['origin_sha','steps','batch_size','seed','branch_lr','task_lr','target_label_supervision_percent']:assert locks[0][key]==locks[1][key],key
assert (D/'INPUT_TRACE.jsonl').read_bytes()==(ROOT/'outputs/phase459/phase459_20261106/A/network_run/INPUT_TRACE.jsonl').read_bytes()
accept=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text())['B'];weight=D/f"branch_{b['step']}.pth";assert hashlib.sha256(weight.read_bytes()).hexdigest()==accept['sha'][weight.name]
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,*x['IoU'],x['mIoU'],0])
lines=['# Phase460：无遮挡地表参照解码器','','目标监督0%；CloudSEN训练8490、原始验证535；6000步、固定argmax。普通续训对照复用459A，相同训练批次签名已核对。','','新增196876参数：四尺度软区域预测、全图地表参照注意力、自身/参照差异与云上下文融合，注入像素解码器。训练解码器及分支，骨干与BN固定。区域预测只在训练时接受源标签监督，推理不输入标签。','','依据：[CANet（ICCV2021）](https://arxiv.org/abs/2108.09894)的非阴影上下文传递。与451末端检索不同，本轮在解码器前进行有监督参照读取和联合训练。','','| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
lines+=['',f"最佳训练点{b['step']}步，相对450三类差值：{b['IoU_delta']}；地表/云保护条件：{b['protected']}，达到70%目标：{b['success']}。",f"保护规则最终选点：{final['choices']['best_protected']}步。",'', '独立batch1的全部混淆矩阵已本地重算；新参数实际更新、骨干/BN固定、起点复现、两臂输入批次与权重SHA验收通过。107ROI配对区间见MATCHED_COMPARISON.json。单seed、同验证集选点，属于探索结果。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,target_supervision_percent=0),indent=2))
with (R/'RESULT.md').open('a',encoding='utf-8') as f:f.write('\n结论：未达到70%目标，保留450。新结构最佳影65.3778，较450低0.1815点；地表/云提高0.1376/0.1590点。相对普通续训459A最佳影仅+0.0438点，107ROI区间[-0.3619,0.4608]。\n\n整层关闭影65.1187，完整网络增加0.2592点，区间[-0.2196,0.7963]；仅参照关闭影65.3753，参照读取只增加0.0025点，区间[-0.0062,0.0116]。因此不能把新增层的整体作用归功于地表参照机制。\n\n相对450，新模型云影漏检增加30385像素，误报减少5869像素，减少误报不足以抵消新增漏检。6000步训练完整完成，用时954.35秒；这是续训探索，不是40k全量重训。\n\n新增52个参数张量全部更新；保护规则最优回退step0原450。所有指标来自原图、固定argmax，目标域监督0%。\n')
print(json.dumps({name:dict(IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows},ensure_ascii=False))
