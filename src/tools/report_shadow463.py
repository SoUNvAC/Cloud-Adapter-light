from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase463/phase463_20261106';D=R/'B/network_run'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
control=json.loads((ROOT/'outputs/phase459/phase459_20261106/A/network_run/FINAL_RESULTS.json').read_text())
a=next(x for x in control['results'] if 'best_shadow' in x['roles'])
final=json.loads((D/'FINAL_RESULTS.json').read_text());assert final['passed'];b=next(x for x in final['results'] if 'best_shadow' in x['roles'])
off=json.loads((D/'ABLATION_OFF.json').read_text())['results'][0];ref=json.loads((D/'REFERENCE_ABLATION_OFF.json').read_text())['results'][0];direction=json.loads((D/'DIRECTION_ABLATION_OFF.json').read_text())['results'][0]
rows=[('Phase450',base),('459A普通续训',a),('463新结构',b),('463整层关闭',off),('463仅RGB线索关闭',ref),('463仅方向差异关闭',direction)]
for prefix,x in [('',v) for v in final['results']]+[('OFF_',off),('REFERENCE_OFF_',ref),('DIRECTION_OFF_',direction)]:
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
lines=['# Phase463：方向遮光线索前置空间分支','','目标监督0%；CloudSEN训练8490、原始验证535；6000步、固定argmax。普通续训对照复用459A，相同抽查训练批次签名已核对。','','新增17443参数：1/4网格RGB对数通道共同变化与色差，四方向、三距离，与原空间细节融合；零初始化注入context/detail，再进入四次空间交互。原骨干及BN固定，梯度通过冻结ViT。新增分支与450分支lr5e-5，解码器lr1e-5，0.1区域软三类监督。','','同权重消融：整层关闭、全部RGB线索归零（包括本地RGB/色度及方向差异）；全部RGB归零仍保留原空间特征与可学习偏置；另做仅方向差异归零，保留本地RGB/色度的消融。均为同权重零训练干预。','','依据：方向上下文 https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Direction-Aware_Spatial_Context_CVPR_2018_paper.html；材质一致边界 https://arxiv.org/abs/2409.06848。这里只使用观测RGB差，不宣称物理光照不变或复现原论文。','','| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
lines+=['',f"最佳训练点{b['step']}步，相对450三类差值：{b['IoU_delta']}；地表/云保护条件：{b['protected']}，达到70%目标：{b['success']}。",f"保护规则最终选点：{final['choices']['best_protected']}步。",'', '独立batch1的全部混淆矩阵已本地重算；新增参数实际更新、骨干/BN固定、起点复现、协议与抽查批次、权重SHA验收通过。107ROI配对区间见MATCHED_COMPARISON.json。单seed、同验证集选点，属于探索结果。']
lines+=['', '结论：未达到70%，保留450。新结构影65.410190，比450低0.149101点；地表/云分别提高0.127847/0.177719点。相对普通续训459A影+0.076162点，107ROI区间[0.000289,0.185519]，接近零的小幅差异不构成结构突破。', '', '整层开启对云影仅+0.012351点，区间[-0.081003,0.128462]；全部RGB线索开启-0.003152点，区间[-0.016155,0.009112]；方向差异开启-0.003362点，区间[-0.008665,0.001406]。当前没有方向线索的实质收益证据。', '', '相对450，漏检增加17939像素，误报增加5920。距影边界8像素以内误报减少80713个，但更远处误报增加86633个，未转化为整体收益。', '', '新增14个参数张量全部更新。训练用时1091.53秒，峰值显存见UPDATE_CHECK.json；原骨干/BN不变，初始端点、梯度、忽略标签、独立混淆矩阵和权重SHA验收通过。2026-11-06启动、11-07完成，目录保留启动日期。6000步续训，不是40k全量重训。', '', '下一步决策：停止在这个冻结主干续训设置下重复叠加类似RGB差异支路。当前结果说明这一实现未带来净收益，不把它推成云影分割的性能上限。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,target_label_supervision_percent=0),indent=2))
print(json.dumps({name:dict(IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
