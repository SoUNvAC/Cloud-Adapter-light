from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase461/phase461_20261106';D=R/'B/network_run'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
control=json.loads((ROOT/'outputs/phase459/phase459_20261106/A/network_run/FINAL_RESULTS.json').read_text())
a=next(x for x in control['results'] if 'best_shadow' in x['roles'])
final=json.loads((D/'FINAL_RESULTS.json').read_text());assert final['passed'];b=next(x for x in final['results'] if 'best_shadow' in x['roles'])
off=json.loads((D/'ABLATION_OFF.json').read_text())['results'][0];ref=json.loads((D/'REFERENCE_ABLATION_OFF.json').read_text())['results'][0];shuffled=json.loads((D/'SHUFFLED_ABLATION.json').read_text())['results'][0]
rows=[('Phase450',base),('459A普通续训',a),('461新结构',b),('461整层关闭',off),('461仅参照关闭',ref),('461参照错位',shuffled)]
for prefix,x in [('',v) for v in final['results']]+[('OFF_',off),('REFERENCE_OFF_',ref),('SHUFFLED_',shuffled)]:
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
lines=['# Phase461：稀疏地表对应与遮光差异','','目标监督0%；CloudSEN训练8490、原始验证535；6000步、固定argmax。普通续训对照复用459A，相同训练批次签名已核对。','','新增166156参数：四尺度top8地表参照，语义和色度匹配；仅从语义/观测RGB对数/色度/相对纹理差、置信度和余弦差生成解码特征。训练解码器及分支，骨干与BN固定。区域预测只在训练时接受源标签监督，推理不输入标签。','','依据：[CANet（ICCV2021）](https://arxiv.org/abs/2108.09894)的非阴影上下文传递。460零更新诊断发现参照注意力熵0.94～0.97、top8质量仅5.8%～8.2%；本轮采用稀疏匹配并去掉直接局部输入。RGB差异只是观测线索，不是物理透射率。','','| 模型 | 地表IoU | 云IoU | 云影IoU | mIoU |','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' |')
lines+=['',f"最佳训练点{b['step']}步，相对450三类差值：{b['IoU_delta']}；地表/云保护条件：{b['protected']}，达到70%目标：{b['success']}。",f"保护规则最终选点：{final['choices']['best_protected']}步。",'', '独立batch1的全部混淆矩阵已本地重算；新参数实际更新、骨干/BN固定、起点复现、两臂输入批次与权重SHA验收通过。107ROI配对区间见MATCHED_COMPARISON.json。单seed、同验证集选点，属于探索结果。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,target_supervision_percent=0),indent=2))
with (R/'RESULT.md').open('a',encoding='utf-8') as f:f.write('\n结论：未达到70%云影IoU，保留450。最佳训练点影65.3607，较450低0.1986点；地表/云提高0.1269/0.1594点。较普通续训459A最佳影+0.0266点，107ROI区间[-0.3711,0.4404]；较460影-0.0171点。\n\n同权重完整/整层关闭/参照关闭/参照错位的云影IoU为65.3607/65.1549/65.3629/65.3623。参照开启相对关闭-0.0023点，区间[-0.0121,0.0049]；正确对应相对错位-0.0016点，区间[-0.0081,0.0042]。当前实现仍无实质参照机制收益，不能把整层作用归因于稀疏地表对应。\n\n错位干预：在每尺度查询网格上展平参照特征、参照RGB线索及置信度，固定循环平移17个位置，保留各自分布；不是另行训练，也不是完全随机置乱。参照关闭时成对差异归零，新增卷积的可学习偏置仍保留。\n\n新增52个参数张量全部更新。两组协议及抽查批次签名、初始端点、骨干/BN固定、全部独立混淆矩阵、最佳权重SHA验收通过。6000步续训用时957.45秒，不是40k全量重训；单seed探索。\n')
print(json.dumps({name:dict(IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows},ensure_ascii=False))
