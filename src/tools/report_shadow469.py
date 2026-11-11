from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase469/phase469_20261107'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
accept=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text());assert accept['identical_training_batches']
comparison=json.loads((R/'MATCHED_COMPARISON.json').read_text());audit=json.loads((R/'MODULE_AUDIT.json').read_text());assert audit['passed']
rows=[('450完整网络',base)];selections={};times={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles']);rows.append((arm+('加性V适配' if arm=='A' else '相对V调制'),best));selections[arm]=final['choices']
 complete=json.loads((d/'COMPLETE.json').read_text());assert complete['steps']==6000;times[arm]=complete['seconds']
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','B关闭新增分支'),('ABLATION_RELATIVE_OFF.json','RELATIVE_OFF_','B改为加性V修正')]:
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
lines=['# Phase469：VFM内部云遮挡／地表遮光双路适配','','目标监督0%；CloudSEN训练8490/原始验证535；450起点，两组各6000步，batch4、seed45201，固定argmax。','','在DINOv3最后4个block（索引8–11）的原生QKV内部加入两路rank4适配。云路加性修改QKV，云影路仅修改V；三类软门控由token预测，地表保留原路径。5个前缀token的QKV不直接修改。','','A：云影路为门控加性V残差；B：云影路为原V乘以0.5*tanh(低秩输出)的门控相对修正。两组新增参数均41484，所有共享参数初始化一致。B的相对调制是隐特征约束，不是物理光照系数。','','原网络全部权重（含VFM、CloudAdapter/MsRE、解码器、450分支）和BN冻结，仅训练新增参数lr1e-4。沿用固定450参考的分割与保持损失，两组均增加0.05三类token软占比监督，排除255。新层零初始化，起点复现450。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','训练耗时/秒：'+json.dumps(times),'','同权重B关闭全部新增VFM层，以及将相对V调制替换为加性V修正；后者改变函数形式及幅度，不等于重新训练的A。','','验收：实际初始化、训练批次、原权重与BN冻结、新增参数更新、535图独立混淆矩阵、权重与传输哈希通过。107ROI配对bootstrap2000次；单seed验证集选点的6000步续训探索。']
for k in ['B_minus_A_IoU_delta','CI95','B_vfm_on_minus_off','B_relative_on_minus_off']:
 lines+=['',k+': '+json.dumps(comparison[k])]
a_result=rows[1][1];b_result=rows[2][1]
lines+=['','最佳训练点：B较450三类IoU变化为'+json.dumps(b_result['IoU_delta'])+'；A变化为'+json.dumps(a_result['IoU_delta'])+'。']
boundary=json.loads((R/'B/network_run/BOUNDARY_DIAG.json').read_text());err=boundary['delta469minus450']
lines+=['','B对450：漏检净变化'+str(sum(err['FN']))+'像素，误报净变化'+str(sum(err['FP']))+'像素；按0–3、3–8、8以上像素边界距离分解为'+json.dumps(err)+'。','', '关闭实验以对应脚本中的enabled/relative_enabled开关复算；仅加载权重不会恢复Python开关状态。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (R/'RESULT.md').open('a',encoding='utf-8') as f:
 f.write('\n结论：本轮未获得云影净收益，保留Phase450。A/B最佳训练点均为1000步，地表、云IoU略升，云影分别下降0.201071、0.209185个百分点；6000步末点也未超过450。B较A云影低0.008114点，配对区间跨0。关闭全部新增层精确恢复450；同权重相对调制改加性后，云影仅差0.000470点，未显示该算子的优势。B少漏检78166像素，但多误报166437像素，是本轮净损失的直接来源。该结果否定当前冻结基座、双路低秩软门控方案的有效性，不能外推为所有VFM内部适配均无效。\n')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*469.py'))+list((ROOT/'src/tools').glob('*469.sh'))+[ROOT/'src/research_plans/phase469.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
