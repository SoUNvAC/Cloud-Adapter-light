from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase468/phase468_20261107'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
accept=json.loads((R/'TRAINED_WEIGHT_ACCEPTANCE.json').read_text());assert accept['identical_training_batches']
comparison=json.loads((R/'MATCHED_COMPARISON.json').read_text());audit=json.loads((R/'MODULE_AUDIT.json').read_text());assert audit['passed']
rows=[('450完整网络',base)];selections={};times={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=json.loads((d/'FINAL_RESULTS.json').read_text());assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles']);rows.append((arm+('普通细节桥' if arm=='A' else '亮度色度分频桥'),best));selections[arm]=final['choices']
 complete=json.loads((d/'COMPLETE.json').read_text());assert complete['steps']==6000;times[arm]=complete['seconds']
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','B关闭新增分支'),('ABLATION_FREQUENCY_OFF.json','FREQUENCY_OFF_','B频带信息置零')]:
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
lines=['# Phase468：高分辨率亮度—色度细节解码','','目标监督0%；原始CloudSEN训练8490/验证535，450起点；A、B各6000步、batch4、seed45201，固定argmax。','','A：1/4空间CNN细节24维＋ViT语义16维＋RGB细节24维，64→32→384生成零初始化特征残差，加入最细解码特征。新增23128参数。B：增加三个尺度、三个方向的非降采样Haar式细节；各频带拆成共同亮度变化与色度残量，由空间/语义及频带强度学习门控。新增总计25657参数，比A多2529。','','两组只训练新桥lr1e-4和原解码器lr1e-5；原骨干、450分支及BN固定。零初始化保持450起点，原450语义读取与解码器接收同一组更新特征。全三类监督和保持约束使用固定450预测，不使用额外目标标签或辅助标签。','','设计动机是将真实1/4细节接入解码器，而不是仅上采样16像素patch特征；亮度/色度分频为可学习外观线索，不是物理照度估计。Haar细节思想参考[CDWMamba](https://doi.org/10.3390/rs17111874)和[WavShadow预印本](https://arxiv.org/abs/2411.05747)，不声称复现其模型或该基础算子为原创。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','训练耗时/秒：'+json.dumps(times),'','同权重关闭整桥与频带信息置零，均使用B最佳训练权重；后者保留所有已学参数/偏置及普通RGB/CNN细节。各步A/B比较见MATCHED_COMPARISON.json。','','验收：起点复现、共享初始化一致、所有新参数梯度与真实更新、原骨干/450分支/BN冻结、训练批次一致、535图独立混淆矩阵、权重和传输SHA通过。107ROI配对重抽样2000次。单seed、同验证集选点探索，6000步继续训练。B-A包含2529参数差异，不能只归于频带变换本身。']
for k in ['B_minus_A_IoU_delta','CI95','B_bridge_on_minus_off','B_frequency_on_minus_off']:
 lines+=['',k+': '+json.dumps(comparison[k])]
a_result=rows[1][1];b_result=rows[2][1]
lines+=['','最佳训练点：B较450三类IoU变化为'+json.dumps(b_result['IoU_delta'])+'；A变化为'+json.dumps(a_result['IoU_delta'])+'。']
boundary=json.loads((R/'B/network_run/BOUNDARY_DIAG.json').read_text());err=boundary['delta468minus450']
lines+=['','B对450：漏检净变化'+str(sum(err['FN']))+'像素，误报净变化'+str(sum(err['FP']))+'像素；按0–3、3–8、8以上像素边界距离分解为'+json.dumps(err)+'。','', '关闭实验以对应脚本中的enabled开关或bands置零干预复算；仅加载权重不会恢复Python开关状态。']
causal=json.loads((R/'B/network_run/BOUNDARY_CAUSAL.json').read_text());assert causal['passed'] and causal['forwards']==0
lines+=['','结论：保持450。B最佳地表/云分别提高0.166161/0.264980点，mIoU提高0.069410点，但云影下降0.222911点，未达到70%。B对普通细节桥A的云影增量仅0.005271点，107ROI区间[-0.009888,0.018788]。','','B整桥开启相对关闭，云影-0.069008点，区间[-0.162505,0.021609]。频带信息开启相对置零，云影+0.014321点，区间[-0.008883,0.039403]。没有明确的整体云影正收益。','','零前向的同权重错误定位：整桥开启，相对关闭，0–3像素边界带误报减少51217、漏检增加17445；频带开启相对置零，0–3像素边界漏检减少15686，但8像素以外误报增加13445、漏检增加1548。频带具有局部边缘作用，尚未转化为全图收益。不能将这些未审阅误报直接命名为水体或暗地表。','','A/B最佳训练点均为3000步；保护规则最佳保留step0。验证集用于探索和选点，区间不替代多seed稳定性验证。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*468.py'))+list((ROOT/'src/tools').glob('*468.sh'))+[ROOT/'src/research_plans/phase468.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
