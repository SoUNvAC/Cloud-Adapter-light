from pathlib import Path
import json,csv
R=Path(__file__).resolve().parents[2]/'outputs/phase446/phase446_20261106/network_run'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
f=read(R/'FINAL_RESULTS.json');lock=read(R/'LOCK.json');done=read(R/'COMPLETE.json')
previous=read(R.parents[2]/'phase445/phase445_20261106/network_run/FINAL_RESULTS.json')['results'][0]
lines=['# Phase446：双判别头局部遮光网络结果','',
'在445环形地表参照上，将单一有符号云影修正头拆成独立补漏、纠错两头，按冻结431的影/非影预测路由；增加0.1倍路由内影/非影均衡二分类损失。两头均零初始化、有符号输出。',
f"原431及BN冻结，从头训练新增{lock['branch_parameters']}参数；源训练8490图，原始验证535图；目标域标签适配 **0%**。6000步、batch4、seed44501、学习率2e-4，与445同预算。耗时{done['seconds']:.1f}秒（含训练与7轮验证），另做最佳权重单张完整复测。",'',
'|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
base=f['results'][0]['baseline'];rows=[('原431',base),('Phase445',previous)]+[(f"Phase446 step{v['step']}",v) for v in f['results']]
for n,v in rows:lines.append(f"|{n}|{v['IoU'][0]:.4f}|{v['IoU'][1]:.4f}|{v['IoU'][2]:.4f}|{v['mIoU']:.4f}|")
summary=[]
for v in f['results']:
 t=v['transition'];recovered=sum(t[2][j][2] for j in [0,1]);lost=sum(t[2][2][j] for j in [0,1]);removed=sum(t[c][2][j] for c in [0,1] for j in [0,1]);added=sum(t[c][j][2] for c in [0,1] for j in [0,1])
 d=[a-b for a,b in zip(v['IoU'],previous['IoU'])]
 lines+=['',f"step{v['step']}：相对431三类IoU变化{v['IoU_delta']}；相对445变化{d}个百分点。非影保护通过={v['protected']}，云影70%目标通过={v['success']}。",f'原漏影找回{recovered}、原正确影损伤{lost}，净找回{recovered-lost}；消除原误报{removed}、新增误报{added}，误报净变化{added-removed}。']
 summary.append(dict(step=v['step'],delta_vs445=d,recovered=recovered,lost=lost,removedFP=removed,addedFP=added))
 # Independent stdlib reconstruction of all image confusion totals and all metrics.
 per=read(R/f"INDEPENDENT_{v['step']}.json")['per_image'];assert len(per)==535
 cm=[[sum(x['confusion'][a][b] for x in per) for b in range(3)] for a in range(3)]
 assert cm==v['confusion']
 iou=[100*cm[c][c]/(sum(cm[c])+sum(x[c] for x in cm)-cm[c][c]) for c in range(3)]
 assert max(abs(a-b) for a,b in zip(iou,v['IoU']))<1e-10
 assert v['state_unchanged']
lines+=['','本轮同时改变双头结构与训练损失，评价整体方案。验证集每1000步选检查点；独立复测指重新加载和计算，仍使用同一验证集。ROI区间见FINAL_RESULTS.json。','', '推理代码：src/tools/infer_shadow446.py；新增网络：src/tools/shadow_branch446.py；训练：src/tools/train_shadow446.py。整网权重保留远端independent/step_x/combined_network.pth，本地保存所有轻量分支权重及原431权重来源。']
(R/'READOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,images=535,summary=summary),indent=2))
with (R/'MAIN.csv').open('w',newline='',encoding='utf-8-sig') as h:
 w=csv.writer(h);w.writerow(['model','surfaceIoU','cloudIoU','shadowIoU','mIoU']);w.writerows([[n,*v['IoU'],v['mIoU']] for n,v in rows])
print(json.dumps(summary,indent=2))
