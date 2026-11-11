from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase450/phase450_20261106/network_run'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
f=read(R/'FINAL_RESULTS.json');lock=read(R/'LOCK.json');done=read(R/'COMPLETE.json')
previous=read(ROOT/'outputs/phase449/phase449_20261106/network_run/FINAL_RESULTS.json')['results'][0]
lines=['# Phase450：图像驱动的遮光候选结果','', '完整保留449并新增图像亮度下降方向的±2/6/12像素采样，比较对数RGB、色度、纹理与影/地表概率。方向无需原模型预测出云影边界。新增1073参数，零初始化输出，其他损失相同。',f"原431和BN完全冻结，新增分支{lock['branch_parameters']}参数从头训练6000步；batch4、seed44501、学习率2e-4，训练8490张原始源图、验证535张原始源图。目标标签监督0%。训练含验证耗时{done['seconds']:.1f}秒。",'', '|模型|步数|地表IoU|云IoU|云影IoU|mIoU|保护通过|','|---|---:|---:|---:|---:|---:|---|']
reference448=read(ROOT/'outputs/phase448/phase448_20261106/network_run/FINAL_RESULTS.json')['results'][0]
base=f['results'][0]['baseline'];rows=[('原431',base),('449预测边界分支',previous),('448方向参照',reference448)]+[(f"450 {'/'.join(v['roles'])}",v) for v in f['results']]
for n,v in rows:lines.append(f"|{n}|{v.get('step','—')}|{v['IoU'][0]:.4f}|{v['IoU'][1]:.4f}|{v['IoU'][2]:.4f}|{v['mIoU']:.4f}|{v.get('protected','—')}|")
summary=[]
for v in f['results']:
 t=v['transition'];recovered=sum(t[2][j][2] for j in [0,1]);lost=sum(t[2][2][j] for j in [0,1]);removed=sum(t[c][2][j] for c in [0,1] for j in [0,1]);added=sum(t[c][j][2] for c in [0,1] for j in [0,1]);d=[a-b for a,b in zip(v['IoU'],previous['IoU'])]
 lines+=['',f"step{v['step']}：相对431三类IoU变化{v['IoU_delta']}；相对449变化{d}个百分点。70%目标通过={v['success']}。",f'找回漏影{recovered}、新增漏影{lost}，净找回{recovered-lost}；消除误报{removed}、新增误报{added}，净增加误报{added-removed}。']
 summary.append(dict(step=v['step'],delta_vs449=d,recovered=recovered,lost=lost,removedFP=removed,addedFP=added))
 per=read(R/f"INDEPENDENT_{v['step']}.json")['per_image'];assert len(per)==535
 cm=[[sum(x['confusion'][a][b] for x in per) for b in range(3)] for a in range(3)];assert cm==v['confusion'];assert [[sum(t[a][j][b] for j in range(3)) for b in range(3)] for a in range(3)]==cm
 iou=[100*cm[c][c]/(sum(cm[c])+sum(x[c] for x in cm)-cm[c][c]) for c in range(3)];assert max(abs(a-b) for a,b in zip(iou,v['IoU']))<1e-10;assert v['state_unchanged']
lines+=['','每1000步验证并选检查点，最终重载权重单张完整复测；同一验证集、单seed。ROI区间见FINAL_RESULTS.json。网络代码src/tools/shadow_branch450.py，训练代码src/tools/train_shadow450.py。本地保存全部轻量分支权重；整网权重和逐图预测另存远端D盘。']
(R/'READOUT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,images=535,summary=summary),indent=2))
with (R/'MAIN.csv').open('w',newline='',encoding='utf-8-sig') as h:
 w=csv.writer(h);w.writerow(['model','surfaceIoU','cloudIoU','shadowIoU','mIoU']);w.writerows([[n,*v['IoU'],v['mIoU']] for n,v in rows])
print(json.dumps(summary,indent=2))
