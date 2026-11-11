"""Fixed-40K comparison, with a separate within-checkpoint early-path intervention."""
import csv,json,re
from pathlib import Path
R=Path('/home/scv/shared/phase501_20261111');B=Path('/home/scv/shared/phase496_20261110')
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
rows=[]
for name,root in [('496',B),('500',Path('/home/scv/shared/phase500_20261111')),('501',R)]:
    for item in json.loads((root/'RESULT.json').read_text()):
        if item['variant']!='texture':continue
        m=item['metrics']
        rows.append(dict(model=name,checkpoint=item['checkpoint'],surface=m['IoU_surface'],cloud=m['IoU_cloud'],shadow=m['IoU_shadow'],mIoU=m['mIoU_exact'],shadow_FN=m['FN_shadow'],shadow_FP=m['FP_shadow'],path=item['path'],sha256=item['sha256']))
assert len(rows)==6
with (R/'MAIN.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
final={r['model']:r for r in rows if r['checkpoint']=='final'}
delta={k:final['501'][k]-final['500'][k] for k in ['surface','cloud','shadow','mIoU','shadow_FN','shadow_FP']}
(R/'FINAL_DELTA.json').write_text(json.dumps(delta,indent=2))
curve=[];step=0;classes={}
logs=(R/'texture.log').read_text(errors='replace')
for line in logs.splitlines():
    h=re.search(r'Iter\(train\) \[\s*(\d+)/40000\]',line)
    if h:step=int(h[1])
    h=re.search(r'\|\s*(surface_visible|cloud|shadow)\s*\|\s*([0-9.]+)',line)
    if h:classes[h[1]]=float(h[2])
    h=re.search(r'Iter\(val\) \[268/268\].*?mIoU: ([0-9.]+)',line)
    if h:curve.append(dict(step=step,**classes,mIoU=float(h[1])))
assert len(curve)==20
with (R/'CURVE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
audit=json.loads((R/'INDEPENDENT_AUDIT.json').read_text())
assert audit['passed'] and audit['input_trace_matched_steps']==list(range(1,11))+list(range(2000,40001,2000))
lines=['# Phase501：去掉成对读取，保留高分辨率细节','',
'原始CloudSEN训练8490、验证535，目标监督0%，单seed42901。冻结官方DINOv3，任务模块全新初始化训练40K，没有从500训练检查点续训。','',
'改动：删除方向预测、两侧采样、成对编码和选择；融合由32维缩为16维，移除成对通道。保留500的真实1/2 CNN细节、语义融合及三类掩膜解码。共同层初始化相同；高分辨率分支沿用500初始值，移除零输入通道的常数项折入偏置，单元测试核对初始函数等价。','',
'| 模型 | 检查点 | 地表IoU | 云IoU | 云影IoU | mIoU | 云影FN | 云影FP |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model']} | {r['checkpoint']} | {r['surface']:.4f} | {r['cloud']:.4f} | {r['shadow']:.4f} | {r['mIoU']:.4f} | {r['shadow_FN']:,} | {r['shadow_FP']:,} |")
lines+=['','固定40K差值（501−500）：'+', '.join(f'{k} {v:+.4f}' for k,v in delta.items())+'。']
d496={k:final['501'][k]-final['496'][k] for k in delta}
(R/'DELTA_VS496.json').write_text(json.dumps(d496,indent=2))
lines+=['','固定40K差值（501−496）：'+', '.join(f'{k} {v:+.4f}' for k,v in d496.items())+'。']
bd=json.loads((R/'BOUNDARY_DELTA_VS500.json').read_text())
lines+=['','| 距真值边界 | FN变化（501−500） | FP变化（501−500） |','|---|---:|---:|']
for b,caption in [('within3','≤3px'),('3to8','3–8px'),('beyond8','>8px')]:
    lines.append(f"| {caption} | {bd[b]['FN']:+,} | {bd[b]['FP']:+,} |")
m=json.loads((R/'HIGHRES_OFF.json').read_text())['metrics']
lines+=['',f"同权重关闭高分辨率细节残差、保留双线性升采样：地表/云/影/mIoU={m['IoU_surface']:.4f}/{m['IoU_cloud']:.4f}/{m['IoU_shadow']:.4f}/{m['mIoU_exact']:.4f}；完整501相对关断，云影{final['501']['shadow']-m['IoU_shadow']:+.4f}点。这是推理干预，不替代重训对照。"]
cost=json.loads((R/'COST.json').read_text())['median_ms']
new=audit['boundary_parameters'];total=audit['texture']['trainable_parameters']
assert new==6160
lines+=['',f'细节分支{new:,}参数，可训练总数{total:,}；相对500减少1653参数。GPU前向496={cost["phase496"]:.3f}ms、500={cost["phase500"]:.3f}ms、501={cost["phase501"]:.3f}ms；501比500变化{100*(cost["phase501"]/cost["phase500"]-1):+.2f}%。batch1/512/FP32，20次预热、32次轮换计时，不含IO。',
'','验收：细节分支运行图无成对模块和grid_sample，零初始化残差、1/2输出、折叠函数等价、全部新增参数更新、冻结VFM、共同初始化、30个抽查批次、代码锁及最终/最优重载通过。单seed，不调阈值。权重留远端，路径和SHA见MAIN.csv。']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
print((R/'RESULT.md').read_text())
