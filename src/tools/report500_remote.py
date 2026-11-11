"""Fixed-40K comparison, with a separate within-checkpoint early-path intervention."""
import csv,json,re
from pathlib import Path
R=Path('/home/scv/shared/phase500_20261111');B=Path('/home/scv/shared/phase496_20261110')
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
rows=[]
for name,root in [('496',B),('500',R)]:
    for item in json.loads((root/'RESULT.json').read_text()):
        if item['variant']!='texture':continue
        m=item['metrics']
        rows.append(dict(model=name,checkpoint=item['checkpoint'],surface=m['IoU_surface'],cloud=m['IoU_cloud'],shadow=m['IoU_shadow'],mIoU=m['mIoU_exact'],shadow_FN=m['FN_shadow'],shadow_FP=m['FP_shadow'],path=item['path'],sha256=item['sha256']))
assert len(rows)==4
with (R/'MAIN.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
final={r['model']:r for r in rows if r['checkpoint']=='final'}
delta={k:final['500'][k]-final['496'][k] for k in ['surface','cloud','shadow','mIoU','shadow_FN','shadow_FP']}
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
lines=['# Phase500：1/2分辨率边界两侧联合解码','',
'原始CloudSEN训练8490、验证535；目标监督0%，单seed42901。冻结官方DINOv3，任务模块重新初始化40K。496为已完成对照。','',
'496诊断：72.89%漏影位于真值边界内8像素，该区域占云影像素58.78%；约50.63%误报距云影超过8像素。边界距离按最近相反有效类别的像素中心计算，不把图像外沿人为视为阴影边界。','',
'改动：保留496骨干及纹理适配器；取空间CNN真实1/2细节，联合原128维掩膜语义特征，学习4个采样方向、各2/6/12像素距离的两侧读取与选择。新的细节残差写入1/2掩膜特征，原查询解码三类。没有真值边界输入、辅助损失或阈值调整。共同层初始化相同；结构与分辨率共同改变，初始全模型预测不要求等同496。','',
'| 模型 | 检查点 | 地表IoU | 云IoU | 云影IoU | mIoU | 云影FN | 云影FP |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model']} | {r['checkpoint']} | {r['surface']:.4f} | {r['cloud']:.4f} | {r['shadow']:.4f} | {r['mIoU']:.4f} | {r['shadow_FN']:,} | {r['shadow_FP']:,} |")
lines+=['','固定40K差值（500−496）：'+', '.join(f'{k} {v:+.4f}' for k,v in delta.items())+'。']
bd=json.loads((R/'BOUNDARY_DELTA.json').read_text())
lines+=['','| 距真值边界 | 云影FN变化 | 云影FP变化 |','|---|---:|---:|']
for b,caption in [('within3','≤3px'),('3to8','3–8px'),('beyond8','>8px')]:
    lines.append(f"| {caption} | {bd[b]['FN']:+,} | {bd[b]['FP']:+,} |")
lines+=['','负数表示错误减少。完整按场景统计见DIAGNOSIS496_SCENES.json与DIAGNOSIS500_SCENES.json。']
for file,caption in [('PAIR_OFF.json','只关闭成对读取，保留学习的中心细节支路'),('HIGHRES_OFF.json','关闭全部新增细节残差，保留掩膜特征双线性升采样')]:
    m=json.loads((R/file).read_text())['metrics']
    lines+=['',f"{caption}：地表/云/影/mIoU={m['IoU_surface']:.4f}/{m['IoU_cloud']:.4f}/{m['IoU_shadow']:.4f}/{m['mIoU_exact']:.4f}；完整500相对该关断的云影差值{final['500']['shadow']-m['IoU_shadow']:+.4f}点。"]
cost=json.loads((R/'COST.json').read_text())['median_ms']
a=cost['phase496'];b=cost['phase500']
new=audit['boundary_parameters'];total=audit['texture']['trainable_parameters']
lines+=['',f'新增{new:,}参数，可训练总数{total:,}。GPU前向496={a:.3f}ms，500={b:.3f}ms（{100*(b/a-1):+.2f}%）。batch1/512/FP32，20次预热和32次交替计时，不含IO。',
'','验收：零初始化残差、1/2输出尺寸、采样坐标与越界屏蔽、全部新增参数更新、冻结VFM、共同初始化、30个抽查批次、代码签名与最终/最优检查点重载通过。关断用于检查推理依赖，不替代重训消融；本轮是单seed探索。权重路径与SHA见MAIN.csv。']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
print((R/'RESULT.md').read_text())
