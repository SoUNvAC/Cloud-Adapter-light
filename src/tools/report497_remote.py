"""Compare the fixed 40K experiment to completed same-seed Phase496."""
import csv,json,re
from pathlib import Path
R=Path('/home/scv/shared/phase497_20261110');B=Path('/home/scv/shared/phase496_20261110')
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
rows=[]
for name,root in [('496',B),('497',R)]:
    for item in json.loads((root/'RESULT.json').read_text()):
        if item['variant']!='texture':continue
        m=item['metrics']
        rows.append(dict(model=name,checkpoint=item['checkpoint'],surface=m['IoU_surface'],cloud=m['IoU_cloud'],shadow=m['IoU_shadow'],mIoU=m['mIoU_exact'],shadow_FN=m['FN_shadow'],shadow_FP=m['FP_shadow'],path=item['path'],sha256=item['sha256']))
assert len(rows)==4
with (R/'MAIN.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
final={r['model']:r for r in rows if r['checkpoint']=='final'}
delta={k:final['497'][k]-final['496'][k] for k in ['surface','cloud','shadow','mIoU','shadow_FN','shadow_FP']}
(R/'FINAL_DELTA.json').write_text(json.dumps(delta,indent=2))
curve=[];step=0;classes={}
logs=(R/'texture.log').read_text(errors='replace')
if (R/'resume.log').exists():logs+='\n'+(R/'resume.log').read_text(errors='replace')
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
lines=['# Phase497：同材质候选参照','',
'CloudSEN原始验证集535张，目标监督0%。单seed42901，从相同官方DINOv3-S+/16冻结基座重新初始化任务模块并训练40K；直接复用已完成的496同seed对照，没有多seed训练。','',
'**改动**：用16维语义特征在16/32像素偏移、各8方向的候选中分配参照权重，加入可学习空参照并屏蔽越界候选。原固定环形比较替换为亮度、色度、纹理方向/能量、微纹理幅度的相似参照比较。保留子patch细节重分配、空间CNN、交互、输出残差及分割头；无MsRE、无450分支，不调分类阈值。语义相似度是材质相似的代理，不是经标注确认的材质对应。','',
'| 模型 | 检查点 | 地表IoU | 云IoU | 云影IoU | mIoU | 云影FN | 云影FP |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model']} | {r['checkpoint']} | {r['surface']:.4f} | {r['cloud']:.4f} | {r['shadow']:.4f} | {r['mIoU']:.4f} | {r['shadow_FN']:,} | {r['shadow_FP']:,} |")
lines+=['','固定40K差值（497−496）：'+', '.join(f'{k} {v:+.4f}' for k,v in delta.items())+'。',
'可训练参数：1,494,597 → 1,501,895，新增7,298（约0.49%）。新纹理适配器44,291参数。',
'结构、边界掩码、空参照与梯度测试通过；共同层初始化、冻结基座、抽查批次和代码签名核验通过。所有24个纹理模块参数张量在真实预检中更新。固定40K为主，验证最优另列。']
if (R/'COST.json').exists():
    cost=json.loads((R/'COST.json').read_text())['median_ms'];a,b=cost['phase496'],cost['phase497']
    lines+=['',f'GPU前向（batch1/512/FP32，不含IO和预处理；20次预热、32次交替计时）：496 {a:.3f} ms，497 {b:.3f} ms，变化 {100*(b/a-1):+.2f}%。']
for file,label in [('UNIFORM_REFERENCE.json','均匀候选权重，保留学到的空参照置信度'),('TEXTURE_OFF.json','关闭整个纹理模块')]:
    if (R/file).exists():
        m=json.loads((R/file).read_text())['metrics']
        lines+=['',f"推理诊断：{label}，地表/云/云影/mIoU={m['IoU_surface']:.4f}/{m['IoU_cloud']:.4f}/{m['IoU_shadow']:.4f}/{m['mIoU_exact']:.4f}。完整497减该诊断的云影差值={final['497']['shadow']-m['IoU_shadow']:+.4f}点。"]
lines+=['','关断/均匀参照均为同一检查点的零训练干预，不能代替移除后重训消融。本轮同时改变参照选择和比较描述量，整体对照不能单独归因语义匹配。权重留远端；路径与SHA见MAIN.csv。']
if (R/'RESUME_ACCEPTANCE.json').exists():
    lines += ['', '运行中远端环境重启，日志停在28,600步；从完整28K检查点恢复模型、优化器与调度器，继续至40K。原检查点未存随机数状态，因此不声称与未中断运行逐位等价；数据批次核对见RESUME_TRACE_CHECK.jsonl。']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
print((R/'RESULT.md').read_text())
