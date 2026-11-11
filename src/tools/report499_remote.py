"""Fixed-40K comparison, with a separate within-checkpoint early-path intervention."""
import csv,json,re
from pathlib import Path
R=Path('/home/scv/shared/phase499_20261110');B=Path('/home/scv/shared/phase496_20261110')
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
rows=[]
for name,root in [('496',B),('499',R)]:
    for item in json.loads((root/'RESULT.json').read_text()):
        if item['variant']!='texture':continue
        m=item['metrics']
        rows.append(dict(model=name,checkpoint=item['checkpoint'],surface=m['IoU_surface'],cloud=m['IoU_cloud'],shadow=m['IoU_shadow'],mIoU=m['mIoU_exact'],shadow_FN=m['FN_shadow'],shadow_FP=m['FP_shadow'],path=item['path'],sha256=item['sha256']))
assert len(rows)==4
with (R/'MAIN.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
final={r['model']:r for r in rows if r['checkpoint']=='final'}
delta={k:final['499'][k]-final['496'][k] for k in ['surface','cloud','shadow','mIoU','shadow_FN','shadow_FP']}
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
lines=['# Phase499：RGB协同纹理与环形对比','',
'CloudSEN原始验证535张，目标监督0%。单seed42901；官方DINOv3-S+/16冻结，任务模块全新初始化训练40K。直接复用496同seed结果，没有多seed复现。','',
'改动：保留496全部路径，在原分辨率3/9/21窗口统计RGB协方差六项、总方差和颜色比例，再池化至1/4，与环形邻域比较。60维证据经轻量卷积和零初始化投影进入第一层高分辨率解码特征。新增15184参数；无MsRE、无498早期连接，不改损失或阈值。RGB统计不解释为物理反射率。','',
'| 网络 | 检查点 | 地表IoU | 云IoU | 云影IoU | mIoU | 云影FN | 云影FP |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model']} | {r['checkpoint']} | {r['surface']:.4f} | {r['cloud']:.4f} | {r['shadow']:.4f} | {r['mIoU']:.4f} | {r['shadow_FN']:,} | {r['shadow_FP']:,} |")
lines+=['','固定40K差值（499−496）：'+', '.join(f'{k} {v:+.4f}' for k,v in delta.items())+'。']
a=json.loads((B/'texture/INITIAL.json').read_text())['trainable_parameters']
b=json.loads((R/'texture/INITIAL.json').read_text())['trainable_parameters']
assert b-a==15184
lines += ['',f'可训练参数：{a:,} → {b:,}，新增{b-a:,}（{100*(b/a-1):.2f}%）。']
cost=json.loads((R/'COST.json').read_text())['median_ms'];a,b=cost['phase496'],cost['phase499']
lines+=['',f'GPU前向：496 {a:.3f} ms，499 {b:.3f} ms（{100*(b/a-1):+.2f}%）。batch1、512、FP32，20次预热、32次交替计时，不含IO与预处理。']
m=json.loads((R/'COVARIANCE_OFF.json').read_text())['metrics']
lines+=['',f"同一40K权重只关闭颜色协方差分支：地表/云/影/mIoU={m['IoU_surface']:.4f}/{m['IoU_cloud']:.4f}/{m['IoU_shadow']:.4f}/{m['mIoU_exact']:.4f}。完整499相对关断，云影{final['499']['shadow']-m['IoU_shadow']:+.4f}点，mIoU{final['499']['mIoU']-m['mIoU_exact']:+.4f}点。",'该关断保留496原有的全部分支，衡量对新增路径的推理依赖，不代替移除后重训消融。']
if delta['shadow']>0:
    conclusion='本轮固定40K云影IoU提高。'+('地表、云也提高。' if delta['surface']>0 and delta['cloud']>0 else '其他类别存在取舍，见主表。')
else:
    conclusion='本轮没有提高固定40K云影IoU，保留496。'
lines+=['',conclusion+' 单seed结构探索，不做稳定性结论。',
'',f"结构诊断：完整模型相对关闭新分支，云影净变化{final['499']['shadow']-m['IoU_shadow']:+.4f}点。该数值反映本权重的推理依赖；与496的重训差值一起判断，不将关断等同重训消融。",
'','验收：原有参数初始化一致；新分支初始为零；同灰度不同颜色的合成输入产生不同协方差证据；平坦/黑色输入数值有限；全部纹理参数真实更新；冻结基座、优化器覆盖、30个抽查批次及训练代码签名通过。最终与最优检查点均重载535图复评。权重留远端，路径和SHA见MAIN.csv。']
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
print((R/'RESULT.md').read_text())
