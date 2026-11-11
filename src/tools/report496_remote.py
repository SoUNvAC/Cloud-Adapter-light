"""Build a compact final report from independently reloaded checkpoint metrics."""
from pathlib import Path
import json,csv,re,hashlib
R=Path('/home/scv/shared/phase496_20261110')
assert json.loads((R/'STATUS.json').read_text())['state']=='complete'
rows=json.loads((R/'RESULT.json').read_text())
assert len(rows)==4
params={v:json.loads((R/v/'INITIAL.json').read_text())['trainable_parameters'] for v in ['control','texture']}
curves=[]
for v in ['control','texture']:
    step=0;classes={}
    for line in (R/(v+'.log')).read_text(errors='replace').splitlines():
        hit=re.search(r'Iter\(train\) \[\s*(\d+)/40000\]',line)
        if hit:step=int(hit[1])
        hit=re.search(r'\|\s*(surface_visible|cloud|shadow)\s*\|\s*([0-9.]+)',line)
        if hit:classes[hit[1]]=float(hit[2])
        hit=re.search(r'Iter\(val\) \[268/268\].*?mIoU: ([0-9.]+)',line)
        if hit:
            curves.append(dict(variant=v,step=step,**classes,mIoU=float(hit[1])))
    assert sum(c['variant']==v for c in curves)==20
with (R/'CURVE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['variant','step','surface_visible','cloud','shadow','mIoU']);w.writeheader();w.writerows(curves)
table=[]
for r in rows:
    m=r['metrics']
    table.append(dict(variant=r['variant'],checkpoint=r['checkpoint'],surface=m['IoU_surface'],cloud=m['IoU_cloud'],shadow=m['IoU_shadow'],mIoU=m['mIoU_exact'],trainable_parameters=params[r['variant']],sha256=r['sha256']))
with (R/'MAIN.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
final={r['variant']:r for r in table if r['checkpoint']=='final'}
delta={k:final['texture'][k]-final['control'][k] for k in ['surface','cloud','shadow','mIoU']}
lines=['# Phase496：移除MsRE后的40K重训','',
'CloudSEN验证535张，目标域监督 **0%**。相同官方DINOv3-S+/16冻结基座；全部任务模块重新初始化。两组同数据、seed42901、batch2、AdamW学习率1e-4、40K训练。','',
'**改动**：新组完全删除MsRE及LocalReadWrite；用493B池化前微纹理幅度、488B环形纹理证据、480/482子patch细节重分配组成36,993参数纹理适配器，作用于首级高分辨率输出。保留空间CNN、交叉注意力、输出残差和Mask2Former头。未加载旧431任务权重或450分支；不做阈值调整。','',
'| 网络 | 检查点 | 地表IoU | 云IoU | 云影IoU | mIoU |','|---|---|---:|---:|---:|---:|']
for r in table:
    lines.append(f"| {'原结构（含MsRE）' if r['variant']=='control' else '新结构（无MsRE）'} | {r['checkpoint']} | {r['surface']:.4f} | {r['cloud']:.4f} | {r['shadow']:.4f} | {r['mIoU']:.4f} |")
lines += ['', '**固定40K差值（新−原，百分点）**：'+', '.join(f'{k} {v:+.4f}' for k,v in delta.items())+'。',
f"可训练参数：{params['control']:,} → {params['texture']:,}，减少{100*(1-params['texture']/params['control']):.2f}%。",'',
'验收：10步真实预检、共同初始化签名、优化器覆盖、全部21个纹理参数更新、冻结基座不变及批次签名核验通过；最终与最优检查点分别重载，在完整535张上复评。',
'本轮是单seed的整个替代包比较；MsRE与配套读写连接同时移除，不能将效果单独归因某一组件。验证最优在同一验证集上选择；以预先固定40K比较为主。',
'权重保留远端，各检查点路径和SHA见RESULT.json；曲线见CURVE.csv。']
if (R/'COST.json').exists():
    cost=json.loads((R/'COST.json').read_text())
    a,b=cost['median_ms']['control'],cost['median_ms']['texture']
    lines += ['', f'GPU前向计时（batch1，512×512，FP32，不含IO/预处理）：原结构 {a:.3f} ms，新结构 {b:.3f} ms，变化 {100*(b/a-1):+.2f}%。20次预热、32次交替计时取中位数。']
if (R/'TEXTURE_OFF.json').exists():
    off=json.loads((R/'TEXTURE_OFF.json').read_text())['metrics']
    on=final['texture']
    lines += ['', f"新组40K检查点关闭整个纹理适配器后：地表IoU {off['IoU_surface']:.4f}，云 {off['IoU_cloud']:.4f}，云影 {off['IoU_shadow']:.4f}，mIoU {off['mIoU_exact']:.4f}。开启−关闭云影 {on['shadow']-off['IoU_shadow']:+.4f}点、mIoU {on['mIoU']-off['mIoU_exact']:+.4f}点。这是同一已训练网络的推理关断诊断，不等同于移除模块后重新训练的消融。"]
(R/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
(R/'FINAL_DELTA.json').write_text(json.dumps(delta,indent=2))
print((R/'RESULT.md').read_text())
