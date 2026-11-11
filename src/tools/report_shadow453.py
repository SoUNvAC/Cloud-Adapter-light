from pathlib import Path
import json,csv
root=Path(__file__).resolve().parents[2];out=root/'outputs/phase453/phase453_20261106';r=out/'A/network_run'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
f=read(r/'FINAL_RESULTS.json');h=read(r/'HISTORY.json');b=read(root/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
assert f['passed'] and h[-1]['step']==6000 and read(r/'AUGMENTATION_AUDIT.json')['passed']
trace=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
assert trace(r/'INPUT_TRACE.jsonl')==trace(root/'outputs/phase452/phase452_20261106/A/network_run/INPUT_TRACE.jsonl')
lines=['# Phase453：遮光配对监督','', '网络沿用450，原431与BN冻结；目标监督0%。从450 step6000出发续训6000步，batch4、seed45201、分支学习率5e-5。原图损失加0.5倍合成配对图损失，前向预算约为452A的两倍。','仅在源训练地表标签内做RGB图像空间遮光，衰减20%–45%，核心标为影，过渡带忽略；并非物理反射率模拟。验证535张全部为未经改动的真实图。','', '|模型|步数|地表IoU|云IoU|云影IoU|mIoU|保护450地表/云|','|---|---:|---:|---:|---:|---:|---|',f"|450起点|0|{b['IoU'][0]:.4f}|{b['IoU'][1]:.4f}|{b['IoU'][2]:.4f}|{b['mIoU']:.4f}|是|"]
for v in f['results']:
 per=read(r/f"INDEPENDENT_{v['step']}.json")['per_image'];assert len(per)==535
 cm=[[sum(z['confusion'][a][b] for z in per) for b in range(3)] for a in range(3)];assert cm==v['confusion'] and v['state_unchanged']
 iou=[100*cm[c][c]/(sum(cm[c])+sum(row[c] for row in cm)-cm[c][c]) for c in range(3)];assert max(abs(a-b) for a,b in zip(iou,v['IoU']))<1e-10
 lines.append(f"|453 {'/'.join(v['roles'])}|{v['step']}|{v['IoU'][0]:.4f}|{v['IoU'][1]:.4f}|{v['IoU'][2]:.4f}|{v['mIoU']:.4f}|{v['protected']}|")
d=read(r/'BOUNDARY_DIAG.json')['delta453minus450']
control=read(root/'outputs/phase452/phase452_20261106/A/network_run/FINAL_RESULTS.json')
for v in control['results']:
 lines.append(f"|452A {'/'.join(v['roles'])}|{v['step']}|{v['IoU'][0]:.4f}|{v['IoU'][1]:.4f}|{v['IoU'][2]:.4f}|{v['mIoU']:.4f}|{v['protected']}|")
end=h[-1]
lines+=['',f"固定6000步终点（batch4）：地表 {end['IoU'][0]:.4f}、云 {end['IoU'][1]:.4f}、云影 {end['IoU'][2]:.4f}、mIoU {end['mIoU']:.4f}。", '452A为相同起点、优化器、原图抽样和步数的普通续训参照；453额外增加配对前向及监督，因此不是等算力对照，也不能单独分离新增样本与损失权重的作用。']
lines+=['',f"最佳训练点相对450：距真实云影边界0–3、3–8、>8像素的漏检变化分别为 {d['FN']}；误报总变化 {sum(d['FP'])}。负数表示错误减少。",'', 'step0若入选只代表回退原450。最佳模型由同一验证集选择，再独立重载batch1复测；单seed。ROI配对区间见FINAL_RESULTS.json，不包含训练seed与模型选择不确定性。原图批次哈希与452A一致，合成增强独立随机发生器不改变原图抽样。']
(out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,input_traces_match452A=True,images=535,results=f['results'],boundary_delta=d),indent=2))
with (out/'MAIN.csv').open('w',newline='',encoding='utf-8-sig') as file:
 w=csv.writer(file);w.writerow(['roles','step','surface','cloud','shadow','mIoU','protected']);w.writerows([[','.join(v['roles']),v['step'],*v['IoU'],v['mIoU'],v['protected']] for v in f['results']])
print(json.dumps(dict(results=f['results'],boundary=d)))
