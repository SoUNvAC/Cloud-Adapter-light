from pathlib import Path
import json,csv
root=Path(__file__).resolve().parents[2];out=root/'outputs/phase457/phase457_20261106';r=out/'A/network_run'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
f=read(r/'FINAL_RESULTS.json');h=read(r/'HISTORY.json');b=read(root/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
assert f['passed'] and h[-1]['step']==6000
trace=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
assert trace(r/'INPUT_TRACE.jsonl')==trace(root/'outputs/phase452/phase452_20261106/A/network_run/INPUT_TRACE.jsonl')
lines=['# Phase457：去除局部捷径的云—影关联调制层','', '450增加9313参数的去除局部捷径的云—影关联调制层，原431与BN冻结；目标监督0%。从450 step6000出发续训6000步，batch4、seed45201、旧分支学习率5e-5、新模块2e-4。沿用450原损失；原图抽样与452A相同。','模块沿8方向、4距离移位云概率图，以归一化云影候选空间匹配和场景特征学习32种位移及空关联权重；融合方向投影、距离、有效覆盖和空关联证据，注入器仅读取13通道关联证据，生成空间中心化的缩放和偏移，调制归一化旧特征；不再直接拼接旧局部特征或类别概率。末层零初始化，精确复现450；新增9313参数，整个分支48461参数。不使用合成图或排序损失，固定三类argmax。验证535张全部为未经改动的真实图。','', '|模型|步数|地表IoU|云IoU|云影IoU|mIoU|保护450地表/云|','|---|---:|---:|---:|---:|---:|---|',f"|450起点|0|{b['IoU'][0]:.4f}|{b['IoU'][1]:.4f}|{b['IoU'][2]:.4f}|{b['mIoU']:.4f}|是|"]
for v in f['results']:
 per=read(r/f"INDEPENDENT_{v['step']}.json")['per_image'];assert len(per)==535
 cm=[[sum(z['confusion'][a][b] for z in per) for b in range(3)] for a in range(3)];assert cm==v['confusion'] and v['state_unchanged']
 iou=[100*cm[c][c]/(sum(cm[c])+sum(row[c] for row in cm)-cm[c][c]) for c in range(3)];assert max(abs(a-b) for a,b in zip(iou,v['IoU']))<1e-10
 lines.append(f"|457 {'/'.join(v['roles'])}|{v['step']}|{v['IoU'][0]:.4f}|{v['IoU'][1]:.4f}|{v['IoU'][2]:.4f}|{v['mIoU']:.4f}|{v['protected']}|")
d=read(r/'BOUNDARY_DIAG.json')['delta457minus450']
control=read(root/'outputs/phase452/phase452_20261106/A/network_run/FINAL_RESULTS.json')
for v in control['results']:
 lines.append(f"|452A {'/'.join(v['roles'])}|{v['step']}|{v['IoU'][0]:.4f}|{v['IoU'][1]:.4f}|{v['IoU'][2]:.4f}|{v['mIoU']:.4f}|{v['protected']}|")
end=h[-1]
lines+=['',f"固定6000步终点（batch4）：地表 {end['IoU'][0]:.4f}、云 {end['IoU'][1]:.4f}、云影 {end['IoU'][2]:.4f}、mIoU {end['mIoU']:.4f}。", '452A为相同起点、优化器、原图抽样和步数的普通续训参照；457增加结构及新模块参数组，编码器前向数量相同，但模块计算量增加。']
lines+=['',f"最佳训练点相对450：距真实云影边界0–3、3–8、>8像素的漏检变化分别为 {d['FN']}；误报总变化 {sum(d['FP'])}。负数表示错误减少。",'', 'step0若入选只代表回退原450。最佳模型由同一验证集选择，再独立重载batch1复测；单seed。ROI配对区间见FINAL_RESULTS.json，不包含训练seed与模型选择不确定性。原图批次哈希与452A一致。']
(out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,input_traces_match452A=True,images=535,results=f['results'],boundary_delta=d),indent=2))
with (out/'MAIN.csv').open('w',newline='',encoding='utf-8-sig') as file:
 w=csv.writer(file);w.writerow(['roles','step','surface','cloud','shadow','mIoU','protected']);w.writerows([[','.join(v['roles']),v['step'],*v['IoU'],v['mIoU'],v['protected']] for v in f['results']])
print(json.dumps(dict(results=f['results'],boundary=d)))
