from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase452/phase452_20261106'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
lines=['# Phase452：冻结续训与开放任务模块对照','', '两组从同一450 step6000出发，各续训6000步；原始源训练8490图、验证535图，目标监督0%。A只训新增分支，B另开放空间CNN、交互、融合细化、局部读写、输出残差与分割头。DINOv3和全部BN统计/仿射保持不变，保护损失使用冻结原431。','新增分支峰值学习率5e-5，B任务模块2e-6；batch4、seed45201、200步warmup后衰减；模型基座保持eval，B允许梯度穿过冻结ViT。','', '|模型/选择|续训步数|地表IoU|云IoU|影IoU|mIoU|保护450|','|---|---:|---:|---:|---:|---:|---|',f"|起点450|0|{base['IoU'][0]:.4f}|{base['IoU'][1]:.4f}|{base['IoU'][2]:.4f}|{base['mIoU']:.4f}|是|"]
summary={};audit=[];traces={}
for arm in ['A','B']:
 d=R/arm/'network_run';f=read(d/'FINAL_RESULTS.json');h=read(d/'HISTORY.json');lock=read(d/'LOCK.json');assert h[-1]['step']==6000
 traces[arm]=[json.loads(x) for x in (d/'INPUT_TRACE.jsonl').read_text().splitlines()]
 summary[arm]=dict(best=f['results'],fixed6000=h[-1],lock=lock,boundary=read(d/'BOUNDARY_DIAG.json'))
 for v in f['results']:
  lines.append(f"|{arm} {'/'.join(v['roles'])}|{v['step']}|{v['IoU'][0]:.4f}|{v['IoU'][1]:.4f}|{v['IoU'][2]:.4f}|{v['mIoU']:.4f}|{v['protected']}|")
  per=read(d/f"INDEPENDENT_{v['step']}.json")['per_image'];assert len(per)==535
  cm=[[sum(z['confusion'][a][b] for z in per) for b in range(3)] for a in range(3)];assert cm==v['confusion']
  iou=[100*cm[c][c]/(sum(cm[c])+sum(row[c] for row in cm)-cm[c][c]) for c in range(3)];assert max(abs(a-b) for a,b in zip(iou,v['IoU']))<1e-10 and v['state_unchanged']
  audit.append(dict(arm=arm,step=v['step'],passed=True))
assert traces['A']==traces['B'],'Batch order/augmentation mismatch'
lines+=['','## 相同6000步终点（batch4验证）','', '|组别|地表IoU|云IoU|影IoU|mIoU|','|---|---:|---:|---:|---:|']
for arm,v in summary.items():
 x=v['fixed6000'];lines.append(f"|{arm}|{x['IoU'][0]:.4f}|{x['IoU'][1]:.4f}|{x['IoU'][2]:.4f}|{x['mIoU']:.4f}|")
lines+=['','## 最佳训练模型的错误变化（相对450）','', '|组别|0—3px漏影变化|3—8px漏影变化|深部>8px漏影变化|总误报变化|','|---|---:|---:|---:|---:|']
for arm,v in summary.items():
 x=v['boundary']['delta452minus450'];lines.append(f"|{arm}|{x['FN'][0]}|{x['FN'][1]}|{x['FN'][2]}|{sum(x['FP'])}|")
lines+=['','负数表示错误减少。保护门槛是450地表/云IoU；step0若被选中，仅代表回退原模型。最佳点与保护点完整重载batch1复测，仍使用同一验证集；单seed，不包含模型选择或训练seed不确定性。两组全部记录批次及增强哈希一致；编码器、BN和参考模型不变、开放模块真实更新验收通过。','权重含新增分支及任务模块的完整更新状态（student_patch），保存在远端各组branch_step.pth，不能只加载分支权重。独立目录另存整网。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,input_traces_equal=True,checks=audit),indent=2))
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as file:
 w=csv.writer(file);w.writerow(['arm','roles','step','surface','cloud','shadow','mIoU','protected']);w.writerows([[a,','.join(v['roles']),v['step'],*v['IoU'],v['mIoU'],v['protected']] for a,x in summary.items() for v in x['best']])
print(json.dumps({a:[dict(step=v['step'],roles=v['roles'],IoU=v['IoU'],protected=v['protected']) for v in x['best']] for a,x in summary.items()}))
