from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase447/phase447_20261106'
paths={'445 单头/原损失':ROOT/'outputs/phase445/phase445_20261106/network_run','446 双头/均衡损失':ROOT/'outputs/phase446/phase446_20261106/network_run','447A 双头/原损失':OUT/'A/network_run','447B 单头/均衡损失':OUT/'B/network_run'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
lines=['# Phase447：拆开双头与均衡损失','', '两组均完成6000步，源域8490图训练、535图验证，目标标签监督0%。原431及BN冻结，batch4、seed44501、峰值学习率2e-4。A仅保留双头；B仅保留0.1倍路由内类别均衡二分类损失。','', '## 相同步数：6000步（batch4验证）','', '|方案|地表IoU|云IoU|影IoU|mIoU|','|---|---:|---:|---:|---:|']
fixed={};best={};audit=[]
for n,r in paths.items():
 h=read(r/'HISTORY.json');assert h[-1]['step']==6000;v=h[-1];fixed[n]=v
 lines.append(f"|{n}|{v['IoU'][0]:.4f}|{v['IoU'][1]:.4f}|{v['IoU'][2]:.4f}|{v['mIoU']:.4f}|")
lines+=['','## 各自最佳点：重新加载权重，batch1完整复测','', '|方案|步数|地表IoU|云IoU|影IoU|mIoU|非影保护|','|---|---:|---:|---:|---:|---:|---|']
for n,r in paths.items():
 f=read(r/'FINAL_RESULTS.json');assert f['passed'];v=next(x for x in f['results'] if 'best_shadow' in x['roles']);best[n]=v
 for x in f['results']:
  lines.append(f"|{n} ({','.join(x['roles'])})|{x['step']}|{x['IoU'][0]:.4f}|{x['IoU'][1]:.4f}|{x['IoU'][2]:.4f}|{x['mIoU']:.4f}|{x['protected']}|")
  if n.startswith('447'):
   per=read(r/f"INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535
   cm=[[sum(z['confusion'][a][b] for z in per) for b in range(3)] for a in range(3)];assert cm==x['confusion']
   iou=[100*cm[c][c]/(sum(cm[c])+sum(row[c] for row in cm)-cm[c][c]) for c in range(3)];assert max(abs(a-b) for a,b in zip(iou,x['IoU']))<1e-10
   assert x['state_unchanged'];t=x['transition'];assert [[sum(t[a][j][b] for j in range(3)) for b in range(3)] for a in range(3)]==cm
   audit.append(dict(model=n,step=x['step'],passed=True))
lines+=['','## 最佳点错误交换','', '|方案|找回漏影|新增漏影|消除误报|新增误报|','|---|---:|---:|---:|---:|']
for n,v in best.items():
 t=v['transition'];counts=[sum(t[2][j][2] for j in [0,1]),sum(t[2][2][j] for j in [0,1]),sum(t[c][2][j] for c in [0,1] for j in [0,1]),sum(t[c][j][2] for c in [0,1] for j in [0,1])]
 lines.append('|'+n+'|'+'|'.join(str(x) for x in counts)+'|')
keys=list(paths);s=[fixed[k]['IoU'][2] for k in keys];effects=dict(dual_under_original=s[2]-s[0],balanced_under_single=s[3]-s[0],dual_under_balanced=s[1]-s[3],balanced_under_dual=s[1]-s[2],interaction=s[1]-s[2]-s[3]+s[0])
lines+=['','同一步数影IoU差值（百分点）：'+json.dumps(effects,ensure_ascii=False),'','各自最佳点反映候选性能，相同6000步用于比较改动；均为同一验证集选点、单seed探索。ROI区间见各组FINAL_RESULTS.json。原模型零初始化预测一致、冻结状态哈希和新增参数更新验收通过。','']
OUT.mkdir(parents=True,exist_ok=True);(OUT/'RESULT.md').write_text('\n'.join(lines),encoding='utf-8');(OUT/'SUMMARY.json').write_text(json.dumps(dict(fixed=fixed,best=best,effects=effects),indent=2));(OUT/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,checks=audit),indent=2))
with (OUT/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as h:
 w=csv.writer(h);w.writerow(['selection','model','step','surface','cloud','shadow','mIoU']);w.writerows([[sel,n,v['step'],*v['IoU'],v['mIoU']] for sel,data in [('fixed6000',fixed),('best',best)] for n,v in data.items()])
print(json.dumps(dict(effects=effects,best={k:dict(step=v['step'],IoU=v['IoU'],protected=v['protected']) for k,v in best.items()}),ensure_ascii=False))
