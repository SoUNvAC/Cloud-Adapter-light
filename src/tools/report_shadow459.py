from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase459/phase459_20261106'
base=json.loads((ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json').read_text())['results'][0]
rows=[dict(model='Phase450',step=6000,**{k:base[k] for k in ['IoU','mIoU']},protected=True)]
for arm in ['A','B']:
 d=R/arm/'network_run';r=json.loads((d/'FINAL_RESULTS.json').read_text());assert r['passed']
 for x in r['results']:
  rows.append(dict(model=arm+':'+','.join(x['roles']),**x))
  p=json.loads((d/f"INDEPENDENT_{x['step']}.json").read_text())['per_image'];assert len(p)==535
  cm=[[sum(z['confusion'][i][j] for z in p) for j in range(3)] for i in range(3)]
  iou=[100*cm[i][i]/(sum(cm[i])+sum(c[i] for c in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(a-b) for a,b in zip(iou,x['IoU']))<1e-9
 a=json.loads((d/'LOCK.json').read_text());assert a['target_label_supervision_percent']==0
traces=[(R/a/'network_run/INPUT_TRACE.jsonl').read_text().splitlines() for a in ['A','B']];assert traces[0]==traces[1]
off=json.loads((R/'B/network_run/ABLATION_OFF.json').read_text())['results'][0];rows.append(dict(model='B:injection_off',**off))
with (R/'MAIN.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','protected','target_supervision_percent'])
 for x in rows:w.writerow([x['model'],x['step'],*x['IoU'],x['mIoU'],x['protected'],0])
text=['# Phase459：云内容／遮光状态分解','','目标域监督 **0%**。原始CloudSEN验证535张，固定argmax。两组从450起点各续训6000步，数据顺序与损失主项相同。','','| 模型 | 步数 | 地表IoU | 云IoU | 云影IoU | mIoU | 保护地表与云 |','|---|---:|---:|---:|---:|---:|---|']
for x in rows:text.append('| '+x['model']+' | '+str(x['step'])+' | '+' | '.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+' | '+str(x['protected'])+' |')
text+=['','A：解码器＋原450分支续训。B：四尺度特征进入像素解码器前，增加内容/遮光双路、软交互和特征注入，新增263432参数；辅助监督为云/地表、非云区域的遮光占比、特征重建。固定VFM、空间适配器和BN。解码器lr1e-5，分支lr5e-5。','', '思路参考[SDDNet（ACM MM2023）](https://arxiv.org/abs/2308.08935)的背景/阴影分解，改为遥感三类任务的多尺度解码入口分解。本轮未证明物理因素已解耦。','', '验收：两臂训练批次签名一致；独立batch1评测的535张混淆矩阵已本地复算。按107个ROI重抽样的区间见FINAL_RESULTS.json。权重在同一验证集选点，属于单seed探索。']
(R/'RESULT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_training_batches=True,recomputed_images_per_checkpoint=535,target_supervision_percent=0),indent=2))
with (R/'RESULT.md').open('a',encoding='utf-8') as f:f.write('\n结论：未达到70%云影IoU，继续保留450。B最佳训练点影65.3001，比450低0.2592点；A最佳影65.3340。两组保护条件下的最佳选择均为step0原450。\n\n同权重B关闭注入影65.0251，开启增加0.2749点，ROI区间[-0.3424,0.9244]；新路径参与预测，但整体未超越450。B与A各自最佳点影差-0.0340，区间[-0.3880,0.3382]。固定6000步B比A高0.0513点，二者均低于450。\n\n完成零初始化、梯度、全部新增64个参数张量更新、冻结骨干/BN、535图独立重算、两臂批次签名与本地权重SHA验收。结构与辅助监督是联合改动，未拆分二者贡献。\n\n元数据更正：LOCK.json的BN说明沿用了452B“梯度经过骨干”的文字；实际459两臂全部backbone参数冻结，仅decode_head和branch可训练，已由代码及保存权重检查确认。\n')
print('LOCAL_ACCEPTANCE passed')
