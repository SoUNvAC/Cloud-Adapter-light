from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase474/phase474_20261107'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['identical_training_batches']
comparison=read(R/'MATCHED_COMPARISON.json');assert read(R/'MODULE_AUDIT.json')['passed']
assert all(sha(ROOT/n)==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
rows=[('450完整网络',base)];selections={};times={};bestrows={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json');assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles']);bestrows[arm]=best
 rows.append((arm+('普通组合' if arm=='A' else '条件组合'),best));selections[arm]=final['choices']
 for x in final['results']:
  if x['step']>0 and 'best_protected' in x['roles'] and 'best_shadow' not in x['roles']:rows.append((arm+'受保护选点',x))
 complete=read(d/'COMPLETE.json');assert complete['steps']==6000;times[arm]=complete['seconds']
 update=read(d/'UPDATE_CHECK.json');assert update['step']==6000
 assert all(update[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert all(update['new_changed'].values()) and not any(update['changed'].values())
 assert read(d/'REFERENCE_CACHE_CHECK.json')['passed']
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','B关闭新增解码'),('ABLATION_CONDITIONAL_OFF.json','CONDITIONAL_OFF_','B同权重普通组合'),('ABLATION_SPATIAL_MEAN.json','SPATIAL_MEAN_','B同权重全图均值修正')]:
   x=read(d/fn)['results'][0];rows.append((name,x));checks.append((prefix,x))
   if prefix=='OFF_':assert x['confusion']==base['confusion']
 for prefix,x in checks:
  per=read(d/f"{prefix}INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
  cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
  iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(u-v) for u,v in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
 assert all(sha(d/n)==h for n,h in read(d/'DELIVERY_SHA.json').items())
 w=d/f"branch_{best['step']}.pth";assert sha(w)==accept[arm]['sha'][w.name]
assert (R/'A/network_run/INPUT_TRACE.jsonl').read_bytes()==(R/'B/network_run/INPUT_TRACE.jsonl').read_bytes()
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase474：云内容—地表遮光条件解码','','目标监督0%；CloudSEN训练8490/原始验证535。两组均完成6000步，batch4、seed45201、AdamW lr1e-4，warmup200、poly0.9，固定argmax。','','原450完整网络全部冻结。两条新增路径有独立CNN/四层语义投影：云内容头接收RGB与全局场景，遮光头接收logRGB、色度、纹理及5/13/29局部对比。两组同为31106新参数，初始权重和记录批次一致。','','A修正云/影logit并普通归一化；B修正云/非云、非云内影/地表的对数比，条件组合三类概率。初始输出概率复现450。B遮光头不直接改变云概率，但不保证最终云IoU不变。无阈值搜索和辅助标签损失。','','两组损失相同：固定450定义难例加权三类CE，正确非影KL、非影softIoU保护及修正正则。参考缓存与独立450前向一致性通过。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','并行训练耗时/秒：'+json.dumps(times)]
for k in ['B_minus_A_IoU_delta','CI95','B_decoder_on_minus_off','B_conditional_on_minus_off','B_spatial_minus_mean']:lines+=['',k+': '+json.dumps(comparison[k])]
lines+=['','全图均值干预将两头输出各自替换为每图平均值，保留每图偏置，移除学习修正的空间变化。该干预用于区分空间解码作用与整体偏置作用。']
for arm in ['A','B']:
 d=R/arm/'network_run';err=read(d/'BOUNDARY_DIAG.json')['delta474minus450'];last=read(d/'HISTORY.json')[-1]
 lines+=['',arm+'较450三类IoU变化：'+json.dumps(bestrows[arm]['IoU_delta'])+'；漏检净变化'+str(sum(err['FN']))+'、误报净变化'+str(sum(err['FP']))+'像素。边界距离0–3/3–8/>8像素：'+json.dumps(err),'',arm+'6000步末点：'+json.dumps(dict(IoU=last['IoU'],mIoU=last['mIoU']))]
passed=[name for name,x in rows if name.startswith(('A','B条件','B受保护')) and x['protected'] and x['IoU'][2]>base['IoU'][2]]
lines+=['','决策：'+('保护地表/云条件下影IoU超过450：'+','.join(passed)+'；70%门槛见主表。' if passed else '未获得保护地表/云条件下的云影净提升，保留450。'),'','验收：所有原参数与BN冻结，新参数更新；535图独立混淆矩阵、107ROI配对bootstrap2000次、初始化与记录批次、权重与传输SHA通过。单seed验证集选点探索。B同权重切换组合仅为条件干预；架构比较以独立训练A/B为准。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*474.py'))+list((ROOT/'src/tools').glob('*474.sh'))+[ROOT/'src/research_plans/phase474.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
