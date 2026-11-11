from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase475/phase475_20261107'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['identical_training_batches']
comparison=read(R/'MATCHED_COMPARISON.json');assert read(R/'MODULE_AUDIT.json')['passed']
assert all(sha(ROOT/n)==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
oldfinal=read(ROOT/'outputs/phase474/phase474_20261107/A/network_run/FINAL_RESULTS.json');old=next(x for x in oldfinal['results'] if 'best_shadow' in x['roles'])
assert accept['matches_474A_initial_state_and_recorded_batches']
rows=[('450完整网络',base),('474A原损失',old)];selections={};times={};bestrows={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json');assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles']);bestrows[arm]=best
 rows.append((arm+('取消正确影加权' if arm=='A' else '全部像素权重1'),best));selections[arm]=final['choices']
 for x in final['results']:
  if x['step']>0 and 'best_protected' in x['roles'] and 'best_shadow' not in x['roles']:rows.append((arm+'受保护选点',x))
 complete=read(d/'COMPLETE.json');assert complete['steps']==6000;times[arm]=complete['seconds']
 update=read(d/'UPDATE_CHECK.json');assert update['step']==6000
 assert all(update[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert all(update['new_changed'].values()) and not any(update['changed'].values())
 assert read(d/'REFERENCE_CACHE_CHECK.json')['passed']
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','B关闭新增解码'),('ABLATION_SPATIAL_MEAN.json','SPATIAL_MEAN_','B同权重全图均值修正')]:
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
lines=['# Phase475：云影误报的损失权重对照','','目标监督0%；CloudSEN训练8490/原始验证535。新A/B各6000步，batch4、seed45201、模块seed47401，lr1e-4、warmup200，固定argmax。','','全部使用474A普通组合网络，原450完整网络冻结，仅训练31106个新参数；网络与初始化相同。本轮没有提出新架构。','','474A原对照：云影相关难例权重4、正确影1.5，其余1。新A：难例4，其余1。新B：全部1。其余KL、softIoU保护及修正正则相同。已验证三组初始权重和记录批次相同。','','新A对旧对照检验正确影加权；新B对新A检验难例加权。结论仅覆盖当前网络和训练设置。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','并行训练耗时/秒：'+json.dumps(times)]
lines+=['','同6000步末点评测（训练期间batch4、完整535图；主表为选点后batch1独立复算）：','','|损失|地表IoU|云IoU|云影IoU|mIoU|影漏检变化|影误报变化|','|---|---:|---:|---:|---:|---:|---:|']
for name,d in [('474A原损失',ROOT/'outputs/phase474/phase474_20261107/A/network_run'),('475A取消正确影加权',R/'A/network_run'),('475B全部权重1',R/'B/network_run')]:
 last=read(d/'HISTORY.json')[-1];assert last['step']==6000;c=last['confusion'];b=base['confusion']
 fn=sum(c[2][:2])-sum(b[2][:2]);fp=c[0][2]+c[1][2]-b[0][2]-b[1][2]
 lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*last['IoU'],last['mIoU']])+f'|{fn}|{fp}|')
for k in ['B_minus_A_IoU_delta','CI95','B_decoder_on_minus_off','A_minus_474A','B_minus_474A','B_spatial_minus_mean']:lines+=['',k+': '+json.dumps(comparison[k])]
lines+=['','全图均值干预将两头输出各自替换为每图平均值，保留每图偏置，移除学习修正的空间变化。该干预用于区分空间解码作用与整体偏置作用。']
for arm in ['A','B']:
 d=R/arm/'network_run';err=read(d/'BOUNDARY_DIAG.json')['delta475minus450'];last=read(d/'HISTORY.json')[-1]
 lines+=['',arm+'较450三类IoU变化：'+json.dumps(bestrows[arm]['IoU_delta'])+'；漏检净变化'+str(sum(err['FN']))+'、误报净变化'+str(sum(err['FP']))+'像素。边界距离0–3/3–8/>8像素：'+json.dumps(err),'',arm+'6000步末点：'+json.dumps(dict(IoU=last['IoU'],mIoU=last['mIoU']))]
passed=[name for name,x in rows if name.startswith(('A取消','A受保护','B全部','B受保护')) and x['protected'] and x['IoU'][2]>base['IoU'][2]]
lines+=['','诊断：当前网络中，加权策略明显改变漏检/误报取舍。完全取消像素加权，6000步时相对450的误报由原损失增加257916转为减少32761，但漏检由减少148424转为增加33138。云影IoU仍低于450，说明压住误报扩张不等于获得新的区分能力。取消正确影1.5倍权重也未单独改善末点云影IoU。下一步应回到特征判别能力，避免继续只改变输出偏置。']
lines+=['','决策：'+('保护地表/云条件下影IoU超过450：'+','.join(passed)+'；70%门槛见主表。' if passed else '未获得保护地表/云条件下的云影净提升，保留450。'),'','验收：所有原参数与BN冻结，新参数更新；535图独立混淆矩阵、107ROI配对bootstrap2000次、初始化与记录批次、权重与传输SHA通过。单seed验证集选点探索。同权重开关仅为条件干预；损失权重比较以匹配训练为准。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*475.py'))+list((ROOT/'src/tools').glob('*475.sh'))+[ROOT/'src/research_plans/phase475.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
