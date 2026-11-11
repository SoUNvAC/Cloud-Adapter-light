from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase476/phase476_20261107'
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
 rows.append((arm+('logRGB引导' if arm=='A' else 'log色度引导'),best));selections[arm]=final['choices']
 for x in final['results']:
  if x['step']>0 and 'best_protected' in x['roles'] and 'best_shadow' not in x['roles']:rows.append((arm+'受保护选点',x))
 complete=read(d/'COMPLETE.json');assert complete['steps']==6000;times[arm]=complete['seconds']
 update=read(d/'UPDATE_CHECK.json');assert update['step']==6000
 assert all(update[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert all(update['new_changed'].values()) and update['changed']['decode_head.']>0
 assert all(not v for k,v in update['changed'].items() if k!='decode_head.')
 checks=[('',x) for x in final['results']]
 ablations=[('ABLATION_OFF.json','OFF_',arm+'关闭新增卷积')]
 if arm=='B':ablations.append(('ABLATION_APPEARANCE_OFF.json','APPEARANCE_OFF_','B同权重logRGB引导'))
 for fn,prefix,name in ablations:
  x=read(d/fn)['results'][0];rows.append((name,x));checks.append((prefix,x))
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
lines=['# Phase476：材质引导空间卷积','','目标监督0%；CloudSEN训练8490/验证535。两组各6000步、batch4、seed45201、模块seed47601；新增模块lr1e-3、原解码器lr1e-5，warmup200、固定argmax。','','在空间CNN的1/4细节上，用8邻居特征差分、可学习逐通道卷积核和1×1通道混合提取特征；按RGB引导距离分配邻域权重。修正同步池化进入1/16上下文，位于VFM交互之前。两组各4609新参数。','','A使用logRGB距离；B使用去通道均值的log色度距离，弱化未截断无色增益对引导的影响。材质只是设计动机，相似色度不等于已知同材质或物理反射率。','','原参数仅开放解码器，其他原参数、450分支、BN固定。零差分核初始复现450；新层的梯度穿过冻结VFM回传。两组采用相同不加权CE、KL/softIoU保持约束，初始化与记录批次相同。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','并行训练耗时/秒：'+json.dumps(times)]
for k in ['B_minus_A_IoU_delta','CI95','A_convolution_on_minus_off','B_contrast_on_minus_off','B_appearance_on_minus_off']:lines+=['',k+': '+json.dumps(comparison[k])]
lines+=['','B_contrast_on_minus_off表示空间卷积开关；B_appearance_on_minus_off表示色度引导对同权重logRGB引导。关闭新层保留已训练解码器，不应期待恢复450。']
for arm in ['A','B']:
 d=R/arm/'network_run';err=read(d/'BOUNDARY_DIAG.json')['delta476minus450'];last=read(d/'HISTORY.json')[-1]
 lines+=['',arm+'较450三类IoU变化：'+json.dumps(bestrows[arm]['IoU_delta'])+'；漏检净变化'+str(sum(err['FN']))+'、误报净变化'+str(sum(err['FP']))+'像素。边界距离0–3/3–8/>8像素：'+json.dumps(err),'',arm+'6000步末点：'+json.dumps(dict(IoU=last['IoU'],mIoU=last['mIoU']))]
passed=[name for name,x in rows if name.startswith(('Alog','A受保护','Blog','B受保护')) and x['protected'] and x['IoU'][2]>base['IoU'][2]]
diag=read(R/'CONTRAST_DIAG.json');assert diag['independent_predictions_identical'] and diag['state_unchanged']
lines+=['','B细节特征修正相对RMS逐图均值：'+str(diag['mean_relative_detail_rms'])+'；卷积核绝对值均值：'+str(diag['kernel_abs_mean'])+'；引导温度：'+str(diag['temperature'])+'。该量描述参与幅度，不代表性能收益。','','元数据说明：沿用脚本的UPDATE_CHECK.branch_updated字段仅按A/B标签写入，A为false；验收不依赖该旧字段，独立权重比较确认两组新增3个参数张量均更新。原始记录保留。']
lines+=['','决策：'+('保护地表/云条件下影IoU超过450：'+','.join(passed)+'；70%门槛见主表。' if passed else '未获得保护地表/云条件下的云影净提升，保留450。'),'','验收：原解码器与新增参数更新，其余原参数及BN冻结；535图独立混淆矩阵、107ROI配对bootstrap2000次、初始化与记录批次、权重与传输SHA通过。单seed验证集选点探索。同权重切换引导仅为条件干预；色度相对RGB的结构增量看独立训练A/B。尚未做同设置仅解码器续训对照，整体提升不能全部归因新层。']
export=read(R/'CANDIDATE_EXPORT.json');assert export['passed'] and export['all_predictions_identical_to_B_off'] and export['images']==535
assert sha(R/'B/network_run'/export['file'])==export['sha256']
lines+=['','归因与留存：A/B移除新层后云影IoU均高于开启状态，本轮不支持把云影净收益归因于材质卷积。B移除新层的候选云影IoU为'+f"{export['metrics']['IoU'][2]:.6f}"+'，地表和云也高于450；保留为后续对照候选，450继续保留为原始基线。下一步优先补同设置仅解码器训练，分清联合训练与解码器自身贡献。','','候选权重：B/network_run/'+export['file']+'。文件已经移除material参数，配合原431基座按infer_shadow476.py加载，不依赖隐藏的运行时开关；535图预测逐像素复现关层结果。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*476.py'))+list((ROOT/'src/tools').glob('*476.sh'))+[ROOT/'src/research_plans/phase476.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
