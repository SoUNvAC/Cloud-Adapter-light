from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase472/phase472_20261107'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['identical_training_batches']
comparison=read(R/'MATCHED_COMPARISON.json');assert read(R/'MODULE_AUDIT.json')['passed']
assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
rows=[('450完整网络',base)];selections={};times={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json');assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles'])
 rows.append((arm+('原网络联合续训' if arm=='A' else '主路径遮光对比'),best));selections[arm]=final['choices']
 complete=read(d/'COMPLETE.json');assert complete['steps']==6000;times[arm]=complete['seconds']
 update=read(d/'UPDATE_CHECK.json');assert update['step']==6000
 assert all(update[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert all(update['new_changed'].values())
 assert update['changed']['backbone.refinement.']>0 and update['changed']['decode_head.']>0
 assert not any(v for k,v in update['changed'].items() if k not in ['backbone.refinement.','decode_head.'])
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','B关闭新增模块'),('ABLATION_APPEARANCE_OFF.json','APPEARANCE_OFF_','B同权重关闭外观偏好')]:
   x=read(d/fn)['results'][0];rows.append((name,x));checks.append((prefix,x))
 for prefix,x in checks:
  per=read(d/f"{prefix}INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
  cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
  iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
  assert max(abs(u-v) for u,v in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
 assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in read(d/'DELIVERY_SHA.json').items())
 w=d/f"branch_{best['step']}.pth";assert hashlib.sha256(w.read_bytes()).hexdigest()==accept[arm]['sha'][w.name]
assert (R/'A/network_run/INPUT_TRACE.jsonl').read_bytes()==(R/'B/network_run/INPUT_TRACE.jsonl').read_bytes()
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase472：MsRE主路径邻域遮光对比投影','','目标监督0%；CloudSEN训练8490/原始验证535；450起点，两组各6000步，batch4、seed45201，固定argmax。','','A：原MsRE projector与解码器联合续训。B：相同训练范围，另加6147参数参照选择器；8个间隔2-token邻居，通过语义相似、色度接近及较亮偏好形成软参照，在原MsRE projector输入使用x+0.25*(x-reference)，四阶段共享。无额外零输出投影头。','','原模块lr1e-5，新模块lr1e-4；其他原参数、450分支、DINOv3及BN固定。两组损失完全相同，参考固定450，无额外辅助监督。B强度在前2000步从0升到0.25并固定，写入权重；两组均只从2000步及以后选训练最优点，step0回退。','','RGB是外观描述，软参照不是已知同材质地物或真实无影反射率。关闭B结构仍保留已训练的原投影/解码器，不能期待恢复450；净结构收益主要看B对同条件A，开关检验共同适配后的条件作用。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','训练耗时/秒：'+json.dumps(times)]
lines+=['','6000步末点（训练期间batch4全535图评测；主表为选点后batch1独立复算）：']
for arm in ['A','B']:
 last=read(R/arm/'network_run/HISTORY.json')[-1];assert last['step']==6000
 lines.append(arm+': '+json.dumps(dict(IoU=last['IoU'],mIoU=last['mIoU'])))
for k in ['B_minus_A_IoU_delta','CI95','B_contrast_on_minus_off','B_appearance_on_minus_off']:
 lines+=['',k+': '+json.dumps(comparison[k])]
for a,row in zip(['A','B'],rows[1:3]):
 x=row[1];err=read(R/a/'network_run/BOUNDARY_DIAG.json')['delta472minus450']
 lines+=['',a+'较450三类IoU变化：'+json.dumps(x['IoU_delta'])+'；漏检净变化'+str(sum(err['FN']))+'、误报净变化'+str(sum(err['FP']))+'像素；边界距离0–3/3–8/8以上分解：'+json.dumps(err)+'。']
passed=[name for name,x in rows[1:3] if x['protected'] and x['IoU'][2]>base['IoU'][2]]
diag=read(R/'CONTRAST_DIAG.json');assert diag['independent_predictions_identical'] and diag['state_unchanged'] and diag['strength']==.25
lines+=['','B四阶段投影输入修正RMS/原输入RMS的逐图均值：'+json.dumps(diag['stage_mean_relative_input_rms'])+'；中位数：'+json.dumps(diag['stage_median_relative_input_rms'])+'。该量说明参与幅度，不等于性能收益。']
lines+=['','决策：'+('候选获得保护其他两类条件下的云影提升：'+','.join(passed)+'；是否达到70%见主表。' if passed else '两组均未在保护其他两类的条件下超过450，保留450。'),'', '验收：共享原参数初始化相同、记录的批次签名一致（固定seed及采样器）、新增参数更新、未开放原参数与BN冻结、原投影和解码器更新、535图独立混淆矩阵、权重和传输SHA通过。107ROI配对bootstrap2000次；单seed验证集选点的6000步探索，非独立测试集结论。','','参考：[Adaptive Illumination Mapping for Shadow Detection in Raw Images](https://openaccess.thecvf.com/content/ICCV2023/html/Sun_Adaptive_Illumination_Mapping_for_Shadow_Detection_in_Raw_Images_ICCV_2023_paper.html)。借鉴对比信息动机，本轮不使用RAW照明映射。','','B开关状态由ablate_shadow472.py和ablate_appearance472.py设置，单独加载权重不会恢复Python开关。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*472.py'))+list((ROOT/'src/tools').glob('*472.sh'))+[ROOT/'src/research_plans/phase472.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
