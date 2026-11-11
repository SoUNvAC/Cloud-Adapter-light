from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase470/phase470_20261107'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['identical_training_batches']
comparison=read(R/'MATCHED_COMPARISON.json');assert read(R/'MODULE_AUDIT.json')['passed']
assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
rows=[('450完整网络',base)];selections={};times={}
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json');assert final['passed']
 best=next(x for x in final['results'] if 'best_shadow' in x['roles'])
 rows.append((arm+('等权专家' if arm=='A' else '错误类型竞争路由'),best));selections[arm]=final['choices']
 complete=read(d/'COMPLETE.json');assert complete['steps']==6000;times[arm]=complete['seconds']
 update=read(d/'UPDATE_CHECK.json');assert update['step']==6000
 assert all(update[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert all(update['new_changed'].values()) and not any(update['changed'].values())
 checks=[('',x) for x in final['results']]
 if arm=='B':
  for fn,prefix,name in [('ABLATION_OFF.json','OFF_','B关闭新增模块'),('ABLATION_ROUTING_OFF.json','ROUTING_OFF_','B同权重等权混合')]:
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
lines=['# Phase470：漏检恢复—误报抑制竞争专家','','目标监督0%；CloudSEN训练8490/原始验证535；450起点，两组各6000步，batch4、seed45201，固定argmax。','','改动：在解码器前1/4分辨率特征加入4专家，共44572参数；读取原空间CNN、四级语义、对数RGB及色度描述。四路分别由通用、遗漏云影、地表误报云影、云误报云影的软路由控制。修正后的特征同时传给原解码器和450分支。输出零初始化；原网络全部权重与BN冻结，仅新模块lr1e-4。','','A对照等权混合；B学习竞争路由。两组均用固定450预测和源训练标签生成4类错误监督，增加0.1类别平衡辅助损失，其余分割损失一致。A也训练路由辅助头，但不用于混合。推理不读取标签，也不额外运行450参考网络。','','专家名称描述训练设计，不代表每个专家已学成纯粹独立的错误修正功能；同权重开关检验实际路由作用。','','|模型|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
lines+=['','选点：'+json.dumps(selections),'','训练耗时/秒：'+json.dumps(times)]
lines+=['','6000步末点（训练期间batch4全535图评测；主表为选点后batch1独立复算）：']
for arm in ['A','B']:
 last=read(R/arm/'network_run/HISTORY.json')[-1];assert last['step']==6000
 lines.append(arm+': '+json.dumps(dict(IoU=last['IoU'],mIoU=last['mIoU'])))
for k in ['B_minus_A_IoU_delta','CI95','B_experts_on_minus_off','B_routing_on_minus_off']:
 lines+=['',k+': '+json.dumps(comparison[k])]
for a,row in zip(['A','B'],rows[1:3]):
 x=row[1];err=read(R/a/'network_run/BOUNDARY_DIAG.json')['delta470minus450']
 lines+=['',a+'较450三类IoU变化：'+json.dumps(x['IoU_delta'])+'；漏检净变化'+str(sum(err['FN']))+'、误报净变化'+str(sum(err['FP']))+'像素；边界距离0–3/3–8/8以上分解：'+json.dumps(err)+'。']
passed=[name for name,x in rows[1:3] if x['protected'] and x['IoU'][2]>base['IoU'][2]]
route=read(R/'ROUTING_DIAG.json');assert route['independent_predictions_identical'] and route['state_unchanged']
lines+=['','路由分工（1/4网格，按各错误类型像素占比加权的平均软权重）：','','|实际类型|通用路|漏检影路|地表误报路|云误报路|','|---|---:|---:|---:|---:|']
for label,values in zip(['其他','遗漏云影','地表误报云影','云误报云影'],route['mean_route_by_error_type']):
 assert abs(sum(values)-1)<1e-5
 lines.append('|'+label+'|'+'|'.join(f'{100*v:.2f}%' for v in values)+'|')
lines+=['','该表只表示路由学到一定错误类型区分，不能据此认定对应专家已成功修复错误；实际功能以同权重开关和分割指标判断。']
lines+=['','决策：'+('候选获得保护其他两类条件下的云影提升：'+','.join(passed)+'；是否达到70%见主表。' if passed else '两组均未在保护其他两类的条件下超过450，保留450。'),'', '验收：实际共享初始化与批次一致、新增参数更新、原参数与BN冻结、535图独立混淆矩阵、权重和传输SHA通过。107ROI配对bootstrap2000次；单seed验证集选点的6000步探索，非独立测试集结论。','','参考：[Distraction-Aware Shadow Detection, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Distraction-Aware_Shadow_Detection_CVPR_2019_paper.html)。借鉴显式学习阴影干扰区域，实验实现为错误类型驱动的高分辨率特征专家路由。','','B开关状态由ablate_shadow470.py和ablate_routing470.py设置，单独加载权重不会恢复Python开关。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,matched_batches=True,independent_images=535,ROI_count=107,target_supervision_percent=0),indent=2))
files=list((ROOT/'src/tools').glob('*470.py'))+list((ROOT/'src/tools').glob('*470.sh'))+[ROOT/'src/research_plans/phase470.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
