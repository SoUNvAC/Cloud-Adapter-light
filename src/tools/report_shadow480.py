from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase480/phase480_20261108'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(ROOT/'outputs/phase450/phase450_20261106/network_run/FINAL_RESULTS.json')['results'][0]
previous=read(ROOT/'outputs/phase476/phase476_20261107/B/network_run/ABLATION_OFF.json')['results'][0]
control=next(x for x in read(ROOT/'outputs/phase477/phase477_20261107/A/network_run/FINAL_RESULTS.json')['results'] if 'best_shadow' in x['roles'])
accept=read(R/'TRAINED_WEIGHT_ACCEPTANCE.json');assert accept['passed'] and accept['matches_477_common_initialization_and_recorded_batches'] and accept['matched_AB_new_initial_parameters']
assert read(R/'UNIT_ACCEPTANCE.json')['passed']
assert all(sha(ROOT/n)==h for n,h in read(R/'INITIAL_CODE_SHA.json').items())
rows=[('450完整网络',base),('476B关材质层候选',previous),('477仅解码器',control)];details=[];choices={};candidates=[]
for arm in ['A','B']:
 d=R/arm/'network_run';final=read(d/'FINAL_RESULTS.json');off=read(d/'ABLATION_OFF.json');assert final['passed'] and off['passed']
 done=read(d/'COMPLETE.json');assert done['steps']==6000
 assert all(sha(d/n)==h for n,h in read(d/'DELIVERY_SHA.json').items())
 choices[arm]=final['choices'];u=read(d/'UPDATE_CHECK.json');assert u['step']==6000 and len(u['new_changed'])==13 and all(u['new_changed'].values())
 assert all(u[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert u['changed']['decode_head.']>0 and all(not v for n,v in u['changed'].items() if n!='decode_head.')
 reports=[('on',final,''),('off',off,'OFF_')]
 if arm=='B':reports.append(('strict',read(d/'STRICT_PRECISION.json'),'STRICT_'))
 for kind,f,prefix in reports:
  for x in f['results']:
   per=read(d/f"{prefix}INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
   cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
   iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
   assert max(abs(a-b) for a,b in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
   if x['step']:
    p=d/f"branch_{x['step']}.pth";assert sha(p)==accept['arms'][arm]['weights'][p.name]['sha256']
   rows.append((f"480{arm} {kind} {'/'.join(x['roles'])}",x))
   if kind!='strict' and x['protected'] and x['IoU'][2]>previous['IoU'][2]:candidates.append((arm,kind,x['step'],x['IoU'][2]))
 best=next(x for x in final['results'] if 'best_shadow' in x['roles']);err=read(d/'BOUNDARY_DIAG.json')['delta480minus450']
 details.append(f"{arm}：新分支15265参数，解码器{accept['arms'][arm]['task_parameters']}参数；并行墙钟{done['seconds']:.1f}秒，峰值分配显存{u['peak_memory_bytes']}字节。最佳点较450漏检净变化{sum(err['FN'])}、误报净变化{sum(err['FP'])}像素；边界分解：{json.dumps(err)}")
for arm in ['A','B']:
 e=read(R/arm/'network_run/EFFECT_BOUNDARY_DIAG.json');assert e['passed']
 details.append(arm+'最佳点开启减关闭的边界错误变化：'+json.dumps(e['delta_on_minus_off']))
diag=read(R/'REDISTRIBUTION_DIAG.json');assert diag['independent_predictions_identical'] and diag['state_unchanged'] and len(diag['per_image'])==535
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase480：云影亚patch重分配', '', '在第一层VFM金字塔特征、解码器前新增15265参数分支。原生1/16 VFM特征与1/4 CNN细节分别投影到16维，融合后预测云影覆盖率，软门控16维细节并投影到384维残差。A普通残差；B先减去每个4×4子格均值，保持分支加入前每个4×4块的平均特征。末层无bias且零初始化，两组初始参数相同。', '', 'CloudSEN训练8490/原始验证535；目标监督0%；431+450起点，6000步、batch4、seed45201、分支seed48001、warmup200。新分支lr1e-3，解码器lr1e-5；其余原参数、450分支、VFM与BN固定，固定argmax。', '', 'A/B采用相同分割保持损失，并均加入0.1倍云影覆盖率BCE；覆盖率为原三类标签的4×4有效像素占比，255忽略。A/B是匹配的均值约束对照；477没有新分支和辅助损失，比较只代表整套方案增量，不能单独归因于均值约束。', '', '1000/3000/6000及最佳点535图独立复算；最佳点同权重关分支。107ROI配对bootstrap2000次。450地表/云为保护线，云影目标70%。', '', '|模型|步数|地表IoU|云IoU|云影IoU|mIoU|', '|---|---:|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+str(x.get('step'))+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
comparison=read(R/'MATCHED_COMPARISON.json');lines+=['','|比较|云影增量/百分点|95%区间|','|---|---:|---|']
for k,x in comparison.items():
 if isinstance(x,dict) and 'IoU_delta' in x:lines.append(f"|{k}|{x['IoU_delta'][2]:.6f}|{x['CI95'][2]}|")
strict=read(R/'B/network_run/STRICT_PRECISION.json')['results'][0]
assert strict['maximum_patch_mean_residual']<1e-5 and strict['original_training_unchanged']
details.append('原B严格均值验收通过：'+str(diag['original_absolute_tolerance_passed'])+'；原阈值'+str(diag['original_absolute_tolerance'])+'。严格FP32仅复算注入投影、未重训，最大均值残差'+str(strict['maximum_patch_mean_residual'])+'，该诊断通过原数值线。原结果保留。')
lines+=['',*details,'',f"B分支相对第一层金字塔特征RMS变化均值{diag['mean_relative_detail_rms']:.6f}；535图最大patch均值残差{diag['maximum_patch_mean_residual']:.9g}。均值保持只约束特征，不保证类别IoU不变。",'','超过476关层候选且保护450地表/云的检查点：'+json.dumps(candidates),'',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'','方法参考：[FADE细节引导上采样](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870228.pdf)。本轮测试云影覆盖率引导与patch均值保持组合，不声称上采样或零均值算子原创。','','验收：新分支13个参数张量与原解码器更新，其余原参数/BN固定；共同初始化/记录批次、独立混淆矩阵、权重和传输SHA通过。单seed验证集探索，场景区间不包含重新训练seed波动。','','部署：加载原431基座、student_patch以及infer_shadow480中的分支state_dict。off是显式enabled=False诊断条件，单独state_dict不保存该运行时开关，不能将off结果当作默认部署结果。']
lines+=['','strict为precision_shadow480.py中仅注入卷积关闭TF32的显式诊断设置；其state_dict本身不保存这个执行设置。当前保留的A/3000候选使用默认on推理，无关闭分支或精度覆盖开关。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,scope='Metrics, weights and provenance; original strict conservation is a separate failed check',original_B_strict_conservation_passed=diag['original_absolute_tolerance_passed'],strict_projection_diagnostic_passed=True,images_per_checkpoint=535,ROI_count=107,target_supervision_percent=0,choices=choices),indent=2))
files=list((ROOT/'src/tools').glob('*480.py'))+list((ROOT/'src/tools').glob('*480.sh'))+[ROOT/'src/research_plans/phase480.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
