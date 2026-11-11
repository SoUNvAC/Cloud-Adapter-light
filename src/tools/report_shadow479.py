from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'outputs/phase479/phase479_20261108'
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
 choices[arm]=final['choices'];u=read(d/'UPDATE_CHECK.json');assert u['step']==6000 and len(u['new_changed'])==6 and all(u['new_changed'].values())
 assert all(u[k] for k in ['reference_unchanged','encoder_unchanged','BN_unchanged','frozen_parameters_unchanged'])
 assert u['changed']['decode_head.']>0 and all(not v for n,v in u['changed'].items() if n!='decode_head.')
 for kind,f,prefix in [('on',final,''),('off',off,'OFF_')]:
  for x in f['results']:
   per=read(d/f"{prefix}INDEPENDENT_{x['step']}.json")['per_image'];assert len(per)==535 and len({v['group'] for v in per})==107
   cm=[[sum(v['confusion'][i][j] for v in per) for j in range(3)] for i in range(3)];assert cm==x['confusion']
   iou=[100*cm[i][i]/(sum(cm[i])+sum(v[i] for v in cm)-cm[i][i]) for i in range(3)]
   assert max(abs(a-b) for a,b in zip(iou,x['IoU']))<1e-9 and x['state_unchanged']
   if x['step']:
    p=d/f"branch_{x['step']}.pth";assert sha(p)==accept['arms'][arm]['weights'][p.name]['sha256']
   rows.append((f"479{arm} {kind} {'/'.join(x['roles'])}",x))
   if x['protected'] and x['IoU'][2]>previous['IoU'][2]:candidates.append((arm,kind,x['step'],x['IoU'][2]))
 best=next(x for x in final['results'] if 'best_shadow' in x['roles']);err=read(d/'BOUNDARY_DIAG.json')['delta479minus450']
 details.append(f"{arm}：新分支4992参数，解码器{accept['arms'][arm]['task_parameters']}参数；并行墙钟{done['seconds']:.1f}秒，峰值分配显存{u['peak_memory_bytes']}字节。最佳点较450漏检净变化{sum(err['FN'])}、误报净变化{sum(err['FP'])}像素；边界分解：{json.dumps(err)}")
diag=read(R/'ENVELOPE_DIAG.json');assert diag['independent_predictions_identical'] and diag['state_unchanged'] and len(diag['per_image'])==535
with (R/'MAIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['model','step','surface_IoU','cloud_IoU','shadow_IoU','mIoU','target_supervision_percent'])
 for name,x in rows:w.writerow([name,x.get('step'),*x['IoU'],x['mIoU'],0])
lines=['# Phase479：多尺度暗区包络分支','','空间CNN输出后、四次交互前添加4992参数分支。四分之一尺度logRGB在5/13/29窗口构造暗区描述，每尺度包括RGB正差、平均差及通道离散度；与64维detail拼接，经1×1→DW3×3→1×1生成64维残差，同步加到detail与池化context。末层零初始化。A参考为邻域均值，B参考为膨胀后腐蚀的闭运算；两组新参数初始化相同。','','CloudSEN源训练8490/原始验证535；目标监督0%；450起点，6000步、batch4、seed45201、分支seed47901，新分支lr1e-3、解码器lr1e-5、warmup200。其余原参数、450分支、BN与VFM冻结。与477相同的不加权CE及保持约束，固定argmax。','','1000/3000/6000和最佳点各535图batch1复算；最佳点同权重关分支。107ROI配对bootstrap2000次。450地表/云为保护线，云影目标70%。','','|模型|步数|地表IoU|云IoU|云影IoU|mIoU|','|---|---:|---:|---:|---:|---:|']
for name,x in rows:lines.append('|'+name+'|'+str(x.get('step'))+'|'+'|'.join(f'{v:.6f}' for v in [*x['IoU'],x['mIoU']])+'|')
comparison=read(R/'MATCHED_COMPARISON.json');lines+=['','|比较|云影增量/百分点|95%区间|','|---|---:|---|']
for k,x in comparison.items():
 if isinstance(x,dict) and 'IoU_delta' in x:lines.append(f"|{k}|{x['IoU_delta'][2]:.6f}|{x['CI95'][2]}|")
lines+=['',*details,'',f"B分支相对detail RMS变化均值{diag['mean_relative_detail_rms']:.6f}，中位数{diag['median_relative_detail_rms']:.6f}。特征变化用于确认参与，不单独证明收益。",'','超过476关层候选且保护450地表/云的检查点：'+json.dumps(candidates),'',(R/'INTERPRETATION.md').read_text(encoding='utf-8'),'','方法来源：形态学闭运算/黑顶帽为已有遥感阴影方法；本轮检验其网络内特征融合，不声称算子原创。[相关研究](https://repositorio.unesp.br/entities/publication/68b2edaa-5ffe-4c42-a521-00928a0ec343)。RGB暗区描述并非真实反射率反演。','','验收：新分支和解码器实际更新、其余参数/BN冻结；共同初始化与记录批次匹配；535图独立混淆矩阵、权重及传输SHA通过。单seed验证集探索；场景区间不包含重新训练seed波动。最佳点比较与同一步比较分别列出。','','部署：加载原431基座，再加载student_patch和infer_shadow479中的分支state_dict。off为显式enabled=False的诊断条件，单独保存state_dict不会保存此开关，不应将off诊断权重误当成默认部署模型。']
(R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(R/'LOCAL_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,images_per_checkpoint=535,ROI_count=107,target_supervision_percent=0,choices=choices),indent=2))
files=list((ROOT/'src/tools').glob('*479.py'))+list((ROOT/'src/tools').glob('*479.sh'))+[ROOT/'src/research_plans/phase479.md']
(R/'CODE_SHA.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},indent=2))
print(json.dumps({name:dict(step=x.get('step'),IoU=x['IoU'],mIoU=x['mIoU']) for name,x in rows}))
