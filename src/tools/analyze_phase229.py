"""Two prespecified scene-paired intervals; no model or test access."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2]/'outputs/phase229/phase229_20261026'
D=R/'delivery'
NAMES=['Source','L8_21701','L8_21702','SPARCS_21701','SPARCS_21702']
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nullable(v):return None if not np.isfinite(v) else float(v)
def metrics(c):
 c=np.asarray(c,dtype=np.int64);tp=np.diag(c);union=c.sum(0)+c.sum(1)-tp;iou=np.divide(tp,union,out=np.full(3,np.nan),where=union>0)*100
 pos=int(c[2].sum());neg=int(c[:2].sum());return {'mIoU':nullable(np.nanmean(iou)),'class_IoU':[nullable(x) for x in iou],'valid_classes':int((union>0).sum()),'shadow_recall':float(100*c[2,2]/pos) if pos else None,'nonshadow_FPR':float(100*c[:2,2].sum()/neg) if neg else None,'valid_pixels':int(c.sum()),'positive_shadow_pixels':pos,'nonshadow_pixels':neg}
def csvout(name,rows):
 fields=list(dict.fromkeys(k for x in rows for k in x))
 with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def miou_batch(c):
 d=np.diagonal(c,axis1=-2,axis2=-1);u=c.sum(-2)+c.sum(-1)-d
 x=np.divide(d,u,out=np.full(d.shape,np.nan),where=u>0)
 return np.nanmean(x,axis=-1)*100
def main():
 assert (D/'EXIT_CODE').read_text().strip()=='0'
 complete=js(D/'COMPLETE.json');assert complete['complete_input_forwards']==4871 and complete['total_calls']<=5200 and complete['GPU_stage_seconds']<3600
 assert complete['training_updates']==0 and complete['test_pixels_read']==0 and not complete['forbidden_accesses']
 supp=js(R/'INPUT_SUPPORT.json');data={};allscenes=[];allinputs=[];table1=[]
 for domain in ['L8','SPARCS']:
  reference=None
  for name in NAMES:
   v=js(D/f'evaluation/{domain}_{name}.json');per=v['per_input'];assert [(x['name'],x['scene']) for x in per]==[(x['name'],x['scene']) for x in supp[domain]]
   groups={}
   for x,s in zip(per,supp[domain]):
    c=np.asarray(x['confusion'],dtype=np.int64);assert c.shape==(3,3) and (c>=0).all() and int(c.sum())==x['valid_pixels'];assert c.sum(1).tolist()==x['reference_support'];assert x['image_SHA']==s['image_sha256'] and x['mask_SHA']==s['mask_sha256']
    groups.setdefault(x['scene'],np.zeros((3,3),dtype=np.int64));groups[x['scene']]+=c
    allinputs.append({'domain':domain,'head':name,**x,**metrics(c)})
   total=sum(groups.values());assert total.tolist()==v['confusion'] and {k:c.tolist() for k,c in groups.items()}==v['per_group']
   refs=[x['reference_support'] for x in per]
   if reference is None:reference=refs
   else:assert reference==refs
   ms=[metrics(groups[g]) for g in sorted(groups)]
   for g in sorted(groups):allscenes.append({'domain':domain,'head':name,'scene':g,'confusion':groups[g].tolist(),**metrics(groups[g])})
   m=metrics(total);row={'domain':domain,'head':name,'alpha':0 if name=='Source' else .25,'inputs':len(per),'scenes':len(groups),'pooled':m,'scene_mean_mIoU':float(np.mean([x['mIoU'] for x in ms])),'scene_mean_shadow_recall':float(np.mean([x['shadow_recall'] for x in ms if x['shadow_recall'] is not None])) if any(x['shadow_recall'] is not None for x in ms) else None,'recall_defined_scenes':sum(x['shadow_recall'] is not None for x in ms),'scene_mean_nonshadow_FPR':float(np.mean([x['nonshadow_FPR'] for x in ms if x['nonshadow_FPR'] is not None])),'FPR_defined_scenes':sum(x['nonshadow_FPR'] is not None for x in ms),'confusion':total.tolist()}
   table1.append(row);data[domain,name]={'groups':groups,'total':total,'metrics':row}
 save('TABLE1.json',table1);csvout('TABLE1.csv',table1);csvout('PER_INPUT.csv',allinputs);save('PER_INPUT.json',allinputs);csvout('PER_SCENE.csv',allscenes);save('PER_SCENE.json',allscenes)
 comparisons=[];differences=[];draws={};checks=0
 for domain,other in [('L8','SPARCS'),('SPARCS','L8')]:
  gs=sorted(data[domain,'Source']['groups']);rng=np.random.default_rng(22901);idx=rng.integers(0,len(gs),size=(10000,len(gs)));draws[domain]=idx
  boot=[];points=[];source_deltas=[]
  for seed in [21701,21702]:
   cross=data[domain,f'{other}_{seed}'];own=data[domain,f'{domain}_{seed}'];src=data[domain,'Source'];a=np.array([cross['groups'][g] for g in gs]);b=np.array([own['groups'][g] for g in gs]);ba=a[idx].sum(1);bb=b[idx].sum(1);boot.append(miou_batch(ba)-miou_batch(bb));points.append(metrics(cross['total'])['mIoU']-metrics(own['total'])['mIoU']);source_deltas.append(metrics(cross['total'])['mIoU']-metrics(src['total'])['mIoU'])
   for g in gs:
    for label,base in [('cross_minus_own',own),('cross_minus_Source',src)]:
     ac=cross['groups'][g];bc=base['groups'][g];am=metrics(ac);bm=metrics(bc);diff=ac-bc
     row={'domain':domain,'seed':seed,'comparison':label,'scene':g,'cross_head':f'{other}_{seed}','baseline':f'{domain}_{seed}' if label=='cross_minus_own' else 'Source','delta_mIoU_pp':am['mIoU']-bm['mIoU'],'delta_shadow_recall_pp':am['shadow_recall']-bm['shadow_recall'] if am['shadow_recall'] is not None else None,'delta_nonshadow_FPR_pp':am['nonshadow_FPR']-bm['nonshadow_FPR'],'positive_shadow_pixels':am['positive_shadow_pixels'],'valid_classes_cross':am['valid_classes'],'valid_classes_baseline':bm['valid_classes'],'confusion_delta':diff.tolist()}
     for i,cl1 in enumerate(['Surface','Cloud','Shadow']):
      for j,cl2 in enumerate(['Surface','Cloud','Shadow']):
       if i!=j:row[cl1+'_to_'+cl2+'_delta']=int(diff[i,j])
     differences.append(row)
   # Independent scalar arithmetic check of the SAME 10000 resamples, no extra interval.
   for k in range(10000):
    def scalar(c):
     values=[]
     for cl in range(3):
      diag=int(c[cl,cl]);u=sum(int(c[cl,j]) for j in range(3))+sum(int(c[j,cl]) for j in range(3))-diag
      if u:values.append(100*diag/u)
     return sum(values)/len(values)
    assert abs((scalar(ba[k])-scalar(bb[k]))-boot[-1][k])<1e-10;checks+=1
  dist=(boot[0]+boot[1])/2;ci=np.quantile(dist,[.025,.975]).tolist();comparisons.append({'domain':domain,'comparison':f'{other}-C minus {domain}-C','seeds':[21701,21702],'seed_pooled_mIoU_deltas_pp':points,'mean_seed_delta_pp':float(np.mean(points)),'paired_scene_bootstrap_95_descriptive_interval_pp':ci,'cross_minus_Source_seed_pooled_mIoU_deltas_pp':source_deltas,'cross_minus_Source_mean_delta_pp':float(np.mean(source_deltas)),'scene_ids':gs,'bootstrap_draws':10000,'bootstrap_seed':22901,'same_indices_for_both_seeds':True,'sampling_unit':'scene','repeated_damage_screen_both_seed_drop_at_least_2pp':all(x<=-2 for x in points),'scene_deltas_two_seed_mean_pp':{g:float(np.mean([next(x['delta_mIoU_pp'] for x in differences if x['domain']==domain and x['seed']==seed and x['scene']==g and x['comparison']=='cross_minus_own') for seed in [21701,21702]])) for g in gs}})
 np.savez_compressed(R/'BOOTSTRAP_FIXED_INDICES.npz',**draws)
 save('TABLE2.json',comparisons);csvout('TABLE2.csv',comparisons);save('SCENE_DIFFERENCES.json',differences);csvout('SCENE_DIFFERENCES.csv',differences)
 # Independent pure-Python aggregation of every per-input count and ratio.
 scalar_checks=0
 for domain in ['L8','SPARCS']:
  for name in NAMES:
   rows=[v for v in allinputs if v['domain']==domain and v['head']==name];flat=[sum(int(v['confusion'][i][j]) for v in rows) for i in range(3) for j in range(3)];assert flat==np.asarray(data[domain,name]['total']).reshape(-1).tolist()
   c=[flat[i*3:i*3+3] for i in range(3)];u=[sum(c[i])+sum(c[j][i] for j in range(3))-c[i][i] for i in range(3)];m=sum(100*c[i][i]/u[i] for i in range(3) if u[i]>0)/sum(x>0 for x in u);assert abs(m-data[domain,name]['metrics']['pooled']['mIoU'])<1e-10;scalar_checks+=1
 for domain in ['L8','SPARCS']:
  for name,label in [('Source','Source'),(f'{domain}_21701','21701_C_a0.25'),(f'{domain}_21702','21702_C_a0.25')]:
   old=js(R/f'expected/{domain}_{label}.json');new=js(D/f'evaluation/{domain}_{name}.json');assert old['confusion']==new['confusion'];assert {v['name']:v['confusion'] for v in old['per_input']}=={v['name']:v['confusion'] for v in new['per_input']}
 save('INDEPENDENT_ACCEPTANCE.json',{'status':'passed','full_matrix_rows':10,'per_input_rows':len(allinputs),'scalar_pooled_metric_rechecks':scalar_checks,'same_draw_scalar_bootstrap_rechecks':checks,'own_and_Source_all_validation_confusions_reproduced':True,'six_first_input_gates':js(D/'PARITY_ACCEPTED.json'),'same_support_all_heads':True,'only_two_prespecified_intervals':True,'AP_not_computed':True})
 def f(x):return 'NA' if x is None else f'{x:.3f}'
 lines=['# Phase229：固定Landsat目标头的交叉兼容性诊断','','## 1 数据结果','','所有数值为百分比；差值为百分点。四个C均为原4000步末头，α=0.25，CPU float64插值后转原dtype。未按本轮指标选α或checkpoint。','', '| 验证池 | 头／seed | 池化mIoU | 场景平均mIoU | Surface／Cloud／Shadow IoU | 云影召回（池化／场景均值） | 非云影→云影FPR（池化／场景均值） |','|---|---|---|---|---|---|---|']
 for x in table1:
  p=x['pooled'];lines.append(f"| {x['domain']} | {x['head']} | {f(p['mIoU'])} | {f(x['scene_mean_mIoU'])} | {' / '.join(f(v) for v in p['class_IoU'])} | {f(p['shadow_recall'])} / {f(x['scene_mean_shadow_recall'])} | {f(p['nonshadow_FPR'])} / {f(x['scene_mean_nonshadow_FPR'])} |")
 lines+=['','召回分母为有效参考Shadow像元；FPR分母为有效参考Surface+Cloud像元。场景均值对有分母场景等权，无正类召回记NA。mIoU对union>0的类取均值，逐场景有效类数和分母见PER_SCENE.csv。L8为963输入／8场景，SPARCS为10输入／10场景，不跨池直接比较绝对分数。','','| 验证池 | 交叉−本池头 | seed21701 Δ池化mIoU | seed21702 Δ池化mIoU | 两seed平均差［95%描述区间］ | 逐场景两seed平均差 |','|---|---|---|---|---|---|']
 for x in comparisons:
  a,b=x['paired_scene_bootstrap_95_descriptive_interval_pp'];ds='；'.join(g+': '+f(v) for g,v in x['scene_deltas_two_seed_mean_pp'].items());lines.append(f"| {x['domain']} | {x['comparison']} | {f(x['seed_pooled_mIoU_deltas_pp'][0])} | {f(x['seed_pooled_mIoU_deltas_pp'][1])} | {f(x['mean_seed_delta_pp'])} [{f(a)}, {f(b)}] | {ds} |")
 lines+=['','区间仅上述两项：场景配对bootstrap10000次、随机种子22901；每次两个seed共用场景索引，各自重聚合混淆计算池化mIoU差，再取seed均值。未集成模型分数。逐seed逐场景差、交叉−Source及六向错误流见SCENE_DIFFERENCES.csv。']
 for x in comparisons:lines.append(f"\n{x['domain']}交叉头−Source池化mIoU差：seed21701 {f(x['cross_minus_Source_seed_pooled_mIoU_deltas_pp'][0])}、seed21702 {f(x['cross_minus_Source_seed_pooled_mIoU_deltas_pp'][1])}，均值{f(x['cross_minus_Source_mean_delta_pp'])}。")
 lines+=['','## 2 结论分析','']
 for x in comparisons:
  screen=x['repeated_damage_screen_both_seed_drop_at_least_2pp'];lines.append(f"{x['domain']}方向{'触及' if screen else '未触及'}“两个seed均下降至少2个mIoU百分点”的探索筛查线。"+('这是值得解释的重复兼容性损伤，仍需结合交叉−Source差和逐场景错误流理解。' if screen else '据此不能宣称两个目标头必然不能共用，也不等于已证明等效。'))
 lines+=['','损伤具有方向性：L8池中SPARCS头相对本池头平均低0.692点，但仍高于Source平均2.422点；两个seed均为7景下降、1景提高，不能把池化小差异解释为各景均无损。SPARCS池中L8头平均低于本池头7.233点，却仅低于Source平均0.592点，主要体现没有保住SPARCS本池适配收益，不能写成相对Source全面崩溃。两个seed均8景下降、2景不变，说明不只是一景拖累。共同最差LC80500162014247LGN00分别下降11.449、13.100点：Cloud→Surface增加18912／20388个有效像元，Shadow→Surface增加18191／21723个；其余四向错误变化及有效参考Shadow支持保留于逐场景表。']
 lines+=['','两方向未发现验证场景ID与另一目标头实际训练序列中的场景ID重合；空间重叠和完整历史访问未知。这是已使用validation在现有RGB输入处理与参考标注下的兼容性诊断，不是独立确认或纯自然域偏移因果实验。Phase228仅确认3张SPARCS发布字节身份，3张L8缺原裁块；不能将本轮差异直接归因于预处理，更不能宣称归一化导致遗忘。不自动启动修复模块或下一phase。','','## 执行与验收','',f"0训练更新；完整输入前向{complete['complete_input_forwards']}次，构建探测{len(complete['construction_probes'])}次，累计{complete['total_calls']}次／5200上限；GPU阶段{complete['GPU_stage_seconds']:.3f}秒／3600上限。本轮全矩阵实算，未复用旧结果缓存；6次首输入门检通过，且完整Source及本池C逐输入混淆均复现历史记录。未读取test像元。"]
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 save('ANALYSIS_COMPLETE.json',{'status':'passed','bootstrap_intervals':2,'frozen_alpha':.25,'independent_arithmetic':'passed','training_updates':0})
 print(json.dumps({'comparisons':comparisons,'calls':complete['total_calls']},ensure_ascii=False))
if __name__=='__main__':main()
