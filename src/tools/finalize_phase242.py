"""At most three main tables; full exact matrices and diagnostics are separate CSVs."""
from phase242_common import *
from analyze_phase242 import compare_cells,avg
def fmt(x):return 'NA' if x is None else f'{100*x:.3f}'
def summary_decisions(r):
 out=[]
 for model,state,dec in sorted({(v['model'],v['state'],v['decision']) for v in r['decisions']}):
  rows=[v for v in r['decisions'] if (v['model'],v['state'],v['decision'])==(model,state,dec)];c=np.sum([v['confusion'] for v in rows],axis=0);rates=[v['nonShadow_to_Shadow_FPR'] for v in rows if v['nonShadow_to_Shadow_FPR'] is not None];m=metrics(c);out.append({'model':model,'state':state,'decision':dec,**m,'scene_mean_union_mIoU':avg([v['scene_mIoU_union'] for v in rows]),'scene_mean_reference_present_mIoU':avg([v['scene_mIoU_reference_present'] for v in rows]),'scene_mean_Shadow_FPR':avg(rates),'FPR_above_1pct':sum(v>.01 for v in rates),'worst_scene_FPR':max(rates) if rates else None,'worst_scene_ID':max(rows,key=lambda v:v['nonShadow_to_Shadow_FPR'] if v['nonShadow_to_Shadow_FPR'] is not None else -1)['scene']})
 return out
def main():
 started=time.monotonic();assert js(R/'INDEPENDENT_ACCEPTANCE.json')['passed'];assert not (R/'delivery').exists();matrices=[];summary=[];pixelstats=[];transitions=[];decomp=[];signals=[];scatter=[];loo=[];contributions=[];decisions=[];pooled=[];tops=[];support=[]
 datasets={}
 for split in ('val','test'):
  for domain in ('L8','SPARCS'):
   r=js(R/(split.upper()+'_'+domain+'_RESULTS.json'));datasets[split,domain]=r;base={'split':split,'domain':domain};pooled.extend([{**base,**v} for v in summary_decisions(r)]);decisions.extend([{**base,**v} for v in r['decisions']]);decomp.extend([{**base,**v} for v in r['decompositions']]);tops.extend([{**base,**v} for v in r['top_pairs']]);loo.extend([{**base,**v} for v in r['leave_one']])
   for key,mat in r['matrices'].items():
    model,state=key.split(':');cells=compare_cells(r['matrices']['Source:raw'],mat);matrices.extend([{**base,'model':model,'state':state,**c} for c in cells]);q=mat['summary'];s=r['matrices']['Source:raw']['summary'];summary.append({**base,'model':model,'state':state,**q,'pooled_delta':q['pooled_AUC']-s['pooled_AUC'],'scene_mean_delta':q['scene_mean_AUC']-s['scene_mean_AUC'],'damage_contribution':sum(v['weighted_delta'] for v in cells if v['weighted_delta'] is not None and v['weighted_delta']<0),'gain_contribution':sum(v['weighted_delta'] for v in cells if v['weighted_delta'] is not None and v['weighted_delta']>0)})
    for scene in r['scenes']:contributions.append({**base,'model':model,'state':state,'scene':scene,'Shadow_row':sum(v['weighted_delta'] or 0. for v in cells if v['positive_scene']==scene),'Surface_column':sum(v['weighted_delta'] or 0. for v in cells if v['negative_scene']==scene)})
   # Preserve seed means as arithmetic matrices, never as additional samples/models.
   for state in ('raw','J0','J1'):
    aa=[v for v in matrices if v['split']==split and v['domain']==domain and v['state']==state and v['model']=='C21701'];bb=[v for v in matrices if v['split']==split and v['domain']==domain and v['state']==state and v['model']=='C21702']
    for a,b in zip(aa,bb):matrices.append({**a,'model':'two_seed_mean','AUC':None if a['AUC'] is None else (a['AUC']+b['AUC'])/2,'AP':None if a['AP'] is None else (a['AP']+b['AP'])/2,'C_AUC':None if a['C_AUC'] is None else (a['C_AUC']+b['C_AUC'])/2,'delta_AUC':None if a['delta_AUC'] is None else (a['delta_AUC']+b['delta_AUC'])/2,'weighted_delta':None if a['weighted_delta'] is None else (a['weighted_delta']+b['weighted_delta'])/2})
   for model,sig in r['signals'].items():
    signals.append({**base,'model':model,**{k:v for k,v in sig.items() if k not in ('scatter','leave_one_scene')}});scatter.extend([{**base,'model':model,**v} for v in sig['scatter']]);loo.extend([{**base,'model':model,'kind':'signal','excluded':v['excluded'],'Spearman':v['Spearman'],'evaluable_pairs':v['evaluable_pairs']} for v in sig['leave_one_scene']])
   for scene in r['scenes']:
    rec=js(R/'scene_receipts'/split/domain/(scene+'.json'));metadata=js(R/'SCENE_METADATA.json').get(domain,{}).get(scene,{});support.append({**base,'scene':scene,**metadata,'reference':rec['reference'],'valid':rec['valid'],'P_Shadow':rec['reference'][2],'N_Surface':rec['reference'][0]})
    for model,c in rec['C'].items():
     pixelstats.extend([{**base,'scene':scene,'model':model,'m':c['m'],'original_biome':metadata.get('biome'),'posthoc_reference_strata_not_unlabelled_inputs':True,**v} for v in c['statistics']]);transitions.append({**base,'scene':scene,'model':model,'27cells':c['transition_reference_Source_C'],'changes':c['changes']})
 rows={'MATRICES.csv':matrices,'RANKING_SUMMARY.csv':summary,'PIXEL_STATS.csv':pixelstats,'TRANSITIONS_27.csv':transitions,'DECOMPOSITION.csv':decomp,'SIGNALS.csv':signals,'SIGNAL_SCATTER.csv':scatter,'LEAVE_ONE_SCENE.csv':loo,'ROW_COLUMN_CONTRIBUTIONS.csv':contributions,'PER_SCENE_DECISIONS.csv':decisions,'DECISION_SUMMARY.csv':pooled,'TOP_PAIRS.csv':tops,'SUPPORT.csv':support}
 for name,v in rows.items():csvsave(R/name,v)
 save('FINAL_DATA.json',{'ranking':summary,'decisions':pooled,'signals':signals,'top_pairs':tops,'decomposition':decomp})
 lines=['# Phase242：场景对与同像元评分变化','', '## 1．数据结果','', '仅复用Phase241 raw FP32缓存；网络调用、训练、下载、新阈值与bootstrap均为0。主要支持为L8 test官方yes六景、SPARCS test十景；val分别八景、十景，可评对数按实际正负支持给出。L8 no十景未新增诊断。AUC及决策比率以百分数、差值以百分点列示；log评分变化不乘100。场景对不是独立样本。','', '### 表1　损伤在哪里：test完整有向矩阵汇总','', '|目标|模型|状态|景均AUC|景内加权AUC|跨景AUC|池化AUC|池化差|损伤贡献/收益贡献|','|---|---|---|---|---|---|---|---|---|']
 for v in summary:
  if v['split']=='test':lines.append('|'+ '|'.join([v['domain'],v['model'],v['state']]+[fmt(v[k]) for k in ('scene_mean_AUC','within_weighted_AUC','cross_AUC','pooled_AUC','pooled_delta')]+[fmt(v['damage_contribution'])+'/'+fmt(v['gain_contribution'])])+'|')
 lines+=['','两seed平均加权差固定选择的跨景前三损伤／前三改善对（正Shadow来源→负Surface来源）：']
 for v in tops:lines.append(f"- {v['domain']} {v['kind']}：{v['positive_scene']} → {v['negative_scene']}；P={v['P']}，N={v['N']}，w={v['weight']:.6g}，ΔAUC={fmt(v['delta_AUC'])}，加权贡献={fmt(v['weighted_delta'])}。")
 lines+=['','### 表2　同像元变化（test；完整分位数、三类p变化与27格转移另附）','','|目标|场景|seed|整景δ中位数|Surface/Cloud/Shadow δ中位数|Source对→C错|Source错→C对|','|---|---|---|---|---|---|---|']
 for domain in ('L8','SPARCS'):
  for scene in datasets['test',domain]['scenes']:
   rec=js(R/'scene_receipts/test'/domain/(scene+'.json'))
   for model,c in rec['C'].items():
    v=[s['delta']['median'] for s in c['statistics'][1:]];lines.append('|'+ '|'.join([domain,scene,model,f"{c['m']:.6f}",'/'.join('NA' if x is None else f'{x:.6f}' for x in v),str(sum(x['Source_correct_C_wrong'] for x in c['changes'])),str(sum(x['Source_wrong_C_correct'] for x in c['changes']))])+'|')
 lines+=['','固定六对每seed的SS/CS/SC/CC及Shadow侧／Surface侧分量见DECOMPOSITION.csv；两分量和逐项复现CC−SS。这是跨模型混合评分的对称算术分解，包含尺度差异，不是正负特征更新的因果归因。','', '固定信号T=m_i−m_j，预期越大越有利。相关性、方向一致率及留一景范围（不做场景对p值）：']
 for v in signals:lines.append(f"- {v['domain']} {v['split']} {v['model']}：ρ={v['Spearman']}；方向一致={fmt(v['direction_agreement'])}，可评对={v['evaluable_pairs']}，非平局={v['non_tie_pairs']}，T零={v['T_zero']}，ΔAUC零={v['delta_zero']}，留一ρ范围={v['leave_one_rho_range']}。")
 lines+=['','### 表3　raw/J0/J1实际决策及代价（test，阈值均沿用241 raw冻结值）','','|目标|模型|状态|决策|池化mIoU|景均union/参考类IoU|Cloud/Shadow召回|Surface→Cloud FPR|非Shadow→Shadow FPR|景均Shadow FPR|超1%景数/最差景FPR|','|---|---|---|---|---|---|---|---|---|---|---|']
 for v in pooled:
  if v['split']=='test':lines.append('|'+ '|'.join([v['domain'],v['model'],v['state'],v['decision'],fmt(v['mIoU']),fmt(v['scene_mean_union_mIoU'])+'/'+fmt(v['scene_mean_reference_present_mIoU']),fmt(v['Cloud_recall'])+'/'+fmt(v['Shadow_recall']),fmt(v['Surface_to_Cloud_FPR']),fmt(v['nonShadow_to_Shadow_FPR']),fmt(v['scene_mean_Shadow_FPR']),str(v['FPR_above_1pct'])+'/'+fmt(v['worst_scene_FPR'])])+'|')
 lines+=['','## 2．结论分析','','本地最终交付须结合以上数据补充：损伤集中性与留一敏感性、类相关变化能否支持近似平移、固定信号是否跨val/test同向，以及排序变化和实际决策代价是否一致。不得仅据m的差异称纯加性偏置，不根据test改信号或从J0/J1挑赢家。参考类组成、真类分层与原biome只作事后描述，未经影像证据不能断言雪冰/暗地物致因。','', 'T、J1均需要适配后的C分数；仅在类别标签意义上无标签，仍依赖241发布有效域/Fill掩膜。J1是整景转导式诊断，不是已经验证的新部署算法。所有test已暴露；没有新增显著性或独立确认。J0成对排序、J1景内排序不变是构造性质，不是方法成功。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');d=R/'delivery';d.mkdir()
 cost={'network_calls':0,'gradient_updates':0,'fitted_detection_models':0,'downloads':0,'new_thresholds':0,'new_bootstrap':0,'CPU_seconds_through_report_before_packaging':cpu_seconds()+time.monotonic()-started,'packaging_tail':'full final stage duration in cpu_costs/finalize.json and STAGE_JOURNAL.jsonl; measured after this report','sampled_peak_RSS_bytes':max([js(p).get('peak_RSS_bytes') or 0 for p in (R/'cpu_costs').glob('*.json')]),'new_remote_bytes_before_delivery':sum(p.stat().st_size for p in R.rglob('*') if p.is_file()),'raw_FP32_241_not_copied':True,'limits':js(R/'LOCK.json')['limits']};save('COST.json',cost)
 names=['RESULT.md','FINAL_DATA.json','VAL_LOCK.json','VAL_SEAL.json','VAL_COMPLETE.json','TEST_COMPLETE.json','TEST_ACCESS_STARTED.json','INDEX_REFERENCES.json','RAW_THRESHOLDS.json','MODEL_IDENTITIES.json','LOCK.json','SCENE_METADATA.json','REMOTE_PREFLIGHT.json','KERNEL_ACCEPTANCE.json','CPU_SMOKE_ACCEPTED.json','INDEPENDENT_ACCEPTANCE.json','COST.json','BUNDLE_SHA.json','VAL_L8_RESULTS.json','VAL_SPARCS_RESULTS.json','TEST_L8_RESULTS.json','TEST_SPARCS_RESULTS.json']+list(rows)
 for n in names:shutil.copy2(R/n,d/n)
 for n in ('code','scene_receipts','cpu_costs'):shutil.copytree(R/n,d/n)
 files={p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()};save(d/'SHA256SUMS.json',files);size=sum(p.stat().st_size for p in d.rglob('*') if p.is_file());assert size<=500*1024**2
 save('DELIVERY_COMPLETE.json',{'passed':True,'epoch':time.time(),'formal_bytes':size,'manifest_SHA':sha(d/'SHA256SUMS.json'),'no_full_pixel_arrays_transferred':True});save(d/'DELIVERY_COMPLETE.json',js(R/'DELIVERY_COMPLETE.json'))
if __name__=='__main__':main()
