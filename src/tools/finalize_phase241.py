"""Small formal delivery only; full scores and repeat arrays stay remote."""
from phase241_cpu_common import *
import shutil,csv

def main():
 assert js(R/'INDEPENDENT_ACCEPTANCE.json')['passed'];assert not (R/'delivery').exists()
 from analyze_phase241 import writecsv,mean
 m=js(R/'METRICS.json');p=m['pooled'];rank=m['ranking'];ci=js(R/'BOOTSTRAP.json')
 def fmt(v):return 'NA' if v is None else f'{100*v:.3f}'
 def pick(domain,model,state,support='main',decision='argmax'):
  return next((v for v in p if all(v[k]==x for k,x in [('domain',domain),('split','test'),('model',model),('state',state),('support',support),('decision',decision)])),None)
 lines=['# Phase241：排序与决策审计','', '## 1．数据结果','', '所有数值来自已暴露数据的探索诊断。主要支持为L8官方yes六景、SPARCS发布参考十景；完整L8十六景及no十景只作名义参考。两个seed不是额外测试场景。AP/AUC/IoU/召回/FPR以百分数列示，NLL/Brier不乘100。完整val、逐景与名义口径见CSV。','', '### 表1　主要test支持排序（单元格AP/AUC；完整六任务在RANKING.csv）','', '|目标|模型|状态|聚合|Surface OvR|Cloud OvR|Shadow OvR|Shadow/Surface|Cloud/Surface|Shadow/Cloud|可评景数（同任务顺序）|','|---|---|---|---|---|---|---|---|---|---|---|']
 tasks=['Surface_OvR','Cloud_OvR','Shadow_OvR','Shadow_Surface','Cloud_Surface','Shadow_Cloud']
 for domain in ('L8','SPARCS'):
  for model in ('Source','C21701','C21702'):
   for state in ('raw','calibrated'):
    for scope in ('scene_mean','pooled'):
     rs=[next((v for v in rank if all(v[k]==x for k,x in [('domain',domain),('split','test'),('model',model),('state',state),('support','main'),('scope',scope),('task',task)])),None) for task in tasks]
     lines.append('|'+ '|'.join([domain,model,state,scope]+['NA' if v is None else fmt(v['AP'])+'/'+fmt(v['AUC']) for v in rs]+[','.join(str(v['evaluable_scenes']) if v else '0' for v in rs)])+'|')
 lines+=['','主要比较景均Shadow/Surface AUC，C−Source（百分点；条件95%区间）：']
 for v in ci:
  if v['metric']=='Shadow_Surface_AUC' and v['contrast']=='C-A' and v['decision'] is None:
   raw={x['model']:x['AUC'] for x in rank if x['domain']==v['domain'] and x['split']=='test' and x['state']=='raw' and x['scope']=='scene_mean' and x['support']=='main' and x['task']=='Shadow_Surface'}
   vals=[raw.get('C'+str(s)) for s in (21701,21702)] if v['seed']=='two_seed_difference_mean' else [raw.get('C'+str(v['seed']))];point=mean(vals);point=None if point is None or raw.get('Source') is None else point-raw['Source'];bounds='NA' if v['CI95'] is None else '['+', '.join(fmt(t) for t in v['CI95'])+']'
   lines.append(f"- {v['domain']} seed={v['seed']}：{fmt(point)}，{bounds}；全NA重采样{v['NA_replicates']}次。")
 lines+=['','### 表2　2×2指定偏置对齐（主要test支持）','','|目标|seed|A/B/C/D或差值|池化mIoU|景均union mIoU|景均参考出现类IoU|NLL|Brier|','|---|---|---|---|---|---|---|---|']
 algebra=[]
 keys=['mIoU_fixed3','scene_mean_mIoU_union','scene_mean_mIoU_reference_present','NLL','Brier']
 for domain in ('L8','SPARCS'):
  for seed in (21701,21702):
   rows=[pick(domain,'Source','raw'),pick(domain,'Source','calibrated'),pick(domain,'C'+str(seed),'raw'),pick(domain,'C'+str(seed),'calibrated')]
   for label,row in zip('ABCD',rows):
    lines.append('|'+ '|'.join([domain,str(seed),label]+[('NA' if row is None or row[k] is None else (fmt(row[k]) if j<3 else f'{row[k]:.6f}')) for j,k in enumerate(keys)])+'|')
   for name,coeff in [('B-A',[-1,1,0,0]),('D-C',[0,0,-1,1]),('C-A',[-1,0,1,0]),('D-B',[0,-1,0,1]),('(B-A)-(D-C)',[-1,1,1,-1])]:
    result={k:None if any(r is None or r[k] is None for r in rows) else float(sum(c*r[k] for c,r in zip(coeff,rows))) for k in keys};algebra.append({'domain':domain,'seed':seed,'contrast':name,**result});lines.append('|'+ '|'.join([domain,str(seed),name]+[fmt(result[k]) if j<3 else ('NA' if result[k] is None else f'{result[k]:.6f}') for j,k in enumerate(keys)])+'|')
 lines+=['','### 表3　实际决策（主要test支持；FPR为发布参考口径）','','|目标|模型|状态|决策|Cloud召回|Shadow召回|Surface→Cloud FPR|非Shadow→Shadow FPR|池化mIoU|景均Shadow FPR|超1%景数/最差景FPR|','|---|---|---|---|---|---|---|---|---|---|---|']
 for domain in ('L8','SPARCS'):
  for model in ('Source','C21701','C21702'):
   for state in ('raw','calibrated'):
    for decision in ('argmax','frozen_threshold'):
     v=pick(domain,model,state,decision=decision)
     if v:lines.append('|'+ '|'.join([domain,model,state,decision]+[fmt(v[k]) for k in ['Cloud_recall','Shadow_recall','Surface_to_Cloud_FPR','nonShadow_to_Shadow_FPR','mIoU_fixed3','scene_mean_nonShadow_to_Shadow_FPR']]+[str(v['FPR_scenes_above_1pct'])+'/'+fmt(v['worst_scene_FPR'])])+'|')
 lines+=['','## 2．结论分析','','排序变化先看上述预先固定的C−Source景均AUC及区间，不以单一池化AP或mIoU替代。B−A描述指定NLL偏置对齐的兑现量；D−B是指定对齐后的剩余差异，不能解释为特征贡献或真正上界。校准和checkpoint选择共用val，区间条件于固定模型、校准、阈值，不覆盖训练随机性。','', '原始argmax mIoU混合了排序、类别决策和参考支持差异；是否收窄原判断应同时检查排序主差值、2×2与错误流。L8 no十景存在未知漏标，不能据其名义FPR宣称真实误报或用人审反馈自动改标签。本轮不进入独立确认或新增方法。','', '完整原始FP32分数留在远端scores/并带索引SHA；本地只交小结果。失败、实际调用和资源均在COST.json；若任一校准失败按NA保留，未替换配方。']
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');writecsv(R/'ALIGNMENT_DELTAS.csv',algebra)
 size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file());save('COST.json',{'calls':len(ledger()),'call_kinds':dict(collections.Counter(v['kind'] for v in ledger())),'GPU_stage_seconds':prior_seconds(),'CPU_processing_seconds':cpu_seconds(),'remote_bytes_before_delivery':size,'caps':{'calls':9600,'GPU_seconds':14400,'CPU_seconds':43200,'remote_bytes':60*1024**3,'formal_bytes':1024**3},'training':0,'downloads':0,'score_dtype':'float32','CPU_peak_bytes_sampled_stages':max([v['CPU_peak_bytes'] for v in [js(p) for p in (R/'cpu_costs').glob('*.json')] if v['CPU_peak_bytes'] is not None]),'compile_smoke_RSS':'not sampled','calendar_stop_cancelled':True})
 d=R/'delivery';d.mkdir();names=['RESULT.md','PER_SCENE.csv','RANKING.csv','POOLED.csv','ALIGNMENT_DELTAS.csv','BOOTSTRAP.csv','BOOTSTRAP.json','BOOTSTRAP_INDICES.json','METRICS.json','CALIBRATION_POSITIONS.json','CALIBRATION_LOCK.json','THRESHOLD_LOCK.json','ALIGNMENT_SEAL.json','KERNEL_SELFTEST.json','INDEPENDENT_ACCEPTANCE.json','COST.json','VAL_SCORE_INDEX.jsonl','TEST_SCORE_INDEX.jsonl','VAL_COMPLETE.json','TEST_COMPLETE.json','VAL_BACKEND.json','TEST_BACKEND.json','MODEL_IDENTITIES.json','SUPPORT.csv','LOCK.json','BUNDLE_SHA.json','CALL_LEDGER.jsonl','REMOTE_PREFLIGHT.json']
 for name in names:shutil.copy2(R/name,d/name)
 for name in ('stage_costs','cpu_costs','code'):
  shutil.copytree(R/name,d/name)
 manifest={p.relative_to(d).as_posix():sha(p) for p in d.rglob('*') if p.is_file()};save(d/'SHA256SUMS.json',manifest);total=sum(p.stat().st_size for p in d.rglob('*') if p.is_file());assert total<1024**3
 save('DELIVERY_COMPLETE.json',{'passed':True,'epoch':time.time(),'formal_bytes':total,'manifest_SHA':sha(d/'SHA256SUMS.json'),'full_scores_stay_remote':True});save(d/'DELIVERY_COMPLETE.json',js(R/'DELIVERY_COMPLETE.json'))
if __name__=='__main__':main()
