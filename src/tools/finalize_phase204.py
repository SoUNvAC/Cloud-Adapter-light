"""Complete Phase204 cache-only delivery; never regenerate sealed signals."""
import csv,hashlib,json,shutil,sys,time
from pathlib import Path
R=Path('D:/Cloud-Adapter-light');O=R/'outputs/phase204/phase204_20261019';S=R/'src/research_results/phase204_20261019'
sys.path.insert(0,str(R/'src/tools'))
import analyze_phase203_local as old
import numpy as np
import os
from bounded_phase202_local import usage
def read(n):
 with (O/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,s):
 with (O/n).open('x',encoding='utf8') as f:f.write(s)
def js(n,v):write(n,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rows):
 with (O/n).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fmt(v,scale=1,digits=3):return f'{float(v)*scale:.{digits}f}' if v!='' and v is not None else 'NA'
def md(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)
def main():
 started=time.time();assert (O/'EXIT_CODE').read_text().strip()=='0';assert old.load(O/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed'
 seal=old.load(O/'PRE_REFERENCE_SEAL.json');assert old.sha(O/'UNLABELLED_SIGNALS.csv')==seal['signals_sha256'] and old.sha(O/'ALL_RANKINGS.csv')==seal['rankings_sha256']
 # Full coordinates for the already-saved Catalogue new-error index arrays.
 coordinate_files=[]
 for p in sorted(O.glob('*_new_errors.npy')):
  ix=np.load(p,allow_pickle=False);out=p.with_name(p.stem+'_coordinates.csv')
  with out.open('x',encoding='utf-8-sig',newline='') as f:
   w=csv.writer(f);w.writerow(['flat_index','row20','col20']);w.writerows((int(i),int(i//1022),int(i%1022)) for i in ix)
  coordinate_files.append(dict(file=out.name,count=len(ix),sha256=old.sha(out)))
 # List every changed PixBox reference point in the four-state binary error flows.
 plan=old.load(old.D/'EVALUATION_INPUT_PLAN.json')['rows'];pg,_=old.pix_groups([r for r in plan if r['pool']=='PixBox']);refpath=R/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json';refs={q['point_ids'][0]:q for q in old.load(refpath)};pixrows=[]
 points=[{} for _ in range(3)]
 for r in plan:
  if r['pool']!='PixBox':continue
  for arm,d in zip(('R','C122','C123'),points):
   for q in old.pix_old(arm,r['product_id']):d[q['point_id']]=q
 bygroup={}
 for gid,g in pg.items():
  qs=[q for pid,q in points[0].items() if q['group_id']==gid and refs[pid]['reference']!='ignore' and all(d[pid]['valid'] and d[pid].get('coverage',True) and d[pid].get('author_valid',True) for d in points)];assert len(qs)==len(g['truth']);bygroup[gid]=qs
  base=g['scores']['shared_R_00'].astype(np.float64)>=old.T;y=g['truth']==2
  for seed in (12201,12301):
   for state in ('10','01','11'):
    new=g['scores'][f'{seed}_{state}'].astype(np.float64)>=old.T
    for i in np.flatnonzero(base!=new):
     q=qs[i];pid=q['point_id'];pixrows.append(dict(group_id=gid,tile=q['product_id'].split('_T')[-1][:5],seed=seed,state=state,point_id=pid,product_id=q['product_id'],row20=q['row20'],col20=q['col20'],reference=refs[pid]['reference'],old_correct=bool(base[i]==y[i]),new_correct=bool(new[i]==y[i]),score00=float(g['scores']['shared_R_00'][i]),score_new=float(g['scores'][f'{seed}_{state}'][i])))
 table('PIXBOX_CHANGED_POINT_FLOWS.csv',pixrows)
 # Retain all four-state AP/error metrics accepted by Phase203 without recomputing scores.
 for pool in ('Catalogue','PixBox'):
  src=old.O/f'{pool}_PER_GROUP.csv';dst=O/f'{pool}_FOUR_STATE_PER_GROUP.csv';assert not dst.exists();shutil.copyfile(src,dst)
 truth=read('ALL_GROUP_TRUTH.csv');rank=read('ALL_RANKINGS.csv');warning=read('TABLE1_WARNING.csv');focus=read('TABLE2_FOCUS.csv');strata=read('REFERENCE_SCORE_STRATA.csv');checks=0
 for pool,tile in [('Catalogue','35RPM'),('Catalogue','52UDG'),('PixBox','33VVE'),('PixBox','44SKJ')]:
  rr=[r for r in truth if r['pool']==pool and r['tile']==tile];assert len(rr)==2 and all(float(r['delta_AP_pp'])>0 for r in rr);checks+=len(rr)
 for row in focus:
  sub=[r for r in read('FOCUS_SUBTASKS.csv') if all(r[k]==row[k] for k in ('pool','group_id','seed','state'))];allr=next(r for r in sub if r['negative_class']=='All');assert abs(float(allr['AP'])-float(row['AP']))<1e-14;checks+=1
 # Independently verify risk top-quota FPR coverage and the class-flow binary counts.
 for w in warning:
  tt={r['group_id']:r for r in truth if r['pool']==w['pool'] and r['seed']==w['seed']};rr=[r for r in rank if r['pool']==w['pool'] and r['seed']==w['seed'] and r['signal']==w['signal'] and r['selected']=='True'];den=sum(max(0,float(r['delta_FPR'])) for r in tt.values());num=sum(max(0,float(tt[r['group_id']]['delta_FPR'])) for r in rr);assert (w['positive_delta_FPR_coverage']=='')==(den==0)
  if den:assert abs(float(w['positive_delta_FPR_coverage'])-num/den)<1e-14
  checks+=1
 flow=read('ALL_ERROR_FLOWS.csv')
 for key in {(r['group_id'],str(r['seed']),r['state']) for r in pixrows}:
  for ref,cl in [('negative_surface','Surface'),('negative_cloud','Cloud'),('positive_ground_cloud_shadow','Shadow')]:
   rows=[r for r in pixrows if (r['group_id'],str(r['seed']),r['state'])==key and r['reference']==ref];expected=next(r for r in flow if r['pool']=='PixBox' and (r['group_id'],r['seed'],r['state'],r['reference'])==(*key,cl));assert sum(r['old_correct'] and not r['new_correct'] for r in rows)==int(expected['correct_to_wrong']);assert sum(not r['old_correct'] and r['new_correct'] for r in rows)==int(expected['wrong_to_correct']);checks+=2
 table1=md(['池','seed','信号','复核/全部','AP损伤总数','检出/漏检','入选未损伤/AP未知','随机期望','四固定组检出','正ΔFPR覆盖'],[[r['pool'],r['seed'],r['signal'],r['reviewed']+'/'+r['total'],r['AP_damage_total'],r['detected']+'/'+r['missed'],r['selected_known_not_damaged']+'/'+r['selected_AP_unknown'],fmt(r['random_expected']),r['detected_fixed'] or '无',fmt(r['positive_delta_FPR_coverage'],100)+'%' if r['positive_delta_FPR_coverage'] else 'NA（分母0）'] for r in warning])
 table2rows=[]
 for r in focus:
  ss=[s for s in strata if all(s[k]==r[k] for k in ('pool','group_id','seed','state')) and s['reference']=='Shadow'];mn=min(ss,key=lambda s:float(s['delta_median']));layer=f"00影分数{fmt(mn['score00_min'])}–{fmt(mn['score00_max'])}层中位变化{fmt(mn['delta_median'])}（n={mn['count']}）"
  table2rows.append([r['pool']+'/'+r['tile'],r['seed'],r['state'],fmt(r['AP00'],100)+' / '+fmt(r['AP'],100),fmt(r['Surface_AUC_delta'],100),fmt(r['Cloud_AUC_delta'],100),r['new_FN']+'/'+r['rescued_TP'],r['new_FP']+'/'+r['removed_FP'],layer])
 table2=md(['池/组','seed','状态','原AP/当前AP (%)','Surface AUC差(pp)','Cloud AUC差(pp)','新增FN/救回TP','新增FP/消除FP','影分数下降最多的固定层'],table2rows)
 write('TABLE1_WARNING.md','# 表1：固定信号预警筛查\n\n'+table1+'\n每池分别排序，Catalogue固定7/32，PixBox固定2/10。AP未知不是未损伤。NA未填0。固定四组为32TML/40XDR/19FDV/54TWN；每行仅评价本池两组。\n')
 write('TABLE2_FOCUS.md','# 表2：重点组及固定改善参照\n\n'+table2+'\n00=原R；10=适配W＋原BN状态；01=原W＋适配BN状态；11=完整适配。分层最负中位变化仅是对预先固定所有十分位层的描述，不用于预警或挑选点。完整层及各参考类见REFERENCE_SCORE_STRATA.csv；子任务AP/AUC不可解释为原AP的贡献比例。\n')
 intro='''# Phase204：损伤定位与无类别标签预警筛查

## 1 数据结果

仅复用Phase123/124及203缓存：Catalogue 32相关组（7组有AP支持）、PixBox 16产品/10相关组（8组有AP支持）。V1 RegNet原R与C100 seed12201/12301分别评价；无E/V4。无新增训练、拟合、前向、下载、阈值搜索或bootstrap。原固定阈值0.7693686485290528。

无类别标签端先保存UNLABELLED_SIGNALS.csv及ALL_RANKINGS.csv并封存SHA，之后才连接本地参考。Catalogue仍依赖既定参考有效格网；PixBox信号含9500个共同image-valid位置，包括931个ignore；评价才排ignore，保留379正/8190负。两个seed共用支持，不视为新增独立组。当前池已暴露，封存实现步骤不恢复盲法或独立性。

### 固定预警结果

'''
 details='''
RANK主信号两seed均只检出40XDR。32TML排名15/16（32组中），19FDV均第6、54TWN均第9（10组中），均在复核名额外。Catalogue的RANK前7有6个AP未知组，不能当作6个安全误报或阴性；PixBox前2包含一个AP未知和一个已知未损伤组。seed12201还漏32SPJ、22WES；seed12301的这两组不满足−1pp损伤定义。

CHANGE在Catalogue优先选中改善参照35RPM，在PixBox两seed前2均为改善参照33VVE、44SKJ，未检出任何AP损伤。ENTROPY也未检出AP损伤。全部信号无NA。RANK的两seed排名相关为Catalogue0.971、PixBox0.976；稳定排名不等于稳定识别损伤。

条件于AP可判子集的RANK AUROC为Catalogue0.900（2/7受损）、PixBox0.313/0.333（4/8、2/8受损）。AUPRC为0.833、0.482/0.268。仅这7/8组可判AP，不能将这些条件指标泛化到全部场景。其余两信号的完整条件AUROC/AUPRC、基率和数量见TABLE1_WARNING.csv。

Catalogue两seed及PixBox seed12301没有正ΔFPR，覆盖率分母为0，保留NA；PixBox seed12201的正ΔFPR覆盖为RANK0%、CHANGE0%、ENTROPY30.012%。全部逐组连续ΔFPR见ALL_GROUP_TRUTH.csv。这是组均误报增量覆盖，不是像元损失或安全概率。

### 重点组定位

'''
 focus11=[r for r in focus if r['tile'] in ('32TML','19FDV') and r['state']=='11']
 compact=md(['组','seed','ΔAP(pp)','ΔAUC Surface/Cloud(pp)','新增FN/救回TP','新增FP/消除FP'],[[r['tile'],r['seed'],fmt(float(r['AP'])-float(r['AP00']),100),fmt(r['Surface_AUC_delta'],100)+' / '+fmt(r['Cloud_AUC_delta'],100),r['new_FN']+'/'+r['rescued_TP'],r['new_FP']+'/'+r['removed_FP']] for r in focus11])
 conclusion='''
32TML参考云影110205格。11在00影分数约0.487–0.571层下降最多，中位分数变化−0.373/−0.268；高分层亦下降。10的Surface子任务AUC下降−9.174/−12.269pp，而Cloud AUC上升0.676/0.616pp。11的Surface AUC下降−1.860/−2.867pp，Cloud AUC为−0.670/+0.080pp。完整11净增加漏影12270/14325格，同时消除的FP多于新增FP。

19FDV参考云影仅50点，每个固定十分位层只有5点。11在00最高分约0.944–0.999层下降最多，中位变化−0.540/−0.494；00分数约0.794–0.811的5点层反而上升0.162/0.008。10该最高分层下降−0.706/−0.696。11两seed均新增13个FN、救回2个TP，净漏影增加11点；Surface和Cloud AUC均下降，但Cloud下降幅度分别4.092/1.123pp，跨seed不宜归结为同一单一负类。

四个固定改善参照35RPM、52UDG、33VVE、44SKJ的11相对00的AP在两seed均为正，支持/身份复现，无替换参照。其中35RPM、44SKJ虽AP改善仍净丢失固定阈值TP，显示排序与决策损失必须分开。

完整00/10/01/11表见TABLE2_FOCUS.md；所有组AP及四流见两个FOUR_STATE_PER_GROUP.csv和ALL_ERROR_FLOWS.csv；全部参考内固定分层见REFERENCE_SCORE_STRATA.csv。Catalogue新增错误完整flat_indices及20m坐标已保存（1022列原格网）；每类最多12个等序号间隔位置仅供定位，不做地物裁决。PixBox提供所有改变二元决策的参考点坐标/分数/四流，不推断连通区或面积。

## 2 结论分析

32TML更一致地体现为云影与Surface参考的排序受损，仅W交换仍保持并加重这一模式；19FDV同样存在W相关损伤及高分云影下移，但完整11的两负类排序均受影响，且只有50个正点。这里的Surface/Cloud是参考类别，不是暗水体、植被或标签错误等物理成因证据。状态交换也不能直接证明训练时冻结BN会改善。

三个固定信号未提供可用的通用损伤预警：主RANK漏掉32TML、19FDV、54TWN；两个对照在固定名额下没有检出任何AP损伤，且CHANGE把明确改善组排在前面。RANK对Catalogue可判子集有区分信息，但其全池固定名额结果及跨池失败不足以支持损伤预警成功。不能事后选ENTROPY条件AUROC代替主要信号或放宽复核名额。

结束本轮三个信号配方；当前证据不足以直接投入风险学习器、自动回退或声称已泛化。若未来另行授权预测研究，仍须独立场景与预先确定的支持和终点；本轮不增加特征、拟合或新phase。两seed分别报告，7/8个AP组的小样本限制与历史暴露不变。
'''
 write('RESULT.md',intro+table1+details+compact+conclusion)
 short='''# Phase204 交付摘要

## 1 数据结果

纯本地缓存分析，Catalogue32组/PixBox10相关组，两seed分别列；零训练/拟合/前向/下载/阈值搜索/新bootstrap。信号和排名在参考连接前封存，但历史池已暴露。

| 固定前20%筛查 | Catalogue seed12201/12301 | PixBox seed12201/12301 |
|---|---|---|
| RANK检出/全部AP损伤 | 1/2、1/2 | 0/4、0/2 |
| CHANGE检出/全部AP损伤 | 0/2、0/2 | 0/4、0/2 |
| ENTROPY检出/全部AP损伤 | 0/2、0/2 | 0/4、0/2 |

主RANK两seed只检出40XDR，漏32TML、19FDV、54TWN；seed12201还漏32SPJ、22WES。Catalogue前7中6组AP未知；PixBox前2中1组AP未知。无影组不能当安全阴性。CHANGE把改善的35RPM或33VVE/44SKJ排在最前。全部排名/随机期望/条件AUROC与AUPRC/误报增量覆盖见表1。

'''+compact+'''
两重点组都增加漏影。32TML的11损伤在Surface排序更一致，00中等分云影层下降最多；19FDV仅50正点，11对Surface/Cloud排序均下降，最高分云影层下降最多。10保留W相关损伤，但19FDV的Cloud方向不完全一致。四固定改善参照AP方向复现。

## 2 结论分析

本轮未找到可用的全池损伤预警；稳定分数变化/排名不能当真实损伤。主RANK在Catalogue的7个可判AP组中有区分信息，但固定名额漏检与PixBox失败阻止泛化主张。结束当前三个固定信号配方，不增加特征或自动训练风险模型/回退。

Catalogue支持仍由既定参考有效格网限定；PixBox信号用9500点含ignore，参考评价用379正/8190负。历史暴露未消除，AP仅7/8组；分层不是物理成因，稀疏点不是面积。没有独立判读就不能归因暗地表或标签错误。尚不足以直接推进预测方法投入；未来独立研究需要另行授权。

完整数据与解释见outputs/phase204/phase204_20261019/RESULT.md；表1、表2、全排名/四流/分层/位置及输入代码锁和独立验收均已交付。无SSH、无work_dirs同步、无自动push或后续phase。
'''
 write('README.md',short)
 js('MISSING_AND_LIMITS.json',dict(required_cache_missing=[],optional_image_maps='not generated; not required',reference_support='Catalogue inherited reference-valid grid; PixBox risk9500 incl931ignore, evaluation379positive/8190negative',AP_groups={'Catalogue':7,'PixBox':8},historically_exposed=True,independent_validation=False,physical_cause_not_identified=True,no_deployment_guarantee=True,threshold=old.T,NA='AP without both classes; FPR increase coverage with zero denominator',no_extra_models_features_or_thresholds=True))
 js('FINAL_ACCEPTANCE.json',dict(status='passed',additional_checks=checks,reference_comparator_directions_reproduced=True,all_changed_PixBox_decisions_listed=len(pixrows),Catalogue_coordinate_files=coordinate_files,signals_sha256=old.sha(O/'UNLABELLED_SIGNALS.csv'),rankings_sha256=old.sha(O/'ALL_RANKINGS.csv'),reference_file_sha256=old.sha(refpath),code_sha256={str(p.relative_to(R)):old.sha(p) for p in (R/'src/tools').glob('*phase204*.py')},seconds=time.time()-started))
 runtime=old.load(O/'RUN_STATUS.json');cost=old.load(O/'COMPUTE_RECEIPT.json');js('COST_RECEIPT.json',dict(cache_compute_seconds=cost['seconds'],bounded_analysis_and_independent_audit_seconds=runtime['seconds'],final_delivery_seconds=time.time()-started,peak_analysis_bytes=runtime['peak_bytes'],finalizer_peak_bytes=usage(os.getpid()),finalizer_bound='only small PixBox reference/score tables and <=149k-element Catalogue index array at a time',new_bytes_before_completion_manifest=sum(p.stat().st_size for p in O.rglob('*') if p.is_file()),neural_calls=0,fits=0,downloads=0,SSH=0,bootstrap=0,CPU_budget_seconds=7200,memory_budget_bytes=16*1024**3,new_file_budget_bytes=4*1024**3))
 S.mkdir(parents=True,exist_ok=True)
 for n in ('README.md','RESULT.md','TABLE1_WARNING.md','TABLE2_FOCUS.md','TABLE1_WARNING.csv','TABLE2_FOCUS.csv','ALL_RANKINGS.csv','ALL_GROUP_TRUTH.csv','SEED_RANK_CONSISTENCY.csv','MISSING_AND_LIMITS.json','INDEPENDENT_ACCEPTANCE.json','FINAL_ACCEPTANCE.json','COST_RECEIPT.json'):
  dst=S/n;assert not dst.exists();shutil.copyfile(O/n,dst)
 # Update only this phase's current handoff/status; retain its initial handoff verbatim.
 shutil.copyfile(O/'HANDOFF.md',O/'HANDOFF_INITIAL.md')
 (O/'HANDOFF.md').write_text('''# Phase204 COMPLETE

Current delivery: completed. Analysis EXIT_CODE0;1646 independent checks plus FINAL_ACCEPTANCE.json passed. All required outputs and summary exist. No new model calls, fits, downloads, SSH, bootstrap or work_dirs transfer. Signals/rankings SHA unchanged. Read RESULT.md and README.md; data and conclusions are separate. Delete phase204-hourly-delivery monitor; do not restart analysis or start another phase.

Original handoff retained as HANDOFF_INITIAL.md.
''',encoding='utf8')
 runtime.update(status='completed',delivery_complete=True);(O/'RUN_STATUS.json').write_text(json.dumps(runtime,indent=2)+'\n')
 files={p.name:old.sha(p) for p in O.iterdir() if p.is_file()};js('DELIVERY_COMPLETE.json',dict(status='complete',files=files,summary=str(S),neural_calls=0,fit=0,automatic_push=False));assert sum(p.stat().st_size for p in O.rglob('*') if p.is_file())<4*1024**3
 print(json.dumps(dict(status='complete',checks=checks,new_bytes=sum(p.stat().st_size for p in O.rglob('*') if p.is_file()))))
if __name__=='__main__':main()

