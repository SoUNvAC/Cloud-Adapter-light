"""Finalize accepted Phase206 tables and separate measured results from interpretation."""
import ast,csv,hashlib,json,shutil
from pathlib import Path
R=Path('D:/Cloud-Adapter-light');O=R/'outputs/phase206/phase206_20261019';D=O/'delivery';S=R/'src/research_results/phase206_20261019'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def put(n,text):
 with (O/n).open('x',encoding='utf8') as f:f.write(text+'\n')
def save(n,value):put(n,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def fmt(v):
 if v is None or v=='':return 'NA'
 if isinstance(v,float):return f'{v:.3f}'
 return str(v).replace('|','/')
def md(cols,data):return '| '+' | '.join(cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(c)) for c in cols)+' |' for r in data)
def main():
 assert load(O/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed';assert not (O/'DELIVERY_COMPLETE.json').exists();tables={i:[r for pool in ('Catalogue','PixBox') for r in rows(pool+f'_TABLE{i}.csv')] for i in (1,2,3)}
 extra_checks=0
 for pool in ('Catalogue','PixBox'):
  summary={r['strategy']:r for r in load(O/(pool+'_ALL_METRICS.json'))}
  for row in rows(pool+'_TABLE1.csv'):
   base=summary[row['strategy']]
   for k in ('macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_negative_shadow_fraction_pp','max_group_negative_shadow_fraction_pp'):
    assert abs(float(row[k])-base[k])<1e-10;extra_checks+=1
  for row in rows(pool+'_TABLE3.csv'):
   seed=int(row['seed']);state=row['member_state'];name='M_'+str(seed)+'_10' if state=='10' else 'C4-'+str(seed);a,b=summary[name],summary['E4']
   for k in ('macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_negative_shadow_fraction_pp'):
    assert abs(float(row['delta_'+k])-(a[k]-b[k]))<1e-10;extra_checks+=1
   rr=rows(pool+'_MEMBER_'+state+'_VS_E4_GROUPS.csv');assert int(row['both_seed_damaged_groups'])==sum(q['both_damaged']=='True' for q in rr);extra_checks+=1
 supplement=dict(status='passed',independent_checks=extra_checks,table1_recomputed_from_full_metrics=True,table3_deltas_and_repeated_counts_recomputed=True)
 if (O/'TABLE1_TABLE3_ACCEPTANCE.json').exists():assert load(O/'TABLE1_TABLE3_ACCEPTANCE.json')==supplement
 else:save('TABLE1_TABLE3_ACCEPTANCE.json',supplement)
 for i in (1,2,3):
  keys={1:['pool','seed','family','state','macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_negative_shadow_fraction_pp','max_group_negative_shadow_fraction_pp','Catalogue_pooled_mIoU_pp'],2:['pool','group_id','tile','positive','seed','family','AP00_pp','AP10_pp','AP01_pp','AP11_pp','10-11_pp','01-00_pp','10-00_pp','I_pp'],3:['pool','seed','member_state','delta_macro_AP_pp','delta_pooled_AP_pp','delta_macro_recall_pp','delta_macro_negative_shadow_fraction_pp','both_seed_damaged_groups']}[i]
  data=[{k:(float(v) if k in keys and k not in ('pool','seed','family','state','group_id','tile','member_state','positive','both_seed_damaged_groups') and v!='' else v) for k,v in row.items()} for row in tables[i]]
  put(f'TABLE{i}.md',f'# 表{i}\n\nAP、召回、误报比例以百分数呈现，差值为百分点。NA保留；PixBox误报只指负参考点云影比例，非全景FPR。\n\n'+md(keys,data))
 intervals=[r for pool in ('Catalogue','PixBox') for r in load(O/(pool+'_INTERVALS.json'))['rows']];icols=['pool','family','seed','contrast','mean_delta_AP_pp','CI95_pp','n_groups','improved','stable','damaged','worst_delta_AP_pp'];put('PAIRED_GROUP_INTERVALS.md','# 固定比较与描述性区间\n\n每相关组等权，bootstrap10000、seed206019；两个seed分别报告，不扩大独立组数。\n\n'+md(icols,intervals))
 counts=load(O/'COHORT_REPEATED_COUNTS.json');put('REPEATED_CHANGES.md','# 两seed共同变化\n\n改善>1pp、下降<−1pp、稳定在±1pp内。只算有正参考的组；其余组继续进入负参考误报评价。\n\n'+md(list(counts[0]),counts))
 old=[r for r in tables[2] if any(t in r['tile'].split(',') for t in ('32TML','40XDR','19FDV','54TWN'))];put('OLD_FOUR_GROUPS.md','# 固定旧四组跟踪\n\n54TWN只有7个正参考。未用这些组选择状态或阈值。\n\n'+md(['pool','tile','positive','seed','family','AP00_pp','AP10_pp','AP01_pp','AP11_pp','10-11_pp','I_pp'],old))
 flows=[];no_shadow=[];undefined=[]
 for pool in ('Catalogue','PixBox'):
  for r in rows(pool+'_PER_GROUP.csv'):
   mat=ast.literal_eval(r['confusion']);item=dict(pool=pool,strategy=r['strategy'],group_id=r['group_id'],tile=r['tile'],positive=int(r['positive']),negative=int(r['negative']),confusion=mat,Shadow_to_Surface=mat[2][0],Shadow_to_Cloud=mat[2][1],Surface_to_Shadow=mat[0][2],Cloud_to_Shadow=mat[1][2],precision=float(r['precision']) if r['precision'] else None,recall=float(r['recall']) if r['recall'] else None,negative_shadow_fraction=float(r['negative_shadow_fraction']),AP=float(r['AP']) if r['AP'] else None);flows.append(item)
   if not item['positive']:no_shadow.append(item)
   if item['precision'] is None:undefined.append(dict(pool=pool,strategy=r['strategy'],group_id=r['group_id'],reason='zero predictedShadow; precision undefined, AP still computed if positive/negative references exist'))
 save('ALL_CONFUSION_ERROR_FLOWS.json',dict(rows=flows,classes=['Surface','Cloud','Shadow'],row='reference',column='prediction',PixBox='point support only, notsceneIoU/FPR'))
 save('NO_REFERENCE_SHADOW_GROUPS.json',no_shadow);save('UNDEFINED_PRECISION.json',undefined)
 limits=dict(inference_completed=True,all_14_common_support=True,Catalogue_groups=32,Catalogue_AP_groups=7,Catalogue_positive=655961,Catalogue_negative=32767527,PixBox_products=16,PixBox_related_groups=10,PixBox_AP_groups=8,PixBox_positive=379,PixBox_negative=8190,PixBox_ignore=931,old_54TWN_positive=7,author_full_training_overlap='unknown; reusedhistoricallyexposedpools,notindependentconfirmation',model_change='stateexchangeonly,notnewtraining; cannotinferfreezingBNduringtrainingcausally',AP_nonadditive='interactionisfunctionalmetriccontrast,notpercentcausaldamage',deployment='no modelselected, no thresholdchanged, no claimeddeploymentlatencysavings',unexplained=['physicalsurfacecause','labelsemantics','normalizationcausality','otherlearningrates/lengths','MsREbenefit'],undefined_precision='NAwhennoShadowpredictions; neither0norNaN',CPU_recovery='firstPixBoxJSONfailedonundefinedprecision; archivedfailure; fixNAonly; completedCataloguepreservedSHA;0newforwards; initialCPUcostretained',old205_failedgate_memory='unknown, notfabricated')
 save('MISSING_AND_LIMITS.json',limits)
 stats={pool:{r['strategy']:r for r in load(O/(pool+'_ALL_METRICS.json'))} for pool in ('Catalogue','PixBox')}
 ap_table=[]
 for pool in stats:
  for family in ('member','ensemble'):
   for seed in (12201,12301):
    ns={'00':'R4' if family=='member' else 'E4','11':('C4-' if family=='member' else 'E4-')+str(seed),'10':('M_' if family=='member' else 'E_')+str(seed)+'_10','01':('M_' if family=='member' else 'E_')+str(seed)+'_01'}
    ap_table.append(dict(pool=pool,family=family,seed=seed,AP00=stats[pool][ns['00']]['macro_AP_pp'],AP10=stats[pool][ns['10']]['macro_AP_pp'],AP01=stats[pool][ns['01']]['macro_AP_pp'],AP11=stats[pool][ns['11']]['macro_AP_pp']))
 data='''# Phase206 数据结果

W=decoder/head全部32个可学习参数张量（含BN affine）；B=10个decoder BN的30个运行状态buffer。00原模型，10适配W+原B，01原W+适配B，11完整适配。成员和完整集成分别有自己的00。全程eval/no_grad，N4每patch一次共用，按raw四类平均再官方拼接/后处理；决策四类argmax后合并Cloud，不扫阈值。

数据沿用Catalogue32相关组、PixBox16产品/10相关组；分别7/8组有云影可计算AP。原六策略全42组先在本地复现；32TML/19FDV共12项成员/集成重建抽查通过。输入与全部支持保持原205。

逐组等权宏AP（%）：

'''+md(['pool','family','seed','AP00','AP10','AP01','AP11'],ap_table)+'''

运行决策：成员10在Catalogue两seed宏云影召回仅0.016741%/0.011408%，池化三类mIoU15.1623%/15.9999%；原R4为90.5845%/82.1387%。PixBox成员10两seed的8个有影组全部零召回，负参考点云影误报亦为0，精确率为NA。原R4宏召回67.9187%。因此必须同时查看AP、实际召回和错误流。

集成10在Catalogue宏AP88.9042%/88.9790%，低于原E4的93.2619%；PixBox为80.9118%/82.7445%，原E4为81.0462%，但其宏召回62.2445%/64.0913%仍低于原74.1519%，池化AP83.7590%/86.4384%也低于原89.1595%。

完整交付：TABLE1.md含32行四状态汇总（池化AP、宏召回、宏/最大组负参考误报和Catalogue mIoU）；TABLE2.md含168行全组机制对照；TABLE3.md为8行单成员相对原完整E4。所有区间见PAIRED_GROUP_INTERVALS.md，两seed共同变化见REPEATED_CHANGES.md，固定旧四组见OLD_FOUR_GROUPS.md。全部逐组混淆/流向在ALL_CONFUSION_ERROR_FLOWS.json，原始csv保留未舍入值。

成本：6002累计网络调用（重建272+主5730），0训练/0拟合；GPU执行阶段1445.074684秒。独立6689项验收通过。CPU首次汇总遇到无预测云影时精确率未定义的JSON错误，已保留失败记录，改为NA，仅补CPU未完成部分，未追加任何神经调用。累计CPU333.750229秒。
'''
 put('DATA_RESULTS.md',data)
 analysis='''# Phase206 结论分析

1. **BN运行状态参与了功能退化，但恢复旧BN不是稳定修复。** 01−00在两池、两seed的成员宏AP均下降，描述性95%区间均低于0；共同下降为Catalogue5/7、PixBox8/8。完整集成01也下降。说明运行状态置换能改变排序，但它与参数训练时共同适应，不能据此推算BN的因果损伤占比。

2. **适配参数在原BN下仍受损。** 成员10−00在Catalogue两seed为−14.81/−18.91pp，PixBox为−20.18/−15.34pp；描述性区间均低于0。Catalogue7/7、PixBox6/8组两seed共同下降。更严重的是成员10的云影argmax决策近乎或完全消失；误报归零伴随漏影，不能称修复。10相对11的成员宏AP改善方向还在PixBox两seed间翻转。

3. **W和B存在明显的状态失配与非线性交互，强度随seed和集成变化。** PixBox seed12201交互均值：成员+23.87pp、集成+13.55pp，描述性区间均高于0；seed12301区间跨0。Catalogue也不统一。完整集成10相对11的AP点估计两池两seed均提高，但部分区间跨0；只有Catalogue12301、PixBox12301的该差值区间完全高于0。不能忽略召回和池化排序的代价，也不能把这些历史池上的区间当独立确认。

4. **单成员10/11没有在这些固定结果中形成相对原E4的替代依据。** 两池两seed的单成员宏AP和实际宏召回均低于完整原E4。PixBox集成10的宏AP接近原E4，不等于单模型接近，也不抵消其召回/池化AP损失。本轮没有配对部署测速，不报告速度收益，不挑最好状态部署。

这轮支持“参数变化与BN运行状态共同相关”的混合解释，不支持把广泛退化单独归给BN，也不证明训练中冻结BN必然改善。物理地物、标签语义、归一化机制及其他训练设置仍unknown。54TWN仅7个正参考，AP独立组只有7/8组；作者V4完整训练覆盖unknown。没有新增模型、特征、阈值或后续phase。
'''
 put('CONCLUSIONS.md',analysis);put('RESULT.md','# Phase206\n\n## 1．数据结果\n\n'+data.replace('# Phase206 数据结果\n','',1)+'\n## 2．结论分析\n\n'+analysis.replace('# Phase206 结论分析\n','',1))
 readme='''# Phase206 已完成

只做V4参数/BN状态交换，0训练、0拟合、0阈值搜索。数据与结论分开见DATA_RESULTS.md、CONCLUSIONS.md，完整表1/2/3及逐组csv均保留。

**BN有关吗？** 只换适配BN（01）会降低两池两seed成员/集成宏AP，但恢复原BN（10）不是稳定修复；成员实际云影召回崩溃。

**参数还受损吗？** 原BN下，成员10相对原R4在Catalogue7/7、PixBox6/8组两seed共同AP下降。原W+适配B和适配W+原B都不能消除损伤。

**是否交互？** 存在seed/池/集成依赖的非线性交互。例如PixBox12201成员I=+23.87pp、集成+13.55pp；另一个seed区间跨0。不能拆成BN的因果损伤比例，也不证明训练时冻结BN更好。

**能用单模型替E4吗？** 单成员10/11在两池两seed的宏AP和实际召回均不及原完整E4。成员10在PixBox全部8个有影组零召回。完整集成10的部分AP恢复伴随召回/池化AP代价；本轮不选部署模型、不宣称测速收益。

6002调用、GPU阶段1445.07秒，独立6689项验收通过。CPU精确率NA序列化错误已修复并保留失败，未增神经调用；Catalogue首次完成结果未改。参考仅本地；没有work_dirs同步、历史清理、自动push或下一phase。54TWN7正参考、历史暴露与作者完整训练覆盖unknown等限制不变。
'''
 assert len(readme)<2200;put('README.md',readme)
 inputs={'protocol_sha256':sha(R/'src/research_plans/PHASE206_V4_PARAMETER_BN_STATE_CHECK_20261019.md'),'archive_sha256':sha(O/'phase206_delivery.zip'),'old205_completion_sha256':sha(R/'outputs/phase205/phase205_20261019/DELIVERY_COMPLETE.json'),'code':{p.name:sha(p) for p in (R/'src/tools').glob('*phase206*') if p.is_file()},'model_state_sources':'delivery/infer/STATE_<seed>_<10|01>.json','actual_weight_audit':load(D/'ACTUAL_WEIGHT_ACCEPTANCE.json'),'reference':'local Catalogue mask members/SHA in input plan; PixBox fixed referenceb1448f3fbd05c9e77af7ed0ea50dbde4be7480a35a4c532a2fec45789af25f96'}
 save('FINAL_PROVENANCE.json',inputs)
 S.mkdir(exist_ok=False)
 selected=[p for p in O.iterdir() if p.is_file() and p.suffix in ('.csv','.md','.json') and (p.name.startswith(('Catalogue_','PixBox_')) or p.name in ['RESULT.md','DATA_RESULTS.md','CONCLUSIONS.md','README.md','TABLE1.md','TABLE2.md','TABLE3.md','PAIRED_GROUP_INTERVALS.md','REPEATED_CHANGES.md','OLD_FOUR_GROUPS.md','COHORT_REPEATED_COUNTS.json','INDEPENDENT_ACCEPTANCE.json','CPU_ACCEPTANCE.json','COST_RECEIPT.json','MISSING_AND_LIMITS.json','UNDEFINED_PRECISION.json','NO_REFERENCE_SHADOW_GROUPS.json','ALL_CONFUSION_ERROR_FLOWS.json','FINAL_PROVENANCE.json'])]
 for p in selected:shutil.copyfile(p,S/p.name)
 handoff=O/'HANDOFF.md';text=handoff.read_text(encoding='utf8');shutil.copyfile(handoff,O/'HANDOFF_BEFORE_COMPLETE.md');a=text.index('Current delivery:');b=text.index('\n',a);text=text[:a]+'Current delivery: COMPLETE. Remote6002calls/1445.0746843270317s, allscores sealed; localCPU recovery completed withNA precision andoriginal266.36s preserved (cumulative333.750229s). All Catalogue first-pass bytes unchanged. Independent6689checks passed. FullTable1/2/3, pairedgroupintervals,errorflows,stateidentity,cost/limits,data-vs-analysis andsummary delivered. No more model/CPU analysis/receiver runs. Deletephase206-hourly, notifyonce. No207 orotherphase.'+text[b:];handoff.write_text(text,encoding='utf8')
 actual=load(O/'DELIVERY_RECEIPT.json')['remote_new_bytes']+sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+sum(p.stat().st_size for p in S.rglob('*') if p.is_file());assert actual<6000000000
 save('COMBINED_STORAGE_RECEIPT.json',dict(remote_bytes=load(O/'DELIVERY_RECEIPT.json')['remote_new_bytes'],actual_combined_bytes_before_final_manifest=actual,manifest_allowance_bytes=1000000,conservative_final_upper_bytes=actual+1000000,cap=6000000000))
 manifest=dict(status='complete',root_files={p.relative_to(O).as_posix():sha(p) for p in O.iterdir() if p.is_file() and p.name!='DELIVERY_COMPLETE.json'},summary_files={p.relative_to(S).as_posix():sha(p) for p in S.iterdir() if p.is_file()},neural_calls=6002,new_fits=0,independent_acceptance_sha256=sha(O/'INDEPENDENT_ACCEPTANCE.json'),data_conclusions_separate=True)
 save('DELIVERY_COMPLETE.json',manifest)
 for n,h in manifest['root_files'].items():assert sha(O/n)==h
 for n,h in manifest['summary_files'].items():assert sha(S/n)==h
 print(json.dumps(dict(status='complete',root_files=len(manifest['root_files']),summary_files=len(manifest['summary_files']),combined_bytes=actual,README_characters=len(readme))))
if __name__=='__main__':main()
