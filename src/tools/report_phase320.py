"""Summarize actual two-part diagnostics without upgrading label semantics or oracle scores."""
from phase320_common import *
from collections import defaultdict
def f(x):return 'NA' if x is None else f'{float(x):.4f}'
def main():
 guard();assert load(ROOT/'INDEPENDENT_QA.json')['status']=='passed';ea=load(ROOT/'ERROR_ACCEPTANCE.json');ta=load(ROOT/'THRESHOLD_ACCEPTANCE.json');composition=read(ROOT/'ERROR_COMPOSITION.csv') if (ROOT/'ERROR_COMPOSITION.csv').exists() else [];summary=read(ROOT/'ERROR_SCENE_SUMMARY.csv') if (ROOT/'ERROR_SCENE_SUMMARY.csv').exists() else [];matches=read(ROOT/'THRESHOLD_MATCH.csv') if (ROOT/'THRESHOLD_MATCH.csv').exists() else [];ranking=read(ROOT/'RANKING_SUMMARY.csv') if (ROOT/'RANKING_SUMMARY.csv').exists() else []
 text=['# Phase320：错误构成与阈值取舍','', '本轮只读既有缓存，gzs CPU执行，零训练、拟合、神经前向和下载。以下为已暴露开发数据上的事后诊断；GT类别只能说明参考标成什么，不能证明错标原因或标注错误。阈值点均为使用val真值的 **oracle_diagnostic**，不能作为部署泛化成绩。','', '## 错Shadow实际对应什么',''];opportunity=['# Phase320 错误机会盘点','', '原生参考能证实类别，不能直接提供GT外的识别信号。下面的“错误”指相对现有参考的Shadow伪标签错误，不指参考标注错误。',''];outcomes={};lead=[];thresholdlead=[]
 for pool in ['SPARCS','L8']:
  if ea['pools'].get(pool,{}).get('status')!='passed':text.append(pool+'：not_testable，见输入缺项。');continue
  a=ea['pools'][pool];poolrows=[r for r in composition if r['pool']==pool and r['level']=='pool'];errors=sorted([r for r in poolrows if int(r['wrong_Shadow'])>0],key=lambda r:-int(r['wrong_Shadow']));scenes=[r for r in summary if r['pool']==pool and r['level']=='scene'];tot=next(r for r in summary if r['pool']==pool and r['level']=='pool');parts=[]
  for r in errors:
   n=sum(int(c['wrong_Shadow'])>0 for c in composition if c['pool']==pool and c['level']=='scene' and c['native_class']==r['native_class']);parts.append(f"{r['native_class']} {r['wrong_Shadow']}（错误中{f(r['share_of_all_errors_pct'])}%，{n}scene）")
  text.append(f"{pool}：原A_Sh {a['A_Sh']}，有效正确 {a['T']}、错误 {a['E']}、无效参考 {a['invalid']}，precision {f(a['precision_pct'])}%。错误构成："+'；'.join(parts)+'。')
  text.append(f"参考边界2像元范围内E {tot['near_either_boundary_E']}，远离两边界E {tot['far_both_boundaries_E']}，边界上下文未知E {tot['boundary_context_unknown_E']}；三者互斥回加错误总数。几何状态沿319原件，完整池包含input_not_qualified。")
  opportunity.append(f"{pool}："+'；'.join(parts)+f"。未细分Surface错误 {tot['unresolved_Surface_errors']}；这部分只能称land/clear，不能拆成暗地表、地形阴影、水体或标签画错。边界未知E {tot['boundary_context_unknown_E']}；近边界不能称标注偏差。")
  if errors:
   label=errors[0]['native_class'];affected=sorted([c for c in composition if c['pool']==pool and c['level']=='scene' and c['native_class']==label and int(c['wrong_Shadow'])>0],key=lambda c:-int(c['wrong_Shadow']));opportunity.append('最多错误类别 '+label+' 的具体受影响scene（描述示例，完整清单保留）：'+'；'.join(c['unit']+' E='+c['wrong_Shadow'] for c in affected[:3])+'。待验证条件：能否仅靠既有影像或独立固定辅助输入识别这类错误，同时保留同区域正确Shadow；本轮只有GT类别证据，尚未验证该条件。')
  lead.append(pool+'主要错误为'+'、'.join(r['native_class']+' '+r['wrong_Shadow'] for r in errors[:2])+f"；无法细分Surface原因 {tot['unresolved_Surface_errors']}像元")
  outcomes[pool]=dict(error_composition_status='passed',E=a['E'],T=a['T'],invalid=a['invalid'])
 text+=['','L8仅192可称thin cloud；SPARCS的cloud没有薄厚细分。SPARCS shadow_over_water是正确Shadow的一种，不并入water错误。water、Surface子类与“无法细分Surface”描述存在包含关系，不能把场景表列相加成新总数。2像元是各自评估网格的距离，不合并为米制距离。','', '## PL调一个阈值能否找回CB召回','', '|池|CB原macro召回% / FPR%|PL同误报预算最佳召回%|实际FPR%|召回差pp|PL达到CB召回时最低FPR%|是否同预算复现CB召回|','|---|---:|---:|---:|---:|---:|---|']
 for pool in POOLS:
  if ta['pools'].get(pool,{}).get('status')!='passed':text.append(f'|{pool}|NA|NA|NA|NA|NA|not_testable|');outcomes.setdefault(pool,{})['threshold_status']='not_testable';continue
  a=next(r for r in matches if r['pool']==pool and r['arm']=='PL' and r['target']=='CB_original_FPR' and r['point_kind']=='selected');b=next(r for r in matches if r['pool']==pool and r['arm']=='PL' and r['target']=='CB_original_recall' and r['point_kind']=='selected');target=lock()['threshold'][pool];yes=next(c for c in ta['checks'] if c['pool']==pool and c['arm']=='PL')['exact_fraction_selected_budgets']['CB_original_FPR']['matches_or_exceeds_CB_at_no_greater_FPR'];text.append(f"|{pool}|{f(target['CB_original_macro_recall_pct'])} / {f(target['CB_original_macro_FPR_pct'])}|{f(a['macro_recall_pct'])}|{f(a['macro_FPR_pct'])}|{f(a['recall_minus_CB_original_pp'])}|{f(b['macro_FPR_pct'])}|{'可以（仅本池oracle二分类取舍）' if yes else '不够（单个全局阈值）'}|");outcomes.setdefault(pool,{})['threshold_status']='passed';outcomes[pool]['PL_threshold_can_match_CB_original_binary_tradeoff']=yes;thresholdlead.append(pool+('可' if yes else '不能')+'在不更高误报下达到CB原召回')
 text+=['','主结果逐个不同float32分数精确排序，所有同分整块纳入，端点全选／不选明确。两项选择的并列规则、紧邻预算两侧的真实阶梯点、块大小及全部scene见THRESHOLD_MATCH、THRESHOLD_SCENE；没有插值或逐scene阈值。原argmax由原pred严格复现，不能当作t=0.5。阈值读出仅Shadow对非Shadow，三类mIoU为NA。','', '|池|臂|macro AP%|pooled AP%|不同float32分数数|','|---|---|---:|---:|---:|']
 for r in ranking:text.append(f"|{r['pool']}|{r['arm']}|{f(r['macro_AP_pct'])}|{f(r['pooled_AP_pct'])}|{r['exact_unique_score_count']}|")
 text+=['','AP沿317的并列阶梯定义；macro AP排除只有单一二分类真值的scene，macro召回只用有正例scene，macro FPR只用有负例scene，所有scene仍在表内。整数混淆、原AP、宏平均和支持均已复现；L8963图按原8scene归组。阈值可复现默认点不代表CB所有训练效果等同阈值移动；不能复现也不证明云特有创新。','', '## 下一步边界','', '本轮新增证据是错误类别／边界／几何状态的完整交叉计数及精确阈值对照。它们尚未验证任何不读GT即可使用的新区分信息；水体多、薄云多或边界集中都不能自动证明几何有效或参考画错。建议先整理本次CB、Flip与几何探索的阶段报告；若据具体类别与受影响scene提出新假设，必须另写固定输入、对照和预算，本轮不自动执行。原方法窗口及317—319历史结论不重写，不把本轮视作全项目停止或投稿承诺。']
 opportunity+=['','目前能从既有输入直接使用的是教师预测及原几何状态；本轮没有验证新的GT外水体／薄云／地形判别器，也没有光谱规则、人工真值或新模型。类别构成是潜在线索，其可识别性与代价仍未知。应先保留受影响scene清单（ERROR_COMPOSITION scene行和ERROR_SCENE_SUMMARY），连同本轮阈值取舍讨论；不得因某类占比大而重调几何或自动训练。']
 hints=[];oldsc=read(P317/'PER_SCENE_METRICS.csv');selectedsc=read(ROOT/'THRESHOLD_SCENE.csv') if (ROOT/'THRESHOLD_SCENE.csv').exists() else []
 for pool in POOLS:
  if outcomes.get(pool,{}).get('PL_threshold_can_match_CB_original_binary_tradeoff') is not False:continue
  examples=[]
  for r in selectedsc:
   if r['pool']!=pool or r['arm']!='PL' or r['target']!='CB_original_FPR' or r['recall_pct']=='NA':continue
   cb=next(v for v in oldsc if v['pool']==pool and v['strategy']=='CB' and v['scene']==r['scene'])
   if cb['recall_pct']!='NA' and float(cb['recall_pct'])>float(r['recall_pct']):examples.append(dict(scene=r['scene'],CB_recall_pct=float(cb['recall_pct']),PL_global_threshold_recall_pct=float(r['recall_pct'])))
  hints.append(dict(pool=pool,hypothesis='CB may alter ranking or scene heterogeneity beyond a single global PL threshold; not a cloud-specific innovation claim',affected_positive_recall_gap_scenes=examples,requires='separate fixed protocol, compatible controls and bounded repeat; exposed single seed is not confirmation'))
 if hints:text+=['','可讨论的有限后续线索（不是自动立项）：'+ '；'.join(h['pool']+'中PL单阈值不足，CB局部召回差涉及 '+', '.join(e['scene'] for e in h['affected_positive_recall_gap_scenes']) for h in hints)+'。待验证假设是排序/scene异质性改变能否在固定重复与兼容对照下保留；原全局阈值下逐scene误报成本同表保留，不因该线索改写总体历史结论。']
 text.insert(2,'；'.join(lead)+'。PL单阈值：'+'；'.join(thresholdlead)+'。'+('存在需另行固定方案讨论的局部排序/scene异质性线索，见文末。' if hints else '尚无新的GT外区分信号，建议先整理阶段报告。')+'\n')
 (ROOT/'RESULT.md').write_text('\n'.join(text)+'\n',encoding='utf-8');(ROOT/'ERROR_OPPORTUNITY.md').write_text('\n'.join(opportunity)+'\n',encoding='utf-8');write(ROOT/'CONCLUSIONS.json',dict(status='complete',pools=outcomes,not_a_new_method=True,oracle_only=True,new_GT_free_discriminator_verified=False,possible_limited_followup=hints,no_automatic_next_phase=True,prior_candidate_conclusions_unchanged=True));print('Phase320 both diagnostics reported with limits',flush=True)
if __name__=='__main__':main()
