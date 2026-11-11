"""Seal Phase209 fixed tables, independent acceptance and bounded claims."""
import csv,json,hashlib,shutil,time,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase209/phase209_20261020';S=R/'src/research_results/phase209_20261020'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,text):
 with p.open('x',encoding='utf8') as f:f.write(text)
def f(v):return 'NA' if v is None or v=='' else f'{float(v):.3f}'
def tab(headers,data):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in data)
def main():
 assert load(O/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed';A=load(O/'TABLE_A.json');B=load(O/'TABLE_B.json');diff=load(O/'B_RECALL_DIFFERENCES.json');lock=load(O/'THRESHOLD_FROZEN.json');receipt=load(O/'DELIVERY_RECEIPT.json');status=load(O/'calibration/RUN_STATUS.json')
 result='# Phase209 数据结果\n\n原R4调阈值能否解释适配器收益？本轮零训练、零拟合、零新增评价前向。模型与输入固定为208的V4 R4/M/B，E4仅引用208。Catalogue32相关组（7组可算AP），PixBox16产品/10相关组（8组可算AP）；PixBox误报只表示参考点支持。\n\n'
 result+='## A表：同误报水平的oracle诊断\n\n使用各评价池标签扫描R4唯一全局阈值，不是部署阈值，不用于B表。ties整批处理，离散预算缺口不插值。召回差是适配器默认−R4匹配点，单位百分点。\n\n'
 result+=tab(['池','目标来源','目标误报%','R4实际误报%','缺口pp','适配器召回%','R4召回%','召回差pp','95%区间','R4最大组误报%'],[[q['pool'],q['target_source'],f(q['target_mean_negative_pp']),f(q['R4_actual_mean_negative_pp']),format(q['budget_gap_pp'],'.9g'),f(q['adapter_default_macro_recall_pp']),f(q['R4_matched_macro_recall_pp']),f(q['delta_recall_pp']),'['+', '.join(f(x) for x in q['CI95_pp'])+']',f(q['R4_max_group_negative_pp'])] for q in A])
 result+='\nA表bootstrap10000次、seed209020，每次分别在有影/无影两层整组重采样，保持7+25及8+2组结构；每次重新求适配器目标与R4匹配阈值。区间条件于该支持结构，不固定原扫描阈值。所有40000次重选阈值及计数已独立复核。精确预算缺口、阈值和每次抽样权重见JSON/CSV。\n\n'
 result+='## B表：53组唯一阈值的迁移\n\n'+f"只在原53校准相关组完整负类支持上选择阈值`t={lock['threshold']!r}`，规则为Shadow分数按float64比较`>=t`，逐组等权平均误报≤1%的最低可行全局阈值。校准实际平均误报{lock['macro_negative_fraction']*100:.9f}%。100 fit/53 calibration/32 development的组ID互不相交；这些历史已暴露数据不因此变为独立确认。阈值封存后才运行B表；不使用A表oracle阈值。\n\n"
 result+=tab(['池','决策','宏AP%','宏召回%','平均误报%','单侧95%上界%','最大组误报%','超1%组数','零召回组数'],[[q['pool'],q['strategy'],f(q['macro_AP_pp']),f(q['macro_recall_pp']),f(q['macro_negative_pp']),f(q['negative_upper95_pp']),f(q['max_group_negative_pp']),q['groups_over1pct'],q['zero_recall_groups']] for q in B])
 result+='\nB表误报上界为全部32/10组非分层bootstrap均值的95分位数（10000次、seed209020）。预算是相关组平均值，非每景约束；点估计≤1%也不意味着上界≤1%。B表不同方法实际误报并不相等。\n\n'
 result+=tab(['池','固定召回比较','平均差pp','配对组95%区间'],[[q['pool'],q['contrast'],f(q['mean_delta_recall_pp']),'['+', '.join(f(x) for x in q['CI95_pp'])+']'] for q in diff])
 result+='\nB表召回区间条件于已冻结的53组阈值，不包括重新校准的不确定性；按有影/无影层重采样，召回仅7/8有影组。无影组AP和召回NA，继续参加误报统计。\n\n'
 result+=tab(['池','决策','池化AP%','池化召回%','池化精确率%','池化误报%','Catalogue二类Shadow IoU%','Catalogue三类mIoU%'],[[q['pool'],q['strategy']]+[f(q[k]) for k in ['pooled_AP_pp','pooled_recall_pp','pooled_precision_pp','pooled_negative_pp','Shadow_IoU_pp','Catalogue_three_class_mIoU_pp']] for q in B])
 result+='\n原R4改阈值前后连续评分、AP完全不变。新阈值三类mIoU为NA：现有raw_class/parent_class只有第一候选，无法恢复被移除Shadow像元的非Shadow条件argmax；未补前向、未统一填Surface、未用标签回填。不能据此声称超过208的M三类mIoU 86.459%。\n\n'
 result+='## 真实成本与缺项\n\n'+f"本轮新增{receipt['total_calls']}次R4网络调用，其中53组主评分212次、构建{status['constructor_calls']}次；GPU阶段{receipt['GPU_stage_seconds']:.3f}秒。零训练、零拟合、零评价集新前向。所有调用、失败、输入/模型/代码SHA及资源见COST_PROVENANCE.json。校准分数包SHA/ZIP CRC/逐文件校验通过；独立验收见INDEPENDENT_ACCEPTANCE.json。完整逐组/各次bootstrap/校准53组计数保留CSV。\n"
 conclusions='# Phase209 结论分析\n\n'
 conclusions+='## 同误报诊断回答了什么\n\n'
 for pool in ['Catalogue','PixBox']:
  qm=next(q for q in A if q['pool']==pool and q['target_source']=='M');qb=next(q for q in A if q['pool']==pool and q['target_source']=='B')
  conclusions+=f"{pool}：M在同误报水平下相对R4召回差为{qm['delta_recall_pp']:+.3f}个百分点，95%区间[{qm['CI95_pp'][0]:.3f}, {qm['CI95_pp'][1]:.3f}]；B为{qb['delta_recall_pp']:+.3f}个百分点，区间[{qb['CI95_pp'][0]:.3f}, {qb['CI95_pp'][1]:.3f}]。"
  conclusions+='这说明原R4改变工作点已可解释相当一部分默认召回/误报变化；剩余差异方向依池而异，当前小样本不能确立稳定的适配器额外优势。oracle使用评价标签，不能把这些工作点称为可部署或迁移成功。\n\n'
 conclusions+='## 冻结阈值迁移回答了什么\n\n'
 for pool in ['Catalogue','PixBox']:
  q=next(q for q in B if q['pool']==pool and q['strategy']=='R4_calibrated');m=next(q for q in B if q['pool']==pool and q['strategy']=='M');b=next(q for q in B if q['pool']==pool and q['strategy']=='B')
  conclusions+=f"{pool}：R4冻结阈值的平均召回{q['macro_recall_pp']:.3f}%，实际平均误报{q['macro_negative_pp']:.3f}%、单侧95%上界{q['negative_upper95_pp']:.3f}%；M为召回{m['macro_recall_pp']:.3f}%/误报{m['macro_negative_pp']:.3f}%，B为召回{b['macro_recall_pp']:.3f}%/误报{b['macro_negative_pp']:.3f}%。"
  conclusions+=('本批点估计满足1%预算；' if q['macro_negative_pp']<=1 else '迁移后的点估计超过1%预算；')+('上界也在1%以内。' if q['negative_upper95_pp']<=1 else '上界超过1%，不能认定平均误报稳定满足预算。')+'这些工作点误报不同，不能仅凭召回高低排出总体胜负。\n\n'
 conclusions+='R4调阈值不会改变AP，因此不能解释208中评分排序本身的增减；它检验的是决策工作点解释。适配器是否有剩余排序收益仍须结合208的M−B宏AP小幅增量、跨池不确定性及损伤报告，不能用本轮oracle或新的阈值mIoU缺项去证明内部机制。\n\n'
 conclusions+='## 限制和可说到哪里\n\n单seed、历史多次使用的评价池；组ID边界互斥不等于与官方完整训练数据独立（覆盖仍未知）。相关组是统计单位，AP/召回有效组仅7/8。PixBox是采样点，54TWN仅7个正参考；不称全景FPR/mIoU。新R4阈值三类mIoU缺失是缓存信息不足，不是0，也不推断它高于或低于M。E4未新增校准或评价，仅引用208的原结果。\n\n本轮证据倾向先把阈值/校准作为必要低成本基线，同时保留少量、依池而异的适配剩余收益作为未定问题。不能据此宣称MsRE有效/无效或已经替代E4。不自动调预算、阈值、波段、窗口或新增seed/Phase210。\n'
 write(O/'DATA_RESULTS.md',result);write(O/'CONCLUSIONS.md',conclusions);write(O/'RESULT.md',result+'\n'+conclusions)
 readme='# Phase209\n\n完成：原53组R4评分、唯一阈值锁定、A/B表及独立验收；零训练、零拟合、零评价新前向。\n\n数据结果见DATA_RESULTS.md；结论分析见CONCLUSIONS.md；合并报告RESULT.md。\n\nA表是使用评价标签的oracle同误报诊断，不能部署；B表阈值只来自53组校准，评价误报按实际报告。新阈值三类mIoU因缺非Shadow条件argmax为NA。\n\n全组计数见BASELINE_GROUPS.csv、A_GROUPS.csv、B_GROUPS.csv、CALIBRATION_53_COUNTS.csv；40000次A表重选阈值抽样轨迹完整保留。COST_PROVENANCE.json与DELIVERY_COMPLETE.json封存来源、成本及SHA。\n\n未创建定时任务，未自动push，未启动下一phase。\n'
 write(O/'README.md',readme);S.mkdir(exist_ok=False)
 for n in ['RESULT.md','DATA_RESULTS.md','CONCLUSIONS.md','README.md']:shutil.copyfile(O/n,S/n)
 costs=dict(total_new_network_calls=receipt['total_calls'],calibration_main_calls=212,constructor_calls=status['constructor_calls'],new_fits=0,new_training=0,new_evaluation_forward_calls=0,GPU_stage_seconds=receipt['GPU_stage_seconds'],remote_CPU_peak_bytes=status['cpu_peak_bytes'],remote_GPU_reserved_peak_bytes=status['gpu_peak_bytes'],remote_minimum_host_free_bytes=status['minimum_host_free_bytes'],CPU_stage_resources={p.parent.name:load(p) for p in O.glob('cpu_*/STATUS.json')},remote_new_bytes=receipt['remote_new_bytes'],threshold_lock_sha256=sha(O/'THRESHOLD_FROZEN.json'),execution_lock=load(O/'EXECUTION_LOCK.json'),calibration_score_seal=load(O/'calibration/SCORES_SEALED.json'),local_code_sha256={n:sha(R/'src/tools'/n) for n in ['prepare_phase209.py','execute_phase209.py','analyze_phase209.py','bounded_phase209_local.py','validate_phase209.py','finalize_phase209.py']},network_retries='initial restricted SSH could not resolve saved alias; escalated existing config succeeded; initial SCP kex closed then bounded retry succeeded; no duplicate scoring launch',CPU_bootstrap_recovery=load(O/'CPU_BOOTSTRAP_RECOVERY.json'),missing=['new R4 threshold three-class mIoU: nonShadow conditional argmax not cached','complete official V4 training overlap unknown','unobserved resource peaks between samples unknown'],scope='history-exposed exploratory fixed configuration; no timer, push or new phase')
 costs['helper_and_delivery_code_sha256']={n:sha(R/'src/tools'/n) for n in ['analyze_phase208_local.py','analyze_phase203_local.py','analyze_phase202_local.py','bounded_phase202_local.py','bounded_phase209_validation.py','accept_phase209_transfer.py','receive_phase209.ps1','deploy_phase209.py']}
 costs['evaluation_score_seals_sha256']={n:sha(R/p) for n,p in [('original_R4','outputs/phase205/phase205_20261019/delivery/infer/SCORES_SEALED.json'),('M_B','outputs/phase208/phase208_20261020/delivery/infer/SCORES_SEALED.json')]}
 costs['combined_bytes_before_final_receipts']=costs['remote_new_bytes']+sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+sum(p.stat().st_size for p in S.rglob('*') if p.is_file());assert costs['combined_bytes_before_final_receipts']+10000000<4000000000
 write(O/'COST_PROVENANCE.json',json.dumps(costs,indent=2,allow_nan=False))
 finish()
def finish():
 assert load(O/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed' and (O/'RESULT.md').exists() and (O/'COST_PROVENANCE.json').exists()
 receipt=load(O/'DELIVERY_RECEIPT.json');lock=load(O/'THRESHOLD_FROZEN.json');costs=load(O/'COST_PROVENANCE.json')
 if (O/'HANDOFF.md').exists():shutil.copyfile(O/'HANDOFF.md',O/'HANDOFF_PRE_FINALIZATION.md')
 (O/'HANDOFF.md').write_text('# Phase209 handoff\n\nCurrent delivery: COMPLETE. R4 sole53 calibration pass EXIT0, baseline/calibration/B evaluation and independent validation passed; reports and manifests sealed. Do not rerun fits/scores or change thresholds. No scheduled automation created.\n\nSee RESULT.md and DELIVERY_COMPLETE.json.\n',encoding='utf8')
 manifest=dict(status='complete',files={str(p.relative_to(O)).replace('\\','/'):sha(p) for p in O.rglob('*') if p.is_file()},summary_files={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file()},new_network_calls=receipt['total_calls'],new_fits=0,combined_bytes_before_manifest=costs['remote_new_bytes']+sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+sum(p.stat().st_size for p in S.rglob('*') if p.is_file()))
 write(O/'DELIVERY_COMPLETE.json',json.dumps(manifest,indent=2,allow_nan=False))
 for n,h in manifest['files'].items():assert sha(O/n)==h
 print(json.dumps(dict(status='complete',total_calls=receipt['total_calls'],combined_bytes=manifest['combined_bytes_before_manifest'],threshold=lock['threshold'])))
if __name__=='__main__':finish() if '--finish' in sys.argv else main()
