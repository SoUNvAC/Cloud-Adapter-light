"""Finalize frozen Phase205 results; no model calls, fitting or threshold changes."""
import csv,hashlib,json,os,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase205/phase205_20261019';S=R/'src/research_results/phase205_20261019'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(n):
 with (O/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,s):
 with (O/n).open('x',encoding='utf8') as f:f.write(s)
def save(n,v):write(n,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def fmt(x,scale=1):return 'NA' if x is None or x=='' else f'{float(x)*scale:.3f}'
def md(h,rows):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)
def main():
 started=time.time();assert (O/'LOCAL_EXIT_CODE').read_text().strip()=='0';assert load(O/'CPU_ACCEPTANCE.json')['status']=='passed' and load(O/'INDEPENDENT_FINAL_ACCEPTANCE.json')['status']=='passed' and load(O/'N4_RETENTION_ACCEPTANCE.json')['status']=='passed'
 summary=[];damage=[];intervals=[]
 for pool in ('Catalogue','PixBox'):summary+=read(pool+'_SUMMARY.csv');damage+=read(pool+'_REPEATED_DAMAGE.csv');intervals+=load(O/(pool+'_INTERVALS.json'))['rows']
 assert len(summary)==12 and len(damage)==84 and len(intervals)==8
 t1=md(['池','策略','全部/AP组','宏AP (%)','池化AP (%)','宏召回 (%)','宏误报/点误报 (%)','Catalogue池化mIoU (%)'],[[r['pool'],r['strategy'],r['groups']+'/'+r['AP_groups'],fmt(r['macro_AP_pp']),fmt(r['pooled_AP_pp']),fmt(r['official_decision_macro_recall_pp']),fmt(r['macro_negative_shadow_fraction_pp']),fmt(r['Catalogue_pooled_mIoU_pp'])] for r in summary])
 def table2(rows):return md(['池/组','正参考','原AP (%)','seed12201 ΔAP(pp)','seed12301 ΔAP(pp)','双seed均<-1pp','召回差12201/12301(pp)','误报差12201/12301(pp)'],[[r['pool']+'/'+r['tile'],r['positive'],fmt(r['baseline_AP'],100),fmt(r['seed12201_delta_AP_pp']),fmt(r['seed12301_delta_AP_pp']),r['both_damage'] or 'NA',fmt(r['seed12201_delta_recall_pp'])+'/'+fmt(r['seed12301_delta_recall_pp']),fmt(r['seed12201_delta_negative_shadow_pp'])+'/'+fmt(r['seed12301_delta_negative_shadow_pp'])] for r in rows])
 repeated=[r for r in damage if r['both_damage']=='True'];oldfour=[r for r in damage if r['tile'] in ('32TML','40XDR','19FDV','54TWN')];members=[r for r in repeated if r['family']=='member'];ensembles=[r for r in repeated if r['family']=='ensemble'];ci=md(['池','比较','seed','平均ΔAP(pp)','配对组bootstrap95%CI(pp)','AP组','改善/稳定/下降'],[[r['pool'],r['family'],r['seed'],fmt(r['mean_delta_AP_pp']),fmt(r['CI95_pp'][0])+' / '+fmt(r['CI95_pp'][1]),r['AP_groups'],f"{r['improved_gt1pp']}/{r['stable_within1pp']}/{r['damaged_lt_minus1pp']}"] for r in intervals])
 def listing(rows):return '、'.join(r['pool']+'/'+r['tile'] for r in rows) if rows else '未观察到'
 decision=('成员和完整双模型均出现两seed重复损伤，支持这个固定V4适配设置下存在有限的局部损伤现象。' if members and ensembles else '只在单成员观察到重复损伤，完整双模型未观察到相同标准的重复损伤；应分别解释成员和集成结果。' if members else '只在完整双模型观察到重复损伤；本批成员和集成的结果不同，不能据此归因单一机制。' if ensembles else '单成员与完整双模型均未观察到两seed重复损伤；结束将V1机制或预警线直接推广到V4的本轮尝试。')
 foot='''\n官方决策均为四类raw argmax后合并Thick/Thin为Cloud。AP采用作者最终Shadow confidence（softmax+0.001并裁剪），未扫阈值、未沿用V1校准阈值。Catalogue误报为既定有效格网的组均非Shadow→Shadow比例；PixBox仅为抽样参考点的组均误报比例，不是全景FPR，且不计算全景mIoU。无影组AP=NA仍保留误报统计。\n'''
 write('TABLE1_SIX_STRATEGIES.md','# 表1：固定六策略\n\n'+t1+foot)
 for family in ('member','ensemble'):write('TABLE2_'+family.upper()+'.md','# 表2：'+family+'全组重复损伤\n\n'+table2([r for r in damage if r['family']==family])+foot+'\nΔAP<-1个百分点是预先固定的描述分类，不是显著性检验。NA不填0。\n')
 write('PAIRED_GROUP_INTERVALS.md','# 描述性配对相关组区间\n\n'+ci+'\n每个seed分别在同一相关组内配对；10000次，seed205019。仅7/8个AP组，未将两个seed或像元作为新增独立样本。\n')
 write('OLD_FOUR_GROUPS.md','# 历史四组跟踪：成员\n\n'+table2([r for r in oldfour if r['family']=='member'])+'\n# 历史四组跟踪：完整集成\n\n'+table2([r for r in oldfour if r['family']=='ensemble'])+'\n54TWN仅7个正参考点；这不是面积或总体错误率估计。\n')
 gate=load(O/'gate_receipts/gate/GATE_ACCEPTANCE.json');white=load(O/'gate_receipts/gate/SMP_WHITELIST.json');aux=load(O/'N4_RETENTION_ACCEPTANCE.json');auxreceipt=load(O/'N4_transfer/DELIVERY_RECEIPT.json');cpu=load(O/'CPU_ACCEPTANCE.json');local=load(O/'LOCAL_STATUS.json')
 support=load(O/'COMMON_SUPPORT_ACCEPTANCE.json');assert all(r['loss']==0 and r['six_strategies_identical'] for r in support['rows']);assert support['PixBox_reference_negative']==8190 and support['PixBox_reference_positive']==379
 mainstatus=load(O/'delivery/CONTROLLER_STATUS.json');assert mainstatus['total_calls']==6600 and aux['all_actual_calls']==7746 and aux['combined_stage_seconds']<14400
 gpupeak=max(load(O/'delivery'/stage/'RUN_STATUS.json')['gpu_peak_reserved'] for stage in ('gate','train_12201','train_12301','infer'));rampeak=max(load(O/'delivery'/stage/'RUN_STATUS.json')['rss_peak_bytes'] for stage in ('gate','train_12201','train_12301','infer'));n4state=load(O/'N4_delivery/RUN_STATUS.json');gpupeak=max(gpupeak,n4state['gpu_peak_reserved']);rampeak=max(rampeak,n4state['rss_peak_bytes']);assert gpupeak<=20*1024**3 and max(rampeak,local['peak_bytes'])<=16*1024**3
 costs=dict(all_actual_neural_calls=7746,main_six_policy_calls=6600,training_calls=2000,main_inference_calls=4584,all_gate_calls=16,failed_gate_calls_included=5,N4_retention_repeated_inference_calls=1146,N4_recovery_reason='omitted N4 solo score artifact, no additional comparison or fit; same model/input/window',GPU_stage_seconds_including_charged_failures_and_wait=aux['combined_stage_seconds'],main_stage_seconds=mainstatus['stage_seconds_total'],N4_stage_seconds=n4state['stage_seconds'],successful_GPU_peak_reserved_bytes=gpupeak,successful_remote_CPU_peak_bytes=rampeak,local_analysis_peak_bytes=local['peak_bytes'],local_CPU_seconds=cpu['seconds'],local_bounded_seconds=local['seconds'],old_failed_gate_peak_memory='unknown: original handler did not sample at failure; not invented',receiver_recovery='Get-FileHash unavailable; exact already-downloaded full partial reused via .NET streaming SHA256; no retransmission',extra_fits=0,threshold_search=0,new_imagery_or_weight_downloads=0)
 save('COST_RECEIPT.json',costs)
 misses=dict(required_data_missing=[],full_author_V4_training_manifest='not obtained; overlap unknown',OCM_private_hard_negative_and_scribble_coverage='not publicly complete',old_failed_gate_peak_memory='not sampled at exception; unknown',independent_domain_validation=False,model_scope='only RegNet decoder+head adapted; EdgeNeXt frozen',two_seeds_do_not_estimate_all_randomness=True,AP_groups={'Catalogue':7,'PixBox':8},PixBox_not_area_or_fullscene=True,calibration53_not_read=True,no_BN_swaps_or_risk_probes=True,N4_single_scores_recovered_separately=True)
 save('MISSING_AND_LIMITS.json',misses)
 description=f'''# Phase205：V4适配后的重复场景损伤检查

## 1 数据结果

使用官方V4 SMP Unet的R4（tu-regnety_004）与N4（tu-edgenext_small），固定HF revision c9a4fb88188709127aa25cfe51ae7fd41b0132f8；实际权重SHA与Phase200原锁匹配。R4共{white['total_parameters']}参数，仅decoder/head的{white['trainable_parameters']}参数可训练，decoder/head含10个BN模块，encoder参数与buffer冻结。N4全程固定。

两个适配模型分别原样复用Phase122/123的100fit组和seed12201/12301各1000步裁剪/翻转序列；只保存末步，不早停或选checkpoint。R4 loader/直接重建最大误差{gate['R4_official_vs_direct_max_error']}；原E4官方链路与CPU独立集成拼接复算误差{gate['E4_official_vs_independent_max_error']:.9g}。初末步、参数/BN白名单、loss及encoder不变断言均已保存。CPU端点复核确认初始decoder/head逐张量匹配官方R4，末步仅包含许可状态。

评价固定Catalogue32相关组（7组AP支持）与PixBox16产品/10相关组（8组AP支持）。六策略支持一致，额外支持损失0；PixBox原9500个共同image-valid位置，排除931个ignore后379正/8190负，未使用Phase202额外掩膜。未读取53校准输入，未新增ALCD、65a/65b/65c或Phase200案例评价。历史数据已暴露；作者V4完整训练产品/footprint及OCM私有补充未知，逐产品覆盖状态见TRAINING_COVERAGE.csv。

### 六策略结果

'''
 counts=md(['池','成员重复损伤/AP组','集成重复损伤/AP组'],[[pool,str(sum(r['pool']==pool for r in members))+'/'+str(next(r['AP_groups'] for r in summary if r['pool']==pool)),str(sum(r['pool']==pool for r in ensembles))+'/'+str(next(r['AP_groups'] for r in summary if r['pool']==pool))] for pool in ('Catalogue','PixBox')])
 current_direction=('本轮各池、两seed的成员和集成宏AP点估计均下降。' if all(r['mean_delta_AP_pp']<0 for r in intervals) else '各比较的宏AP方向和区间分别保留。')
 interpretation=f'''\n## 2 结论分析

{decision}

成员重复损伤组：{listing(members)}。完整集成重复损伤组：{listing(ensembles)}。历史四组和其余固定组分别列于OLD_FOUR_GROUPS.md及完整表2，不用新出现组冒充原四组的地物机制复现。

{current_direction}固定N4的集成减轻了多数宏AP损失，但没有消除跨seed重复损伤。召回、误报与mIoU应结合表1解释：误报下降不能单独替代排序或召回评价。配对区间中，PixBox集成seed12201跨零；这项均值不确定性与逐组重复损伤计数分别陈述。

这里的−1pp只是预定描述边界，配对bootstrap也只是描述性区间，不能宣称独立确认或机制因果。两seed不估计所有训练随机性的失败概率，54TWN只有7个正参考点。Source训练集、V4扩展训练覆盖和本项目历史评价的相交状态未完全排除，不能称独立域外验证。

该方案只适配RegNet成员，保持N4固定；V4的SMP结构、可训练容量及公开训练来源均与此前V1不同，不能将本轮结果无条件解释为此前BN机制的复现。没有新增BN交换、风险探针或修复策略，也未自动启动后续阶段。
'''
 repeated_table=table2(repeated) if repeated else '两类比较均没有满足“两seed均ΔAP<-1pp”的组。\n'
 resource=f'''\n### 调用与交付记录

实际累计7746次调用：训练2000、固定六策略推理4584、门检16（含失败5）、补齐原N4最终分数1146。补齐仅恢复遗漏的N4单成员归档分数，原六策略封存输出未改变，未增加比较或拟合。总GPU阶段计账{aux['combined_stage_seconds']:.3f}s，小于14400s；包含既有失败及被计入的挂起等待。全部训练在2026-10-20北京时间截止前完成。成功进程GPU reserved峰值{gpupeak/1024**3:.3f}GiB，远端CPU峰值{rampeak/1024**3:.3f}GiB，本地CPU峰值{local['peak_bytes']/1024**3:.3f}GiB；早期失败门检的完整峰值未采样，明确保留unknown。

分数包、N4单独归档包及最终连续7746条调用账本均核对SHA；所有权重留在shared，未传work_dirs或自动push。完整原值见CSV/CONFUSION/INTERVALS，主AP独立复算、决策混淆复算及最终表格验收见CPU_ACCEPTANCE.json/INDEPENDENT_FINAL_ACCEPTANCE.json/N4_RETENTION_ACCEPTANCE.json。
'''
 write('RESULT.md',description+t1+foot+'\n### 跨seed重复损伤组\n\n'+counts+'\n'+repeated_table+'\n### 逐组配对差的描述性区间\n\n'+ci+resource+interpretation)
 mini=md(['池/比较','重复损伤组'],[[pool+'/'+family,listing([r for r in repeated if r['pool']==pool and r['family']==family])] for pool in ('Catalogue','PixBox') for family in ('member','ensemble')])
 oldset={'32TML','40XDR','19FDV','54TWN'}
 history_summary=f"旧四组中，成员重复损伤：{listing([r for r in members if r['tile'] in oldset])}；集成：{listing([r for r in ensembles if r['tile'] in oldset])}。其他固定组中，成员重复损伤：{listing([r for r in members if r['tile'] not in oldset])}；集成：{listing([r for r in ensembles if r['tile'] not in oldset])}。"
 short=f'''# Phase205 结果摘要

## 1 数据结果

官方V4 R4/N4原权重与runtime复用；仅适配R4 decoder/head（2829572/6732716参数），N4冻结。两个固定seed分别原100fit组/1000步；原Catalogue32组、PixBox16产品/10组，AP支持7/8组。未读53校准输入或扫阈值。六策略官方四类argmax后合并Cloud，同一有效支持；PixBox379正/8190负。

{mini}

{history_summary}

重复损伤定义为相对本版本R4或E4，两seed均ΔAP<-1pp。完整六策略宏/池化AP、召回、误报及Catalogue三类mIoU见TABLE1；全部组含NA/稳定/改善/下降见TABLE2_MEMBER/TABLE2_ENSEMBLE；原四组见OLD_FOUR_GROUPS。PixBox仅点支持误报，不代表全景FPR或面积；54TWN仅7个正参考。

## 2 结论分析

{decision}

{current_direction}集成减轻多数损失，仍保留重复损伤；PixBox集成seed12201均值区间跨零。−1pp非显著性检验，区间仅描述。两seed、7/8个AP组与已暴露历史池不能支持独立泛化或全部随机性结论。完整V4训练覆盖/私有OCM交集未知；本轮只适配一个成员，结构和容量与V1不同，未检验BN因果或修复方法。

累计7746调用（含5次失败门检、1146次补齐N4归档），GPU阶段{aux['combined_stage_seconds']:.1f}s/14400s。原六策略输出未改，零额外拟合/阈值搜索/新下载影像权重，未自动启动后续phase。完整报告RESULT.md的数据结果与解释分开；早期失败完整内存峰值unknown如实保留。
'''
 write('README.md',short)
 for n in ('TRAINING_COVERAGE.csv','TRAINING_COVERAGE_SOURCES.json','V4_WEIGHT_LOCK.json'):
  assert (O/n).exists()
 save('MODEL_AND_TRAINING_IDENTITY.json',dict(weight_lock=load(O/'V4_WEIGHT_LOCK.json'),gate=gate,trainable_parameters=white['trainable_parameters'],total_parameters=white['total_parameters'],decoder_head_BN=10,initial_final_endpoint_audit=load(O/'delivery/ENDPOINT_CPU_STATE_ACCEPTANCE.json'),training_sha={str(seed):load(O/f'delivery/train_{seed}/TRAINING_RECEIPT.json')['final_model_sha256'] if 'final_model_sha256' in load(O/f'delivery/train_{seed}/TRAINING_RECEIPT.json') else load(O/f'delivery/train_{seed}/TRAINING_RECEIPT.json')['final_sha256'] for seed in (12201,12301)},training_schedule_sha={str(seed):sha(O/f'{seed}_PAIRED_CROP_SCHEDULE.json') for seed in (12201,12301)}))
 save('FINAL_PROVENANCE.json',dict(main_score_manifest_sha256=sha(O/'delivery/DELIVERY_MANIFEST.json') if (O/'delivery/DELIVERY_MANIFEST.json').exists() else load(O/'DELIVERY_RECEIPT.json')['manifest_sha256'],main_archive_sha256=load(O/'DELIVERY_RECEIPT.json')['sha256'],auxiliary_archive_sha256=auxreceipt['sha256'],input_lock_sha256=sha(O/'INPUT_LOCK.json'),code_sha256={p.name:sha(p) for p in (R/'src/tools').glob('*phase205*.py')},support_lock_sha256=sha(O/'COMMON_SUPPORT_ACCEPTANCE.json'),no_automatic_push=True,reference_only_local=True))
 S.mkdir(parents=True,exist_ok=True)
 small=['README.md','RESULT.md','TABLE1_SIX_STRATEGIES.md','TABLE2_MEMBER.md','TABLE2_ENSEMBLE.md','OLD_FOUR_GROUPS.md','PAIRED_GROUP_INTERVALS.md','COST_RECEIPT.json','MISSING_AND_LIMITS.json','MODEL_AND_TRAINING_IDENTITY.json','FINAL_PROVENANCE.json','CPU_ACCEPTANCE.json','INDEPENDENT_FINAL_ACCEPTANCE.json','N4_RETENTION_ACCEPTANCE.json','COMMON_SUPPORT_ACCEPTANCE.json','TRAINING_COVERAGE.csv','TRAINING_COVERAGE_SOURCES.json','V4_WEIGHT_LOCK.json']+[p.name for p in O.glob('Catalogue_*.csv')]+[p.name for p in O.glob('PixBox_*.csv')]+[p.name for p in O.glob('*_INTERVALS.json')]+[p.name for p in O.glob('*_CONFUSION.json')]
 for n in small:assert not (S/n).exists();shutil.copyfile(O/n,S/n)
 previous=O/'HANDOFF_BEFORE_COMPLETE.md';assert not previous.exists();shutil.copyfile(O/'HANDOFF.md',previous);(O/'HANDOFF.md').write_text('# Phase205 COMPLETE\n\nCurrent delivery: complete. Two fixed fits and six policies accepted; N4 missing archival scores recovered and accepted. Final7746calls/2684.5397005201085s; no more forward or fitting. Original six-policy outputs and manifests unchanged. RESULT.md has separate data and analysis, full tables/identity/support/cost/independent receipts saved; summary src/research_results/phase205_20261019. Delete phase205-hourly, notify once, no nextphase. Historical handoff preserved as HANDOFF_BEFORE_COMPLETE.md.\n',encoding='utf8')
 localbytes=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());summarybytes=sum(p.stat().st_size for p in S.rglob('*') if p.is_file());sourcebytes=sum(p.stat().st_size for p in (R/'src/tools').glob('*phase205*') if p.is_file());combined=auxreceipt['root_new_bytes']+localbytes+summarybytes+sourcebytes+94452+1048576;assert combined<6000000000
 save('COMBINED_STORAGE_RECEIPT.json',dict(remote_root_new_bytes_reported=auxreceipt['root_new_bytes'],local_outputs_bytes=localbytes,summary_bytes=summarybytes,new_source_bytes=sourcebytes,external_remote_bootstrap_bytes=94452,allowance_for_final_manifests_receipts_and_copy_bytes=1048576,combined_new_bytes_upper_bound=combined,cap=6000000000));shutil.copyfile(O/'COMBINED_STORAGE_RECEIPT.json',S/'COMBINED_STORAGE_RECEIPT.json')
 save('DELIVERY_COMPLETE.json',dict(status='completed',files={p.name:sha(p) for p in O.iterdir() if p.is_file()},summary_files={p.name:sha(p) for p in S.iterdir() if p.is_file()},all_actual_calls=7746,combined_new_bytes_upper_bound=combined,GPU_stage_seconds=aux['combined_stage_seconds'],CPU_delivery_seconds=time.time()-started,no_automatic_next_phase=True))
 print(json.dumps(dict(status='completed',member_repeated=listing(members),ensemble_repeated=listing(ensembles),combined_bytes=combined)))
if __name__=='__main__':main()
