"""Two-page diagnostic report, resource receipt and final immutable SHA seal."""
from pathlib import Path
import csv,json,hashlib,shutil,time
from analyze_phase216 import peak
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase220/phase220_20261024';RAW=O/'delivery';SUMMARY=ROOT/'src/research_results/phase220_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert js(O/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed';done=js(RAW/'COMPLETE.json');assert done['units']==4800 and done['GPU_seconds']<3600
 a=rd('TABLE_A_POOLED.csv');b=rd('TABLE_B_WEIGHT_ALLOCATION.csv');facts=js(O/'ANSWER_FACTS.json');receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');resources=[done[v] for v in ['val','replay']];alloc=max(v['GPU_allocated_peak'] for v in resources);reserved=max(v['GPU_reserved_peak'] for v in resources);cpu=max(v['CPU_peak_bytes'] for v in resources);local=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());packages=sum(p.stat().st_size for p in O.glob('*.zip'));combined=receipt['remote_scope_bytes']+receipt['bytes']+local+packages
 assert max(alloc,reserved)<20*1024**3 and cpu<16*1024**3 and peak()<16*1024**3 and combined<2_000_000_000 and shutil.disk_usage(ROOT).free>4*1024**3
 historical={}
 for phase in [217,218,219]:
  d=ROOT/f'outputs/phase{phase}/phase{phase}_20261023';m=js(d/'DELIVERY_MANIFEST.json');assert sha(d/'DELIVERY_MANIFEST.json')==js(d/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((d/n).stat().st_size==v['bytes'] and sha(d/n)==v['SHA'] for n,v in m.items());historical[str(phase)]={'manifest_SHA':sha(d/'DELIVERY_MANIFEST.json'),'all_unchanged':len(m)}
 audit=ROOT/'outputs/research_audits/phase219_class_damage_20261024';assert all(sha(audit/n)==h for n,h in js(RAW/'ORIGINAL_CLASS_AUDIT_SHA.json').items())
 save('COST_AND_IDENTITIES.json',{'units':4800,'calls':js(RAW/'REMOTE_COLLECTION_RECEIPT.json')['calls'],'GPU_stage_seconds':done['GPU_seconds'],'zero_training_backward_fits_target_evaluation_bootstrap':True,'GPU_allocated_peak':alloc,'GPU_reserved_peak':reserved,'remote_CPU_peak_bytes':cpu,'local_CPU_peak_bytes':peak(),'combined_persistent_upper_bytes_before_final_seal':combined,'remote_scope_bytes':receipt['remote_scope_bytes'],'archive_remote_bytes':receipt['bytes'],'local_scope_bytes_before_final_seal':local,'uploaded_packages_upper_bytes':packages,'persistent_cap_bytes':2_000_000_000,'local_free_bytes':shutil.disk_usage(ROOT).free,'model_lock':js(RAW/'MODEL_LOCK.json'),'code_lock':js(RAW/'CODE_LOCK.json'),'support_lock_SHA':sha(RAW/'SUPPORT_LOCK.json'),'expected_historical_margins_SHA':sha(RAW/'EXPECTED_VAL.json'),'val_vs_replay_runtime_distinction':{v:done[v]['flags'] for v in ['val','replay']},'original217_218_219_unchanged':historical,'class_audit_unchanged':True,'Source_file_and_all6_endpoints_unchanged':True,'no_timer_download_auto_push_or_next_phase':True})
 at='\n'.join(f"|{v['seed']}|{v['band']}|{int(v['valid_pixels']):,}|{int(v['D_wrong_G_right']):,}|{int(v['D_right_G_wrong']):,}|{int(v['G_minus_D_errors']):+,}|{100*float(v['D_error_rate']):.2f}/{100*float(v['G_error_rate']):.2f}/{100*float(v['S_error_rate']):.2f}|" for v in a if v['reference_class']=='Shadow' and v['teacher_reference_correctness']=='all')
 bt='\n'.join(f"|{'val参考类' if v['pool']=='source_val535' else '回放教师预测类×4000序列'}|{v['class']}|{100*float(v['high_confidence_fraction']):.2f}%|{float(v['G_mean_weight']):.4f}|{float(v['S_mean_weight']):.4f}|{int(v['support']):,}|" for v in b if (v['pool']=='source_val535' and v['basis']=='reference_valid') or (v['pool']=='replay256' and v['repetition']=='original4000_sequence'))
 localization=[];protected=[];classline=[]
 for s in ['21701','21702']:
  f=facts[s];hi=f['by_band']['high'];lo=f['by_band']['low'];total=f['Shadow_net_errors'];share=lo['G_minus_D_errors']/total if total else None;localization.append(f"{s}云影净错误{total:+,}：低置信区{lo['G_minus_D_errors']:+,}"+(f"（占净增量{share*100:.2f}%）" if share is not None else '（总净量为0，不算比例）')+f"，高置信区{hi['G_minus_D_errors']:+,}。")
  tc=f['teacher_correct_Shadow_by_band'];protected.append(f"{s}教师原正确云影的净新增错误="+' / '.join(f"{k}:{tc[k]['G_minus_D_errors']:+,}" for k in ['high','low'])+f"，合计{sum(v['G_minus_D_errors'] for v in tc.values()):+,}。")
  classes=[]
  for c in ['Surface','Cloud','Shadow']:classes.append(c+':'+f"{sum(int(v['G_minus_D_errors']) for v in a if v['seed']==s and v['reference_class']==c and v['teacher_reference_correctness']=='all'):+,}")
  classline.append(s+' '+', '.join(classes))
 rp=next(v for v in b if v['pool']=='replay256' and v['repetition']=='original4000_sequence' and v['class']=='Shadow');vr=next(v for v in b if v['pool']=='source_val535' and v['basis']=='reference_valid' and v['class']=='Shadow');predicate='低于' if float(rp['G_mean_weight'])<float(rp['S_mean_weight']) else '不低于'
 low_all=all(facts[s]['by_band']['low']['G_minus_D_errors']>facts[s]['Shadow_net_errors']/2 for s in ['21701','21702']);correct_all=all(sum(v['G_minus_D_errors'] for v in facts[s]['teacher_correct_Shadow_by_band'].values())>0 for s in ['21701','21702'])
 report=f'''# Phase220：置信度加权的云影损伤定位

## 1．数据结果

零训练、反传、拟合、调参或新bootstrap。复用217的D、219的G/S两seed末步，α=1；仅原CloudSEN源val535图/107 ROI和原256张无标签源train回放RGB。Source固定；不运行SPARCS目标、官方源test、L8或Omni。源val使用219评价runtime，回放使用219权重预计算的训练runtime，各自复现旧SHA，未混用。

逐图保存整数J[置信区,参考,教师,D,G,S]，shape=(2,3,3,3,3,3)。表A摘录参考Shadow，错误率为D/G/S；Surface、Cloud、教师对错分层和全部107 ROI见完整CSV。D错→G对与D对→G错为本轮同像元直接计数。

|seed|置信区|参考Shadow支持|D错→G对|D对→G错|G−D净错误|D/G/S错误率%|
|---|---|---:|---:|---:|---:|---|
{at}

{''.join(localization)} 净增量占比可受另一置信区抵消影响，不是错误率或独立像元样本量。

{''.join(protected)} 所有参考类总净变化：{'；'.join(classline)}。Shadow→Surface/Cloud以及Surface/Cloud→Shadow的新增、移除流向完整保存，不从边际表推断直接转移。

表B为类别权重暴露。val按参考类，仅作诊断重建；回放按教师预测类、原4000步重复次数加权，两个seed共用序列不重复计数。每图一次的unique256及val教师预测类另在CSV。

|池及类别依据|类别|高置信占比|G平均权重|S平均权重|支持|
|---|---|---:|---:|---:|---:|
{bt}

旧四焦点和预先锁定的两seed共同下降>1pp的13 ROI均交附表，不代替全体。固定首图克隆教师、7次正式整网路径与共享路径完全相同；六学生逐图/逐ROI混淆、教师预测/高置信掩膜、四正确性状态和旧类别分解全部复现。CPU另一套显式六轴索引求和验收通过。实际4800单位（val4280、首图额外门检8、回放512），GPU阶段{done['GPU_seconds']:.1f}s；GPU allocated/reserved峰值{alloc/1024**3:.2f}/{reserved/1024**3:.2f}GiB，持久文件合计上界{combined/1e9:.3f}GB。旧217–219及本地审计逐文件SHA未变；完整输入、模型和代码SHA及资源见成本回执。

## 2．结论探讨

新增云影错误{'在两个seed均主要落在低置信区' if low_all else '未在两个seed都呈现主要落在低置信区的同向结果'}；{'教师原本判对的云影也出现净损伤' if correct_all else '教师原正确云影的净损伤并非两个seed同向'}，不能只解释成教师错误被少纠正。低置信区是逐图相对排名的一半，不等于所有像元置信度绝对很低。

val参考云影高置信占比{100*float(vr['high_confidence_fraction']):.2f}%，G平均权重{float(vr['G_mean_weight']):.4f}；原回放教师预测云影的G平均权重{float(rp['G_mean_weight']):.4f}，{predicate}S的{float(rp['S_mean_weight']):.4f}。S保持每图权重直方图，类别分配是否接近1由表B实际值判断。回放类别是教师预测，不能当作真实云影训练支持；平均权重也不等于实际梯度贡献。

本轮定位了相关结构，没有训练干预来识别其原因。不能证明类别权重变化导致损伤、置信度已校准或类均衡训练一定改善；这些是后续可检验解释，本轮不启动新方法。13 ROI为事后已知受损子集，不是独立验证；两个seed不扩成214独立组。
'''
 (O/'RESULT.md').write_text(report,encoding='utf-8');SUMMARY.mkdir(parents=True,exist_ok=True)
 names=['RESULT.md','TABLE_A_POOLED.csv','TABLE_B_WEIGHT_ALLOCATION.csv','FOUR_FOCUS_ROIS.csv','THIRTEEN_POSTHOC_DAMAGE_ROIS.csv','COHORT_SUMMARIES.csv','ANSWER_FACTS.json','COST_AND_IDENTITIES.json','INDEPENDENT_CPU_ACCEPTANCE.json','JOINT_AXES.json']
 for n in names:shutil.copy2(O/n,SUMMARY/n)
 (O/'HANDOFF.md').write_text('Current delivery: COMPLETE. Frozen220 diagnostics4800 units, no fits/backwards/target/bootstrap. All6 old student outputs and teacher confidence/replay training masks reproduced; explicit CPU6-axis verification passed. RESULT data/interpretation separate, full107ROI/class/teacher tiers, direct error flows and unique/4000 replay allocation tables delivered. Old217-219 and class audit unchanged. No timer/auto push/new phase.\n',encoding='utf-8')
 files=[p for p in RAW.rglob('*') if p.is_file()]+[p for p in (O/'local_code').rglob('*') if p.is_file()]+[p for p in O.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','LOCAL_STATUS.json']];m={str(p.relative_to(O)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};save('DELIVERY_MANIFEST.json',m);assert all((O/n).stat().st_size==v['bytes'] and sha(O/n)==v['SHA'] for n,v in m.items());save('DELIVERY_COMPLETE.json',{'status':'passed','artifacts':len(m),'manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'units':4800,'GPU_seconds':done['GPU_seconds'],'epoch':time.time()});print(json.dumps({'status':'complete','report':str(O/'RESULT.md')}))
if __name__=='__main__':main()
