"""Independent Phase203 scalar/hash audit and small result delivery; zero forwards."""
import csv,hashlib,json,re,shutil,sys
from pathlib import Path
R=Path('D:/Cloud-Adapter-light');O=R/'outputs/phase203/phase203_20261018';D=O/'delivery';S=R/'src/research_results/phase203_20261018'
sys.path.insert(0,str(R/'outputs/phase202/phase202_20261017/software'))
import numpy as np
ST=('shared_R_00','12201_11','12301_11','12201_10','12201_01','12301_10','12301_01')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def write(p,t):
 with Path(p).open('x',encoding='utf8') as f:f.write(t)
def num(v):return 'NA' if v is None or v=='' else f'{float(v):.3f}'
def tile(p):return '/'.join(sorted(set(re.findall(r'_T([0-9A-Z]{5})_',p))))
def table(h,rr):return '| '+' | '.join(h)+' |\n|'+'|'.join(['---']*len(h))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rr)
def main():
 assert not (O/'DELIVERY_COMPLETE.json').exists();assert load(O/'LOCAL_CPU_ACCEPTANCE.json')['status']=='passed';assert sha(R/'src/tools/analyze_phase203_local.py')==load(O/'LOCAL_CPU_ACCEPTANCE.json')['code_sha256']
 lock=load(D/'EXECUTION_LOCK.json');manifest=load(O/'INPUT_DELIVERY_LOCK.json')['manifest'];checks=0
 for n,h in manifest['files'].items():assert sha(D/n)==h,n;checks+=1
 for n,h in lock['source_sha256'].items():assert sha(D/n)==h,n;checks+=1
 for n,h in lock['accepted_source_sha256'].items():assert sha(R/n)==h,n;checks+=1
 assert sha(R/'src/research_plans/PHASE203_DECODER_PARAMETER_BN_MECHANISM_20261018.md')==lock['protocol_sha256']
 white=load(D/'WHITELIST.json');parameters=set(white['trainable_names']);buffers=set(white['decoder_state_names'])-parameters;assert len(parameters)==46 and len(buffers)==15
 identities=[];reproductions=[];stages=[]
 for state in ST:
  q=load(D/state/'STATE_IDENTITY.json');mode=state.rsplit('_',1)[1];assert q['encoder_sha256']==white['initial_encoder_sha256'] and q['eval'] and q['trainable_parameters']==0;assert set(q['parameters'])==parameters and set(q['BN_buffers'])==buffers
  for n,source in q['tensor_source'].items():assert source==('adapted' if mode[0 if n in parameters else 1]=='1' else 'original');checks+=1
  identities.append(q)
  for pool in ('Catalogue','PixBox'):
   stages.append(dict(load(D/state/('R_'+pool)/'RUN_STATUS.json'),state=state,pool=pool))
   if state in ST[:3]:
    rr=load(D/state/(pool+'_REPRODUCTION.json'));assert rr['passed'] and all(x['same_support'] and x['max_Shadow_confidence_error']<=1e-5 for x in rr['rows']);reproductions.extend(rr['rows']);checks+=1
 assert len(reproductions)==6 and max(q['max_Shadow_confidence_error'] for q in reproductions)==0
 assert sum(load(D/t/'COMPLETE.json')['all_calls'] for t in ST)+7==4865
 save(O/'MODEL_SOURCES_AND_REPRODUCTION.json',dict(states=identities,reconstruction_receipts=reproductions,encoder_bytes_independently_verified_at_model_construction=True,all_eval_and_buffers_unchanged=True,W_includes_BN_affine=True,counters_not_prediction_cause=True))
 summaries=[];mechanisms=[];per=[];intervals=[]
 for pool,ng,na in [('Catalogue',32,7),('PixBox',10,8)]:
  rr=read(O/(pool+'_PER_GROUP.csv'));ss=read(O/(pool+'_SUMMARY.csv'));mm=read(O/(pool+'_MECHANISM.csv'));ci=load(O/(pool+'_INTERVALS.json'))['rows'];assert len(rr)==ng*8 and len(mm)==ng*2 and len(ss)==8
  for s in ss:
   part=[v for v in rr if v['seed']==s['seed'] and v['state']==s['state']];assert len(part)==ng and len({v['group_id'] for v in part})==ng;assert int(s['AP_groups'])==na
   for field,key in [('AP','macro_AP_pp'),('recall','fixed_threshold_macro_recall_pp'),('FPR','macro_FPR_pp')]:
    values=[float(v[field]) for v in part if v[field]!=''];assert abs(sum(values)/len(values)*100-float(s[key]))<1e-10;checks+=1
   assert abs(max(float(v['FPR']) for v in part if v['FPR'])*100-float(s['worst_group_FPR_pp']))<1e-10
   for v in part:
    assert int(v['TP'])+int(v['Shadow_to_nonShadow'])==int(v['positive']);assert int(v['FP'])==int(v['nonShadow_to_Shadow'])==int(v['Surface_FP'])+int(v['Cloud_FP']);assert int(v['Surface_negative'])+int(v['Cloud_negative'])==int(v['negative']);assert float(v['threshold'])==.7693686485290528
    if not int(v['positive']):assert v['AP']==''
    checks+=4
  for v in mm:
   if v['AP00_pp']=='':assert all(v[k]=='' for k in ('AP10_pp','AP01_pp','AP11_pp','11-00','10-11','01-00','10-00','I','W','B'));continue
   a,b,c,d=(float(v['AP'+k+'_pp']) for k in ('00','10','01','11'));expected={'11-00':d-a,'10-11':b-d,'01-00':c-a,'10-00':b-a,'I':d-b-c+a,'W':((b-a)+(d-c))/2,'B':((c-a)+(d-b))/2}
   for k,e in expected.items():assert abs(float(v[k])-e)<1e-10;checks+=1
   assert abs(float(v['W'])+float(v['B'])-float(v['11-00']))<1e-10
  for v in ci:
   values=np.array([float(m[v['comparison']]) for m in mm if int(m['seed'])==v['seed'] and m[v['comparison']]!='']);assert len(values)==na;idx=np.random.default_rng(203018).integers(na,size=(10000,na));draw=np.array([sum(float(values[i]) for i in row)/na for row in idx]);assert abs(float(values.mean())-v['mean_pp'])<1e-10 and np.max(np.abs(np.quantile(draw,[.025,.975])-v['CI95_pp']))<1e-10;checks+=2
  summaries+=ss;mechanisms+=mm;per+=rr;intervals+=ci
 csvnames=['Catalogue_SUMMARY.csv','Catalogue_PER_GROUP.csv','Catalogue_MECHANISM.csv','PixBox_SUMMARY.csv','PixBox_PER_GROUP.csv','PixBox_MECHANISM.csv']
 save(O/'INDEPENDENT_FINAL_ACCEPTANCE.json',dict(status='passed',additional_scalar_and_hash_checks=checks,prior_local_independent_checks=550,all_original_AP_reproduced=True,encoder_and_tensor_source_checks=True,full42groups_retained_per_seed=True,no_positive_AP_NA=True,bootstrap10000_seed203018_reproduced=True,algebraic_not_training_causal_decomposition=True,no_new_forward_or_fit=True))
 t1=table(['池','seed','状态','AP组','宏AP%','池化AP%','固定阈值宏召回%','宏误报%','最大组误报%'],[[v['pool'],v['seed'],v['state'],v['AP_groups'],num(v['macro_AP_pp']),num(v['pooled_AP_pp']),num(v['fixed_threshold_macro_recall_pp']),num(v['macro_FPR_pp']),num(v['worst_group_FPR_pp'])] for v in summaries]);write(O/'TABLE1_FOUR_STATES.md',t1)
 t2=table(['池/组','正参考','seed','AP00%','AP10%','AP01%','AP11%','11−00pp','10−11pp','01−00pp','10−00pp','交互I pp','W pp','B pp'],[[v['pool']+'/'+tile(v['products']),v['positive'],v['seed']]+[num(v[k]) for k in ('AP00_pp','AP10_pp','AP01_pp','AP11_pp','11-00','10-11','01-00','10-00','I','W','B')] for v in mechanisms]);write(O/'TABLE2_ALL_GROUP_MECHANISM.md',t2)
 targets=[v for v in mechanisms if tile(v['products']) in ('32TML','40XDR','19FDV','54TWN')];assert len(targets)==8
 targettable=table(['池/组','正参考','seed','11−00pp','恢复原BN：10−11pp','只换BN：01−00pp','只换W：10−00pp','交互I pp'],[[v['pool']+'/'+tile(v['products']),v['positive'],v['seed']]+[num(v[k]) for k in ('11-00','10-11','01-00','10-00','I')] for v in targets]);write(O/'FOUR_DAMAGE_GROUPS.md',targettable)
 cite=table(['池','seed','差值','平均pp','配对相关组95%描述区间'],[[v['pool'],v['seed'],v['comparison'],num(v['mean_pp']),'['+', '.join(num(x) for x in v['CI95_pp'])+']'] for v in intervals]);write(O/'DESCRIPTIVE_INTERVALS.md',cite)
 cost=dict(neural_calls=4865,cap=5000,new_main_calls=sum(int(s.get('main_calls',0)) for s in stages),construction_calls_including_first_failed7=sum(int(s.get('construction_calls',0)) for s in stages)+7,warmup_calls=sum(int(s.get('warmup_calls',0)) for s in stages),training_calls=0,fits=0,active_stage_wall_seconds=load(D/'RUN_STATUS.json')['elapsed_seconds'],excluded_confirmed_stopped_idle_seconds=5585.188312530518,original_start_epoch=1792376465.008057,GPU_active_stage_cap_seconds=7200,remote_CPU_peak_bytes=max(int(s.get('rss_peak_bytes',0)) for s in stages),GPU_reserved_peak_bytes=max(int(s.get('gpu_reserved_peak_bytes',0)) for s in stages),local_CPU_peak_bytes=load(O/'LOCAL_ANALYSIS_STATUS.json')['peak_bytes'],local_CPU_analysis_seconds=load(O/'LOCAL_CPU_ACCEPTANCE.json')['elapsed_seconds'],local_archive_sha256=load(O/'DELIVERY_RECEIPT.json')['sha256'],local_archive_bytes=668404251,work_dirs_transferred=False,new_image_or_weight_downloads=0)
 assert cost['new_main_calls']+cost['construction_calls_including_first_failed7']+cost['warmup_calls']==4865
 cost['local_new_logical_bytes']=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());cost['combined_new_logical_bytes_before_summary']=cost['local_new_logical_bytes']+1342864907;assert cost['combined_new_logical_bytes_before_summary']<6_000_000_000;assert cost['local_CPU_peak_bytes']<16*1024**3 and cost['remote_CPU_peak_bytes']<16*1024**3 and cost['GPU_reserved_peak_bytes']<20*1024**3
 save(O/'COST_RECEIPT.json',cost)
 # Preserve prior local-only state audit writeup; final README is deliberately short.
 assert (O/'README.md').exists() and not (O/'README_STATE_AUDIT_ONLY.md').exists();shutil.copyfile(O/'README.md',O/'README_STATE_AUDIT_ONLY.md')
 readme='''# Phase203：decoder参数与BN运行状态的固定交换

00=原R；10=适配可学习参数+原BN统计；01=原参数+适配BN统计；11=原适配模型。W包含BN仿射参数。没有训练、拟合或新阈值；encoder逐字节不变，所有网络eval。六个00/11重建抽查分数最大差均0；550项本地独立核对及额外交付验收通过。

**数据结果。** 恢复原BN（10−11）的逐组平均AP变化：Catalogue两seed为−0.418/−1.427pp，PixBox为+4.184/−0.991pp；相应95%描述区间均跨零。7/8有影相关组分别重抽10000次，seed203018，两个seed不增加组数。全部32/10组继续参与误报分析，无影AP为NA，54TWN仅7个正参考。

四个重复损伤组：54TWN的10−11为+61.966/+34.630pp，01−00为−54.811/−35.505pp；32TML的10−11为−2.181/−6.993pp，01−00为+0.172/+3.901pp；19FDV的10−11为−4.448/+0.506pp，01−00为+3.274/+5.092pp；40XDR的10−11为+4.448/+2.349pp，但10−00仍为−1.974/−3.471pp。

**解释。** 54TWN呈现跨seed一致的BN状态相关损伤模式，但极稀疏支持限制推广；32TML/19FDV更支持可学习decoder变化相关，40XDR涉及W、B及交互。不能用单一BN原因解释全部四组。恢复原BN对35RPM/32NKF/48XWG/22WES/44SKJ在两seed均另有AP下降>1pp；Catalogue及PixBox宏召回相对11也均下降。不能据此推荐统一恢复BN，更不能声称训练时冻结BN必然更好。

全部状态使用原R阈值0.7693686485290528，仅描述错误流，不判新方案预算合格。PixBox379正/8190负是采样点参考，非全景误报；Catalogue655961正/32767527负。详见RESULT、TABLE1、TABLE2、全逐组CSV及区间表。W/B对称分解只是模型函数的代数分解，不是训练因果贡献百分比；非线性/共适应失配、512训练与1000推理归一化问题、参考语义机制仍unknown。历史已暴露池探索不是独立确认；未测E/V4，不自动启动下一phase。

累计4865调用（含全部工程故障），GPU阶段累计执行4785.80秒；已确认停机等待5585.19秒单列，原运行消耗未重置，不能称连续两小时内完成。全部小表/来源/校验/成本已落盘，权重与分数不入Git，无work_dirs同步。
'''
 (O/'README.md').write_text(readme,encoding='utf8')
 result='''# Phase203实验结果

## 1 数据结果

采用原R（Omni V1 RegNet单成员）、C122/C123 decoder末步，固定32 Catalogue development代表与PixBox16产品/10相关组。保留原Phase123/124共同支持；Catalogue655961正/32767527负，PixBox379正/8190负，忽略931个未定义参考点，无新增输入损失。未读53校准输入，未改变波段、窗口、归一化、精度、掩膜或阈值。全部encoder382张量状态保持原R，decoder46参数与15个BN buffer按精确白名单交换，不依据性能选择状态。

四状态总表（AP/召回/误报均为百分数；两seed分别列）：

'''+t1+'''
六个原00/11重建分数最大差均为0，原R/C122/C123在全部组的AP与支持精确复现。四个已知重复损伤组的比较：

'''+targettable+'''
完整84行逐组机制表在TABLE2_ALL_GROUP_MECHANISM.md，336行固定阈值错误流/分数中位数在两个PER_GROUP.csv。无正类AP均NA；每组同时列FN、FP以及Surface/Cloud负类分层。所有状态阈值同为0.7693686485290528，float64比较，未在新组合上校准。

恢复原BN（10−11）的宏AP变化和95%描述区间：Catalogue seed12201 −0.418 [−2.011,1.435]pp、seed12301 −1.427 [−3.628,0.401]pp；PixBox +4.184 [−7.042,21.891]pp、−0.991 [−10.101,10.770]pp。区间以相关组为单位，10000次，seed203018；7/8组，不把两个seed当独立组。全部七种对比及W/B代数分解区间均完整保留，无事后选择。

其他组的代价：恢复原BN时35RPM/32NKF/48XWG/22WES/44SKJ两seed均AP下降>1pp；其中44SKJ分别−14.190/−18.743pp、22WES−5.265/−14.602pp。Catalogue宏召回从11的45.141/41.051%降为10的42.633/36.617%；PixBox从43.258/40.847%降为40.013/31.671%。误报同时下降，排序/尺度/阈值效应不能混为一个概念。10的池化AP也不始终优于11：PixBox seed12201为84.150%对84.307%，seed12301为84.693%对85.971%。

执行与校验：4865次调用=主推理、构建与预热之和（成本JSON逐项）；无训练/拟合/新影像权重下载。三次工程修复保留全部失败、调用与原时间戳；已完成48景未重复推理。累计GPU阶段执行4785.80秒，退出后等待5585.19秒单列，不宣称连续2小时墙钟完成。CPU分析158.96秒，550项独立核对及最终标量/哈希复算通过，精确计数和内存峰值见COST_RECEIPT。受控shared包668404251bytes长度/SHA及325文件逐一通过，参考仅本地连接；这不构成work_dirs备份。

## 2 结论分析

1. **54TWN支持BN运行状态相关的损伤模式。** 两seed只换BN均显著降低AP，恢复原BN均大幅恢复，且仅W变化相对00反而更高。此组只有7正参考，少数点排序变动可产生很大的AP变化；它不能单独支持全数据或新区域结论。
2. **32TML与19FDV更支持可学习decoder变化相关。** 32TML的10比11更差，两seed的01均未造成AP下降；19FDV的10仍比00低18.697/15.123pp，而01均改善。BN无法解释这两组的全部损伤，19FDV恢复效应还随seed变化。
3. **40XDR含共同作用。** 恢复原BN有跨seed改善，但仅W变化仍受损，01单独也下降（第二seed幅度较小）；交互项约−1.981/−1.947pp。不能把它归于单一统计量，也不能把这些非线性差值换算成训练中BN的因果责任百分比。
4. **统一修复方案没有得到本轮支持。** 四组表现异质，恢复原BN对其他组造成代价，宏AP恢复随池/seed不稳定且召回下降。所有冻结阈值结果只是错误流描述，不是为新组合校准后的预算合格判定。

W与B共同适应，状态交换可能制造失配；本轮是现存模型函数的定位实验，不是训练时冻结BN的对照训练。encoder字节未更新排除该次encoder参数/缓冲更新，但不能回答输入归一化及encoder响应变化、标签语义或地物成因。历史已暴露评价池、极少相关组与54TWN稀疏参考限制均保留。未测双模型E/V4，不复活旧失败主检验，不自动追加状态/窗口/层级或启动下一phase。
'''
 write(O/'RESULT.md',result)
 save(O/'MISSING_AND_LIMITS.json',dict(required_output_missing=[],unmeasured=['training-time BN freeze effect','512/1000 normalization mechanism','reference or land-cover semantic cause','E adaptation','V4'],unknowns_retained=True,historically_exposed_pools=True,AP_related_groups={'Catalogue':7,'PixBox':8},sparse_54TWN_positives=7))
 S.mkdir(parents=True,exist_ok=True)
 names=['README.md','RESULT.md','TABLE1_FOUR_STATES.md','TABLE2_ALL_GROUP_MECHANISM.md','FOUR_DAMAGE_GROUPS.md','DESCRIPTIVE_INTERVALS.md','MODEL_SOURCES_AND_REPRODUCTION.json','LOCAL_CPU_ACCEPTANCE.json','INDEPENDENT_FINAL_ACCEPTANCE.json','COMMON_SUPPORT_ACCEPTANCE.json','COST_RECEIPT.json','MISSING_AND_LIMITS.json','DELIVERY_RECEIPT.json']+csvnames+['Catalogue_INTERVALS.json','PixBox_INTERVALS.json']
 for n in names:assert not (S/n).exists();shutil.copyfile(O/n,S/n)
 provenance=dict(protocol_sha256=lock['protocol_sha256'],execution_lock_sha256=sha(D/'EXECUTION_LOCK.json'),canonical_model_sources=lock['model_source_sha256'],input_manifest_sha256=sha(O/'INPUT_DELIVERY_LOCK.json'),source_sha256={p.name:sha(p) for p in (R/'src/tools').glob('*phase203*') if p.is_file()},shared_reference_labels_uploaded=0)
 save(O/'PROVENANCE.json',provenance);shutil.copyfile(O/'PROVENANCE.json',S/'PROVENANCE.json');names.append('PROVENANCE.json')
 done=dict(status='completed',neural_calls=4865,new_calls_after_remote_complete=0,new_fits=0,files={n:sha(O/n) for n in names},independent_final_checks=checks,no_automatic_next_phase=True)
 save(O/'DELIVERY_COMPLETE.json',done);shutil.copyfile(O/'DELIVERY_COMPLETE.json',S/'DELIVERY_COMPLETE.json')
 p=O/'EXECUTION_STATUS.json';previous=p.read_text(encoding='utf8');write(O/'EXECUTION_STATUS_BEFORE_COMPLETION.json',previous);p.write_text(json.dumps(dict(status='completed',neural_calls=4865,delivery_complete_sha256=sha(O/'DELIVERY_COMPLETE.json'),local_CPU_acceptance=True,independent_final_acceptance=True),indent=2)+'\n',encoding='utf8')
 h=O/'HANDOFF.md';old=h.read_text(encoding='utf8');h.write_text(old.replace('# Phase203 current delivery\n','# Phase203 current delivery\n\n## FINAL COMPLETE — 2026-10-19\nAll seven states, remoteEXIT0,4865calls, localTRANSFER SHA and325file hashes,550localchecks and independent finalscalar/hash/bootstrap checks passed. RESULT andREADME data/interpretation, TABLE1/TABLE2/known4groups/all336operatingrows/84mechanismrows/intervals/modelandcost/provenance delivered. Smallsummary src/research_results/phase203_20261018. No further modelcall/fit/restart. Deletephase203-hourly after finalnotification. Fullresults do NOT support universalBNcause; see group-specific interpretation. Oldnotesbelow historical.\n',1),encoding='utf8')
 print(json.dumps(dict(status='completed',independent_final_checks=checks,small_files=len(names),calls=4865,resource_cost=cost)))
if __name__=='__main__':main()
