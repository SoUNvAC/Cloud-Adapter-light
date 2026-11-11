"""Two-page baseline report: data separate from interpretation, no gamma selection."""
from pathlib import Path
import json,csv,hashlib,shutil,time
from analyze_phase216 import peak
R=next(p for p in Path(__file__).resolve().parents if (p/'src/tools').is_dir());O=R/'outputs/phase227/phase227_20261025';D=O/'delivery';S=R/'src/research_results/phase227_20261025'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def write(n,rr):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rr for k in v)));w.writeheader();w.writerows(rr)
def main():
 acceptance=js(O/'INDEPENDENT_CPU_ACCEPTANCE.json');assert acceptance['status']=='passed';lock=js(D/'EXPERIMENT_LOCK.json');complete=js(D/'COMPLETE.json');old={};failures={k:js(D/k/'FAILURE_COST.json') for k in ['bootstrap_omission_01','bootstrap_omission_02']};assert all(v['new_model_calls']==v['training_updates']==v['fits_started']==0 for v in failures.values())
 for phase,v in lock['history'].items():
  p=R/f'outputs/phase{phase}/phase{phase}_{v["date"]}';m=js(p/'DELIVERY_MANIFEST.json');assert sha(p/'DELIVERY_MANIFEST.json')==v['manifest_SHA'] and all((p/n).stat().st_size==z['bytes'] and sha(p/n)==z['SHA'] for n,z in m.items());old[phase]={'manifest_SHA':v['manifest_SHA'],'unchanged_files':len(m)}
 receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');combined=receipt['remote_scope_bytes']+receipt['bytes']+sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+(O/'bootstrap.zip').stat().st_size;res=[js(D/job/'TRAINING_COMPLETE.json') for job in lock['jobs']]+[js(D/domain/'evaluation/COMPLETE.json') for domain in ['SPARCS','L8']];cpu=max(v['CPU_peak_bytes'] for v in res);gpu=max(v['GPU_allocated_peak'] for v in res);reserved=max(v['GPU_reserved_peak'] for v in res);assert combined<4_000_000_000 and max(cpu,peak())<16*1024**3 and max(gpu,reserved)<20*1024**3 and shutil.disk_usage(R).free>=4*1024**3;assert complete['units']==102752 and complete['GPU_seconds']<21600
 save('COST_AND_IDENTITIES.json',{'new_units':102752,'archive_manifest_repair':receipt.get('archive_repair'),'CPU_only_comparison_iterator_fix':js(O/'POSTPROCESS_PREFLIGHT_FIX.json'),'preflight_omission_failures':failures,'preflight_import_CPU_peak':'unknown','GPU_seconds':complete['GPU_seconds'],'fits':8,'updates_each':4000,'gammas':[1,10],'formula':'FP32 0.5 SUM squared displacement of exactly1316516 parameters; no mean/epsilon/clip','counts':js(D/'REMOTE_CPU_ACCEPTANCE.json')['counts'],'training_new_source_RGB':0,'training_source_labels':0,'training_source_teacher_forward':0,'training_source_replay_cache':0,'evaluation_source_reference_images_per_candidate':535,'evaluation_source_ROIs':107,'evaluation_source_teacher_head_forward':1070,'extra_teacher_head_calls_after_lost_cache':535,'evaluation_failure_cost':js(D/'RESUME03_BUDGET.json'),'failed_evaluation_exact_CPU_GPU_peak':'unknown; periodic samples retained','source_val_used_for_alpha_selection':True,'original_Source_was_source_supervised':True,'not_SFUDA':True,'AdamW_original_weight_decay_groups':[0,.05],'legacy_counter_shared_D_gate_full':'W10 shared/full equality check,4 units across domains','historical_C_D_H_training_cost':js(D/'HISTORICAL_TRAINING_COST.json'),'historical_costs_not_added_twice':'C/D/H are reused historical fits; source initial pretraining cost unknown','initial_Source_training_cost':'unknown','CPU_remote_peak_bytes':cpu,'CPU_local_peak_bytes':peak(),'GPU_allocated_peak':gpu,'GPU_reserved_peak':reserved,'combined_new_persistent_upper_before_seal':combined,'anchor_head_SHA':sha(D/'anchor_head.pth'),'new_head_SHA_and_distance':acceptance['eight_numpy_endpoint_sum_and_distance_checks'],'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','experiment_code_lock_SHA':sha(D/'CODE_LOCK.json'),'experiment_input_sequence_lock_SHA':sha(D/'EXPERIMENT_LOCK.json'),'old217_to226_unchanged':old,'new_timer_download_push_nextphase':False})
 aa=rows('TABLE_A_FIXED_POINTS.csv');ci=rows('PAIRED_CI.csv');cov=rows('FINITE_DH_COVERAGE.csv');corr=rows('TEACHER_CORRECTNESS_ALPHA1.csv');ep=acceptance['eight_numpy_endpoint_sum_and_distance_checks'];write('ENDPOINT_DISTANCES.csv',[{'job_arm':k,**v} for k,v in ep.items()]);table='\n'.join(f"|{v['domain']}|{v['arm']}|{float(v['test_mIoU']):.3f}|{float(v['pool_loss_pp']):.3f}|{float(v['T11_pp']):.3f}|{float(v['worst_ROI_loss_pp']):.3f}|{float(v['ROI_loss_gt5']):.1f}|" for v in aa if v['seed']=='mean_of_two' and float(v['alpha'])==1)
 def interval(domain,comparison,key):
  v=next(v for v in ci if v['domain']==domain and v['comparison']==comparison and v['context']=='alpha1' and v['scope']=='mean_of_two' and v['metric']==key);return f"{float(v['point']):+.3f} [{float(v['CI95_low']):+.3f},{float(v['CI95_high']):+.3f}]"
 bt='\n'.join(f"|{domain}|{a}−{b}|{interval(domain,a+'-'+b,'test_mIoU_diff_pp')}|{interval(domain,a+'-'+b,'source_pool_loss_diff_pp')}|{interval(domain,a+'-'+b,'T11_diff_pp')}|" for domain in ['SPARCS','L8'] for a in ['W1','W10'] for b in ['D','H']);cover='；'.join(f"{domain}/{a}未被D/H任一固定点弱覆盖{sum(v['weakly_covered_by_any_DH_point']=='False' for v in cov if v['domain']==domain and v['arm']==a)}/10（两seed各5点）" for domain in ['SPARCS','L8'] for a in ['W1','W10'])
 conclusions=[]
 for domain in ['SPARCS','L8']:
  for a in ['W1','W10']:
   ds=[]
   for key,name,lower in [('test_mIoU_diff_pp','目标',False),('source_pool_loss_diff_pp','源平均损失',True),('T11_diff_pp','源尾部',True)]:
    v=next(v for v in ci if v['domain']==domain and v['comparison']==a+'-D' and v['context']=='alpha1' and v['scope']=='mean_of_two' and v['metric']==key);lo,hi=float(v['CI95_low']),float(v['CI95_high']);benefit=hi<0 if lower else lo>0;harm=lo>0 if lower else hi<0;ds.append(name+('改善区间不跨零' if benefit else '恶化区间不跨零' if harm else '差区间跨零，非等效结论'))
   conclusions.append(domain+'/'+a+'相对D：'+'、'.join(ds)+'。')
 text=f'''# Phase227：训练无源回放的head L2-SP式基线

## 1．数据结果

SPARCS train60scene、val/test各10；L8原yes train1560/18scene、val963/8、完整test2167/16（yes6/no10）；源val535图/107ROI。新增每目标×两个原seed×W1/W10共8次4000步，从同一Source初始化，仅更新原1,316,516个head参数。目标样本/增强/标签SHA/RNG/监督loss/AdamW参数组（weight_decay .05/0）/lr均与217/222匹配。Omega严格为FP32的0.5×所有head参数平方位移之和，gamma=1/10；不是mean、不重标定，初始Omega及锚定梯度均0，独立theta0不共享可写存储。保留原weight decay后额外锚定，不声称原论文全部细节复现。

原C/D/H只读复用。下表末步α1、两seed各自算指标后平均，单位pp。源损失=Source−候选；T11为最差11个带符号ROI损失均值。

|目标|臂|test mIoU|源池化损失|T11|最差ROI损失|>5pp ROI数|
|---|---|---:|---:|---:|---:|---:|
{table}

固定主比较为端点W1−D、W10−D；H补充同时列出。目标差高为好，源损失/T11差低为好。10000次bootstrap、seed217023，同10/16scene与107ROI索引用于两seed，逐seed重算后平均；L8 yes/no另列6/10组。

|目标|比较|test差[95%CI]|源池化损失差[95%CI]|T11差[95%CI]|
|---|---|---:|---:|---:|
{bt}

α.25及B1/3/5且T11≤5的源侧先锁工作点完整交付，不替代端点、不选gamma。全α三类IoU/分母/NA、召回、名义误报、混淆/六向错误/全ROI和L8 yes/no保留。两目标分别做全部400有限配对，含不利结果；{cover}。该已看test的描述不是重新部署选择或连续前沿证明。α1 J[参考,教师,D,W1,W10]共2140图/520020格独立验收，教师正确/错误分层直接转移及纠错代价见TABLE_C_DIRECT_ALPHA1、TEACHER_CORRECTNESS_ALPHA1；其余α仅净混淆。实际Omega/距离及目标/正则loss轨迹见LOSS_SUMMARY、ENDPOINT_DISTANCES，不用初始零梯度校准系数。

新增102752单位、GPU累计{complete['GPU_seconds']:.2f}秒。两次准备故障分别遗漏身份JSON与L8输入锁，均0调用/0更新退出；补齐原元数据恢复，已完成SPARCS不重跑，原日志保留，导入期CPU峰值unknown。评价还因W10源结果误命名停止；全部10170次源评价和GPU时间保留，源分数/联合计数只读复用，补535次教师head重建丢失缓存，计入原预算，失败期精确峰值unknown。**训练源RGB/标签/回放缓存/教师前向均0**，32000更新及8次零更新门检只用目标完整网络32008单位。全部训练结束后才源val评价/选α及教师诊断；正常535次共同教师head及故障丢失缓存后的535次补算均复用正式冻结特征，合计1070次，非额外整池网络。Source本身有源监督；历史C/D/H训练/回放成本单列，原Source初训成本unknown。包内验收CPU峰值字段并发写导致清单失配，原包保留、头/分数/指标原字节未改，仅修复交付清单；本地比较迭代器在分析前修正，零模型调用。独立CPU/NumPy验收通过，含参数集合/公式/梯度、优化器、32000步目标配对、访问阻断/冻结、Source/旧D复现、全候选/规则、完整bootstrap、联合边际/直接转移恒等式及全部SHA。成本/输入/模型/配置/序列/代码回执见COST_AND_IDENTITIES。

## 2．结论分析

{' '.join(conclusions)} 两系数和两个目标均完整保留，不合并总增益、不按有利工作点替代端点。参数锚定是否竞争力足够需同时看上述目标—源损伤取舍、逐组严重损伤与纠错代价。

这检验的是两个预先固定sum系数，不能据此宣布整个L2-SP有效或无效；若有竞争力，它构成经典强基线而非原创。训练无源回放不等于整个协议无源数据：源val用于后续选择，且目标训练有标签，因此不是SFUDA。test已有暴露、107源ROI地理独立性未证实，CI不含重选工作点或训练seed总体不确定性。跨零不称非劣/等效，不追加第三系数、seed或后续phase。
'''
 (O/'RESULT.md').write_text(text,encoding='utf-8');S.mkdir(parents=True,exist_ok=True)
 for n in ['RESULT.md','TABLE_A_FIXED_POINTS.csv','FIXED_COMPARISONS.csv','PAIRED_CI.csv','ENDPOINT_DISTANCES.csv','COST_AND_IDENTITIES.json','INDEPENDENT_CPU_ACCEPTANCE.json']:shutil.copy2(O/n,S/n)
 save('LOCAL_STATUS.json',{'stage':'complete','epoch':time.time(),'new_model_calls':0});(O/'HANDOFF.md').write_text('Current delivery: COMPLETE. Eight4000-step W1/W10 fits with no source replay, both target domains all fixed points, independent CPU/NumPy acceptance complete. See RESULT.md; no nextphase.\n',encoding='utf-8');manifest={str(p.relative_to(O)):{'SHA':sha(p),'bytes':p.stat().st_size} for p in O.rglob('*') if p.is_file() and p.suffix.lower() not in ['.zip','.gz'] and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','RECEIVER.console.log','RECEIVER.error.log','LOCAL_EXIT_CODE'] and not p.name.endswith('.log')};save('DELIVERY_MANIFEST.json',manifest);save('DELIVERY_COMPLETE.json',{'status':'complete','manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'files':len(manifest),'epoch':time.time(),'independent_acceptance':'passed','no_next_phase':True});print(json.dumps({'status':'complete','files':len(manifest)}))
if __name__=='__main__':main()
