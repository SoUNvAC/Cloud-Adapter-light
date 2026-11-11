"""Seal SPARCS replication, with separate L8 descriptive comparison, no pooled claim."""
from pathlib import Path
import json,csv,hashlib,shutil,time
from analyze_phase216 import peak
R=next(p for p in Path(__file__).resolve().parents if (p/'src/tools').is_dir());O=R/'outputs/phase226/phase226_20261025';D=O/'delivery';S=R/'src/research_results/phase226_20261025'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def write(n,rr):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rr for k in v)));w.writeheader();w.writerows(rr)
def main():
 assert js(O/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed';lock=js(D/'EXPERIMENT_LOCK.json');complete=js(D/'COMPLETE.json');old={}
 for phase,v in lock['history'].items():
  p=R/f'outputs/phase{phase}/phase{phase}_{v["date"]}';m=js(p/'DELIVERY_MANIFEST.json');assert sha(p/'DELIVERY_MANIFEST.json')==v['manifest_SHA'] and all((p/n).stat().st_size==z['bytes'] and sha(p/n)==z['SHA'] for n,z in m.items());old[phase]={'unchanged_files':len(m),'manifest_SHA':v['manifest_SHA']}
 receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');combined=receipt['remote_scope_bytes']+receipt['bytes']+sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+(O/'bootstrap.zip').stat().st_size;resources=[js(D/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]]+[js(D/'evaluation/COMPLETE.json')];cpu=max(v['CPU_peak_bytes'] for v in resources);gpu=max(v['GPU_allocated_peak'] for v in resources);reserved=max(v['GPU_reserved_peak'] for v in resources);assert combined<4_000_000_000 and max(cpu,peak())<16*1024**3 and max(gpu,reserved)<20*1024**3 and shutil.disk_usage(R).free>=4*1024**3;assert complete['units']==74508 and complete['GPU_seconds']<14400
 heads={}
 for seed in [21701,21702]:
  for a in ['R','Q']:
   p=D/f'seed{seed}/heads/{a}_iter4000.pth';assert sha(p)==js(D/f'seed{seed}/{a}_TRAIN_COMPLETE.json')['head_SHA'];heads[f'{seed}/{a}']={'SHA':sha(p),'bytes':p.stat().st_size}
 dep=js(D/'HISTORICAL_DEPENDENCY_COST.json');save('COST_AND_IDENTITIES.json',{'new_units':complete['units'],'GPU_seconds':complete['GPU_seconds'],'fits':4,'updates_each':4000,'counts':js(D/'REMOTE_CPU_ACCEPTANCE.json')['counts'],'legacy_counter_shared_D_gate_full':'Q shared/full equality gate, two units; original sealed name retained','historical_dependency':dep,'original217_whole_cost':js(D/'historical/PHASE217_COMPLETE_COST.json'),'CPU_remote_peak_bytes':cpu,'CPU_local_peak_bytes':peak(),'GPU_allocated_peak':gpu,'GPU_reserved_peak':reserved,'combined_new_persistent_upper_before_seal':combined,'new_head_only_delivery':heads,'old217_to225_unchanged':old,'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','experiment_lock_SHA':sha(D/'EXPERIMENT_LOCK.json'),'code_lock_SHA':sha(D/'CODE_LOCK.json'),'frozen_weights_SHA':sha(D/'weights/SHA256.json'),'weight_use_SHA':sha(D/'WEIGHT_USE_ACCEPTED.json'),'source_selection_SHA':sha(D/'evaluation/SOURCE_SELECTION_LOCK.json'),'D_reproduction_SHA':sha(D/'evaluation/HISTORICAL_D_REPRODUCTION.json'),'initial_Source_training_cost':'unknown','no_timer_new_download_push_nextphase':True})
 aa=rows('TABLE_A_FIXED_POINTS.csv');ci=rows('PAIRED_CI.csv');cov=rows('FINITE_DH_COVERAGE.csv');corr=rows('TEACHER_CORRECTNESS_ALPHA1.csv');pm=rows('ALL_POOL_METRICS.csv')
 table='\n'.join(f"|{v['alpha']}|{v['arm']}|{float(v['test_mIoU']):.3f}|{float(v['pool_loss_pp']):.3f}|{float(v['T11_pp']):.3f}|{float(v['worst_ROI_loss_pp']):.3f}|{float(v['ROI_loss_gt5']):.1f}|" for v in aa if v['seed']=='mean_of_two')
 def interval(c,ctx,k):
  v=next(v for v in ci if v['comparison']==c and v['context']==ctx and v['scope']=='mean_of_two' and v['metric']==k);return f"{float(v['point']):+.3f} [{float(v['CI95_low']):+.3f},{float(v['CI95_high']):+.3f}]"
 bt='\n'.join(f"|R−{a}|{ctx}|{interval('R-'+a,ctx,'test_mIoU_diff_pp')}|{interval('R-'+a,ctx,'source_pool_loss_diff_pp')}|{interval('R-'+a,ctx,'T11_diff_pp')}|" for ctx in ['alpha025','alpha1'] for a in ['Q','D','H'])
 l8=R/'outputs/phase225/phase225_20261024';lc=list(csv.DictReader((l8/'PAIRED_CI.csv').open(encoding='utf-8-sig')));cross=[{'domain':domain,**v} for domain,vs in [('SPARCS',ci),('L8',lc)] for v in vs if v['comparison']=='R-Q' and v['context']=='alpha025' and v['scope']=='mean_of_two' and v['metric'] in ['T11_diff_pp','test_mIoU_diff_pp','source_pool_loss_diff_pp']];write('SEPARATE_DOMAIN_PRIMARY_COMPARISON.csv',cross)
 lpm=list(csv.DictReader((l8/'ALL_POOL_METRICS.csv').open(encoding='utf-8-sig')));decisions=[]
 for domain,vs in [('SPARCS',pm),('L8',lpm)]:
  for a in ['R','Q']:
   z=[v for v in vs if v['arm']==a and float(v['alpha'])==.25 and v['pool']=='test' and v['support']=='full'];assert len(z)==2;decisions.append({'domain':domain,'arm':a,'alpha':.25,'seed':'mean_of_two',**{k:sum(float(v[k]) for v in z)/2 for k in ['mIoU','Shadow_recall','nonShadow_to_Shadow_FPR']}})
 write('SEPARATE_DOMAIN_DECISION_COMPARISON.csv',decisions)
 l8decision='；'.join(f"{v['arm']}召回{100*v['Shadow_recall']:.3f}%/名义误报{100*v['nonShadow_to_Shadow_FPR']:.3f}%" for v in decisions if v['domain']=='L8')
 def conclusion(v,less):
  low,high=float(v['CI95_low']),float(v['CI95_high']);return '改善区间不跨零' if (high<0 if less else low>0) else '恶化区间不跨零' if (low>0 if less else high<0) else '区间跨零，方向不确定，非等效结论'
 judgments='；'.join(f"{v['domain']} {v['metric']}={float(v['point']):+.3f} [{float(v['CI95_low']):+.3f},{float(v['CI95_high']):+.3f}]：{conclusion(v,v['metric']!='test_mIoU_diff_pp')}" for v in cross if v['metric']!='source_pool_loss_diff_pp')
 cover='；'.join(f"{seed}/{a}：未被D/H任一固定点弱覆盖{sum(v['weakly_covered_by_any_DH_point']=='False' for v in cov if v['seed']==str(seed) and v['arm']==a)}/5" for seed in [21701,21702] for a in ['R','Q'])
 ct='；'.join(f"{v['seed']} D→R：教师原正确少错{int(v['teacher_correct_new_fewer_errors']):+,}、原错误少纠正{int(v['teacher_wrong_new_fewer_corrections']):+,}" for v in corr if v['ROI']=='__pooled__' and v['old_arm']=='D' and v['new_arm']=='R')
 decision=[]
 for a in ['D','H','R','Q']:
  z=[v for v in pm if v['arm']==a and float(v['alpha'])==.25 and v['pool']=='test'];decision.append(f"{a}召回{100*sum(float(v['Shadow_recall']) for v in z)/2:.3f}%、误报{100*sum(float(v['nonShadow_to_Shadow_FPR']) for v in z)/2:.3f}%")
 report=f'''# Phase226：图像级回放加权的SPARCS固定复核

## 1．数据结果

仅SPARCS：60训练scene密集监督，val/test各10scene；源val535图/107 ROI、源train回放256 RGB/249 ROI，4000步原序列。新增R/Q×21701/21702各4000步，从同一Source初始化，仅更新1,316,516 head参数。原217目标样本、增强、标签SHA、模型RNG、优化器/lr和原均匀KL函数固定，λ=T=1；唯一变化为冻结CSV标量乘整图KL。R来自SPARCS历史D教师分歧、Q同15/16回放层内固定打乱，沿用phase225命名空间；没有用L8权重或重新抽取。C/D来自217、H来自224，只读复用。

α为Source→末步head插值强度。下表两seed分别计算后平均，单位pp；最差ROI损失均值不表示同一ROI。T11为最差11个带符号损失均值，不截负数。

|α|臂|test mIoU|源池化损失|T11|最差ROI损失|>5pp ROI数|
|---|---|---:|---:|---:|---:|---:|
{table}

主比较固定为α.25 R−Q。下列目标差越高越好，源池化损失/T11差越低越好；10000次配对bootstrap、seed217023，同10scene/107ROI索引用于两seed，逐seed重算池化和尾部后平均。

|比较|固定α|test差[95%CI]|源池化损失差[95%CI]|T11差[95%CI]|
|---|---|---:|---:|---:|
{bt}

α.25实际决策（池化、再平均两seed）：{'；'.join(decision)}。源B1/3/5且T11≤5最大可行α已先锁后连test，所有选择点及辅助指标见全表。所有R/Q5点对D/H10点共200配对含不利结果；{cover}。这是已看test的有限点描述，不证明连续前沿。α1逐图J五轴/243格、1070图全部验收：{ct}。α.25仅净混淆，不借用α1直接转移。

实际新增{complete['units']:,}单位、GPU累计{complete['GPU_seconds']:.2f}秒；4次4000步。本配方依赖历史217两次D训练，另有{dep['total_D_training_units']:,}训练单位/{dep['total_D_training_recorded_seconds']:.2f}秒；共享构建/门检归属unknown，原217全轮成本另存不重复相加。220类别暴露为已有信息，原Source初训成本unknown。独立CPU验收通过：权重/原日志SHA、全部16000步目标和权重、函数/冻结、Source和旧D复现、全候选支持/NA/六向错误、T11/规则、完整重采样、五轴边际/直接转移恒等式。成本和所有输入/代码/模型/序列SHA见COST_AND_IDENTITIES.json。

## 2．结论分析

固定主比较与L8开发结果分别报告：{judgments}。L8原固定α.25决策为{l8decision}，与上述SPARCS代价分别比较。不合并总增益；两域共用107源ROI，不是两批新独立样本。完整SEPARATE_DOMAIN_PRIMARY_COMPARISON.csv、DECISION_COMPARISON及每seed原值保留。区间跨零不能称非劣/等效，也不据此自动关闭方向。

算法已看过L8开发结果、SPARCS test亦有历史暴露，本轮属于配方迁移复核，不是未暴露test独立确认。R/Q仍改变教师预测类别暴露：R约.911/1.123/1.051，Q约.987/1.018/1.007（Surface/Cloud/Shadow）；不是排除类别/场景共变的纯风险因果对照。教师分歧不是真实错误风险；保护教师原正确也可能减少教师错误纠正。权重均值1不保证加权KL/梯度匹配，H非梯度匹配。CI不含历史权重生成、训练seed总体或重选α不确定性；保留不利结果，不追加seed/权重/臂或后续phase。
'''
 (O/'RESULT.md').write_text(report,encoding='utf-8');S.mkdir(parents=True,exist_ok=True)
 for n in ['RESULT.md','TABLE_A_FIXED_POINTS.csv','FIXED_COMPARISONS.csv','PAIRED_CI.csv','SEPARATE_DOMAIN_PRIMARY_COMPARISON.csv','COST_AND_IDENTITIES.json','INDEPENDENT_CPU_ACCEPTANCE.json']:shutil.copy2(O/n,S/n)
 save('LOCAL_STATUS.json',{'stage':'complete','epoch':time.time(),'new_model_calls':0});(O/'HANDOFF.md').write_text('Current delivery: COMPLETE. SPARCS-only four R/Q4000-step fits, fixed all-point evaluation, independent local CPU acceptance and separate L8 descriptive comparison complete. No nextphase. See RESULT.md.\n',encoding='utf-8');manifest={str(p.relative_to(O)):{'SHA':sha(p),'bytes':p.stat().st_size} for p in O.rglob('*') if p.is_file() and p.suffix.lower() not in ['.zip','.gz'] and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','RECEIVER.console.log','RECEIVER.error.log','LOCAL_EXIT_CODE'] and not p.name.endswith('.log')};save('DELIVERY_MANIFEST.json',manifest);save('DELIVERY_COMPLETE.json',{'status':'complete','manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'files':len(manifest),'epoch':time.time(),'independent_acceptance':'passed','no_next_phase':True});print(json.dumps({'status':'complete','files':len(manifest)}))
if __name__=='__main__':main()
