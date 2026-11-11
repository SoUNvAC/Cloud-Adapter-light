"""Seal concise Phase225 data and interpretation separately; no next experiment."""
from pathlib import Path
import json,csv,hashlib,shutil,time
from analyze_phase216 import peak
R=next(p for p in Path(__file__).resolve().parents if (p/'src/tools').is_dir());O=R/'outputs/phase225/phase225_20261024';D=O/'delivery';S=R/'src/research_results/phase225_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert js(O/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed';lock=js(D/'EXPERIMENT_LOCK.json');complete=js(D/'COMPLETE.json');old={}
 for phase,v in lock['history'].items():
  p=R/f'outputs/phase{phase}/phase{phase}_{v["date"]}';m=js(p/'DELIVERY_MANIFEST.json');assert sha(p/'DELIVERY_MANIFEST.json')==v['manifest_SHA'] and all((p/n).stat().st_size==z['bytes'] and sha(p/n)==z['SHA'] for n,z in m.items());old[phase]={'unchanged_files':len(m),'manifest_SHA':v['manifest_SHA']}
 receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');combined=receipt['remote_scope_bytes']+receipt['bytes']+sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+(O/'bootstrap.zip').stat().st_size;resources=[js(D/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]]+[js(D/'evaluation/COMPLETE.json')];cpu=max(v['CPU_peak_bytes'] for v in resources);gpu=max(v['GPU_allocated_peak'] for v in resources);reserved=max(v['GPU_reserved_peak'] for v in resources);assert combined<4_000_000_000 and max(cpu,peak())<16*1024**3 and max(gpu,reserved)<20*1024**3 and shutil.disk_usage(R).free>=4*1024**3;assert complete['units']==124268 and complete['GPU_seconds']<21600
 heads={}
 for s in [21701,21702]:
  for a in ['R','Q']:
   p=D/f'seed{s}/heads/{a}_iter4000.pth';assert sha(p)==js(D/f'seed{s}/{a}_TRAIN_COMPLETE.json')['head_SHA'];heads[f'{s}/{a}']={'SHA':sha(p),'bytes':p.stat().st_size}
 dependency=js(D/'HISTORICAL_DEPENDENCY_COST.json');save('COST_AND_IDENTITIES.json',{'new_units':complete['units'],'GPU_seconds':complete['GPU_seconds'],'fits':4,'updates_each':4000,'counts':js(D/'REMOTE_CPU_ACCEPTANCE.json')['counts'],'legacy_counter_shared_D_gate_full':'actually Q shared-head/full-segmentor equality check, two units, name retained from sealed implementation','historical_dependency':dependency,'original222_whole_cost':js(D/'historical/PHASE222_COMPLETE_COST.json'),'CPU_remote_peak_bytes':cpu,'CPU_local_peak_bytes':peak(),'GPU_allocated_peak':gpu,'GPU_reserved_peak':reserved,'combined_new_persistent_upper_before_seal':combined,'new_head_only_delivery':heads,'old217_to224_unchanged':old,'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','experiment_lock_SHA':sha(D/'EXPERIMENT_LOCK.json'),'code_lock_SHA':sha(D/'CODE_LOCK.json'),'frozen_weights_SHA':sha(D/'weights/SHA256.json'),'source_selection_SHA':sha(D/'evaluation/SOURCE_SELECTION_LOCK.json'),'weight_use_SHA':sha(D/'WEIGHT_USE_ACCEPTED.json'),'historical_D_reproduction_SHA':sha(D/'evaluation/HISTORICAL_D_REPRODUCTION.json'),'initial_Source_training_cost':'unknown','no_new_timer_download_push_or_nextphase':True})
 aa=rows('TABLE_A_FIXED_POINTS.csv');ci=rows('PAIRED_CI.csv');coverage=rows('FINITE_DH_COVERAGE.csv');corr=rows('TEACHER_CORRECTNESS_ALPHA1.csv')
 table='\n'.join(f"|{v['alpha']}|{v['arm']}|{float(v['test_mIoU']):.3f}|{float(v['pool_loss_pp']):.3f}|{float(v['T11_pp']):.3f}|{float(v['worst_ROI_loss_pp']):.3f}|{float(v['ROI_loss_gt5']):.1f}|" for v in aa if v['seed']=='mean_of_two')
 def interval(other,context,key):
  v=next(v for v in ci if v['comparison']=='R-'+other and v['context']==context and v['scope']=='mean_of_two' and v['metric']==key);return f"{float(v['point']):+.3f} [{float(v['CI95_low']):+.3f},{float(v['CI95_high']):+.3f}]"
 bt='\n'.join(f"|R−{a}|{context}|{interval(a,context,'test_mIoU_diff_pp')}|{interval(a,context,'source_pool_loss_diff_pp')}|{interval(a,context,'T11_diff_pp')}|" for context in ['alpha025','alpha1'] for a in ['Q','D','H'])
 cover='；'.join(f"{s}/{a}：未被D/H任一固定点弱覆盖{sum(v['weakly_covered_by_any_DH_point']=='False' for v in coverage if v['seed']==str(s) and v['arm']==a)}/5" for s in [21701,21702] for a in ['R','Q'])
 ct='；'.join(f"{v['seed']} {v['old_arm']}→{v['new_arm']}：教师原正确少错{int(v['teacher_correct_new_fewer_errors']):+,}、原错误少纠正{int(v['teacher_wrong_new_fewer_corrections']):+,}" for v in corr if v['ROI']=='__pooled__' and v['old_arm']=='D' and v['new_arm']=='R')
 mainci=[v for v in ci if v['comparison']=='R-Q' and v['context']=='alpha025' and v['scope']=='mean_of_two'];t=next(v for v in mainci if v['metric']=='T11_diff_pp');q=next(v for v in mainci if v['metric']=='test_mIoU_diff_pp');tail='源尾部改善的区间全部低于0' if float(t['CI95_high'])<0 else '源尾部恶化的区间全部高于0' if float(t['CI95_low'])>0 else '源尾部差值区间跨零，方向尚不确定';target='目标收益区间全部高于0' if float(q['CI95_low'])>0 else '目标损伤区间全部低于0' if float(q['CI95_high'])<0 else '目标差值区间跨零，不能据此称等效'
 report=f'''# Phase225：冻结历史教师分歧的图像级回放加权

## 1．数据结果

仅L8，新增R/Q×seed21701/21702各4000步，从原Source初始化，只更新1,316,516 head参数。原222样本、增强、标签SHA、RNG、优化器、学习率和4000步源回放序列不变。λ=T=1；唯一loss变化是整图CSV标量×原均匀KL。R按冻结历史D分歧加权，Q在15/16重复次数层内固定打乱；256图/249 ROI，全程4000步图权重均值均1。没有重新排序或在线改权重。C/D来自222、H来自224，只读复用。

目标train1560输入/18scene，val963/8，完整test2167/16（yes6/no10）；源val535图/107 ROI。下表两seed分别算指标后平均，单位pp；最差ROI数值也为两seed各自最差损失均值，具体成员见全表。Source为α0。名义误报与各类支持/NA、六向错误和全部逐组结果随表交付。

|α|臂|test mIoU|源池化损失|T11|最差ROI损失|>5pp ROI数|
|---|---|---:|---:|---:|---:|---:|
{table}

固定配对差及95%区间如下；**主诊断为α.25的R−Q**。目标差越高越好，源损失/T11差越低越好。配对bootstrap10000、seed217023；源107 ROI、目标16scene，yes/no另抽6/10；同索引作用于两seed，不扩充独立组数。

|比较|固定工作点|test差|源池化损失差|T11差|
|---|---|---:|---:|---:|
{bt}

原B1/3/5且T11≤5最大可行α已先按源侧冻结；所有臂选择见SOURCE_RULE_ALL_ARMS.csv，选择点差及yes/no区间完整保留。全部R/Q各5点与D/H合计10点共200配对，含不利结果；{cover}。这是已看test的有限点描述，不是重新选择部署或连续前沿证明。α1联合243格/图×1070图独立逐格验收：{ct}。α.25只有净混淆差，不能借用α1直接转移。

实际新增{complete['units']:,}单位，GPU累计{complete['GPU_seconds']:.2f}秒；4次fit×4000步。本方法另依赖历史两次D训练：{dependency['total_D_training_units']:,}训练单位、记录训练时间{dependency['total_D_training_recorded_seconds']:.2f}秒；共享构建/门检不能独立归属，原222全轮成本另存，不重复相加。220教师类别暴露也为历史信息，原Source初训成本unknown。独立CPU验收、逐步权重/公式、目标SHA、冻结/原D复现、全部10000重采样、联合边际/正确性恒等式已通过；完整成本与输入/模型/配置/序列/代码SHA见COST_AND_IDENTITIES.json。

## 2．结论分析

固定低强度R−Q：{tail}；{target}。其余D/H对照及α1固定补充需一并阅读，不能从多个工作点挑最有利者替代主诊断。教师原正确少错与原错误少纠正揭示保护/纠错取舍，不能把更接近教师直接称为真实风险降低。

R/Q还改变教师预测类别暴露（Surface/Cloud/Shadow的R均值约.893/1.151/1.045，Q约.969/1.038/1.043），不构成排除所有类别和场景共变的纯因果比较；图权重均值1也不保证加权KL或梯度强度相同，H不是梯度匹配。区间不含训练seed总体、历史权重生成、重选α的不确定性。单目标探针不证明跨域泛化，不增加权重搜索或后续实验。
'''
 (O/'RESULT.md').write_text(report,encoding='utf-8');S.mkdir(parents=True,exist_ok=True)
 for n in ['RESULT.md','TABLE_A_FIXED_POINTS.csv','FIXED_COMPARISONS.csv','PAIRED_CI.csv','COST_AND_IDENTITIES.json','INDEPENDENT_CPU_ACCEPTANCE.json']:shutil.copy2(O/n,S/n)
 save('LOCAL_STATUS.json',{'stage':'complete','epoch':time.time(),'new_model_calls':0});(O/'HANDOFF.md').write_text('Current delivery: COMPLETE. Four original4000-step R/Q fits; all fixed evaluations and independent local CPU acceptance completed. No auto next phase. See RESULT.md.\n',encoding='utf-8');manifest={str(p.relative_to(O)):{'SHA':sha(p),'bytes':p.stat().st_size} for p in O.rglob('*') if p.is_file() and p.suffix.lower() not in ['.zip','.gz'] and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','RECEIVER.console.log','RECEIVER.error.log','LOCAL_EXIT_CODE'] and not p.name.endswith('.log')};save('DELIVERY_MANIFEST.json',manifest);save('DELIVERY_COMPLETE.json',{'status':'complete','manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'files':len(manifest),'epoch':time.time(),'independent_acceptance':'passed','no_next_phase':True});print(json.dumps({'status':'complete','files':len(manifest)}))
if __name__=='__main__':main()
