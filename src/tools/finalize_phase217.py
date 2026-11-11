"""Phase217: bounded separated report, resource receipts and SHA delivery seal."""
from pathlib import Path
import csv,json,hashlib,shutil,statistics,time
from analyze_phase216 import peak
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase217/phase217_20261023';RAW=OUT/'delivery';SUMMARY=ROOT/'src/research_results/phase217_20261023'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert js(OUT/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed'
 complete=js(RAW/'COMPLETE.json');assert complete['units']==48893 and complete['GPU_seconds']<14400
 endpoints=rd('ENDPOINTS.csv');joint=rd('JOINT_RULE.csv');ci=rd('PAIRED_CI.csv');candidates=rd('ALL_CANDIDATES.csv');common=rd('COMMON_DAMAGE_ROIS.csv');worst=rd('WORST_ROIS.csv');dom=rd('FINITE_SET_DOMINANCE.csv')
 def interval(context,scope,metric):
  r=next(x for x in ci if x['context']==context and x['scope']==scope and x['metric']==metric)
  return f"{float(r['point']):+.3f} [{float(r['CI95_low']):+.3f}, {float(r['CI95_high']):+.3f}]"
 def row(r):
  return f"| {r['seed']} | {r['C_alpha']} / {r['D_alpha']} | {float(r['C_test_mIoU']):.3f} / {float(r['D_test_mIoU']):.3f} | {float(r['D_minus_C_test_mIoU_pp']):+.3f} | {float(r['C_pool_loss_pp']):.3f} / {float(r['D_pool_loss_pp']):.3f} | {float(r['C_T11_pp']):.3f} / {float(r['D_T11_pp']):.3f} | {float(r['C_worst_ROI_loss_pp']):.3f} / {float(r['D_worst_ROI_loss_pp']):.3f} |"
 et='\n'.join(row(r) for r in endpoints)
 same_budgets=all(len({tuple((k,v) for k,v in r.items() if k!='context') for r in joint if r['seed']==s})==1 for s in {r['seed'] for r in joint})
 displayjoint=[r for r in joint if r['context']=='B1'] if same_budgets else joint
 jt='\n'.join(f"| {'1/3/5' if same_budgets else r['context']} | "+row(r)[2:] for r in displayjoint)
 displaycontexts=['endpoint_alpha1','B1'] if same_budgets else ['endpoint_alpha1','B1','B3','B5']
 it='\n'.join(f"| {'B1/3/5' if same_budgets and ctx=='B1' else ctx} | {interval(ctx,'mean_of_two','test_mIoU_diff_pp')} | {interval(ctx,'mean_of_two','source_pool_loss_diff_pp')} | {interval(ctx,'mean_of_two','T11_diff_pp')} |" for ctx in displaycontexts)
 loss={};train=[];resources=[]
 for s in [21701,21702]:
  resources.append(js(RAW/f'seed{s}/TRAINING_COMPLETE.json'))
  for a in ['C','D']:
   r=js(RAW/f'seed{s}/{a}_TRAIN_COMPLETE.json');train.append({'seed':s,'arm':a,**r})
   ls=[json.loads(x) for x in (RAW/f'seed{s}/{a}_LOSSES.jsonl').read_text().splitlines()]
   loss[f'{s}/{a}']={k:{'first100':statistics.mean(x[k] for x in ls[:100]),'last100':statistics.mean(x[k] for x in ls[-100:]),'min':min(x[k] for x in ls),'max':max(x[k] for x in ls)} for k in ['target_loss','source_KL','total_loss']}
 save('LOSS_SUMMARY.json',loss)
 costs=[]
 for s in [21701,21702]:
  c=next(r for r in train if r['seed']==s and r['arm']=='C')['train_seconds'];d=next(r for r in train if r['seed']==s and r['arm']=='D')['train_seconds'];costs.append({'seed':s,'C_seconds':c,'D_seconds':d,'extra_seconds':d-c,'D_over_C':d/c})
 gpu=complete['GPU_seconds'];resources.append(js(RAW/'evaluation/COMPLETE.json'));gpu_peak=max(x['GPU_allocated_peak'] for x in resources);reserved_peak=max(x['GPU_reserved_peak'] for x in resources);cpu_peak=max(x['CPU_peak_bytes'] for x in resources)
 assert max(gpu_peak,reserved_peak)<20*1024**3 and cpu_peak<16*1024**3
 remote=js(OUT/'REMOTE_DELIVERY_RECEIPT.json');localbytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());uploaded_code_archive_upper=sum(p.stat().st_size for p in OUT.glob('*.zip'));combined=remote['remote_scope_bytes']+remote['bytes']+localbytes+uploaded_code_archive_upper
 assert combined<4_000_000_000 and shutil.disk_usage(ROOT).free>4*1024**3 and peak()<16*1024**3
 save('COST_AND_IDENTITY.json',{'units':48893,'unit_breakdown':js(RAW/'REMOTE_CPU_ACCEPTANCE.json')['calls'],'GPU_stage_seconds':gpu,'GPU_allocated_peak':gpu_peak,'GPU_reserved_peak':reserved_peak,'remote_CPU_peak_bytes':cpu_peak,'per_seed_training':costs,'mean_extra_training_seconds':statistics.mean(r['extra_seconds'] for r in costs),'worst_extra_training_seconds':max(r['extra_seconds'] for r in costs),'remote_scope_bytes':remote['remote_scope_bytes'],'remote_archive_bytes':remote['bytes'],'local_scope_bytes_before_final_seal':localbytes,'combined_persistent_bytes_before_final_seal':combined,'persistent_cap_bytes':4_000_000_000,'local_peak_working_set_bytes':peak(),'local_free_bytes':shutil.disk_usage(ROOT).free,'new_local_model_calls':0,'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','old216_manifest_SHA':js(OUT/'LOCAL_PREFLIGHT.json')['phase216_manifest_SHA'],'two_locked_target_sequences':js(RAW/'BOTH_TARGET_SEQUENCES_LOCKED.json'),'all_four_training_receipts':train,'evaluation':js(RAW/'evaluation/COMPLETE.json'),'no_additional_training_or_seed':True})
 accounted=js(OUT/'COST_AND_IDENTITY.json');accounted['remote_uploaded_code_archives_upper_bytes']=uploaded_code_archive_upper;accounted['combined_is_upper_bound_including_uploaded_packages']=True;save('COST_AND_IDENTITY.json',accounted)
 rawend=[r for r in endpoints if r['seed']!='mean_of_21701_21702'];mean=endpoints[-1]
 same_source=all(float(r['D_minus_C_pool_loss_pp'])<0 and float(r['D_minus_C_T11_pp'])<0 for r in rawend)
 endpoint_text=('两个新seed在完整端点均减少源池化损失和T11，216的保源方向在这两次目标训练中重现。' if same_source else '两个新seed未同时重现源池化损失与T11下降；必须保留方向差异，216的保源现象不能称已重复。')
 target_text='；'.join(f"seed{r['seed']}目标差{float(r['D_minus_C_test_mIoU_pp']):+.3f}pp" for r in rawend)
 budget_details=[]
 for b in ['B1','B3','B5']:
  rs=[r for r in joint if r['context']==b and r['seed']!='mean_of_21701_21702'];budget_details.append(f"{b}两seed目标差"+' / '.join(f"{float(r['D_minus_C_test_mIoU_pp']):+.3f}" for r in rs)+'pp')
 dominance='；'.join(f"seed{s}被D有限二维点覆盖的C α="+str(sorted({float(r['C_alpha']) for r in dom if r['seed']==str(s) and r['D_dominates_C']=='True'})) for s in [21701,21702])
 ccom=sum(r['context']=='alpha1' and r['arm']=='C' for r in common);dcom=sum(r['context']=='alpha1' and r['arm']=='D' for r in common)
 report=f'''# Phase217：固定源预测保持的两seed配对复核

## 1．数据结果

新增seed21701／21702；每seed C普通目标监督、D同监督＋源预测KL(λ=1,T=1)，各4000步、相同1,316,516个head参数、同初始Source。仅末步；SPARCS训练60景全部密集标签，val10／test10分别评价。两seed增强序列均开训前锁定，C/D逐步配对；D额外使用同216的256张源train RGB和同4000次回放，零源train标签。Source原监督train池8490、已知parent4000步，继承初始化更早成本未知。源评价535图／107 ROI，CloudSEN官方test未读。本轮未覆盖Source／回放池随机性，目标test此前已查看，属于探索复核。

mIoU单位%，损失单位pp=Source−候选，负数不截断；T11为107 ROI最差11个带符号损失均值。平均行先按seed计算指标再等权平均，α栏按21701／21702顺序列出。

| seed | C / D α | C / D test mIoU | D−C目标pp | C / D源池化损失 | C / D T11 | C / D最差ROI损失 |
|---|---|---|---:|---|---|---|
{et}

原联合规则只用源数据，池化损失≤B且T11≤5，取最大可行α；选择锁先于目标结果连接。以下每个预算完整保留，同选工作点不是多次独立证据。

| B | seed | C / D α | C / D test mIoU | D−C目标pp | C / D源池化损失 | C / D T11 | C / D最差ROI损失 |
|---|---|---|---|---:|---|---|---|
{jt}

两seed配对差均值及条件95%区间：共享10 scene／107 ROI抽样索引，逐seed聚合混淆重算指标再平均差值；源T11每次重排。10000次，seed217023。不会扩成20 scene或214 ROI。

| 工作点 | 目标mIoU差pp | 源池化损失差pp | T11差pp |
|---|---|---|---|
{it}

端点共同损伤>5pp ROI：C={ccom}、D={dcom}。各seed最差ROI、共同名单及旧216最差ROI追踪均完整保存，不从ID推断地物。全20候选、60池化行、2540逐组行、六方向错误流及50对有限二维支配见CSV；val／test不混合，21601单列历史，不进入新seed均值。

实际48,893单位调用：目标训练16,000、D源backbone／教师head／学生head各8,000，门检10，候选评价8,880，公共Source三次抽查3。GPU阶段{gpu:.1f}s；C/D训练秒数为{costs[0]['C_seconds']:.1f}/{costs[0]['D_seconds']:.1f}及{costs[1]['C_seconds']:.1f}/{costs[1]['D_seconds']:.1f}，D额外均值{statistics.mean(r['extra_seconds'] for r in costs):.1f}s、最差{max(r['extra_seconds'] for r in costs):.1f}s。调用≤60,000、GPU≤4h，资源采样／宿主余量和合计新增字节见成本回执。四fit、初始KL／梯度／冻结、8000步标签裁剪翻转、全插值、Source复现、参考行和与36组区间均独立CPU验收；训练warn-only固定，评价显式恢复原非确定性CuDNN设置，不宣称位级可重复。

## 2．结论探讨

{endpoint_text}{target_text}。均值目标差{float(mean['D_minus_C_test_mIoU_pp']):+.3f}pp，源池化损失差{float(mean['D_minus_C_pool_loss_pp']):+.3f}pp、T11差{float(mean['D_minus_C_T11_pp']):+.3f}pp。未观测到明显的端点目标代价，但本轮目标差区间跨零，不能称目标收益成立、等效或非劣。D端点T11为10.426／11.054，均未通过5pp约束；最差ROI均为ROI_0714，损失17.329／16.550pp，保源改善不等于每ROI安全。

联合规则的目标收益未在两seed同向重现：21701为+3.492pp，21702为−0.257pp。D(α=0.5)的T11在21701为4.988而在21702为5.081，后者按原5pp门限正确排除，回退到0.25；不放宽门限或补α。两seed均值虽为+1.617pp、条件区间[0.667,2.769]，不能据此称训练seed间收益稳定。21701所选D的源池化损失／T11／最差ROI均高于所选C；预算通过也不保证源损伤更低。三个预算重复同一工作点。

{dominance}。这是有限点估计支配，不代表连续前沿。旧C最差ROI_0734在新D中损失5.388／6.196pp；旧D最差ROI_4813仍损伤9.521／11.455pp，且比新C更差。完整名单不隐藏残余或转移损伤，不从ID推断地物。

额外256源RGB、每seed4000次回放及约51.7%训练时间增量，换来两次同向、幅度较大的端点保源改善，值得保留为当前配方的保源基线；联合规则下更高目标收益尚未两seed重复，不能作为稳定部署收益结论。区间条件于实际两模型、固定Source／回放池和冻结α，不含训练seed总体或重新选择的不确定性。回放图与CloudSEN官方test有1个同产品、无同ROI，跨EPSG地理关系未知；完整独立性未宣称。不追加seed、λ、网格或新方法。
'''
 (OUT/'RESULT.md').write_text(report,encoding='utf-8')
 SUMMARY.mkdir(parents=True,exist_ok=True)
 for n in ['RESULT.md','ENDPOINTS.csv','ALL_CANDIDATES.csv','JOINT_RULE.csv','PAIRED_CI.csv','COMMON_DAMAGE_ROIS.csv','WORST_ROIS.csv','HISTORICAL216_WORST_ROI_TRACK.csv','FINITE_SET_DOMINANCE.csv','COST_AND_IDENTITY.json','INDEPENDENT_CPU_ACCEPTANCE.json','LOSS_SUMMARY.json']:shutil.copy2(OUT/n,SUMMARY/n)
 (OUT/'HANDOFF.md').write_text('Current delivery: COMPLETE. Four fixed fits and evaluation sealed; independent CPU acceptance passed. No follow-up training, timer or automatic push. See RESULT.md and DELIVERY_MANIFEST.json.\n',encoding='utf-8')
 # Snapshot only immutable research deliverables, never a still-open receiver log/status.
 names=[p for p in RAW.rglob('*') if p.is_file()]+[p for p in (OUT/'local_code').rglob('*') if p.is_file()]+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in ['LOCAL_STATUS.json','RECEIVER_STARTED.json','DELIVERY_COMPLETE.json','DELIVERY_MANIFEST.json']]
 manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in names}
 save('DELIVERY_MANIFEST.json',manifest)
 assert all((OUT/n).stat().st_size==r['bytes'] and sha(OUT/n)==r['SHA'] for n,r in manifest.items())
 save('DELIVERY_COMPLETE.json',{'status':'passed','epoch':time.time(),'artifacts':len(manifest),'manifest_SHA':sha(OUT/'DELIVERY_MANIFEST.json'),'units':48893,'GPU_seconds':gpu,'no_followup_phase':True})
 print(json.dumps({'status':'complete','report':str(OUT/'RESULT.md'),'endpoint_means':mean},ensure_ascii=False))
if __name__=='__main__':main()
