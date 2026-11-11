"""Measured endpoint/intervention report and complete immutable221 delivery seal."""
from pathlib import Path
import csv,json,hashlib,statistics,shutil,time
from analyze_phase216 import peak
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/phase221/phase221_20261024';RAW=O/'delivery';SUMMARY=ROOT/'src/research_results/phase221_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert js(O/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed' and js(O/'INDEPENDENT_JOINT_ACCEPTANCE.json')['status']=='passed';complete=js(RAW/'COMPLETE.json');assert complete['units']==75018 and complete['GPU_seconds']<14400
 endpoint=rows('TABLE_A_ENDPOINTS.csv');diff=rows('TABLE_B_DIFFERENCES.csv');ci=rows('PAIRED_CI.csv');direct=rows('TABLE_C_POOLED.csv');weight=rows('CLASS_WEIGHT_DISTRIBUTIONS.csv');chosen=rows('JOINT_RULE.csv');old={}
 for phase,date in [(217,'20261023'),(218,'20261023'),(219,'20261023'),(220,'20261024')]:
  d=ROOT/f'outputs/phase{phase}/phase{phase}_{date}';m=js(d/'DELIVERY_MANIFEST.json');assert sha(d/'DELIVERY_MANIFEST.json')==js(d/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((d/n).stat().st_size==r['bytes'] and sha(d/n)==r['SHA'] for n,r in m.items());old[str(phase)]={'manifest_SHA':sha(d/'DELIVERY_MANIFEST.json'),'all_unchanged':len(m)}
 receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');localbytes=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());uploaded=sum(p.stat().st_size for p in O.glob('*.zip'));combined=receipt['remote_scope_bytes']+receipt['bytes']+localbytes+uploaded;resources=[js(RAW/'weight_preparation/COMPLETE.json'),js(RAW/'evaluation/COMPLETE.json')]+[js(RAW/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]];cpu=max(v['CPU_peak_bytes'] for v in resources);alloc=max(v['GPU_allocated_peak'] for v in resources);reserved=max(v['GPU_reserved_peak'] for v in resources)
 assert combined<4_000_000_000 and cpu<16*1024**3 and max(alloc,reserved)<20*1024**3 and peak()<16*1024**3 and shutil.disk_usage(ROOT).free>4*1024**3
 heads={}
 for s in [21701,21702]:
  for a in ['N','P']:
   p=RAW/f'seed{s}/heads/{a}_iter4000.pth';assert sha(p)==js(RAW/f'seed{s}/{a}_TRAIN_COMPLETE.json')['head_SHA'];heads[str(p.relative_to(O))]={'SHA':sha(p),'bytes':p.stat().st_size}
 losses={}
 for s in [21701,21702]:
  for a in ['N','P']:
   lr=[json.loads(v) for v in (RAW/f'seed{s}/{a}_LOSSES.jsonl').read_text().splitlines()];losses[f'{s}/{a}']={k:{'first100':statistics.mean(v[k] for v in lr[:100]),'last100':statistics.mean(v[k] for v in lr[-100:]),'min':min(v[k] for v in lr),'max':max(v[k] for v in lr)} for k in ['target_loss','source_KL_unweighted','source_KL_weighted','total_loss']}
 save('LOSS_SUMMARY.json',losses);save('COST_AND_IDENTITIES.json',{'new_units':75018,'counts':js(RAW/'REMOTE_CPU_ACCEPTANCE.json')['calls'],'GPU_stage_seconds':complete['GPU_seconds'],'new_fits':4,'steps_per_fit':4000,'historical_C_D_G_S_fits':0,'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','new_heads_only_no_full_backbone_transferred':heads,'GPU_allocated_peak_bytes':alloc,'GPU_reserved_peak_bytes':reserved,'remote_CPU_peak_bytes':cpu,'local_CPU_peak_bytes':peak(),'combined_persistent_upper_bytes_before_final_seal':combined,'remote_scope_bytes':receipt['remote_scope_bytes'],'remote_archive_bytes':receipt['bytes'],'local_scope_bytes_before_seal':localbytes,'uploaded_packages_upper_bytes':uploaded,'new_persistent_cap_bytes':4_000_000_000,'local_free_bytes':shutil.disk_usage(ROOT).free,'old217_to220_unchanged':old,'new_code_lock_SHA':sha(RAW/'CODE_LOCK.json'),'weights_manifest_SHA':sha(RAW/'weight_preparation/WEIGHT_MANIFEST.json'),'source_selection_SHA':sha(RAW/'evaluation/SOURCE_SELECTION_LOCK.json'),'source_training_reference_reads':0,'official_source_test_reads':0,'no_new_timer_download_auto_push_or_next_phase':True,'earlier_Source_initialization_training_cost':'unknown'})
 def selected(c,metric):return next(v for v in ci if v['comparison']==c and v['context']=='endpoint_alpha1' and v['scope']=='mean_of_two' and v['metric']==metric)
 def interval(c,k):
  v=selected(c,k);return f"{float(v['point']):+.3f} [{float(v['CI95_low']):+.3f},{float(v['CI95_high']):+.3f}]"
 at='\n'.join(f"|{v['seed']}|{v['arm']}|{float(v['test_mIoU']):.3f}|{float(v['pool_loss_pp']):.3f}|{float(v['T11_pp']):.3f}|{float(v['worst_ROI_loss_pp']):.3f}|{v['ROI_loss_gt5']}|" for v in endpoint)
 bt='\n'.join(f"|{c}|{interval(c,'test_mIoU_diff_pp')}|{interval(c,'source_pool_loss_diff_pp')}|{interval(c,'T11_diff_pp')}|{float(next(v['Shadow_error_delta_pixels'] for v in diff if v['comparison']==c and v['seed']=='mean_of_two')):+,.1f}|" for c in ['N-G','N-D','N-P'])
 ct='\n'.join(f"|{v['seed']}|{v['band']}|{int(v['D_wrong_N_right']):,}|{int(v['D_right_N_wrong']):,}|{int(v['N_minus_D_errors']):+,}|{int(v['teacher_original_correct_net_errors']):+,}|" for v in direct if v['reference_class']=='Shadow')
 rp=[v for v in weight if v['pool']=='replay256' and v['repetition']=='original4000_sequence'];rpt='；'.join(f"{v['class']} G/N/P={float(v['G_mean_weight']):.4f}/{float(v['N_mean_weight']):.4f}/{float(v['P_mean_weight']):.4f}" for v in rp);rv=next(v for v in weight if v['pool']=='source_val535' and v['basis']=='reference' and v['class']=='Shadow')
 choices='；'.join(f"{s}{a}="+'/'.join(next(v['alpha'] for v in chosen if v['seed']==str(s) and v['arm']==a and v['budget_pp']==str(b)) for b in [1,3,5]) for s in [21701,21702] for a in ['N','P'])
 judgments=[]
 for c in ['N-G','N-D','N-P']:
  rr=[v for v in diff if v['comparison']==c and v['seed']!='mean_of_two'];parts=[]
  for key,label,lower in [('Shadow_error_delta_pixels','参考Shadow错误',True),('source_pool_loss_diff_pp','源池化损失',True),('T11_diff_pp','尾部T11',True),('test_mIoU_diff_pp','目标mIoU',False)]:
   vals=[float(v[key]) for v in rr];ok=all(v<0 for v in vals) if lower else all(v>0 for v in vals);bad=all(v>0 for v in vals) if lower else all(v<0 for v in vals);parts.append(label+('两seed均改善' if ok else '两seed均恶化' if bad else '两seed方向不同或含零差'))
  judgments.append(c+'：'+'、'.join(parts)+'。')
 target='；'.join(c+'目标差两seed='+('/'.join(f"{float(v['test_mIoU_diff_pp']):+.3f}" for v in diff if v['comparison']==c and v['seed']!='mean_of_two'))+'pp' for c in ['N-G','N-D','N-P'])
 report=f'''# Phase221：教师预测类别归一化的置信度保持

## 1．数据结果

新增N/P各两原seed、4000步，只更新同1,316,516个head参数；λ=T=1，从原Source重启。N将G按每图教师预测类归一化到均值约1，P在类内固定打乱同直方图。CPUfloat64计算再转float32，无epsilon/clip/重标定。C/D/G/S复用历史。原SPARCS train60密集监督、val10/test10；源回放256 RGB不读标签，源val535/107 ROI。输入/增强/RNG和4000回放序列逐步配对。目标test已查看，仍属探索。

表A为α=1；mIoU为%，源损失=Source−候选，T11为最差11 ROI带符号损失均值，单位pp。

|seed|臂|目标test mIoU|源池化损失|T11|最差ROI损失|>5pp ROI数|
|---|---|---:|---:|---:|---:|---:|
{at}

表B为两seed指标差均值及条件95%区间；10 scene/107 ROI同索引bootstrap10000次seed217023，逐seed重算后平均，T11每次重排。每seed原值见CSV；不扩充组数，不含重新选择或训练seed总体不确定性。

|比较|目标差pp [95%CI]|源损失差pp [95%CI]|T11差pp [95%CI]|Shadow净错误差均值|
|---|---|---|---|---:|
{bt}

表C摘录参考Shadow，D/N/P同像元直接计数；其余类别、教师对错分层和全107 ROI完整交付。

|seed|置信区|D错→N对|D对→N错|N−D净错误|教师原正确区域净错误|
|---|---|---:|---:|---:|---:|
{ct}

原4000回放序列加权的教师预测类平均权重：{rpt}。源val参考Shadow的N/P平均权重为{float(rv['N_mean_weight']):.4f}/{float(rv['P_mean_weight']):.4f}，不要求等于1。唯一256图、val两种类别依据、类支持和训练KL贡献另交CSV；预测类不是参考类，KL贡献不是梯度归因。

原源池B1/3/5且T11≤5规则不变，先锁源侧α再连接目标。N/P所选α：{choices}。全60候选、180池化行、7620逐组行、联合工作点、直接N/P计数、旧四焦点和13个事后ROI附表保留。实际75,018调用（含两次共享P正式路径门检），GPU阶段{complete['GPU_seconds']:.1f}s；新增持久上界{combined/1e9:.3f}GB。四新head已交付，未复制完整backbone。权重/边界/RNG、初始化/冻结、全16000步、插值、Source及历史D、联合边际和bootstrap独立CPU验收通过；旧217–220文件SHA未变。

## 2．结论探讨

{''.join(judgments)} {target}。区间跨零只表示该条件下方向仍不确定，不称等效或自动关闭。

N−G回答修复原加权损伤，N−D才回答是否超过均匀KL；仅回到D附近不能宣称超越D。N−P控制每类权重直方图后检查空间对应的增量。归一化同时改变类间总权重及类内权重绝对幅度，不能完全识别单一机制；教师预测类别平衡也不保证真实类别或梯度平衡。D/N/P转移可直接计算，N/G只用净混淆变化，不冒充G→N逐像元转移。焦点子集不替代107 ROI，不追加seed/阈值/网格或下一phase。
'''
 (O/'RESULT.md').write_text(report,encoding='utf-8');SUMMARY.mkdir(parents=True,exist_ok=True)
 for name in ['RESULT.md','TABLE_A_ENDPOINTS.csv','TABLE_B_DIFFERENCES.csv','TABLE_C_POOLED.csv','CLASS_WEIGHT_DISTRIBUTIONS.csv','TRAINING_CLASS_KL_CONTRIBUTIONS.csv','PAIRED_CI.csv','JOINT_RULE.csv','INDEPENDENT_CPU_ACCEPTANCE.json','INDEPENDENT_JOINT_ACCEPTANCE.json','COST_AND_IDENTITIES.json']:shutil.copy2(O/name,SUMMARY/name)
 (O/'HANDOFF.md').write_text('Current delivery: COMPLETE. Four fixed N/P4000 fits and60 candidates incl historical C/D/G/S. Paired bootstrap and direct D/N/P joint/weight diagnostics independently accepted. New endpoint heads only delivered. Old217-220 unchanged; no timer/automatic push/followup phase. See RESULT.md.\n',encoding='utf-8')
 files=[p for p in RAW.rglob('*') if p.is_file()]+[p for p in (O/'local_code').rglob('*') if p.is_file()]+[p for p in O.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','LOCAL_STATUS.json']];m={str(p.relative_to(O)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};save('DELIVERY_MANIFEST.json',m);assert all((O/n).stat().st_size==v['bytes'] and sha(O/n)==v['SHA'] for n,v in m.items());save('DELIVERY_COMPLETE.json',{'status':'passed','artifacts':len(m),'manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'units':75018,'GPU_seconds':complete['GPU_seconds'],'epoch':time.time()});print(json.dumps({'status':'complete','report':str(O/'RESULT.md')}))
if __name__=='__main__':main()
