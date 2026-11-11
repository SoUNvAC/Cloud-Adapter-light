"""Bounded219 report: measured data separated from conditional interpretations."""
from pathlib import Path
import csv,json,hashlib,shutil,time,statistics
from analyze_phase216 import peak
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/phase219/phase219_20261023';RAW=OUT/'delivery';SUMMARY=ROOT/'src/research_results/phase219_20261023'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert js(OUT/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed';complete=js(RAW/'COMPLETE.json');assert complete['units']==73946 and complete['GPU_seconds']<14400
 endpoints=rd('TABLE_A_ENDPOINTS.csv');diff=rd('TABLE_B_DIFFERENCES.csv');ci=rd('PAIRED_CI.csv');bands=rd('TEACHER_CONFIDENCE_DIAGNOSTIC.csv');joint=rd('JOINT_RULE.csv');common=rd('COMMON_DAMAGE_ROIS.csv')
 training=[];loss={};resources=[js(RAW/'weight_preparation/COMPLETE.json'),js(RAW/'evaluation/COMPLETE.json')]
 for s in [21701,21702]:
  resources.append(js(RAW/f'seed{s}/TRAINING_COMPLETE.json'))
  for a in ['G','S']:
   training.append({'seed':s,'arm':a,**js(RAW/f'seed{s}/{a}_TRAIN_COMPLETE.json')});ls=[json.loads(v) for v in (RAW/f'seed{s}/{a}_LOSSES.jsonl').read_text().splitlines()];loss[f'{s}/{a}']={k:{'first100':statistics.mean(v[k] for v in ls[:100]),'last100':statistics.mean(v[k] for v in ls[-100:]),'min':min(v[k] for v in ls),'max':max(v[k] for v in ls)} for k in ['target_loss','source_KL_unweighted','source_KL_weighted','high_weight_region_KL','low_weight_region_KL','total_loss']}
 save('LOSS_SUMMARY.json',loss)
 remote=js(OUT/'REMOTE_DELIVERY_RECEIPT.json');localbytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());packages=sum(p.stat().st_size for p in OUT.glob('*.zip'));combined=remote['remote_scope_bytes']+remote['bytes']+localbytes+packages;assert combined<4_000_000_000 and shutil.disk_usage(ROOT).free>4*1024**3
 allocated=max(v['GPU_allocated_peak'] for v in resources);reserved=max(v['GPU_reserved_peak'] for v in resources);cpu=max(v['CPU_peak_bytes'] for v in resources);assert max(allocated,reserved)<20*1024**3 and cpu<16*1024**3 and peak()<16*1024**3
 historical={}
 for phase in [216,217,218]:
  d=ROOT/f'outputs/phase{phase}/phase{phase}_20261023';m=js(d/'DELIVERY_MANIFEST.json');assert sha(d/'DELIVERY_MANIFEST.json')==js(d/'DELIVERY_COMPLETE.json')['manifest_SHA'];assert all((d/n).stat().st_size==v['bytes'] and sha(d/n)==v.get('SHA',v.get('sha256')) for n,v in m.items());historical[str(phase)]={'manifest_SHA':sha(d/'DELIVERY_MANIFEST.json'),'all_artifacts_unchanged':len(m)}
 cost={'new_units':73946,'counts':js(RAW/'REMOTE_CPU_ACCEPTANCE.json')['calls'],'new_fits':4,'updates_per_fit':4000,'historical_C_D_refits':0,'GPU_stage_seconds':complete['GPU_seconds'],'GPU_allocated_peak_bytes':allocated,'GPU_reserved_peak_bytes':reserved,'CPU_peak_remote_bytes':cpu,'CPU_peak_local_bytes':peak(),'remote_scope_bytes':remote['remote_scope_bytes'],'remote_archive_bytes':remote['bytes'],'local_scope_before_seal_bytes':localbytes,'uploaded_packages_upper_bytes':packages,'combined_persistent_upper_bytes_before_final_seal':combined,'persistent_cap_bytes':4_000_000_000,'local_free_bytes':shutil.disk_usage(ROOT).free,'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','initial_and_final_heads_shared_only':training,'weight_manifest_SHA':sha(RAW/'weight_preparation/WEIGHT_MANIFEST.json'),'code_lock_SHA':sha(RAW/'CODE_LOCK.json'),'training_runtime':'original217 deterministic warn-only; evaluation original flags allFalse FP32','historical_unchanged':historical,'earlier_Source_initialization_training_cost':'unknown','new_model_calls_local':0,'no_timer_auto_push_or_followup_phase':True};save('COST_AND_IDENTITY.json',cost)
 def drows(comp):return [v for v in diff if v['comparison']==comp]
 def avg(comp,k):return statistics.mean(float(v[k]) for v in drows(comp))
 def interval(comp,key):
  v=next(v for v in ci if v['comparison']==comp and v['context']=='endpoint_alpha1' and v['scope']=='mean_of_two' and v['metric']==key);return f"{float(v['point']):+.3f} [{float(v['CI95_low']):+.3f}, {float(v['CI95_high']):+.3f}]"
 at='\n'.join(f"|{v['seed']}|{v['arm']}|{float(v['test_mIoU']):.3f}|{float(v['pool_loss_pp']):.3f}|{float(v['T11_pp']):.3f}|{float(v['worst_ROI_loss_pp']):.3f}|{v['ROI_loss_gt5']}|" for v in endpoints)
 bt='\n'.join(f"|{v['seed']}|{v['comparison']}|{float(v['test_mIoU_diff_pp']):+.3f}|{float(v['source_pool_loss_diff_pp']):+.3f}|{float(v['T11_diff_pp']):+.3f}|{int(v['teacher_correct_net_benefit_pixels']):+,}|{int(v['teacher_correction_net_cost_pixels']):+,}|{int(v['total_error_delta_pixels']):+,}|" for v in diff if v['comparison'] in ['G-D','G-S'])
 ct='\n'.join(f"|{c}|{interval(c,'test_mIoU_diff_pp')}|{interval(c,'source_pool_loss_diff_pp')}|{interval(c,'T11_diff_pp')}|" for c in ['G-D','G-S'])
 conf='；'.join(f"{v['band']}：错误率{100*float(v['reference_error_rate']):.3f}%，Surface/Cloud/Shadow支持={v['Surface_support']}/{v['Cloud_support']}/{v['Shadow_support']}" for v in bands if v['ROI']=='__pooled__')
 choices='；'.join(f"{s}{a}="+'/'.join(str(next(v['alpha'] for v in joint if v['seed']==str(s) and v['arm']==a and v['budget_pp']==str(b))) for b in [1,3,5]) for s in [21701,21702] for a in ['C','D','G','S'])
 meanings=[]
 for c in ['G-D','G-S']:
  for key,label,prefer in [('test_mIoU_diff_pp','目标mIoU',1),('source_pool_loss_diff_pp','源池化损失',-1),('T11_diff_pp','T11',-1)]:
   vals=[float(v[key]) for v in drows(c)];direction='两个seed同向'+('改善' if all(v*prefer>0 for v in vals) else '恶化') if all(v*prefer>0 for v in vals) or all(v*prefer<0 for v in vals) else '两个seed方向不一致或包含零差';mean=avg(c,key);meanrow=next(v for v in ci if v['comparison']==c and v['context']=='endpoint_alpha1' and v['scope']=='mean_of_two' and v['metric']==key);unc='区间跨零，方向证据不足' if float(meanrow['CI95_low'])<=0<=float(meanrow['CI95_high']) else '条件区间未跨零';meanings.append(f"{c}的{label}差均值{mean:+.3f}pp，{direction}，{unc}。")
 correction='；'.join(f"seed{v['seed']} G−D净保护收益{int(v['teacher_correct_net_benefit_pixels']):+,}、净纠错代价{int(v['teacher_correction_net_cost_pixels']):+,}、总错误变化{int(v['total_error_delta_pixels']):+,}" for v in drows('G-D'))
 commontext='、'.join(f"{a}={sum(v['context']=='alpha1' and v['arm']==a for v in common)}" for a in ['C','D','G','S'])
 report=f'''# Phase219：教师置信度加权的源预测保持

## 1．数据结果

G按冻结教师semantic-softmax置信度稳定排序，每图高低各半赋权1.5/0.5；S在图内固定打乱同权重。新增G/S各两个原seed21701/21702、4000步，从同Source仅更新1,316,516个head参数，λ=T=1；历史C普通微调、D均匀KL复用217。SPARCS密集监督train60、val10、test10；回放256张CloudSEN源train RGB不读标签，源val535图/107 ROI。原4000步增强及回放序列逐步配对，Source官方test未读。目标test已多次查看，属探索方法探针。

表A为α=1。mIoU单位%，源损失=Source−候选、T11=最差11 ROI带符号损失均值，单位pp；负数不截断。

|seed|臂|目标test mIoU|源池化损失|T11|最差ROI损失|损失>5pp组数|
|---|---|---:|---:|---:|---:|---:|
{at}

表B为端点配对差；后3列是固定教师正确/错误区域的净像元计数。

|seed|比较|目标差pp|源损失差pp|T11差pp|教师正确净收益|教师纠错净代价|总错误净变化|
|---|---|---:|---:|---:|---:|---:|---:|
{bt}

两seed指标差均值与条件95%区间：同一10 scene/107 ROI索引，逐seed非线性重算后平均，10000次seed217023，T11逐次重排，不扩充组数。

|比较|目标差pp [95%CI]|源损失差pp [95%CI]|T11差pp [95%CI]|
|---|---|---|---|
{ct}

教师高/低置信区源val诊断：{conf}。每图按几何各131072格划分，参考ignore另排除，支持不必各半。该置信度不是正确概率；回放不报告准确率。全107 ROI及0050/1013/1404/4813的净分解、全部类别支持与错误率见CSV。

联合规则仍为源池损失≤B且T11≤5，取最大α，先封存源侧选择再连接目标结果。B1/3/5所选α顺次为：{choices}。完整40候选、120池化行、5080逐组行及错误流均交付。端点两seed共同>5pp ROI：{commontext}。

实际73,946单位调用（权重预计算512、训练64,000、门检16、候选8,880、Source抽查3、教师诊断535）；GPU阶段{complete['GPU_seconds']:.1f}s。四fit无恢复/重训，GPU峰值allocated/reserved={allocated/1024**3:.2f}/{reserved/1024**3:.2f}GiB，持久文件合计上界{combined/1e9:.3f}GB。全部权重SHA、局部PCG64/稳定tie、初始化/冻结、loss、标签增强、插值、Source复现、混淆/分解与bootstrap经CPU独立验收，旧216–218封存文件未变。参数、输入、成本及采样限制详见回执。

## 2．结论探讨

{''.join(meanings)} 这分别回答目标代价、是否牺牲原保源收益及空间对应优于打乱与否，不能将它们合并成单一成功判定。

{correction}。负纠错代价才表示恢复更多教师纠错；保护收益为负表示教师原正确区域损伤增加。两项差值恒等于总错误变化，但不是D→G逐像元转移，也不能分解非线性mIoU或证明梯度因果。焦点ROI仅事后追踪，不代替全107组。

区间条件于实际两训练seed、固定Source/回放池及已选α，不覆盖训练seed总体或重新选择不确定性；跨零不等于等效。G−S只检验此固定权重与空间对齐的增量，不证明置信度已校准。无论此探针结果是否理想，都不能据此否定或确认整个选择性保持方向；不追加seed、网格、阈值或下一phase。
'''
 (OUT/'RESULT.md').write_text(report,encoding='utf-8');SUMMARY.mkdir(parents=True,exist_ok=True)
 for n in ['RESULT.md','TABLE_A_ENDPOINTS.csv','TABLE_B_DIFFERENCES.csv','PAIRED_CI.csv','ALL_CANDIDATES.csv','JOINT_RULE.csv','FOUR_FOCUS_ROIS.csv','TEACHER_CONFIDENCE_DIAGNOSTIC.csv','COST_AND_IDENTITY.json','INDEPENDENT_CPU_ACCEPTANCE.json']:shutil.copy2(OUT/n,SUMMARY/n)
 (OUT/'HANDOFF.md').write_text('Current delivery: COMPLETE. Four fixed4000 fits, all40 candidates including historical C/D and independent CPU checks delivered. No timer, automatic push or followup phase. See RESULT.md.\n',encoding='utf-8')
 names=[p for p in RAW.rglob('*') if p.is_file()]+[p for p in (OUT/'local_code').rglob('*') if p.is_file()]+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','LOCAL_STATUS.json','RECEIVER_STARTED.json']]
 manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in names};save('DELIVERY_MANIFEST.json',manifest);assert all((OUT/n).stat().st_size==v['bytes'] and sha(OUT/n)==v['SHA'] for n,v in manifest.items());save('DELIVERY_COMPLETE.json',{'status':'passed','artifacts':len(manifest),'manifest_SHA':sha(OUT/'DELIVERY_MANIFEST.json'),'epoch':time.time(),'units':73946,'GPU_seconds':complete['GPU_seconds']});print(json.dumps({'status':'complete','report':str(OUT/'RESULT.md')}))
if __name__=='__main__':main()
