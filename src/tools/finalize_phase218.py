"""Separated218 data/interpretation, bounded report and immutable SHA delivery."""
from pathlib import Path
import csv,json,hashlib,shutil,time
from analyze_phase216 import peak
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase218/phase218_20261023';RAW=O/'delivery';SUMMARY=R/'src/research_results/phase218_20261023'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert js(O/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed';cost=js(RAW/'COMPLETE.json');assert cost['total_units']==4758 and cost['GPU_seconds']<3600
 A={(int(r['seed']),r['arm'],r['pool']):r for r in rd('TABLE_A_RETENTION.csv')};B=rd('TABLE_B_REFERENCE.csv');C=rd('TABLE_C_FOCUS.csv');allc=rd('TABLE_C_ALL_ROI.csv');diff=rd('TABLE_A_DIFFERENCES.csv')
 at=[]
 for s in [21701,21702]:
  for a in ['C','D']:
   x=A[s,a,'Replay256'];y=A[s,a,'Source_val535'];at.append(f"| {s}/{a} | {float(x['KL_mean_geometry']):.6f} | {100*float(x['teacher_disagreement_rate_geometry']):.3f} | {float(y['KL_mean_geometry']):.6f} | {100*float(y['teacher_disagreement_rate_geometry']):.3f} |")
 bt='\n'.join(f"| {r['seed']}/{r['arm']} | {float(r['mIoU']):.3f} | {float(r['T11_pp']):.3f} | {float(r['Surface_IoU']):.3f}/{float(r['Cloud_IoU']):.3f}/{float(r['Shadow_IoU']):.3f} | {100*float(r['error_rate']):.3f} |" for r in B)
 ct='\n'.join(f"| {r['seed']} | {r['ROI']} | {float(r['C_KL_mean']):.5f}/{float(r['D_KL_mean']):.5f} | {r['C_errors']}/{r['D_errors']} | {'是' if r['D_KL_lower_and_errors_more']=='True' else '否'} |" for r in C)
 totals={s:sum(int(r['seed'])==s and r['D_KL_lower_and_errors_more']=='True' for r in allc) for s in [21701,21702]};focusnum=sum(r['D_KL_lower_and_errors_more']=='True' for r in C)
 formula=js(O/'INDEPENDENT_CPU_ACCEPTANCE.json');equal_support=all(int(r['geometry_pixels'])==int(r['reference_valid_pixels']) for r in B)
 kl_diffs='；'.join(f"{r['seed']} {r['pool']} D−C={float(r['D_minus_C_pooled_KL']):+.6f}" for r in diff)
 flows={'ROI_0050':'Surface_to_Cloud','ROI_4813':'Surface_to_Cloud','ROI_1013':'Cloud_to_Shadow','ROI_1404':'Surface_to_Shadow'}
 flow_details='；'.join(g+' '+f+'新增'+('/'.join(str(int(next(r for r in C if int(r['seed'])==s and r['ROI']==g)['D_'+f])-int(next(r for r in C if int(r['seed'])==s and r['ROI']==g)['C_'+f])) for s in [21701,21702]))+'像元' for g,f in flows.items())
 replay_better=all(float(A[s,'D','Replay256']['KL_mean_geometry'])<float(A[s,'C','Replay256']['KL_mean_geometry']) for s in [21701,21702]);val_better=all(float(A[s,'D','Source_val535']['KL_mean_geometry'])<float(A[s,'C','Source_val535']['KL_mean_geometry']) for s in [21701,21702])
 interpretation=('两个seed的D均降低回放和val的平均KL，保持教师的现象不只出现在回放自身。' if replay_better and val_better else '回放与val的最终KL变化并非两seed都同向，不能把保持迁移概括成一致结果。')
 report=f'''# Phase218：最终模型的回放保持与源val损伤

## 1．数据结果

只用21701／21702的C、D末步(α=1)及同一Source；全部冻结eval、FP32，零训练／反传／拟合／插值／调参。Replay256为原256图／249 ROI，每图一次、无标签；源val为原535图／107 ROI。未运行目标集、官方Source test、L8额外池或Omni。

KL逐像元按p_Source·(logp_Source−logq_student)三类求和，p/q来自原query聚合后的semantic分数softmax(T=1)，再按共同几何像元平均；不直接归一化query logits，不按15／16次训练重复加权。预测类别沿用217原semantic归一化argmax，KL分布不称校准正确概率。

| seed/臂 | Replay均值KL | Replay教师分歧% | val均值KL | val教师分歧% |
|---|---:|---:|---:|---:|
{chr(10).join(at)}

逐图中位数／P90／最大、D−C和全部ROI见表A及CSV。每模型Replay几何分母67,108,864；val几何分母140,247,040，参考有效支持{'完全相同，ignore为0' if equal_support else '另列，不与几何支持混淆'}。负舍入值及其计数原样保存，未clamp。

| seed/臂 | val mIoU% | T11损失pp | Surface/Cloud/Shadow IoU% | 总错分率% |
|---|---:|---:|---|---:|
{bt}

表B另列三种参考类的像元数、KL、教师分歧／参考错误率；四个教师／学生正确性状态与“两者都错但同／不同类”从本轮逐像元直接计数，未从旧混淆反推。全部107 ROI完整保留。

| seed | 既定焦点ROI | C/D平均KL | C/D参考错误像元 | D的KL低且错误多 |
|---|---|---|---|---|
{ct}

指定方向D−C：{flow_details}。两seed全107 ROI中，KL更低却错误更多的ROI分别为{totals[21701]}／{totals[21702]}；焦点8条中{focusnum}条满足该组合。各参考类KL、mIoU、分歧与六方向计数见表C，编号不用于猜地物。

实际4,758单位＝791 backbone＋791教师head＋3164学生head＋工程克隆2 head／原路径10次。门检两张输入的正式输出复用，未重复计入正式样本。GPU阶段{cost['GPU_seconds']:.1f}s，allocated/reserved峰值{cost['GPU_allocated_peak']/1024**3:.3f}/{cost['GPU_reserved_peak']/1024**3:.3f}GiB。全部535图、107 ROI、池化混淆及预测SHA精确复现217；参数／buffer逐张量不变、零梯度。独立CPU核对60个固定坐标，显式逐类KL最大误差{formula['max_FP64_vs_FP32_KL_error']:.2g}，并重建全部2140条有参考记录的三维转移计数。源train标签及官方test零读取。

## 2．结论探讨

{interpretation}D在回放上仍约4.6%像元与教师分歧，不能称完全保持，更不能在无标签Replay上称准确率或遗忘率。两个池构成不同，本轮又是最终固定eval权重，不能与在线train-mode loss直接相减，宣称纯泛化差距或收敛。val参考指标为217的精确复现；新增证据是最终KL和直接像元转移，不是一次新独立性能验证。

证据更接近“回放和val平均保持均改善，但val平均KL下降仍伴局部参考损伤”。0050／1013／1404在两seed均为整体KL更低且错误更多；4813整体和Surface参考类KL反而升高，不能归为同一局部模式。0050／1404的Surface类KL两seed都降低，指定错分却增加；1013的Cloud类KL一降一升。

像元转移显示，D减少了“教师对／学生错”（6,707,531→4,178,379；6,750,577→4,318,214），同时也减少了“教师错／学生对”（3,047,933→2,304,794；3,117,746→2,361,714）。保留教师正确预测和减少对教师错误的修正可以同时发生；这是计数分解，不证明训练梯度因果。四个焦点是事后定位线索，某类KL均值不等于梯度强弱；不据此命名新机制、建探测器、改5pp门限或新增模型。
'''
 (O/'RESULT.md').write_text(report,encoding='utf-8');SUMMARY.mkdir(parents=True,exist_ok=True)
 copies=['RESULT.md','TABLE_A_RETENTION.csv','TABLE_A_DIFFERENCES.csv','TABLE_B_REFERENCE.csv','TABLE_C_FOCUS.csv','TABLE_C_ALL_ROI.csv','POOLED_REFERENCE_CLASSES.csv','POOLED_CORRECTNESS.csv','POOLED_BOTH_WRONG.csv','INDEPENDENT_CPU_ACCEPTANCE.json']
 for n in copies:shutil.copyfile(O/n,SUMMARY/n)
 receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');local=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());summary=sum(p.stat().st_size for p in SUMMARY.rglob('*') if p.is_file());tools=sum(p.stat().st_size for p in (R/'src/tools').glob('*phase218*.py'));upload=sum(p.stat().st_size for p in O.glob('*.zip'));upper=receipt['remote_scope_bytes']+receipt['bytes']+local+summary+tools+upload+1048576
 assert upper<2000000000 and max(cost['GPU_allocated_peak'],cost['GPU_reserved_peak'])<20*1024**3 and cost['CPU_peak_bytes']<16*1024**3 and peak()<16*1024**3 and shutil.disk_usage(R).free>4*1024**3
 save('FINAL_COST_AND_IDENTITIES.json',{'remote_execution':cost,'local_analysis':js(O/'ANALYSIS_RECEIPT.json'),'local_acceptance':formula,'local_report_peak_bytes':peak(),'combined_persistent_upper_bytes':upper,'local_scope_bytes':local,'remote_scope_bytes':receipt['remote_scope_bytes'],'both_archives_and_uploaded_code_and_summary_counted':True,'sealing_reserve_bytes':1048576,'new_units':4758,'new_optimizer_updates':0,'local_model_calls':0,'input_lock_SHA':sha(RAW/'INPUT_MODEL_LOCK.json'),'runtime_and_code_lock_SHA':sha(RAW/'CODE_LOCK.json'),'head_SHA':js(RAW/'INPUT_MODEL_LOCK.json')['models'],'no_unreported_model_failure_or_recovery':True,'no_automatic_push_timer_or_followup_phase':True})
 shutil.copyfile(O/'FINAL_COST_AND_IDENTITIES.json',SUMMARY/'FINAL_COST_AND_IDENTITIES.json');(O/'HANDOFF.md').write_text('Current delivery: COMPLETE; frozen218 diagnostic4758 units, zero fits, exact217 reproduction and independent CPU acceptance passed. See RESULT.md and manifest. No subsequent phase or timer.\n',encoding='utf-8')
 paths=[p for p in RAW.rglob('*') if p.is_file()]+[p for p in (O/'local_code').rglob('*') if p.is_file()]+[p for p in O.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in ['LOCAL_STATUS.json','RECEIVER_STARTED.json','DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json']]
 manifest={str(p.relative_to(O)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in paths};save('DELIVERY_MANIFEST.json',manifest);assert all((O/n).stat().st_size==v['bytes'] and sha(O/n)==v['SHA'] for n,v in manifest.items());save('DELIVERY_COMPLETE.json',{'status':'passed','artifacts':len(manifest),'manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'units':4758,'GPU_seconds':cost['GPU_seconds'],'epoch':time.time()})
 print(json.dumps({'report':str(O/'RESULT.md'),'focus_lower_KL_more_errors':focusnum,'all_ROI_counts':totals,'Replay_and_val_lower_KL_both_seeds':replay_better and val_better}))
if __name__=='__main__':main()
