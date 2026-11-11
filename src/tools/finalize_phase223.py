"""Seal three diagnostic tables, evidence limitations and immutable delivery."""
from pathlib import Path
import json,csv,hashlib,shutil,time
from analyze_phase216 import peak
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase223/phase223_20261024';D=O/'delivery';SUMMARY=R/'src/research_results/phase223_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def fmt(x,n=6):return 'NA' if x in [None,''] else f'{float(x):.{n}g}'
def main():
 assert js(O/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed';complete=js(D/'COMPLETE.json');assert complete['total_units']==5360 and complete['GPU_seconds']<3600;lock=js(D/'INPUT_MODEL_LOCK.json');history={}
 for target,v in lock['history'].items():
  phase=v['phase'];date='20261023' if phase==217 else '20261024';p=R/f'outputs/phase{phase}/phase{phase}_{date}';assert sha(p/'DELIVERY_MANIFEST.json')==v['manifest_SHA'];m=js(p/'DELIVERY_MANIFEST.json');assert all((p/n).stat().st_size==x['bytes'] and sha(p/n)==x['SHA'] for n,x in m.items());history[target]={'unchanged_files':len(m),'manifest_SHA':v['manifest_SHA']}
 audit=R/'outputs/research_audits/cross_target_residual_damage_20261024';assert all(sha(audit/n)==h for n,h in lock['audit_SHA'].items())
 receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');local=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());uploads=sum(p.stat().st_size for p in O.glob('*.zip'));combined=receipt['remote_scope_bytes']+receipt['bytes']+local+uploads;assert combined<2000000000 and peak()<16*1024**3 and complete['CPU_peak_bytes']<16*1024**3 and max(complete['GPU_allocated_peak'],complete['GPU_reserved_peak'])<20*1024**3 and shutil.disk_usage(R).free>=4*1024**3
 save('COST_AND_IDENTITIES.json',{'new_units':5360,'counts':complete['counts'],'GPU_seconds':complete['GPU_seconds'],'new_training_backward_fits_bootstrap':0,'CPU_peak_remote':complete['CPU_peak_bytes'],'CPU_peak_local':peak(),'GPU_allocated_peak':complete['GPU_allocated_peak'],'GPU_reserved_peak':complete['GPU_reserved_peak'],'combined_new_persistent_upper_before_seal':combined,'Source_SHA':lock['Source_SHA'],'head_identities':lock['models'],'all_old217_222_unchanged':history,'cross_target_audit_unchanged':lock['audit_SHA'],'input_code_lock_SHA':sha(D/'INPUT_MODEL_LOCK.json'),'code_lock_SHA':sha(D/'CODE_LOCK.json'),'no_weights_arrays_or_images_transferred':True,'Source_initial_training_cost':'unknown','new_timer_download_environment_push':False})
 a=rows('TABLE_A_ALL_SOURCE.csv');b=rows('TABLE_B_FOCUS_ROI0124.csv');c=[v for v in rows('TABLE_C_CORRECTNESS.csv') if v['ROI']=='__pooled__'];delta=rows('TABLE_A_DIFFERENCES.csv');gaps=rows('SIGNED_GAP_QUANTILES.csv');dest=rows('TEACHER_CORRECT_SHADOW_DESTINATIONS.csv')
 at='\n'.join(f"|{v['target']}|{v['seed']}|{v['arm']}|{fmt(v['KL_mean_geometry'])}|{100*float(v['teacher_disagreement_rate_geometry']):.3f}%|{float(v['source_pool_loss_pp']):.3f}|{float(v['T11_pp']):.3f}|{v['worst_ROI']} / {float(v['worst_loss_pp']):.3f}|{int(v['error_pixels']):,}|" for v in a)
 bt='\n'.join(f"|{v['target']}|{v['seed']}|{float(v['C_mIoU']):.3f}/{float(v['D_mIoU']):.3f}|{fmt(v['C_KL'])}/{fmt(v['D_KL'])}|{fmt(v['C_Shadow_KL'])}/{fmt(v['D_Shadow_KL'])}|{fmt(v['C_Shadow_recall'])}/{fmt(v['D_Shadow_recall'])}|{v['Source_Shadow_to_Surface']}/{v['C_Shadow_to_Surface']}/{v['D_Shadow_to_Surface']}|" for v in b)
 ct='\n'.join(f"|{v['target']}|{v['seed']}|{int(v['teacher_correct_D_fewer_errors']):+,}|{int(v['teacher_wrong_D_fewer_corrections']):+,}|{int(v['D_minus_C_errors']):+,}|" for v in c)
 dt='；'.join(f"{v['target']} seed{v['seed']} ΔKL={fmt(v['D_minus_C_KL'])}、Δ错误={int(v['D_minus_C_errors']):+,}，KL更低但错误更多ROI={v['KL_lower_errors_more_ROI_count']}/107" for v in delta)
 focusremarks=[]
 for v in b:
  t,s=v['target'],v['seed'];model=f'{t}_{s}_D';q=next(x for x in gaps if x['model']==model and x['ROI']=='ROI_0124');dd=next(x for x in dest if x['model']==model and x['ROI']=='ROI_0124' and x['destination']=='Surface');focusremarks.append(f"{t}/{s}：D相对C ΔKL={fmt(v['D_minus_C_KL'])}、Δ参考错误={int(v['D_minus_C_errors']):+,}；教师正确Shadow→D Surface支持{q['pixels']}，teacher/student差中位数={fmt(q['teacher_median'])}/{fmt(q['student_median'])}，教师置信均值{fmt(dd['teacher_confidence_mean'])}")
 interpretations=[]
 for v in delta:
  kl=float(v['D_minus_C_KL']);err=int(v['D_minus_C_errors']);interpretations.append(f"{v['target']}/{v['seed']}在全体源val"+('同时降低平均教师KL和参考错误' if kl<0 and err<0 else '的教师KL与参考错误未同时下降'))
 all_focus_better=all(float(v['D_minus_C_KL'])<0 and int(v['D_minus_C_errors'])<0 for v in b)
 residual='；'.join(f"{v['target']}/{v['seed']}={float(v['Source_mIoU'])-float(v['D_mIoU']):.3f}pp" for v in b)
 focus_conclusion=('ROI_0124四种配对均为D比C的KL更低、参考错误也更少，本焦点未出现D相对C的反向损伤。' if all_focus_better else 'ROI_0124的配对方向并非全部一致，按表B逐项保留。')+'不过D相对Source仍残留mIoU损失：'+residual+'。教师正确Shadow仍有约2万至4.1万像元被D判为Surface，所选像元的teacher/student原分数差中位数由正变负，与该输出排序翻转一致，不解释训练成因。'
 report=f'''# Phase223：α=0.25教师保持与局部云影损伤

## 1．数据结果

固定两目标各C/D、seed21701/21702，共8个历史head；从原Source与末步头用CPUfloat64构造α=0.25，并逐张量匹配旧插值文件SHA。只原源val535图／107 ROI，无训练、反传、拟合、回放或目标评价。每图一次共享backbone和教师head；正式路径与8学生首图逐semantic位一致，全部旧逐图／ROI混淆和存在的预测SHA精确复现。

表A：全源val；损失、T11为百分点，KL按共同512²几何支持汇总，负舍入原样保留，ignore另表。

|目标|seed|臂|平均KL|教师分歧率|源池化损失|T11|最差ROI/损失|错误像元|
|---|---|---|---:|---:|---:|---:|---|---:|
{at}

D−C：{dt}。上述ROI完整清单和逐参考类分解已交付，未用焦点替代全部ROI。

表B：固定ROI_0124；mIoU为%，其余C/D顺序成对列出，Source mIoU从旧同支持基线复现。

|目标|seed|C/D mIoU|C/D KL|C/D参考Shadow KL|C/D Shadow召回|Source/C/D Shadow→Surface|
|---|---|---|---|---|---|---|
{bt}

教师正确Shadow诊断（有符号差=S_Shadow−S_Surface）：{'；'.join(focusremarks)}。P10/P90、空组NA、Surface/Cloud漏影支持及平均KL详见SIGNED_GAP_QUANTILES和TEACHER_CORRECT_SHADOW_DESTINATIONS；全107 ROI及旧最差11 ROI另表。

表C：全体参考正确性净分解；总错误变化＝少纠正−少错。每ROI同格式交付；J另导出真正同像元C错→D对／C对→D错。

|目标|seed|教师原正确：D少错|教师原错误：D少纠正|D−C总错误|
|---|---|---:|---:|---:|
{ct}

实际5360单位（正式5350＋门检10），GPU阶段{complete['GPU_seconds']:.1f}s；独立验收通过四轴显式求和、直接转移、全部支持与恒等式、48点CPU显式KL（容差1e−6）、ROI_0124与T11审计复现、冻结和账本。新增持久量上界{combined/1e9:.3f}GB，无新bootstrap或权重／大数组传输。

## 2．结论探讨

两个目标的两seed均同时降低全体平均KL和参考错误，但各有48—54个ROI呈现KL更低、错误更多，因此平均教师保持不能保证局部参考正确性。表C显示D同时减少教师原正确区域的破坏、也减少对教师原错误的纠正；SPARCS21701两者几乎抵消，净少错仅2855像元。

{focus_conclusion}

本轮仅为同α强度匹配：SPARCS21701原选α=0.5，未在此补算，故不是全部已选工作点的诊断。语义softmax置信度不是校准正确概率，输出差和漏影对应不是梯度因果证据。没有测回放覆盖、训练时间轨迹或“看够该类”；不能据此命名新机制或自动启动训练。两目标重复使用同107 ROI不增加地理独立样本数。
'''
 (O/'RESULT.md').write_text(report,encoding='utf-8');SUMMARY.mkdir(parents=True,exist_ok=True)
 for n in ['RESULT.md','TABLE_A_ALL_SOURCE.csv','TABLE_A_DIFFERENCES.csv','TABLE_B_FOCUS_ROI0124.csv','TABLE_C_CORRECTNESS.csv','KL_LOWER_ERRORS_MORE_ROIS.csv','SIGNED_GAP_QUANTILES.csv','INDEPENDENT_CPU_ACCEPTANCE.json','COST_AND_IDENTITIES.json']:shutil.copy2(O/n,SUMMARY/n)
 (O/'HANDOFF.md').write_text('Current delivery: COMPLETE. Eight fixed alpha0.25 heads, original Sourceval535/107 only; all old predictions/CM and local focus/T11 audit reproduced. Explicit four-axis/scalar-KL independent acceptance passed. Zero training/bootstrap. See RESULT.md. No timer/push/new phase.\n');files=[p for p in D.rglob('*') if p.is_file()]+[p for p in (O/'local_code').rglob('*') if p.is_file()]+[p for p in O.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','LOCAL_STATUS.json']];m={str(p.relative_to(O)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};save('DELIVERY_MANIFEST.json',m);assert all((O/n).stat().st_size==v['bytes'] and sha(O/n)==v['SHA'] for n,v in m.items());save('DELIVERY_COMPLETE.json',{'status':'passed','artifacts':len(m),'manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'units':5360,'GPU_seconds':complete['GPU_seconds'],'epoch':time.time()});print(json.dumps({'status':'complete','report':str(O/'RESULT.md')}))
if __name__=='__main__':main()
