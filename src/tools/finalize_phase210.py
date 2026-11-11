"""Seal two main tables, fixed-threshold gap accounting and provenance."""
import json,hashlib,shutil,csv
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase210/phase210_20261020';S=R/'src/research_results/phase210_20261020'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,s):
 with p.open('x',encoding='utf8') as f:f.write(s)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)
def f(v):return f'{v:.3f}'
def main():
 assert load(O/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed';A=load(O/'TABLE1.json');B=load(O/'TABLE2.json');by={r['strategy']:r for r in A};receipt=load(O/'DELIVERY_RECEIPT.json');remote=load(O/'recovery/RUN_STATUS.json');plan=load(O/'MISSING_PLAN.json');seal=load(O/'recovery/RECOVERY_SEALED.json');t=load(O/'THRESHOLD_REFERENCE.json')['threshold'];oldgap=by['M']['pooled_mIoU_pp']-by['R4']['pooled_mIoU_pp'];filled=by['R4_calibrated']['pooled_mIoU_pp']-by['R4']['pooled_mIoU_pp'];remaining=by['M']['pooled_mIoU_pp']-by['R4_calibrated']['pooled_mIoU_pp'];ratio=filled/oldgap*100 if oldgap else None
 labels={'R4':'R4默认','R4_calibrated':'R4冻结阈值','M':'M默认','B':'B默认','E4':'E4默认'}
 data='# Phase210 数据结果\n\n仅原Catalogue32个相关组；V4 R4、20m B04/B03/B8A、原1000/300窗口、float32及共同支持不变。固定引用209阈值`'+repr(t)+'`，原分数转float64按`>=t`判Shadow，其余取拼接后原raw类别0/1/2的argmax，0→Surface、1/2→Cloud。未重新校准、训练或拟合；未运行PixBox。\n\n'
 data+='## 表一：固定五策略\n\n单位%，召回仅7个有影组均值；误报为全部32组均值。\n\n'+table(['决策','逐组平均mIoU','池化mIoU','Surface IoU','Cloud IoU','Shadow IoU','平均召回','平均误报'],[[labels[q['strategy']]]+[f(q[k]) for k in ['macro_mIoU_pp','pooled_mIoU_pp','pooled_Surface_IoU_pp','pooled_Cloud_IoU_pp','pooled_Shadow_IoU_pp','macro_recall_pp','macro_negative_pp']] for q in A])
 data+='\n## 表二：固定比较\n\n差值单位为百分点；主比较为冻结阈值−M。\n\n'+table(['比较','池化mIoU差','配对组95%区间','逐组平均mIoU差','最差组／差值'],[[q['contrast'],f(q['pooled_delta_mIoU_pp']),'['+', '.join(f(x) for x in q['paired_group_CI95_pp'])+']',f(q['macro_delta_mIoU_pp']),q['worst_tile']+' / '+f(q['worst_delta_mIoU_pp'])] for q in B])
 data+='\n32组等概率整组配对bootstrap10000次、seed210020；每次汇总两方法混淆矩阵再重算池化mIoU。阈值始终冻结，区间不含重新校准或训练seed的不确定性。单组某类union=0则从该组均值排除；参考缺类但预测存在时union>0，该类IoU=0。无影组仍参加三类评价。池化值由总混淆矩阵计算。\n\n'
 data+=f"缺失清单：24景、172518个位置；8景无需前向。实际新增{receipt['total_calls']}次原R4调用（96主调用、{remote['constructor_calls']}构建），GPU阶段{receipt['GPU_stage_seconds']:.3f}秒。所有补跑共同支持的四类默认argmax精确复现，Shadow分数最大误差{max(q['max_shadow_score_error'] for q in seal['rows']):.9g}；新增Shadow位置{sum(q['added_shadow_positions'] for q in plan['rows'])}，该数量经过检查，未预设为0。209二类计数、召回、误报和Shadow IoU全部精确复现。\n\n"
 data+='完整32组五策略数值见PER_GROUP.csv；所有原R4→冻结阈值错误流见ERROR_FLOWS.csv（按参考类/旧预测/新预测计数）；每组比较与最差组ID见GROUP_DIFFERENCES.csv及TABLE2.json。原AP沿用209连续分数，未用补推理分数重定义。\n'
 analysis='# Phase210 结论分析\n\n'
 analysis+=f"原R4与M的池化mIoU差为 **{oldgap:.3f}个百分点**。只改原R4这一个冻结阈值，池化mIoU由{by['R4']['pooled_mIoU_pp']:.3f}%变为{by['R4_calibrated']['pooled_mIoU_pp']:.3f}%，增加 **{filled:.3f}个百分点**，相当于原差距的{ratio:.1f}%；相对M还差 **{remaining:.3f}个百分点**。这一比例仅描述本轮差距的数值分解，不是机制贡献率。\n\n"
 main=next(q for q in B if q['contrast']=='R4_calibrated-M');base=next(q for q in B if q['contrast']=='R4_calibrated-B')
 analysis+=f"冻结阈值−M的池化差为{main['pooled_delta_mIoU_pp']:+.3f}个百分点，配对组95%区间[{main['paired_group_CI95_pp'][0]:.3f}, {main['paired_group_CI95_pp'][1]:.3f}]；相对普通B为{base['pooled_delta_mIoU_pp']:+.3f}个百分点，区间[{base['paired_group_CI95_pp'][0]:.3f}, {base['paired_group_CI95_pp'][1]:.3f}]。剩余差距不能全部归因于MsRE核心，普通残差B是必要参照。mIoU接近也不证明评分等效或内部机制相同。\n\n"
 analysis+=f"对应代价：R4冻结阈值平均召回{by['R4_calibrated']['macro_recall_pp']:.3f}%、平均误报{by['R4_calibrated']['macro_negative_pp']:.3f}%；M为{by['M']['macro_recall_pp']:.3f}%/{by['M']['macro_negative_pp']:.3f}%，B为{by['B']['macro_recall_pp']:.3f}%/{by['B']['macro_negative_pp']:.3f}%。三者实际误报不同，不能只凭一个mIoU或召回数字认定全面优胜。209已发现该冻结阈值迁移到Catalogue后平均误报超过1%，本轮不调整阈值去纠正结果。\n\n"
 analysis+=f"池化改善不等于每组改善：R4逐组平均mIoU由{by['R4']['macro_mIoU_pp']:.3f}%变为{by['R4_calibrated']['macro_mIoU_pp']:.3f}%，下降{by['R4']['macro_mIoU_pp']-by['R4_calibrated']['macro_mIoU_pp']:.3f}个百分点。预定缺类规则会放大少量新增错误类的影响：50PLR的参考全为Cloud，冻结阈值仍有443个Shadow误报，M/B为0；其相对M/B约−50个百分点的单组mIoU差不代表一半像元判错。20WNV新增1个Surface错误预测，使该参考缺类的IoU=0被纳入均值，尽管Shadow误报减少，单组mIoU仍下降16.261个百分点。这些组和既定缺类规则全部保留，不事后删组或改定义。\n\n"
 analysis+='本轮只补齐缺失读出信息，没有改善评分、增添数据或选择新阈值。历史已暴露32组、固定训练seed与固定阈值，属于描述性对照；官方完整训练覆盖仍未知。错误流只说明共同支持内类别决策如何变化，不等于人工确认的地物机制。未新增显著性/晋级门槛，也不自动进入下一phase。\n'
 write(O/'DATA_RESULTS.md',data);write(O/'CONCLUSIONS.md',analysis);write(O/'RESULT.md',data+'\n'+analysis)
 readme='# Phase210\n\n完成：补齐原R4固定阈值的Catalogue三类mIoU；24景96次主前向，8景复用。零训练、零拟合、零阈值搜索，无PixBox/53组重新读取。\n\n数据结果见DATA_RESULTS.md，结论见CONCLUSIONS.md，合并报告RESULT.md；两张主表为TABLE1/2.json及CSV。完整逐组数据与错误流见PER_GROUP.csv、ERROR_FLOWS.csv。\n\n阈值引用、缺失位置/补跑清单、真实成本与SHA、重建及独立验收、最终清单分别见THRESHOLD_REFERENCE.json、MISSING_PLAN.json、COST_PROVENANCE.json、RECONSTRUCTION_ACCEPTANCE.json、INDEPENDENT_ACCEPTANCE.json、DELIVERY_COMPLETE.json。\n\n不建立定时任务、不自动push、不进入下一phase。\n'
 write(O/'README.md',readme);S.mkdir(parents=True,exist_ok=False)
 for n in ['RESULT.md','DATA_RESULTS.md','CONCLUSIONS.md','README.md']:shutil.copyfile(O/n,S/n)
 sources=['prepare_phase210.py','execute_phase210.py','execute_phase209.py','analyze_phase210.py','validate_phase210.py','bounded_phase210_local.py','finalize_phase210.py','accept_phase210_transfer.py','receive_phase210.ps1','deploy_phase210.py','bounded_phase202_local.py']
 cost=dict(new_network_calls=receipt['total_calls'],main_calls=96,constructor_calls=remote['constructor_calls'],training=0,fits=0,GPU_stage_seconds=receipt['GPU_stage_seconds'],remote_CPU_peak_bytes=remote['cpu_peak_bytes'],remote_GPU_peak_reserved_bytes=remote['gpu_peak_bytes'],remote_minimum_host_free_bytes=remote['minimum_host_free_bytes'],local_CPU_stages={p.parent.name:load(p) for p in O.glob('cpu_*/STATUS.json')},remote_new_bytes=receipt['remote_new_bytes'],remote_bootstrap_bytes=(O/'phase210_bootstrap.zip').stat().st_size,threshold_source_sha256=plan['threshold_source_sha256'],execution_lock=load(O/'EXECUTION_LOCK.json'),recovery_seal=seal,code_sha256={n:sha(R/'src/tools'/n) for n in sources},input_support='old208 fixed32 group common support; no pixels removed; missing classes only from originalR4 final mosaic',known_failures=[],unknown=['unobserved host-free minima between samples','complete official V4 training overlap'],new_whole_scene_raw_logits_saved=False,calendar_cutoff=None)
 cost['combined_bytes_before_final_receipts']=cost['remote_new_bytes']+cost['remote_bootstrap_bytes']+sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+sum(p.stat().st_size for p in S.rglob('*') if p.is_file())+sum((R/'src/tools'/n).stat().st_size for n in sources if '210' in n);cost['reserved_final_report_and_bytecode_allowance']=10000000;assert cost['combined_bytes_before_final_receipts']+10000000<2000000000
 write(O/'COST_PROVENANCE.json',json.dumps(cost,indent=2,allow_nan=False))
 if (O/'HANDOFF.md').exists():shutil.copyfile(O/'HANDOFF.md',O/'HANDOFF_PRE_FINALIZATION.md')
 (O/'HANDOFF.md').write_text('# Phase210 handoff\n\nCurrent delivery: COMPLETE. All conditional classes recovered; original raw4 decisions reproduced exactly; original Shadow scores within1e-5; all209 binary counts and four208 default confusions reproduced. No more forwards or threshold changes. See RESULT.md and DELIVERY_COMPLETE.json. No timer created.\n',encoding='utf8')
 manifest=dict(status='complete',files={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file()},summary_files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file()},new_network_calls=receipt['total_calls'],filled_gap_pp=filled,remaining_gap_pp=remaining,filled_fraction_percent=ratio,combined_conservative_upper_bytes=cost['combined_bytes_before_final_receipts']+10000000)
 write(O/'DELIVERY_COMPLETE.json',json.dumps(manifest,indent=2,allow_nan=False))
 for n,h in manifest['files'].items():assert sha(O/n)==h
 print(json.dumps(dict(status='complete',filled_gap_pp=filled,remaining_gap_pp=remaining,filled_percent=ratio,calls=receipt['total_calls'])))
if __name__=='__main__':main()
