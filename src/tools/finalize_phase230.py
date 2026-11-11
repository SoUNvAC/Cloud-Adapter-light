"""Write fixed deliverables only after all score/metric validation passed."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase230/phase230_20261026';D=O/'delivery'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def f(x,d=3):return 'NA' if x is None else f'{x:.{d}f}'
def pc(x):return None if x is None else 100*x
def main():
 assert js(O/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed';assert js(O/'CPU_VALIDATION.json')['status']=='passed';assert (O/'LOCAL_EXIT_CODE').read_text().strip()=='0'
 rows=js(O/'TABLE1.json');cmp=js(O/'TABLE2.json');aux=js(O/'ALL_TASK_SUMMARY.json');norm=js(O/'NORMALIZATION_AND_CLAMPS.json');gpu=js(D/'COMPLETE.json');assert gpu['total_calls']==55 and gpu['training_updates']==0
 lines=['# Phase230：Cloud／Surface决策偏移与排序诊断','','## 1 数据结果','','固定SPARCS validation10景；Source及两套普通监督C头各两个seed，四个C均α=0.25。RGB加载、resize、标签映射与ignore口径全部沿用229，分数及参考缓存为512×512。连续分数为229同一wrapper输出的非负归一化三类语义分数，缓存float32；不再softmax，不称为已校准概率或原始logits。','', '| 模型／seed | 有正负支持场景数 | 场景平均AUC／AP（%） | 池化AUC／AP（%） | 二类云召回：场景／池化（%） | 二类地表误报：场景／池化（%） | 三类mIoU（%） |','|---|---|---|---|---|---|---|']
 for x in rows:lines.append(f"| {x['model']} | {x['both_classes_scenes']}/10 | {f(pc(x['scene_mean_AUC']))} / {f(pc(x['scene_mean_AP']))} | {f(pc(x['pooled_AUC']))} / {f(pc(x['pooled_AP']))} | {f(pc(x['scene_mean_binary_cloud_recall']))} / {f(pc(x['pooled_binary_cloud_recall']))} | {f(pc(x['scene_mean_binary_surface_FPR']))} / {f(pc(x['pooled_binary_surface_FPR']))} | {f(x['three_class_mIoU_percent'])} |")
 lines+=['','主任务仅参考Cloud为正类、Surface为负类；Shadow与ignore排除。d=log(max(pCloud,1e−12))−log(max(pSurface,1e−12))，严格d>0判Cloud，d=0归Surface。二类召回分母为参考Cloud，误报分母仅为Surface，不是旧三类“非云影→云影FPR”。场景均值对分母存在场景等权；AUC／AP需正负类同时存在，缺类保留NA。各模型支持、分位数和d=0计数见MAIN_PER_SCENE.csv；召回／误报有效场景数见TABLE1.csv。AP为完整排序的非插值AP；AUC平分ties，不抽像元样本，不计算新置信区间。','', '| scene／seed | Cloud正／Surface负 | L8−Source ΔAUC／ΔAP（pp） | L8−SPARCS ΔAUC／ΔAP（pp） | L8−Source 云类／地表类d中位数差 | Cloud→Surface净错误差（对Source） |','|---|---|---|---|---|---|']
 def line(x):return f"| {x['scene']} / {x['seed']} | {x['Cloud_positive']} / {x['Surface_negative']} | {f(x['L8_minus_Source_AUC_pp'])} / {f(x['L8_minus_Source_AP_pp'])} | {f(x['L8_minus_SPARCS_AUC_pp'])} / {f(x['L8_minus_SPARCS_AP_pp'])} | {f(x['L8_minus_Source_Cloud_d_median'])} / {f(x['L8_minus_Source_Surface_d_median'])} | {x['Cloud_to_Surface_net_error_delta_vs_Source']:+d} |"
 lines += [line(x) for x in cmp]
 lines+=['','以上分数中位数差为自然对数单位；AUC／AP差为百分点。净错误差是同支持三类混淆Cloud行、Surface列之差，并非新增错误像元的逐像元配对数量。','','### 三个既有诊断场景','','| scene／seed | 正／负支持 | ΔAUC／ΔAP：L8−Source（pp） | ΔAUC／ΔAP：L8−SPARCS（pp） | 云类／地表类中位数差 | Cloud→Surface净变化 |','|---|---|---|---|---|---|']
 lines += [line(x) for x in js(O/'FOCUS_THREE_SCENES.json')]
 lines+=['','### 辅助任务，单独报告','','| 任务 | 模型 | 有正负支持场景数 | 场景平均AUC／AP（%） | 池化AUC／AP（%） |','|---|---|---|---|---|']
 for x in aux:
  if x['task']!='Cloud_vs_Surface':lines.append(f"| {x['task']} | {x['model']} | {x['both_classes_scenes']}/10 | {f(pc(x['scene_mean_AUC']))} / {f(pc(x['scene_mean_AP']))} | {f(pc(x['pooled_AUC']))} / {f(pc(x['pooled_AP']))} |")
 lines+=['','Cloud vs Surface+Shadow直接用pCloud；Shadow vs Surface用log(pShadow)−log(pSurface)，排除Cloud／ignore。逐景支持与结果见AUXILIARY_PER_SCENE.csv，辅助指标不替代主任务。','','## 2 结论分析','', (O/'INTERPRETATION.txt').read_text(encoding='utf-8').strip(),'', '这些validation已反复使用；输入生成方式混杂尚未解决。本轮只是现有输入与参考下的观测，既不是独立确认，也不证明纯自然域偏移或真实物理成因。没有按标签选阈值、偏置、校准配方，没有oracle恢复曲线或新训练，不宣称发现新算法。','','## 复现、资源与交付','', f"50张图×模型的argmax SHA与三类混淆全部精确复现229；固定首景5模型重复最大分数差均0。总计55次完整输入前向（主50、重复5、构建0），GPU阶段{gpu['GPU_stage_wall_seconds']:.3f}秒；GPU分配峰值{gpu['GPU_peak_bytes']/1024**2:.2f}MiB、保留峰值{gpu['GPU_reserved_peak_bytes']/1024**2:.2f}MiB；0训练更新、0拟合，未读L8、源域或test像元。",'',f"有效参考位置零和数：{sum(x['zero_sum_valid_pixels'] for x in norm)}（按模型逐景计）。全部位置零和数{sum(x['zero_sum_all_pixels'] for x in norm)}；最大非截断归一化误差{max(x['normalization_max_abs_error_nonclamped'] for x in norm):.9g}。分母截断占比与每通道log截断占比完整保留于NORMALIZATION_AND_CLAMPS.csv，不静默删除位置。",'', '本地独立算法重新计算全部主任务逐景及池化AUC／AP、严格二类决策、分位数和平局计数，并以布尔计数独立重算50份三类混淆；5个手工ties／排序数组与2个单类NA用例通过。分数、映射参考与ignore=255保存在delivery/scores和delivery/labels，完整SHA见delivery/DELIVERY_MANIFEST.json及最终DELIVERY_MANIFEST.json。初次上传、取回各遇连接关闭，均发生于传输开始前；故障记录保留，未重复任何模型调用。']
 (O/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 for n in ['prepare_phase230.py','execute_phase230.py','receive_phase230.py','phase230_metrics.py','analyze_phase230.py','validate_phase230.py','finalize_phase230.py']:shutil.copy2(R/'src/tools'/n,O/'code'/n)
 remote=sum(v['bytes'] for v in js(D/'DELIVERY_MANIFEST.json').values())+(D/'DELIVERY_MANIFEST.json').stat().st_size
 actual=sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+remote;assert actual<512*1024**2
 save(O/'COST_AND_IDENTITY_RECEIPT.json',{'total_calls':55,'main':50,'repeats':5,'construction_probes':0,'cap':80,'GPU_stage_wall_seconds':gpu['GPU_stage_wall_seconds'],'GPU_seconds_cap':1800,'CPU_analysis':js(O/'CPU_VALIDATION.json'),'GPU_resources':gpu,'combined_persistent_bytes_before_final_seal':actual,'cap_new_bytes':512*1024**2,'remote_durable_TAR':False,'score_caches':50,'mapped_label_caches':10,'runtime_SHA_manifest':sha(O/'RUNTIME_CODE_SHA.json'),'code_SHA':{p.name:sha(p) for p in (O/'code').glob('*.py')},'model_receipt_SHA':sha(D/'MODEL_ACCEPTED.json'),'frozen_LOCK_SHA':sha(O/'LOCK.json'),'input_lock_SHA':sha(O/'SPARCS_VAL_INPUTS.json'),'all50_predictions_and_confusions_exact229':True})
 (O/'HANDOFF.md').write_text('Current delivery: COMPLETE.55 total forwards(50main+5fixed repeats),no training/fits/threshold search. All50 predictions SHA/confusions exact229; repeat scores bitwise equal. SPARCS validation10 only; no L8/source/test pixels. Local full-order main/aux analyses and independent all-main metric/confusion acceptance passed. Complete50float32 score caches and10mapped labels retained locally/shared; no remote durable TAR; combined512MiB budget verified. RESULT.md separates data from interpretation. No follow-on phase or automatic restart authorized.\n',encoding='utf-8')
 s=R/'src/research_results/phase230_20261026';assert not s.exists();s.mkdir(parents=True)
 for n in ['RESULT.md','TABLE1.csv','TABLE2.csv','FOCUS_THREE_SCENES.csv','ALL_TASK_SUMMARY.csv','CPU_VALIDATION.json','INDEPENDENT_ACCEPTANCE.json','COST_AND_IDENTITY_RECEIPT.json']:shutil.copy2(O/n,s/n)
 files={str(p.relative_to(O)).replace('\\','/'):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in sorted(O.rglob('*')) if p.is_file() and p not in [O/'DELIVERY_MANIFEST.json',O/'DELIVERY_COMPLETE.json']}
 save(O/'DELIVERY_MANIFEST.json',files)
 for n,v in files.items():assert (O/n).stat().st_size==v['bytes'] and sha(O/n)==v['SHA']
 save(O/'DELIVERY_COMPLETE.json',{'status':'complete','manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'files':len(files),'all_lengths_SHA_verified':True,'next_phase_started':False,'new_model_calls_during_CPU_stage':0})
 print(json.dumps({'status':'complete','files':len(files),'combined_new_bytes':actual,'RESULT':str(O/'RESULT.md')}))
if __name__=='__main__':main()
