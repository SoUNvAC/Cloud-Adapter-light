"""Seal the train-only scalar-control results after full independent CPU acceptance."""
from pathlib import Path
import hashlib,json,shutil,sys
REPO=Path(__file__).resolve().parents[2];R=REPO/'outputs/phase231/phase231_20261026';D=R/'delivery'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def f(v,d=3):return 'NA' if v is None else f'{v:.{d}f}'
def main():
 pending='--pending-bias-audit' in sys.argv
 assert js(R/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed'
 if pending:assert js(R/'BIAS_SEAL_ACCEPTED_LOCAL.json')['status']=='passed'
 else:
  assert (R/'LOCAL_EXIT_CODE').read_text().strip()=='0' and js(R/'BIAS_ACCEPTED_LOCAL.json')['status']=='passed'
  for n,h in js(R/'PRE_RESUME_FROZEN_RESULTS.json').items():assert sha(R/n)==h,'Previously sealed validation results or biases changed: '+n
  delivered=js(D/'DELIVERY_MANIFEST.json');assert len(delivered)==281
  for n,v in delivered.items():assert (D/n).stat().st_size==v['bytes'] and sha(D/n)==v['SHA'],n
 fits=js(R/'TABLE1.json');rows=js(R/'TABLE2.json');scenes=js(R/'TABLE3.json');gpu=js(D/'GPU_COMPLETE.json');cpu=js(D/'FIT_COMPLETE.json')
 lines=['# Phase231：训练侧单Cloud偏置校准对照','','## 1 数据结果','','固定Source及L8两个普通监督C末头，C均α=0.25；网络参数、骨干与buffer不更新。校准使用原SPARCS60个target_train场景，逐景等权、景内Cloud／Surface有效像元等权；Shadow／ignore不参与BCE，不做正负类平衡或抽样。评价仅复用230原10景validation缓存，没有新增validation前向。','', '| 模型 | 校准场景／Cloud正／Surface负支持 | b | L(0)→L(b) | g(b) | 触边 |','|---|---|---|---|---|---|']
 for x in fits:lines.append(f"| {x['model']} | {x['scenes']} / {x['Cloud_positive']} / {x['Surface_negative']} | {x['b']:.9f} | {x['L0']:.9f} → {x['Lb']:.9f} | {x['g_b']:.3e} | {'是' if x['boundary'] else '否'} |")
 lines+=['','唯一拟合规则：float64导数单调二分60次，范围[−5,5]，不扩边界、不搜索其他损失。先封存全部3个b及输入／代码SHA，经本地封存身份核验后才读取validation分数与标签；偏置求解的独立复算是最终验收，不参与validation偏置选择。时间和SHA回执见BIAS_FROZEN.json、BIAS_SEAL.json、BIAS_SEAL_ACCEPTED_LOCAL.json、VALIDATION_ACCESS_STARTED.json。','', '| 模型／决策 | mIoU：池化／场景均值 | Cloud召回：池化／场景均值 | Surface→Cloud FPR：池化／场景均值 | Shadow召回：池化／场景均值 | 非Shadow→Shadow FPR：池化／场景均值 | Cloud／Surface排序 |','|---|---|---|---|---|---|---|']
 for x in rows:
  def pair(k):return f(x['pooled'][k])+' / '+f(x['scene_mean'][k])
  ranking='历史未校准参照，未拟合b' if x['decision']=='historical_uncalibrated_reference' else f"池化AUC/AP {100*x['AUC_AP_pooled']['AUC']:.3f}/{100*x['AUC_AP_pooled']['AP']:.3f}；逐景／池化不变"
  lines.append(f"| {x['model']} / {x['decision']} | {pair('mIoU')} | {pair('Cloud_recall')} | {pair('Surface_to_Cloud_FPR')} | {pair('Shadow_recall')} | {pair('nonShadow_to_Shadow_FPR')} | {ranking} |")
 lines+=['','以上性能单位均为%；差值表为百分点。Cloud召回分母为参考Cloud；Surface→Cloud FPR分母仅为参考Surface；Shadow召回分母为参考Shadow；非Shadow→Shadow FPR分母为参考Surface+Cloud。以上均是三类argmax决策，不能混称230的二类d>0召回。场景均值只对分母存在场景等权；完整有效场景数、每类支持与缺类NA见TABLE2.csv、VAL_PER_SCENE.csv。mIoU沿用union>0类的均值。','', '偏置后q=(pSurface,exp(b)·pCloud,pShadow)，用float64倍率及原父类顺序argmax；不额外softmax。b=0在全部30个图／模型组合精确复现230；30个逐景及3个池化Cloud／Surface AUC、AP完全不变，完整排序与分母见RANKING_INVARIANTS.csv。','', '**监督预算说明：Source与L8头新增使用了60景SPARCS标签来拟合偏置，不能继续称为无SPARCS监督的迁移。两个SPARCS专用头曾用这些场景训练网络；这里只列230历史未校准参照，不能当作同标签预算、同训练过程的等价方法对比。**','', '| scene／seed | 校准L8−原L8 ΔmIoU | 校准L8−校准Source ΔmIoU | Cloud→Surface净变化 | Surface→Cloud净变化 | Shadow召回变化 |','|---|---|---|---|---|---|']
 def row(x):return f"| {x['scene']} / {x['seed']} | {x['cal_L8_minus_raw_L8_mIoU_pp']:+.3f} | {x['cal_L8_minus_cal_Source_mIoU_pp']:+.3f} | {x['Cloud_to_Surface_net_delta']:+d} | {x['Surface_to_Cloud_net_delta']:+d} | {f(x['Shadow_recall_delta_pp'])} |"
 lines += [row(x) for x in scenes]
 lines+=['','六向净错误变化全量见SIX_WAY_NET_ERRORS.csv。净计数差不同于直接新增／恢复的像元数；DIRECT_PIXEL_TRANSITIONS.json保存按参考类×原预测×校准预测的27格直接转移表，可区分同一像元转移和净变化；未保存额外全景预测大数组。','','### 三个既有诊断场景','','| scene／seed | 校准L8−原L8 ΔmIoU | 校准L8−校准Source ΔmIoU | Cloud→Surface净变化 | Surface→Cloud净变化 | Shadow召回变化 |','|---|---|---|---|---|---|']
 focus={'LC80380362013144LGN00','LC81490372013362LGN00','LC80500162014247LGN00'};lines += [row(x) for x in scenes if x['scene'] in focus]
 lines+=['','## 2 简短分析','',(R/'INTERPRETATION.txt').read_text(encoding='utf-8').strip(),'', '这些validation已反复使用，输入生成方式与参考标注混杂仍在。本轮只说明固定训练侧校准在本批数据实际改变了什么；观察性修正不证明物理成因，不构成新网络算法，也不自动授权新修复训练。','','## 资源与复核','',f"完整输入前向{gpu['total_calls']}次（主体180、固定首景重复3、构建0），GPU阶段{gpu['GPU_stage_wall_seconds']:.3f}秒／1800秒预算；网络参数更新0次，新增仅3个标量。GPU分配／保留峰值{gpu['GPU_peak_bytes']/1024**2:.2f}/{gpu['GPU_reserved_peak_bytes']/1024**2:.2f}MiB。原始CPU拟合{cpu['CPU_seconds']:.3f}秒，独立CPU复核及val分析资源见成本回执；校准d为float32缓存，拟合与偏置应用为float64。",'', '独立CPU重解3个固定偏置、复核损失及导数，独立重算全部60个raw／calibrated逐景决策与指标、8行汇总、20行关键差值、33组排序不变量及直接转移表。校准实际输入文件SHA在CALIBRATION_ACTUAL_INPUTS.json，校准缓存及代码完整SHA在交付manifest；训练增强SHA没有冒充文件SHA。未读取任何test、L8或源域像元；validation只读原230缓存。']
 if pending:
  lines.insert(1,'\n**状态：拟合及validation评价已完成，全部validation指标独立复核通过；校准缓存尚缺68个本地文件，偏置求解独立复算未完成。Phase231尚未完整验收。**')
  lines=[v.replace('独立CPU重解3个固定偏置、复核损失及导数，独立重算','偏置求解的独立重解、损失及导数复核尚待网络恢复；已独立重算') for v in lines]
 (R/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 for n in ['build_phase231.py','prepare_phase231.py','execute_phase231.py','controller_phase231.py','fit_phase231.py','phase231_bias.py','receive_phase231.py','validate_phase231_bias.py','phase231_metrics.py','analyze_phase231.py','validate_phase231.py','finalize_phase231.py']:shutil.copy2(REPO/'src/tools'/n,R/'code'/n)
 # Preserve exact remote GPU/fit code and their seal independently of local tooling.
 bias_check=js(R/'BIAS_ACCEPTED_LOCAL.json') if (R/'BIAS_ACCEPTED_LOCAL.json').exists() else None
 cpu_total=cpu['CPU_seconds']+(bias_check['CPU_seconds'] if bias_check else 0)+js(R/'CPU_ANALYSIS_COMPLETE.json')['CPU_seconds']+js(R/'INDEPENDENT_ACCEPTANCE.json')['CPU_seconds'];assert cpu_total<3600
 remote=sum(v['bytes'] for v in js(D/'DELIVERY_MANIFEST.json').values())+(D/'DELIVERY_MANIFEST.json').stat().st_size;combined=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())+remote;assert combined<512*1024**2
 save(R/'COST_AND_IDENTITY_RECEIPT.json',{'GPU':gpu,'original_CPU_fit':cpu,'CPU_cumulative_seconds':cpu_total,'CPU_seconds_cap':3600,'independent_bias_check':bias_check,'validation_analysis':js(R/'CPU_ANALYSIS_COMPLETE.json'),'independent_val_check':js(R/'INDEPENDENT_ACCEPTANCE.json'),'combined_persistent_bytes_before_final_seal':combined,'cap_new_bytes':512*1024**2,'no_durable_TAR_duplicate':True,'cache_d_float32_remote':180,'cache_label_uint8_remote':60,'local_calibration_cache_copy':'incomplete' if pending else 'complete','bias_seal_SHA':sha(D/'BIAS_FROZEN.json'),'actual_calibration_input_lock_SHA':sha(D/'CALIBRATION_ACTUAL_INPUTS.json'),'validation_access_receipt_SHA':sha(R/'VALIDATION_ACCESS_STARTED.json'),'validation_input_receipt_SHA':sha(R/'VALIDATION_INPUTS_USED.json'),'code_SHA':{p.name:sha(p) for p in (R/'code').glob('*.py')},'new_neural_calls_for_validation':0,'network_updates':0,'supervision_added':'60originalSPARCS target_train reference scenes for three scalar biases; no longer unsupervised SPARCS transfer'})
 (R/'HANDOFF.md').write_text('Current delivery: COMPLETE.183 calibration-only forwards,0validation forwards,0network updates.3train-only biases all SHA sealed before local validation cache access. Independent60-bisection bias/loss/gradient check and all validation decision/metric/ranking/flow checks passed. Full RESULT+CSV+180d/60labels caches retained locally/shared,verified hashes,combined512MiB and CPU60min budgets passed. No pending process or new phase authorized.\n',encoding='utf-8')
 if pending:
  (R/'HANDOFF.md').write_text('Current delivery: PARTIAL, NOT COMPLETE. GPU183calls and3train-only biases completed/SEALED. Validation using only original230localcache completed after seals; independent all60decision metrics/8summary/20scene delta/33rank invariants passed. Bias numerical replay pending: SSH/FRP closed all bounded reconnects, local213of281remote files verified;68missing including all60label.npy and8d.npy. Preserve all completed files and partials. Remote complete cache/shared/phase231_20261026 has180d+60label, original manifestSHA f4e996bbe2c4d8240025daf5009fd8d78d3fea34a2cecae6d70d46ee88a5b16d. No GPU/fit/val reruns. On SSH recovery only resume missing68files, run validate_phase231_bias.py once, preserve RESULT_PENDING_BIAS_AUDIT.md, setLOCAL_EXIT0 after acceptance and finalize_phase231.py without pending flag. Existing val metrics must be reused. All failed receiver records retained; no live model or receiver. Final RESULT currently explicitly pending numerical bias audit. No nextphase.\n',encoding='utf-8')
  shutil.copy2(R/'RESULT.md',R/'RESULT_PENDING_BIAS_AUDIT.md')
 else:
  s=REPO/'src/research_results/phase231_20261026';assert not s.exists();s.mkdir(parents=True)
  for n in ['RESULT.md','TABLE1.csv','TABLE2.csv','TABLE3.csv','RANKING_INVARIANTS.csv','SIX_WAY_NET_ERRORS.csv','INDEPENDENT_ACCEPTANCE.json','COST_AND_IDENTITY_RECEIPT.json']:shutil.copy2(R/n,s/n)
 manifest={str(p.relative_to(R)).replace('\\','/'):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in sorted(R.rglob('*')) if p.is_file() and p not in [R/'DELIVERY_MANIFEST.json',R/'DELIVERY_COMPLETE.json',R/'DELIVERY_STATUS.json']};save(R/'DELIVERY_MANIFEST.json',manifest)
 for n,v in manifest.items():assert (R/n).stat().st_size==v['bytes'] and sha(R/n)==v['SHA']
 save(R/('DELIVERY_STATUS.json' if pending else 'DELIVERY_COMPLETE.json'),{'status':'pending_independent_bias_audit' if pending else 'complete','manifest_SHA':sha(R/'DELIVERY_MANIFEST.json'),'files':len(manifest),'all_delivered_file_SHA_and_lengths_verified':True,'all_remote_calibration_files_locally_present':not pending,'next_phase_started':False});print(json.dumps({'status':'pending_independent_bias_audit' if pending else 'complete','files':len(manifest),'combined_new_bytes':combined,'CPU_cumulative_seconds':cpu_total,'result':str(R/'RESULT.md')}))
if __name__=='__main__':main()
