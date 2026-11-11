"""Independent Phase208 cached-output acceptance and final reports. No model execution."""
import ast,csv,json,hashlib,math,time,sys,zipfile,shutil
from pathlib import Path
from collections import Counter
import numpy as np
R=Path('D:/Cloud-Adapter-light');O=R/'outputs/phase208/phase208_20261020';D=O/'delivery';S=R/'src/research_results/phase208_20261020'
checks=0
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def ck(v):
 global checks
 assert v
 checks+=1
def rows(n):
 with (O/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(n,d):
 with (O/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def write(p,t):
 with p.open('x',encoding='utf8') as f:f.write(t)
def table(headers,data):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in data)
def f(v):return 'NA' if v is None or v=='' else f'{float(v):.3f}'
def main():
 start=time.time();ck((O/'LOCAL_EXIT_CODE').read_text().strip()=='0');ck(load(O/'CPU_ACCEPTANCE.json')['status']=='passed')
 lock=load(O/'INPUT_DELIVERY_LOCK.json');ck(sha(O/'phase208_delivery.zip')==lock['archive_sha256']);ck((O/'phase208_delivery.zip').stat().st_size==331684629)
 with zipfile.ZipFile(O/'phase208_delivery.zip') as z:ck(z.testzip() is None)
 for n,h in lock['manifest']['files'].items():ck(sha(D/n)==h)
 events=[json.loads(s) for s in (D/'CALL_LEDGER.jsonl').read_text().splitlines()];ck(len(events)==4342);ck([q['global_call'] for q in events]==list(range(1,4343)))
 ck(Counter(q['stage'] for q in events)==Counter(gate=5,train_M=1000,train_B=1000,infer=2292,benchmark=45))
 gate=load(D/'gate/GATE_ACCEPTANCE.json');init=load(D/'gate/COMMON_INITIALIZATION.json');seal=load(D/'TRAINING_SEAL.json')
 ck(gate['status']=='passed' and gate['optimizer_updates']==0 and gate['backward_per_arm']==1)
 ck(len(init['shared_tensors'])==18 and init['exact_initial_values_equal'] and init['all_cross_arm_storage_independent']);ck(init['M_new_parameters']==970369 and init['B_new_parameters']==944641)
 for g in gate['rows']:
  ck(g['closed_max_error']<=1e-5 and g['enabled_initial_max_error']>0);ck(g['initial_restored'] and g['gradients_cleared'] and g['original_parameters_buffers_unchanged'])
  ck(all(q['has_gradient'] and q['finite'] for q in g['gradient_rows']))
  t=load(D/('train_'+g['arm'])/'TRAINING_RECEIPT.json');ck(t['steps']==t['optimizer_steps']==1000 and t['module_seed']==20801);ck(t['initial_sha256']==g['initial_sha256'] and t['final_sha256']==seal[g['arm']]);ck(t['original_all_eval'] and t['original_parameters_buffers_unchanged']);ck(t['schedule_sha256']=='3baae5c5e13c3a3dbc515410b537d2071db87e5485953ef87ef5a783b10c0aba')
 ck(sha(R/'src/tools/phase207_msre_interface.py')=='237f1a454d1b483b957fe869f17ec917e3547fc10708b5b2b8aa707918a0362c')
 ck(load(D/'infer/SCORES_SEALED.json')['all_model_parameters_buffers_unchanged'])
 for q in load(D/'infer/SCENE_RECEIPTS.json'):ck(q['official_M_confidence_error']<=1e-5 and q['all_postprocessing_error']<=1e-5 and q['coverage_gradient_exact'])
 summaries=[];intervals=[];detail='';old=[];noshadow=[]
 for pool in ['Catalogue','PixBox']:
  per=rows(pool+'_PER_GROUP.csv');diff=rows(pool+'_GROUP_DIFFERENCES.csv');ss=load(O/(pool+'_ALL_METRICS.json'));ck(len(ss)==4 and len(per)==(128 if pool=='Catalogue' else 40));lookup={(r['strategy'],r['group_id']):r for r in per}
  for r in per:
   m=np.asarray(ast.literal_eval(r['confusion']),dtype=np.int64);p=int(r['positive']);n=int(r['negative']);tp=int(r['TP']);fp=int(r['FP']);ck(m.shape==(3,3) and (m>=0).all());ck(m[2].sum()==p and m[:2].sum()==n and m[2,2]==tp and m[:2,2].sum()==fp);ck(int(r['FN'])==p-tp and int(r['TN'])==n-fp);ck(r['AP']=='' if p==0 else 0<=float(r['AP'])<=1);ck(r['precision']=='' if tp+fp==0 else abs(float(r['precision'])-tp/(tp+fp))<1e-12)
   if r['tile'] in ['32TML','40XDR','19FDV','54TWN']:old.append(r)
  for s in ss:
   pr=[r for r in per if r['strategy']==s['strategy']];m=np.sum([ast.literal_eval(r['confusion']) for r in pr],axis=0);ck(m.tolist()==s['confusion']);ck(abs(math.fsum(float(r['AP']) for r in pr if r['AP']!='')/s['AP_groups']*100-s['macro_AP_pp'])<1e-10);ck(s['AP_groups']==(7 if pool=='Catalogue' else 8));ck(s['positive']==(655961 if pool=='Catalogue' else 379) and s['negative']==(32767527 if pool=='Catalogue' else 8190))
   ck(abs(np.mean([float(r['negative_shadow_fraction']) for r in pr])*100-s['macro_negative_shadow_fraction_pp'])<1e-10)
   ck(abs(max(float(r['negative_shadow_fraction']) for r in pr)*100-s['max_group_negative_shadow_fraction_pp'])<1e-10)
   if pool=='Catalogue':
    un=m.sum(0)+m.sum(1)-np.diag(m);ck(abs(np.mean(np.diag(m)[un>0]/un[un>0])*100-s['Catalogue_pooled_mIoU_pp'])<1e-10)
   else:ck(s['Catalogue_pooled_mIoU_pp'] is None)
   no=[r for r in pr if r['AP']==''];noshadow.append([pool,s['strategy'],len(no),f(100*np.mean([float(r['negative_shadow_fraction']) for r in no]))])
  for r in diff:
   l,b=r['contrast'].split('-');a=lookup[l,r['group_id']];b=lookup[b,r['group_id']]
   if a['AP']=='':ck(r['delta_AP_pp']=='' and r['category']=='NA');continue
   delta=100*(float(a['AP'])-float(b['AP']));ck(abs(delta-float(r['delta_AP_pp']))<1e-10)
  for q in load(O/(pool+'_INTERVALS.json'))['rows']:
   v=[float(r['delta_AP_pp']) for r in diff if r['contrast']==q['contrast'] and r['delta_AP_pp']!=''];idx=np.random.default_rng(208020).integers(len(v),size=(10000,len(v)));means=[math.fsum(v[i] for i in ids)/len(v) for ids in idx]
   ck(abs(math.fsum(v)/len(v)-q['mean_delta_AP_pp'])<1e-10 and np.max(np.abs(np.quantile(means,[.025,.975])-q['CI95_pp']))<1e-10);ck(q['improved']==sum(x>1 for x in v) and q['damaged']==sum(x< -1 for x in v));intervals.append(q)
  summaries+=ss
  detail+='\n## '+pool+' full group AP changes (pp)\n\n'+table(['contrast','tile','group_id','positive','delta AP','delta recall','delta negative fraction','category'],[[r['contrast'],r['tile'],r['group_id'],r['positive'],f(r['delta_AP_pp']),f(r['delta_recall_pp']),f(r['delta_negative_shadow_fraction_pp']),r['category']] for r in diff])
  detail+='\n### '+pool+' full confusion / error flows\n\n'+table(['strategy','tile','group_id','positive','negative','AP','recall','negative fraction','precision','Shadow to Surface','Shadow to Cloud','confusion'],[[r['strategy'],r['tile'],r['group_id'],r['positive'],r['negative'],f(None if r['AP']=='' else 100*float(r['AP'])),f(None if r['recall']=='' else 100*float(r['recall'])),f(100*float(r['negative_shadow_fraction'])),f(None if r['precision']=='' else 100*float(r['precision'])),r['Shadow_to_Surface'],r['Shadow_to_Cloud'],r['confusion']] for r in per])
 timings=rows('BENCHMARK_ALL24.csv');rat=load(O/'BENCHMARK_RATIOS.json');bl=load(D/'BENCHMARK_INPUT_LOCK.json');ck(bl['before_any_training'] and bl['pre_normalized'] and len(bl['rows'])==3);ck(len(timings)==24)
 for i,b in enumerate(load(D/'benchmark/BENCHMARK_TIMINGS.json')['rows']):
  ck(b['warmup_per_strategy']==1 and b['timed_repeats']==2);ck(b['order']==(['R4','E4','M','B'][i:]+['R4','E4','M','B'][:i]));ck(b['input_sha256']==bl['rows'][i]['array_sha256'] and not bl['rows'][i]['label_read'])
 for s in ['R4','E4','M','B']:
  rr=[r for r in timings if r['strategy']==s];ck(len(rr)==6 and all(int(r['network_calls'])==(2 if s=='E4' else 1) for r in rr));ck(abs(np.mean([float(r['GPU_milliseconds']) for r in rr])-rat['equal_product_and_repeat_mean_GPU_ms'][s])<1e-10)
 for s in ['R4','E4']:ck(abs(rat['M_over_'+s]-rat['equal_product_and_repeat_mean_GPU_ms']['M']/rat['equal_product_and_repeat_mean_GPU_ms'][s])<1e-12)
 statuses=[load(D/st/'RUN_STATUS.json') for st in ['gate','train_M','train_B','infer','benchmark']];remote=load(D/'CONTROLLER_STATUS.json');local=load(O/'LOCAL_STATUS.json');receipt=load(O/'DELIVERY_RECEIPT.json')
 ck(remote['total_calls']==4342 and remote['stage_seconds_total']<14400);ck(local['peak_bytes']<16*1024**3 and local['seconds']<7200)
 cost=dict(network_calls=4342,fits=2,steps_each=1000,stages=dict(Counter(q['stage'] for q in events)),GPU_stage_seconds=remote['stage_seconds_total'],remote_CPU_peak_bytes=max(q['rss_peak_bytes'] for q in statuses),GPU_reserved_peak_bytes=max(q['gpu_peak_reserved'] for q in statuses),host_min_free_bytes=min(q['host_min_free_bytes'] for q in statuses),local_analysis_seconds=local['seconds'],local_peak_bytes=local['peak_bytes'],remote_new_bytes=receipt['remote_new_bytes'],ledger_sha256=sha(D/'CALL_LEDGER.jsonl'),initialization=init,training_endpoints=seal,source_code_locks=load(D/'CODE_LOCK.json'),input_locks=load(D/'INPUT_LOCK.json'),local_code_lock=load(O/'LOCAL_CODE_LOCK.json'),receiver_failure='Date template mismatch stopped before SCP; logs retained; 0 additional fits/forwards; repaired receiver exited 0',unknown=['independent deployment peak','full author V4 training-data overlap','unmeasured host physical peak between samples'])
 ck(cost['GPU_reserved_peak_bytes']<20*1024**3 and cost['host_min_free_bytes']>=4*10**9)
 S.mkdir(parents=True,exist_ok=True)
 report='# Phase208 数据结果\n\n四策略：R4原V4单成员；E4原官方双成员集成；M为MsRE-inspired接口适配器；B为普通残差对照。原100 fit组、seed20801，两臂各1000步，仅末步评价。Catalogue32相关组（7组可算AP）；PixBox16产品/10相关组（8组可算AP）。默认raw四类argmax后合并Cloud，无阈值扫描。\n\n所有率及AP单位为%，差值单位为百分点；NA不填零。PixBox只代表参考点支持，不称全景FPR或mIoU。\n\n'
 report+=table(['pool','strategy','new params','macro AP','pooled AP','macro recall','macro negative fraction','max group negative fraction','macro precision','pooled recall','pooled precision','pooled negative fraction','Catalogue mIoU','Surface IoU','Cloud IoU','Shadow IoU','zero recall groups'],[[s['pool'],s['strategy'],s['new_parameters']]+[f(s[k]) for k in ['macro_AP_pp','pooled_AP_pp','macro_recall_pp','macro_negative_shadow_fraction_pp','max_group_negative_shadow_fraction_pp','macro_precision_pp','pooled_recall_pp','pooled_precision_pp','pooled_negative_shadow_fraction_pp','Catalogue_pooled_mIoU_pp','Catalogue_Surface_IoU_pp','Catalogue_Cloud_IoU_pp','Catalogue_Shadow_IoU_pp']]+[s['zero_recall_AP_groups']] for s in summaries])
 report+='\n## 固定比较\n\n配对相关组bootstrap10000次，seed208020；单seed、历史已使用评价池的描述结果，不是新独立确认。改善/稳定/下降按±1 AP百分点。\n\n'+table(['pool','contrast','mean delta AP','95% CI','improved/stable/damaged','worst delta AP','max recall loss'],[[q['pool'],q['contrast'],f(q['mean_delta_AP_pp']),'['+', '.join(f(x) for x in q['CI95_pp'])+']',f"{q['improved']}/{q['stable']}/{q['damaged']}",f(q['worst_delta_AP_pp']),f(q['largest_recall_drop_pp'])] for q in intervals])
 report+='\n## 无参考云影组\n\n'+table(['pool','strategy','groups','mean negative fraction %'],noshadow)
 report+='\n## 固定窗口计时\n\n'+table(['strategy','six-repeat mean GPU ms'],[[k,f(v)] for k,v in rat['equal_product_and_repeat_mean_GPU_ms'].items()])+f"\nM/E4={rat['M_over_E4']:.3f}，M/R4={rat['M_over_R4']:.3f}。全部24次原始记录见BENCHMARK_ALL24.csv，各产品均值见BENCHMARK_PRODUCT_MEANS.csv。固定预归一化1000窗口的GPU网络段（E4包含双成员和raw均值），不含整景IO/拼接；不是整景速度或独立部署显存。重复计时波动大，未剔除最慢值，不能声称稳定加速。\n\n## 成本与验收\n\n累计4342调用、GPU阶段1441.335秒；本地分析75.268秒、峰值2.284GB。SHA/CRC/逐文件传输校验和680项AP/混淆检查已过；独立算术/组bootstrap/状态与账本验收见INDEPENDENT_ACCEPTANCE.json。权重留shared，参考仅本地连接；无新推理或拟合。日期模板接收故障已留档，不增加神经调用。\n\n原始成员、产品ID、每组完整数值与错误流见GROUP_RESULTS.md及CSV；原始成本/代码/输入/初始化/端点哈希见COST_PROVENANCE.json。\n"
 conclusion='# Phase208 结论分析\n\nM相对近参数量普通对照B的Catalogue平均AP增量为0.351个百分点，组bootstrap区间[0.071,0.642]；七组变化全部落在±1百分点内。PixBox增量1.017个百分点，区间[-1.080,2.996]，4组改善、3组稳定、1组下降，最差下降5.000个百分点。结果支持本配置存在小幅、依评价池而异的增量，尚不足以称跨池稳定优势。\n\n相对原R4，M的Catalogue宏AP下降0.993个百分点，宏召回下降8.325个百分点；但平均负例误报比例从1.625%降至0.734%，池化mIoU从82.139%升至86.459%。这是误报与漏影的取舍，不能只用mIoU改善替代排序与召回损伤。PixBox的M宏AP接近R4，但宏召回从67.919%降至54.916%，相对E4宏AP下降7.904个百分点。M不能据此替代原E4。\n\nM相对B召回较高，但误报也略高；两个适配器共同表现出的保守云影决策说明普通残差已解释了相当一部分变化。M新增970369参数、B944641参数，存在2.65%参数差，因此本对照不是严格相同函数容量；不能把全部增量归因于某个token机制。M为CNN接口上的MsRE-inspired变体，不等同于原跨骨干方法的直接复现。\n\n平均窗口M比E4慢约22.95%、比R4慢约55.47%，但三产品与重复间波动很大（原始值完整保留）。该小型计时不支持稳定部署速度结论，不外推整景或独立显存。\n\n限制：只有一个训练seed；AP有效相关组仅7/8，不将像元当独立样本；历史development多次暴露；官方V4完整训练覆盖未知；PixBox为点参考且54TWN仅7个正参考。原四组完整保留，不因结果不利删组。当前结果既不能证明方法整体无效，也不能证明优于原官方集成；不自动新增seed、调参或开始Phase209。\n'
 write(O/'DATA_RESULTS.md',report);write(O/'CONCLUSIONS.md',conclusion)
 write(O/'GROUP_RESULTS.md','# 全组与错误流\n\n混淆行=参考、列=预测；顺序Surface/Cloud/Shadow。所有原始精度见CSV。\n'+detail+'\n## 旧四组追踪\n\n'+table(['pool','strategy','tile','positive','AP','recall','negative fraction'],[[r['pool'],r['strategy'],r['tile'],r['positive'],r['AP'] or 'NA',r['recall'] or 'NA',r['negative_shadow_fraction']] for r in old]))
 readme='# Phase208\n\n状态：交付完成，GPU与本地分析均EXIT0。\n\n1. 数据结果：[DATA_RESULTS.md](DATA_RESULTS.md)；完整逐组与错误流：[GROUP_RESULTS.md](GROUP_RESULTS.md)。\n2. 结论分析：[CONCLUSIONS.md](CONCLUSIONS.md)。\n3. 主比较M−B：Catalogue +0.351pp [0.071,0.642]；PixBox +1.017pp [-1.080,2.996]。\n4. 两适配器降低误报但损失召回；M尚不能替代原E4。窗口计时不支持稳定加速。\n5. 独立验收、成本/代码/输入/端点哈希和交付清单分别见INDEPENDENT_ACCEPTANCE.json、COST_PROVENANCE.json、DELIVERY_COMPLETE.json。\n\n单seed、历史评价池、7/8 AP相关组；PixBox仅点指标；无额外训练、参数选择、阈值搜索或后续phase。\n'
 write(O/'README.md',readme);write(O/'RESULT.md',report+'\n'+conclusion)
 for n in ['DATA_RESULTS.md','CONCLUSIONS.md','README.md']:shutil.copyfile(O/n,S/n)
 cost['local_output_bytes_before_final_receipts']=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());cost['combined_bytes_before_final_receipts']=cost['remote_new_bytes']+cost['local_output_bytes_before_final_receipts']+sum(p.stat().st_size for p in S.rglob('*') if p.is_file());ck(cost['combined_bytes_before_final_receipts']+10000000<6000000000)
 save('COST_PROVENANCE.json',cost);save('INDEPENDENT_ACCEPTANCE.json',dict(status='passed',checks=checks,seconds=time.time()-start,code_sha256=sha(Path(__file__)),paired_bootstrap_independent_fsum=True,neural_calls_added=0,fits_added=0,source_scores_AP_confusion_checks=load(O/'CPU_ACCEPTANCE.json')['independent_AP_confusion_checks'],limitations='State and gradient assertions verified against sealed remote receipts; weights retained remote, not independently re-executed locally. Physical host free space sampled, not continuous.'))
 h=O/'HANDOFF.md';txt=h.read_text(encoding='utf8');txt=txt.replace(txt.split('\n')[2],'Current delivery: COMPLETE. Receiver46216 EXIT0; LOCAL_EXIT0; independent validation passed. Reports and manifest sealed. No additional neural calls/fits; do not restart. Automation removed on final delivery.');h.write_text(txt,encoding='utf8')
 files={str(p.relative_to(O)).replace('\\','/'):sha(p) for p in O.rglob('*') if p.is_file()};summary={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file()};total=cost['remote_new_bytes']+sum(p.stat().st_size for p in O.rglob('*') if p.is_file())+sum(p.stat().st_size for p in S.rglob('*') if p.is_file());ck(total+1000000<6000000000)
 save('DELIVERY_COMPLETE.json',dict(status='complete',files=files,summary_files=summary,combined_bytes_before_manifest=total,network_calls=4342,extra_neural_calls=0,independent_acceptance_sha256=sha(O/'INDEPENDENT_ACCEPTANCE.json')))
 for n,v in files.items():ck(sha(O/n)==v)
 print(json.dumps(dict(status='complete',checks=checks,combined_bytes=total,validator_seconds=time.time()-start)))
if __name__=='__main__':main()
