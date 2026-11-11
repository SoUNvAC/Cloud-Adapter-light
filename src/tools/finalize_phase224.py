"""Compact data/interpretation report after independent H-D acceptance."""
from pathlib import Path
import json,csv,hashlib,shutil,time
from analyze_phase216 import peak
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase224/phase224_20261024';D=O/'delivery';SUMMARY=R/'src/research_results/phase224_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert js(O/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed';lock=js(D/'EXPERIMENT_LOCK.json');complete=js(D/'COMPLETE.json');assert complete['units']==100460 and complete['GPU_seconds']<21600;old={}
 for phase,v in lock['history'].items():
  p=R/f'outputs/phase{phase}/phase{phase}_{v["date"]}';assert sha(p/'DELIVERY_MANIFEST.json')==v['manifest_SHA'];m=js(p/'DELIVERY_MANIFEST.json');assert all((p/n).stat().st_size==x['bytes'] and sha(p/n)==x['SHA'] for n,x in m.items());old[phase]={'unchanged_files':len(m),'manifest_SHA':v['manifest_SHA']}
 receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');local=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());uploads=sum(p.stat().st_size for p in O.glob('*.zip'));combined=receipt['remote_scope_bytes']+receipt['bytes']+local+uploads;resources=[js(D/j/'TRAINING_COMPLETE.json') for j in lock['jobs']]+[js(D/'evaluation/COMPLETE.json')];cpu=max(v['CPU_peak_bytes'] for v in resources);gpu=max(v['GPU_allocated_peak'] for v in resources);reserved=max(v['GPU_reserved_peak'] for v in resources);assert combined<4000000000 and cpu<16*1024**3 and peak()<16*1024**3 and max(gpu,reserved)<20*1024**3 and shutil.disk_usage(R).free>=4*1024**3
 heads={}
 for job in lock['jobs']:
  p=D/job/'heads/H_iter4000.pth';assert sha(p)==js(D/job/'H_TRAIN_COMPLETE.json')['head_SHA'];heads[job]={'SHA':sha(p),'bytes':p.stat().st_size}
 save('COST_AND_IDENTITIES.json',{'units':100460,'counts':js(D/'REMOTE_CPU_ACCEPTANCE.json')['counts'],'GPU_seconds':complete['GPU_seconds'],'fits':4,'updates_each':4000,'lambda':4,'T':1,'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','head_only_delivery':heads,'CPU_remote_peak_bytes':cpu,'CPU_local_peak_bytes':peak(),'GPU_allocated_peak':gpu,'GPU_reserved_peak':reserved,'combined_new_persistent_upper_before_seal':combined,'old217_to223_unchanged':old,'input_model_code_lock_SHA':sha(D/'EXPERIMENT_LOCK.json'),'code_SHA':sha(D/'CODE_LOCK.json'),'source_selection_SHA':sha(D/'evaluation/SOURCE_SELECTION_LOCK.json'),'training_gradient_magnitudes':'not recorded; finite checks passed, no causal gradient attribution','initial_Source_training_cost':'unknown','new_timer_download_auto_push':False})
 A=rows('TABLE_A_ENDPOINTS.csv');B=rows('TABLE_B_H_MINUS_D.csv');ci=rows('PAIRED_CI.csv');finite=rows('FINITE_25_PAIRS_ALL.csv');joint=rows('JOINT_RULE.csv');corr=[v for v in rows('TEACHER_CORRECTNESS_ALPHA1.csv') if v['ROI']=='__pooled__'];focus=rows('FOCUS_ROI0124_ALPHA025_AND1.csv')
 at='\n'.join(f"|{v['target']}|{v['arm']}|{float(v['test_mIoU']):.3f}|{float(v['pool_loss_pp']):.3f}|{float(v['T11_pp']):.3f}|{float(v['worst_ROI_loss_pp']):.3f}|{float(v['ROI_loss_gt5']):.1f}|" for v in A if v['seed']=='mean_of_two')
 def interval(t,context,k):
  v=next(v for v in ci if v['target']==t and v['context']==context and v['scope']=='mean_of_two' and v['metric']==k);return f"{float(v['point']):+.3f} [{float(v['CI95_low']):+.3f},{float(v['CI95_high']):+.3f}]"
 bt='\n'.join(f"|{v['target']}|{v['context']}|{interval(v['target'],v['context'],'test_mIoU_diff_pp')}|{interval(v['target'],v['context'],'source_pool_loss_diff_pp')}|{interval(v['target'],v['context'],'T11_diff_pp')}|{v['H_alpha']} / {v['D_alpha']}|" for v in B if v['seed']=='mean_of_two')
 ft='；'.join(f"{t}/{s}：H占优{sum(v['H_dominates_D']=='True' for v in finite if v['target']==t and v['seed']==str(s))}/25，其余{sum(v['H_dominates_D']!='True' for v in finite if v['target']==t and v['seed']==str(s))}/25" for t in ['SPARCS','L8'] for s in [21701,21702])
 ct='；'.join(f"{v['target']}/{v['seed']}：教师原正确少错{int(v['teacher_correct_H_fewer_errors']):+,}，教师原错误少纠正{int(v['teacher_wrong_H_fewer_corrections']):+,}，净错误{int(v['H_minus_D_errors']):+,}" for v in corr)
 focus_text=[]
 for t in ['SPARCS','L8']:
  for s in [21701,21702]:
   focus_text.append(f"{t}/{s} H的α.25/1损失="+'/'.join(f"{float(next(v['loss_pp'] for v in focus if v['target']==t and v['seed']==str(s) and v['arm']=='H' and float(v['alpha'])==a)):.3f}" for a in [.25,1])+'pp')
 judgments=[]
 for t in ['SPARCS','L8']:
  rr=[v for v in B if v['target']==t and v['context']=='endpoint_alpha1' and v['seed']!='mean_of_two'];text=[]
  for k,name,lower in [('source_pool_loss_diff_pp','源平均损失',True),('T11_diff_pp','源尾部损失',True),('test_mIoU_diff_pp','目标完整test mIoU',False)]:
   vv=[float(v[k]) for v in rr];better=all(x<0 for x in vv) if lower else all(x>0 for x in vv);worse=all(x>0 for x in vv) if lower else all(x<0 for x in vv);text.append(name+('两seed均改善' if better else '两seed均恶化' if worse else '两seed方向不同或含零差'))
  judgments.append(t+'：'+'、'.join(text)+'。')
 report=f'''# Phase224：更强均匀源KL（λ=4）固定对照

## 1．数据结果

新增H四次4000步：SPARCS/L8×原seed21701/21702，从同一Source重启，只更新1,316,516个head参数。唯一配方变化是源均匀KL的λ=1→4，T=1；原数据／标签／增强SHA、RNG、256图／249 ROI回放及4000顺序、优化器与调度固定。C/D只复用历史。SPARCS train60scene、val/test各10；L8 train1560/18scene、val963/8、完整test2167/16（yes6/no10）；源val535/107 ROI。

表A摘要为两个seed指标均值，α=1；mIoU为%，损失为pp。最差ROI损失也是两seed各自最差值的均值，完整18行个别seed／类支持另交TABLE_A。

|目标|臂|test mIoU|源池化损失|T11|最差ROI损失|>5pp ROI数均值|
|---|---|---:|---:|---:|---:|---:|
{at}

表B为H−D两seed差均值及条件95%区间；H/Dα依seed21701/21702成对列出。源侧B1/3/5与T11≤5规则先封存再连接目标。10000次seed217023，同组索引配对两seed，逐seed重算后平均，T11逐次重排。

|目标|工作点|目标差pp [CI]|源损失差pp [CI]|T11差pp [CI]|Hα / Dα|
|---|---|---|---|---|---|
{bt}

L8 yes/no两层、各seed原值、完整60候选、逐组混淆／类IoU分母／NA／召回／名义FPR、六方向净错误及全部107 ROI均交付。ROI_0124：{'；'.join(focus_text)}；α.25/1的C/D/H全值另表，α.25仅净混淆变化。

表C有限取舍：{ft}。占优定义为目标不低、源均值损失不高、T11不高且至少一项严格改善；全部100对（含不利点）见FINITE_25_PAIRS_ALL，不据test选部署点。

α=1同像元保护／纠错：{ct}。净错误＝少纠正−少错；原D逐图／ROI／预测SHA复现，三参考类直接D错→H对／D对→H错另表。此计数不分解非线性mIoU。

四fit实际100,460单位，GPU阶段{complete['GPU_seconds']:.1f}s，新增持久量上界{combined/1e9:.3f}GB；只传四个head。独立验收通过配对输入、λ唯一变化、4000步／冻结／梯度有限性／原始与4倍KL记录、全α、Source/D复现、四轴边际和bootstrap。LOSS_COMPARISON给原D/H首末100步及全程原始KL、乘λ项和目标loss；不把训练末KL或未测梯度幅度假定为四倍。

## 2．结论探讨

{' '.join(judgments)} 源平均、尾部、目标收益及教师错误纠正分别解释，不能由保源改善自动宣称综合胜出。H与D整组有限候选的取舍以全100对为限，不证明连续前沿、λ最优性或机制创新。

已查看test上的探索；区间不覆盖训练seed总体、重选α或回放空间关联。源107 ROI不保证地理独立，两目标重复使用它们不增加独立样本。L8 Shadows=no不是核验真无云影，误报仅为发布参考下的名义指标；不平均两目标增益、不追加训练或下一phase。
'''
 (O/'RESULT.md').write_text(report,encoding='utf-8');SUMMARY.mkdir(parents=True,exist_ok=True)
 for n in ['RESULT.md','TABLE_A_ENDPOINTS.csv','TABLE_B_H_MINUS_D.csv','TABLE_C_ALL_CANDIDATES.csv','JOINT_RULE.csv','PAIRED_CI.csv','TEACHER_CORRECTNESS_ALPHA1.csv','COST_AND_IDENTITIES.json','INDEPENDENT_CPU_ACCEPTANCE.json']:shutil.copy2(O/n,SUMMARY/n)
 (O/'HANDOFF.md').write_text('Current delivery: COMPLETE. Four fixed H(lambda4)4000 fits, all alpha/source/targets/strata; historical C/D unchanged. Independent metrics/10000 bootstrap/direct J/training identity accepted. See RESULT.md. No timer/push/new phase.\n');files=[p for p in D.rglob('*') if p.is_file()]+[p for p in (O/'local_code').rglob('*') if p.is_file()]+[p for p in O.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in ['DELIVERY_MANIFEST.json','DELIVERY_COMPLETE.json','LOCAL_STATUS.json']];m={str(p.relative_to(O)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};save('DELIVERY_MANIFEST.json',m);assert all((O/n).stat().st_size==v['bytes'] and sha(O/n)==v['SHA'] for n,v in m.items());save('DELIVERY_COMPLETE.json',{'status':'passed','artifacts':len(m),'manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'units':100460,'GPU_seconds':complete['GPU_seconds'],'epoch':time.time()});print(json.dumps({'status':'complete','report':str(O/'RESULT.md')}))
if __name__=='__main__':main()
