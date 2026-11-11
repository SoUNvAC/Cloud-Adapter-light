"""Seal independently accepted four-fit L8 exploration, no new model calls."""
from pathlib import Path
import json,csv,hashlib,shutil,time
from analyze_phase216 import peak
R=Path(__file__).resolve().parents[2];O=R/'outputs/phase222/phase222_20261024';D=O/'delivery';SUMMARY=R/'src/research_results/phase222_20261024'
def js(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
 assert js(O/'INDEPENDENT_CPU_ACCEPTANCE.json')['status']=='passed';complete=js(D/'COMPLETE.json');assert complete['units']==99190 and complete['GPU_seconds']<21600
 localpre=js(O/'LOCAL_PREFLIGHT.json');old={}
 for phase,v in localpre['historical'].items():
  date='20261023' if phase in ['217','219'] else '20261024';p=R/f'outputs/phase{phase}/phase{phase}_{date}';m=js(p/'DELIVERY_MANIFEST.json');assert sha(p/'DELIVERY_MANIFEST.json')==v['manifest_SHA'];assert all((p/n).stat().st_size==x['bytes'] and sha(p/n)==x['SHA'] for n,x in m.items());old[phase]={'manifest_SHA':v['manifest_SHA'],'unchanged_files':len(m)}
 receipt=js(O/'REMOTE_DELIVERY_RECEIPT.json');localbytes=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());uploads=sum(p.stat().st_size for p in O.glob('*.zip'));combined=receipt['remote_scope_bytes']+receipt['bytes']+localbytes+uploads
 assert combined<4_000_000_000 and peak()<16*1024**3 and shutil.disk_usage(R).free>=4*1024**3
 resources=[js(D/f'seed{s}/TRAINING_COMPLETE.json') for s in [21701,21702]]+[js(D/'evaluation/COMPLETE.json')];cpu=max(v['CPU_peak_bytes'] for v in resources);gpu=max(v['GPU_allocated_peak'] for v in resources);reserved=max(v['GPU_reserved_peak'] for v in resources);assert cpu<16*1024**3 and max(gpu,reserved)<20*1024**3
 heads={}
 for s in [21701,21702]:
  for a in ['C','D']:
   p=D/f'seed{s}/heads/{a}_iter4000.pth';assert sha(p)==js(D/f'seed{s}/{a}_TRAIN_COMPLETE.json')['head_SHA'];heads[str(p.relative_to(O))]={'SHA':sha(p),'bytes':p.stat().st_size}
 save('COST_AND_IDENTITIES.json',{'units':99190,'GPU_stage_seconds':complete['GPU_seconds'],'fits':4,'updates_each':4000,'failed_attempt_units':0,'failed_attempt_updates':0,'failure_preserved':'RECOVERY_01.json, remote preparation_failure_01','counts':js(D/'REMOTE_CPU_ACCEPTANCE.json')['calls'],'Source_SHA':'64dd9a20288c35ab3362b8b1c1dafc216d9c44d60868475177e139d6be859dc9','heads':heads,'remote_CPU_peak_bytes':cpu,'local_CPU_peak_bytes':peak(),'GPU_allocated_peak':gpu,'GPU_reserved_peak':reserved,'combined_new_persistent_upper_before_seal':combined,'old_manifests_verified':old,'Source_initial_training_cost':'unknown','new_timer_download_push':False,'input_SHA':sha(D/'INPUT_SUPPORT_LOCK.json'),'train_input_SHA':sha(D/'L8_TRAIN_INPUT_LOCK.json'),'code_SHA':sha(D/'CODE_LOCK.json'),'source_selection_SHA':sha(D/'evaluation/SOURCE_SELECTION_LOCK.json')})
 end=rows('TABLE_A_ENDPOINTS.csv');ci=rows('PAIRED_CI.csv');diff=rows('TABLE_B_DIFFERENCES.csv');joint=rows('JOINT_RULE.csv');direct=rows('SOURCE_DIRECT_CORRECTNESS.csv');pre=js(D/'INPUT_PREFLIGHT_ACCEPTED.json')
 def interval(k):
  v=next(v for v in ci if v['context']=='endpoint_alpha1' and v['scope']=='mean_of_two' and v['metric']==k);return f"{float(v['point']):+.3f} [{float(v['CI95_low']):+.3f}, {float(v['CI95_high']):+.3f}]"
 at='\n'.join(f"|{v['seed']}|{v['arm']}|{float(v['test_full_mIoU']):.3f}|{float(v['test_yes_mIoU']):.3f}|{float(v['test_no_mIoU']):.3f}|{float(v['pool_loss_pp']):.3f}|{float(v['T11_pp']):.3f}|{float(v['worst_ROI_loss_pp']):.3f}|{v['ROI_loss_gt5']}|" for v in end)
 bt='\n'.join('|'+label+'|'+interval(k)+'|' for label,k in [('完整16景目标mIoU','test_full_mIoU_diff'),('yes六景目标mIoU','test_yes_mIoU_diff'),('no十景目标mIoU','test_no_mIoU_diff'),('源池化损失','source_pool_loss_diff_pp'),('源T11','T11_diff_pp')])
 jt='；'.join(f"{s}{a}="+'/'.join(next(v['alpha'] for v in joint if v['seed']==str(s) and v['arm']==a and v['budget_pp']==str(b)) for b in [1,3,5]) for s in [21701,21702] for a in ['C','D'])
 shadow='；'.join(f"seed{s} C错→D对{next(v['C_wrong_D_right'] for v in direct if v['seed']==str(s) and v['ROI']=='__pooled__' and v['reference_class']=='Shadow' and v['teacher_correctness']=='all')}、C对→D错{next(v['C_right_D_wrong'] for v in direct if v['seed']==str(s) and v['ROI']=='__pooled__' and v['reference_class']=='Shadow' and v['teacher_correctness']=='all')}" for s in [21701,21702])
 rr=[v for v in diff if v['context']=='endpoint_alpha1' and v['seed']!='mean_of_two'];judgments=[]
 for key,label,lower in [('source_pool_loss_diff_pp','源整体损失',True),('T11_diff_pp','源尾部损失',True),('test_full_mIoU_diff','目标完整test mIoU',False),('test_yes_mIoU_diff','有影层mIoU',False),('test_no_mIoU_diff','未标影层mIoU',False)]:
  vals=[float(v[key]) for v in rr];better=all(x<0 for x in vals) if lower else all(x>0 for x in vals);worse=all(x>0 for x in vals) if lower else all(x<0 for x in vals);judgments.append(label+('两seed均改善' if better else '两seed均恶化' if worse else '两seed方向不同或含零差'))
 report=f'''# Phase222：均匀源蒸馏在L8的配对复核

## 1．数据结果

只新增C（目标监督）／D（同监督＋均匀源KL，λ=T=1）两seed21701／21702，各4000步，均从原Source重启，仅更新1,316,516个head参数。固定L8 Shadows=yes train1560输入／{pre['support']['train']['unique_scenes']}scene、val963／8scene，完整test2167／16scene（yes6、no10）；三split场景无交集。源回放256 RGB／249 ROI，沿用原4000顺序且不读其标签；源val535／107 ROI。旧最佳head未替代新C，无新方案或参数搜索。

α=1；mIoU为%，其余损失为百分点。源损失＝Source−候选；T11为最差11 ROI的带符号损失均值。

|seed|臂|test全16景|yes6景|no10景|源池化损失|T11|最差ROI损失|>5pp ROI|
|---|---|---:|---:|---:|---:|---:|---:|---:|
{at}

主要D−C：两个seed指标差均值及条件95%区间。目标按完整16景及各层分别抽样，源按107 ROI；10000次、seed217023，同索引配对两seed，逐seed重算后平均差，T11每次重排。

|指标|差值pp [95%CI]|
|---|---|
{bt}

源参考Shadow直接转移：{shadow}。教师原正确／错误区域及所有类和107 ROI详表保留，不只列改善区域。

源池化B1／3／5且T11≤5的最大α（0回退）在连接目标前锁定：{jt}。各工作点完整／分层目标收益和源代价见JOINT_RULE；所有α、类分母与NA、错误六方向及逐组混淆均交付。SPARCS历史只另表参照，不与L8合并。

4次训练和评价实际99,190调用，GPU阶段{complete['GPU_seconds']:.1f}s，零调用路径规范化故障已保留；无额外拟合。独立验收包括输入与USGS身份、C/D逐步配对、冻结与KL门、Source精确混淆复现、浮点64插值、四轴边际、所有工作点及完整bootstrap。新持久量上界{combined/1e9:.3f}GB，只交付四个head，没有完整backbone或logits。

## 2．结论探讨

{'；'.join(judgments)}。目标与保源两端分别解释；源整体下降不代表尾部没有残留损伤，联合点收益必须结合已锁α和源代价阅读。区间跨零不证明目标非劣或等效。复现D的收益也不构成新方法。

这是已多次查看test上的跨域探索，不是独立确认。USGS Shadows=no不是经核验的真实无云影，所报FPR仅为发布参考下的名义误报；源ROI不保证地理独立、回放空间独立性未知。条件区间不覆盖训练seed总体或重新选α的不确定性；不混入历史seed64／21601，不自动启动下一phase。
'''
 (O/'RESULT.md').write_text(report,encoding='utf-8');SUMMARY.mkdir(parents=True,exist_ok=True)
 for n in ['RESULT.md','TABLE_A_ENDPOINTS.csv','TABLE_B_DIFFERENCES.csv','JOINT_RULE.csv','PAIRED_CI.csv','COST_AND_IDENTITIES.json','INDEPENDENT_CPU_ACCEPTANCE.json']:shutil.copy2(O/n,SUMMARY/n)
 (O/'HANDOFF.md').write_text('Current delivery: COMPLETE. Four fixed L8 C/D4000 fits; all alpha source/val/full16 test and yes6/no10 strata; independent integer/bootstrap/identity acceptance passed. See RESULT.md. No timers, automatic push, or next phase.\n');files=[p for p in D.rglob('*') if p.is_file()]+[p for p in (O/'local_code').rglob('*') if p.is_file()]+[p for p in O.iterdir() if p.is_file() and p.suffix in ['.md','.json','.csv'] and p.name not in ['DELIVERY_COMPLETE.json','DELIVERY_MANIFEST.json','LOCAL_STATUS.json']];m={str(p.relative_to(O)):{'bytes':p.stat().st_size,'SHA':sha(p)} for p in files};save('DELIVERY_MANIFEST.json',m);assert all((O/n).stat().st_size==v['bytes'] and sha(O/n)==v['SHA'] for n,v in m.items());save('DELIVERY_COMPLETE.json',{'status':'passed','artifacts':len(m),'manifest_SHA':sha(O/'DELIVERY_MANIFEST.json'),'units':99190,'GPU_seconds':complete['GPU_seconds'],'epoch':time.time()});print(json.dumps({'status':'complete','report':str(O/'RESULT.md')}))
if __name__=='__main__':main()
