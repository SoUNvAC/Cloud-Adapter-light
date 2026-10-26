"""Delivery review of existing tables only; preserve original scientific outputs and code locks."""
import subprocess
from phase317_common import *
def main():
 assert load(ROOT/'STATUS.json')['status']=='completed' and not (ROOT/'DELIVERY_LOCK.json').exists();assert load(ROOT/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed';resource_check()
 old=load(ROOT/'CPU_RUN_LOCK.json')
 for n,h in old['code'].items():
  if n!='src/tools/transfer_phase317.py':assert sha(R/n)==h,n
 for n,h in load(ROOT/'GPU_CODE_LOCK.json').items():assert sha(R/n)==h,n
 for n in ['RESULT.md','phase317.md','DECISION.csv']:shutil.copyfile(ROOT/n,ROOT/('PRE_DELIVERY_'+n))
 rows=read(ROOT/'METRICS.csv');cost=[]
 def val(pool,a,k):return float(next(r[k] for r in rows if r['pool']==pool and r['strategy']==a))
 for pool in ['CMC','SPARCS','L8']:
  cost.append(dict(pool=pool,CB_minus_PL_Shadow_recall_pp=val(pool,'CB','macro_Shadow_recall_pct')-val(pool,'PL','macro_Shadow_recall_pct'),CB_minus_PL_Shadow_FPR_pp=val(pool,'CB','macro_Shadow_FPR_pct')-val(pool,'PL','macro_Shadow_FPR_pct'),CB_minus_PL_group_mIoU_pp=val(pool,'CB','group_mean_mIoU_pct')-val(pool,'PL','group_mean_mIoU_pct'),CB_minus_PL_Shadow_IoU_pp=val(pool,'CB','Shadow_IoU_pct')-val(pool,'PL','Shadow_IoU_pct'),PL_minus_Source_Shadow_recall_pp=val(pool,'PL','macro_Shadow_recall_pct')-val(pool,'Source','macro_Shadow_recall_pct')))
 assert all(r['CB_minus_PL_Shadow_recall_pp']>0 and r['CB_minus_PL_Shadow_FPR_pp']>0 and r['CB_minus_PL_group_mIoU_pp']<0 for r in cost)
 table(ROOT/'CB_COST_REVIEW.csv',cost)
 text=(ROOT/'RESULT.md').read_text();opening='三份数据都已跑通。普通PL三池均损失Shadow宏召回；简单CB相对PL三池都恢复召回，但三池组均mIoU都下降、Shadow误报都增加，CMC的Shadow IoU还明显下降。因此均衡有作用，却没有解决收益与代价的矛盾。值得继续的是围绕新增误报和Cloud纠正撤销提出一个具体可检验候选；当前尚未冻结新方法，不能把CB改名当创新，也不自动启动下一轮训练。'
 first=text.index('\n\n');second=text.index('\n\n',first+2);text=text[:first+2]+opening+text[second:]
 text+='\n逐池CB−PL（百分点）：'+ '；'.join(f"{r['pool']}召回{r['CB_minus_PL_Shadow_recall_pp']:+.3f}、FPR{r['CB_minus_PL_Shadow_FPR_pp']:+.3f}、组均mIoU{r['CB_minus_PL_group_mIoU_pp']:+.3f}" for r in cost)+'。这些是本seed的训练干预效应，不证明伪标签噪声是根因；本轮没有读取fit真值作这种归因。后续候选必须在同支持下少付新增地表误报/Cloud纠正撤销代价，不能重开已收口SP/CM或无目标加模块。\n'
 text+='\n27格/六向表沿旧通用列名：按comparison字段解释起点和终点；PL→CB行的Source_pred/Source_errors列代表PL，student_pred/student_errors代表CB，不代表原Source策略。\n'
 (ROOT/'RESULT.md').write_text(text,encoding='utf-8')
 decision=read(ROOT/'DECISION.csv')
 for r in decision:
  r['decision']='recall recovers but groupmIoU/FPR tradeoff persists; continue one targetedhypothesis design only, no newmethodtraining authorizedbythisresult'
  r['next_gap']='need a testable revision that avoids CB extra Surfacefalsepositives/Cloudcorrectionwithdrawal at matchedsupport; rootcause not established, no oldSP/CMreopen or threshold/seedscan'
 table(ROOT/'DECISION.csv',decision)
 with (ROOT/'phase317.md').open('a',encoding='utf-8') as f:
  f.write('\n- 具体判读：三池CB−PL的Shadow召回均增加、FPR均增加、组均mIoU均下降，完整数值见CB_COST_REVIEW；CMC ShadowIoU下降，不能只报召回改善。继续的是明确代价约束下的候选假设设计，尚无新方法立项或新增训练。fit真值未用于噪声归因。\n')
  f.write('- 资源：独立CPU统计峰值RSS4746424320字节，参考/统计/独立审计/报告均退出0；原GPU/CPU代码快照和原判读保留。交付仅补整流验收检查点，以便最终回执SSH失败时只补取回执，不重传。resolved_config保存原模型解析配置，实际手工优化器/1000日程以本轮锁、脚本及曲线为准。\n')
 # A complete code closure is small; original dependencies must match the GPU launch commit.
 deps=[R/'src/tools'/n for n in ['execute_phase302.py','worker_phase304.py','phase302_images.py','analyze_phase304.py','transfer_phase302.py','phase304_common.py','prepare_phase302.py']]
 deps+=list((R/'src/cloud_adapter/models').rglob('*.py'))+list((R/'src/configs').rglob('*.py'))
 subprocess.run(['git','diff','--exit-code','28951bd','--','src/cloud_adapter/models','src/configs',*[p.relative_to(R).as_posix() for p in deps[:7]]],cwd=R,capture_output=True,check=True)
 closure=ROOT/'dependencies';code=ROOT/'delivery_code';code.mkdir();dep_hash={}
 for p in deps:
  n=p.relative_to(R).as_posix();target=closure/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);dep_hash[n]=sha(p)
 for p in (R/'src/tools').glob('*phase317*.py'):shutil.copyfile(p,code/p.name)
 write(ROOT/'DEPENDENCY_SHA256.json',dep_hash);assert budget_bytes()<=16000000000
 write(ROOT/'DELIVERY_LOCK.json',dict(status='sealed',science_outputs_unchanged=['all endpoints','predictions','METRICS','27cells','sixflows','native/scenestatistics','reference/support','trainingcurves'],original_CPU_run_sha256=sha(ROOT/'CPU_RUN_LOCK.json'),original_GPU_code_sha256=sha(ROOT/'GPU_CODE_LOCK.json'),delivery_code={p.name:sha(p) for p in code.iterdir()},table_review_source_sha=sha(ROOT/'METRICS.csv'),old_report_preserved=True,no_new_raw_reference_reads=True,no_new_forwards=True))
 print('Phase317 finaldecision reviewed; originalscience preserved; delivery/dependency closure sealed')
if __name__=='__main__':main()
