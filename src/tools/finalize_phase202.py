"""Phase202 small delivery and independent scalar verification. No fits/inference."""
import csv,hashlib,json,shutil
from pathlib import Path
R=Path('D:/Cloud-Adapter-light');O=R/'outputs/phase202/phase202_20261017';D=R/'src/research_results/phase202_20261017'
import sys
sys.path.insert(0,str(O/'software'))
import numpy as np
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def write(p,t):
 with Path(p).open('x',encoding='utf8') as f:f.write(t)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
def num(v):return 'NA' if v is None or v=='' else f'{float(v):.4f}'
def main():
 assert load(O/'LOCAL_STATUS.json')['status']=='completed' and (O/'LOCAL_EXIT_CODE').read_text().strip()=='0'
 models=load(O/'MODELS_FROZEN.json');th=load(O/'THRESHOLDS_FROZEN.json');assert sha(O/'MODELS_FROZEN.json')==th['model_lock_sha256'];assert models['O1_sign']==1
 assert len(models['models'])==4 and len(th['thresholds'])==5
 for p,h in models['model_sha256'].items():assert sha(O/('MODEL_'+p+'.json'))==h;assert models['models'][p]['converged'] and not models['models'][p]['warnings']
 checks=0;summ={};per={};contrast={}
 for pool,n in [('Calibration',53),('Catalogue',32),('PixBox',10)]:
  summ[pool]=read(O/(pool+'_SUMMARY.csv'));per[pool]=read(O/(pool+'_PER_GROUP.csv'));contrast[pool]=load(O/(pool+'_CONTRASTS.json'))['rows'];assert len(per[pool])==5*n
  for s in summ[pool]:
   rr=[r for r in per[pool] if r['setting']==s['setting']];assert len(rr)==n and len({r['group_id'] for r in rr})==n
   for field,key in [('AP','macro_AP_pp'),('recall','macro_recall_pp'),('FPR','macro_FPR_pp')]:
    values=[float(r[field]) for r in rr if r[field]!=''];assert abs(sum(values)/len(values)*100-float(s[key]))<1e-10;checks+=1
   for r in rr:
    assert float(r['threshold'])==th['thresholds'][s['setting']]['threshold'];checks+=1
    if int(r['positive'])==0:assert r['AP']==''
   assert sum(float(r['FPR'])>.01 for r in rr if r['FPR'])==int(s['groups_FPR_gt1pct'])
   assert sum(int(r['TP']) for r in rr)/sum(int(r['positive']) for r in rr)==float(s['pooled_recall'])
   assert sum(int(r['FP']) for r in rr)/sum(int(r['negative']) for r in rr)==float(s['pooled_FPR']);checks+=3
   if pool=='Calibration':
    t=th['thresholds'][s['setting']];assert float(s['macro_FPR_pp'])<=1 and abs(float(s['macro_FPR_pp'])/100-t['macro_FPR'])<1e-15;assert t['unsafe_macro_FPR']>.01 and np.nextafter(t['threshold'],-np.inf)==t['boundary'];assert [int(r['FP']) for r in rr]==t['FP'];checks+=4
  for c in contrast[pool]:
   ox={r['group_id']:float(r['AP']) for r in per[pool] if r['setting']=='OX' and r['AP']};base=[r for r in per[pool] if r['setting']==c['comparison'][3:] and r['AP']];delta=np.array([(ox[r['group_id']]-float(r['AP']))*100 for r in base]);assert abs(float(delta.mean())-c['mean'])<1e-10
   idx=np.random.default_rng(202018).integers(len(delta),size=(10000,len(delta)));draw=np.array([sum(float(delta[i]) for i in row)/len(delta) for row in idx]);assert np.max(np.abs(np.quantile(draw,[.025,.975])-c['CI95']))<1e-10;assert abs(float(np.quantile(draw,.95))-c['upper95'])<1e-10;checks+=3
 f6=read(O/'F6_ALL_GROUPS.csv');assert len(f6)==42
 for r in f6:
  if int(r['positive']):assert (float(r['C122_minus_R_AP_pp'])< -1 and float(r['C123_minus_R_AP_pp'])< -1)==(r['two_seed_AP_class']=='both_damage')
  else:assert r['C122_minus_R_AP_pp']==r['C123_minus_R_AP_pp']==''
  checks+=1
 assert {r['tiles'] for r in f6 if r['two_seed_AP_class']=='both_damage'}=={'32TML','40XDR','19FDV','54TWN'}
 starts=load(O/'FIT_STARTED.json');assert len(starts['sample_rows'])==100;assert sha(O/'FIT_ROW_IDS_PRIVATE.npz')==starts['sample_ids_sha256'];assert starts['code_sha256']==sha(R/'src/tools/analyze_phase202_local.py')
 with np.load(O/'FIT_ROW_IDS_PRIVATE.npz',allow_pickle=False) as z:
  for row in starts['sample_rows']:
   a=z[row['group_id']];assert len(a)==row['sampled']<=8192 and len(np.unique(a))==len(a);assert hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()==row['row_sha256'];checks+=1
 assert (O/'MODELS_FROZEN.json').stat().st_mtime_ns < (O/'THRESHOLDS_FROZEN.json').stat().st_mtime_ns < (O/'Catalogue_SUMMARY.csv').stat().st_mtime_ns
 save(O/'INDEPENDENT_DELIVERY_ACCEPTANCE.json',dict(status='passed',scalar_checks=checks,prior_independent_AP_checks=load(O/'CPU_ACCEPTANCE.json')['independent_AP_checks'],new_fits=0,new_neural_calls=0,model_and_sample_hashes=True,threshold_adjacent_boundary_and_counts=True,bootstrap_scalar_loop_reproduced=True,F6_all42_retained=True))
 f2='各值为百分点；AP使用有正负支持组，误报保留全部负类支持组。描述区间按相关组配对bootstrap10000，seed202018。\n\n'
 f4='冻结阈值仅来自完整53校准组。误报为相关组等权平均；PixBox是采样负类误报比例，不是全景FPR。\n\n'
 for pool in ('Catalogue','PixBox'):
  cr={c['comparison'][3:]:c for c in contrast[pool]};f2+=pool+'\n\n'+table(['评分','AP组数','宏AP','池化AP','冻结宏召回','宏误报','OX−评分AP差 [95%CI]'],[[s['setting'],s['AP_groups'],num(s['macro_AP_pp']),num(s['pooled_AP_pp']),num(s['macro_recall_pp']),num(s['macro_FPR_pp']),num(cr[s['setting']]['mean'])+' ['+', '.join(num(v) for v in cr[s['setting']]['CI95'])+']'] for s in summ[pool]])+'\n'
 for pool in ('Calibration','Catalogue','PixBox'):
  f4+=pool+'\n\n'+table(['评分','阈值','校准宏误报%','本池宏误报%','单侧95%上界%','最大组误报%','>1%组数/总组','宏召回%','零召回组'],[[s['setting'],s['threshold'],num(next(a['macro_FPR_pp'] for a in summ['Calibration'] if a['setting']==s['setting'])),num(s['macro_FPR_pp']),num(s['FPR_upper95_pp']),num(s['worst_group_FPR_pp']),s['groups_FPR_gt1pct']+'/'+s['groups'],num(s['macro_recall_pp']),s['zero_recall_groups']] for s in summ[pool]])+'\n'
 write(O/'TABLE1_F2.md',f2);write(O/'TABLE2_F4.md',f4)
 primary=[next(c for c in contrast[p] if c['primary']) for p in ('Catalogue','PixBox')]
 primarytable=table(['评价池','有影AP组','OX−ON宏AP差pp','95%描述区间pp','改善/稳定/下降'],[[p['pool'],p['n'],num(p['mean']),'['+', '.join(num(v) for v in p['CI95'])+']',f"{p['improve']}/{p['stable']}/{p['decline']}"] for p in primary])
 result='''# Phase202：Omni V1冻结双模型证据核验

## 1 数据结果

固定原100组fit、53组校准；Catalogue32组中7组可计算AP，PixBox16产品/10相关组中8组可计算AP。原E的Catalogue与PixBox历史宏AP/池化AP均复现。Catalogue共同支持655961正、32767527负；PixBox历史E为379正/8190负，新五评分共同支持379正/8188负，2个输入无效负点排除并保留逐点损失表，没有拼接旧分母。

E为Omni V1官方完整双模型最终云影分数；O1为分数logit，ON增加B04/B03/B8A，OX增加B02/B11/B12，X3仅后三波段。四个逻辑回归仅在fit拟合，L2/C=1/lbfgs及8192/组均匀样本固定；按组等权标准化，float64。四模型与五个53组完整负类支持阈值均已封存，之后才评价新读出。O1系数为正，评价池E裁剪对逐组AP增量均为0。

'''+primarytable+'''
完整五评分表见TABLE1_F2.md，五阈值/校准与迁移见TABLE2_F4.md。Catalogue E冻结阈值0.7818138599395753：宏误报1.0197%，单侧95%上界1.5759%，13/32组超过1%，宏召回66.8185%；PixBox同阈值采样误报0.3873%，上界0.6885%，1/10组超过1%，宏召回50.3428%。OX两池宏误报分别0.8008%/0.4473%，上界1.3252%/0.7828%，宏召回69.2580%/61.8586%，各有1个有影组零召回。所有评分的无影组误报和逐组损伤保留在PER_GROUP.csv与SUMMARY.csv，不以平均改善掩盖尾部。

F6仅复用R成员原C122/C123两seed：42组全部保留，32TML、40XDR、19FDV、54TWN均两seed AP下降>1pp，逐seed AP/召回/误报差见TABLE3_F6.md。54TWN只有7正参考；32TML的AP下降同时冻结阈值召回上升，不能把排序损伤等同于召回损伤。原Phase124区间原样保存，不新增F6重抽样。

新增神经主调用1224、构建探测11，共1235；远端推理阶段墙钟185.08秒（含I/O，非精确GPU活跃计时），GPU峰值8.57GiB，CPU峰值1.90GiB。本地四阶段墙钟916.73秒（含传输等待），进程峰值5.87GiB。首个本地等待接口故障在任何拟合前退出，日志保留；修复不增加拟合/神经调用。新增影像/权重下载0；本地隔离安装统计依赖，未改既有环境。CPU独立AP核对475项，另有独立交付标量验收，见验收JSON。

## 2 结论分析

F2：额外Blue/SWIR读出相对同维原生通道读出，在Catalogue与PixBox的宏AP差均为负且区间跨零；本轮没有显示稳定的光谱增量，也不能据小样本跨零断言互补不存在。OX的池化AP分别75.5891%/83.0693%，低于E的77.5952%/83.6448%。固定低容量读出、不同光谱通道控制并不构成物理机制的因果识别。Catalogue使用发布TOA cube；PixBox使用既有对齐面积平均TOA与原生20m B8A点输入，保留两种来源/重采样限制。

F4：原E在Catalogue的宏误报点估计已略超过1%预算，阈值校准不保证迁移；PixBox采样负类平均及其单侧上界在1%以内，但不能推广为全景安全保证。OX在Catalogue点估计达标而上界仍超过1%，也不能声称稳定满足预算。阈值未在评价池重选，所有结果均保留。

F6：R成员该C100适配方案存在四个跨两seed重复AP损伤组，且召回变化不总同向；这是已观察适配方案的条件结果。冻结E没有本轮训练seed，双模型E适配及V4均未测。历史已暴露池上的探索不能作为独立确认，也不恢复旧失败主检验；本轮不自动启动下一phase。
'''
 write(O/'RESULT.md',result)
 readme='F2：OX−ON宏AP差Catalogue−0.2008pp/PixBox−0.8834pp，区间均跨零；未见稳定光谱增量，历史小样本探索不能证明不存在。\nF4：E在Catalogue宏误报1.0197%超1%点预算，PixBox采样误报0.3873%且上界0.6885%；不代表全景安全。\nF6：R适配两seed有4个重复AP损伤组，54TWN仅7正参考；未测双模型E适配及V4，不推广为冻结E随机损伤。\n'
 write(O/'README.md',readme);save(O/'MISSING_ITEMS.json',dict(required_deliverables_missing=[],unmeasured=['V4','dual_model_E_adaptation','new_binary_readout_three_class_mIoU'],scope_limits=['historically exposed pools','7/8 AP groups','PixBox selected negative points not scene FPR','Catalogue published cube and PixBox native point inputs differ']))
 for pool in ('Catalogue','PixBox'):
  old=R/'src/research_results/phase124_20261017'/(pool+'_INTERVALS_REUSED.json');shutil.copyfile(old,O/('F6_'+pool+'_ORIGINAL_INTERVALS.json'))
 index=load(O/'local_inputs/INPUT_INDEX.json');local_bytes=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());conservative_combined=local_bytes+2*sum(r['bytes'] for r in index['rows']);assert conservative_combined<6_000_000_000
 save(O/'COST_RECEIPT.json',dict(neural_calls=1235,main_calls=1224,construction_calls=11,remote_run=index['run_status'],local_run=load(O/'LOCAL_STATUS.json'),local_files_bytes=local_bytes,combined_persistent_conservative_upper_bytes=conservative_combined,remote_export_hardlinks_counted_twice_in_upper=True,new_image_weight_downloads=0,statistics_dependency_downloaded=True,fit_count=4,threshold_count=5,engineering_recovery_fits=0,engineering_recovery_neural_calls=0))
 # Only small aggregate summaries, parameters and receipts; no point references/arrays.
 D.mkdir(parents=True,exist_ok=True)
 names=['README.md','RESULT.md','TABLE1_F2.md','TABLE2_F4.md','TABLE3_F6.md','F6_ALL_GROUPS.csv','F6_INPUT_LOCK.json','F6_RECEIPT.json','MISSING_ITEMS.json','COST_RECEIPT.json','INDEPENDENT_DELIVERY_ACCEPTANCE.json','CPU_ACCEPTANCE.json','BASELINE_REPRODUCED.json','MODELS_FROZEN.json','THRESHOLDS_FROZEN.json','F6_Catalogue_ORIGINAL_INTERVALS.json','F6_PixBox_ORIGINAL_INTERVALS.json']
 names += [p+'_'+s for p in ('Catalogue','PixBox','Calibration') for s in ('SUMMARY.csv','PER_GROUP.csv','CONTRASTS.json')]+['MODEL_'+p+'.json' for p in ('O1','ON','OX','X3')]
 for name in names:assert not (D/name).exists();shutil.copyfile(O/name,D/name)
 receipt=dict(status='completed',delivered_files={name:sha(O/name) for name in names},code_sha256={p.name:sha(p) for p in (R/'src/tools').glob('*phase202*') if p.is_file()},private_input_index_sha256=sha(O/'local_inputs/INPUT_INDEX.json'),new_neural_calls_in_finalization=0,new_fits_in_finalization=0,no_automatic_next_phase=True)
 save(O/'DELIVERY_COMPLETE.json',receipt);shutil.copyfile(O/'DELIVERY_COMPLETE.json',D/'DELIVERY_COMPLETE.json')
 with (O/'HANDOFF.md').open('a',encoding='utf8') as f:f.write('\n## Current delivery — COMPLETE 2026-10-17\nAll four local stages exited0, baseline and475 independent AP checks accepted; final scalar/hash/bootstrap checks passed. RESULT.md/README.md/tables, full95-group x5 rows, F6 all42, model/scaler/threshold locks and cost receipts delivered. Small summaries copied to src/research_results/phase202_20261017. No new model calls or fits for finalization. Do not rerun any stage. Pause phase202-hourly after notifying completion.\n')
 print(json.dumps(dict(status='completed',independent_scalar_checks=checks,files=len(names),combined_new_bytes_upper=conservative_combined)))
if __name__=='__main__':main()
