"""One fixed local reference join and two-model evaluation, independent CPU statistics."""
import collections,shutil,json
import numpy as np
from phase105_common import *
from phase65d_metrics import curve
from phase89_cpu_ap import average_precision_score as independent_ap

OUT=REPO/'outputs/phase105/phase105_20261013'
DATA=REPO/'data/pixbox_s2_cmix_5036991/phase105_20261013'

def operating(rows,name,threshold):
    positive=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rows])
    predicted=np.array([r[name]>=threshold for r in rows]);p=int(positive.sum());n=len(rows)-p
    tp=int((positive&predicted).sum());fp=int((~positive&predicted).sum())
    result=dict(positive=p,negative=n,TP=tp,FP=fp,recall=tp/p if p else None,
        precision=tp/(tp+fp) if tp+fp else None,selected_negative_false_positive_fraction=fp/n if n else None)
    for kind in ('negative_surface','negative_cloud'):
        mask=np.array([r['reference']==kind for r in rows]);count=int(mask.sum());false=int((mask&predicted).sum())
        result[kind+'_n']=count;result[kind+'_FP']=false;result[kind+'_false_positive_fraction']=false/count if count else None
    return result

def main():
    lock=read(RESULT/'task_lock.json');sealed=read(OUT/'SCORES_SEALED.json')
    scorepath=DATA/'point_scores_no_labels.json';assert sha(scorepath)==sealed['score_sha256']
    assert sha(OUT/'evaluation_lock.json')==sealed['evaluation_lock_sha256']
    assert sha(OUT/'forward_receipts.json')==sealed['forward_receipts_sha256']
    assert sha(OUT/'CPU_tree_acceptance.json')==sealed['CPU_tree_acceptance_sha256']
    assert read(OUT/'CPU_tree_acceptance.json')['status']=='passed'
    assert read(OUT/'evaluation_lock.json')['local_task_lock_sha256']==sha(RESULT/'task_lock.json')
    # Durable score seal precedes any local reference join/metrics.
    write(RESULT/'LOCAL_SCORE_SEAL.json',dict(utc=now(),score_sha256=sha(scorepath),remote_seal_sha256=sha(OUT/'SCORES_SEALED.json'),
         locked_before_reference_join_and_metrics=True),True)
    scores=read(scorepath)['rows'];assert len(scores)==len({r['point_id'] for r in scores})==9500
    referencepath=REPO/'data/pixbox_s2_cmix_5036991/phase103_20261013/deduplicated_locations.json'
    assert sha(referencepath)==lock['phase103_reference_sha256']
    refs=read(referencepath);byid={r['point_ids'][0]:r for r in refs};assert len(byid)==9500
    prior=read(REPO/'src/research_results/phase104_20261013/support_acceptance.json')
    with (REPO/'data/pixbox_s2_cmix_5036991/phase104_20261013/point_validity_no_labels.csv').open(encoding='utf-8',newline='') as f:
        import csv
        validity={r['point_id']:r for r in csv.DictReader(f)}
    joined=[];groups=collections.defaultdict(list);shared={}
    for r in scores:
        ref=byid[r['point_id']];v=validity[r['point_id']]
        assert r['input_valid']==(v['input_valid']=='True')
        assert all(str(r[k])==str(ref[k]) for k in ('product_id','group_id','row20','col20'))
        assert np.isfinite(r['Source']) and 0<=r['Source']<=1
        for name in ('G6','G7'):
            assert (r[name] is not None)==r['input_valid']
            if r['input_valid']:assert np.isfinite(r[name]) and 0<=r[name]<=1 and float(np.float32(r[name]))==r[name]
        cell=(r['product_id'],r['row20'],r['col20']);value=(r['Source'],r['G6'],r['G7'])
        if cell in shared:assert value==shared[cell]
        shared[cell]=value
        q=dict(**r,reference=ref['reference'],included=r['input_valid'] and ref['reference']!='ignore')
        joined.append(q)
        if q['included']:groups[r['group_id']].append(q)
    assert len(groups)==10
    all_valid=[r for r in joined if r['included']]
    assert sum(r['reference']=='positive_ground_cloud_shadow' for r in all_valid)==379
    assert sum(r['reference']!='positive_ground_cloud_shadow' for r in all_valid)==8188
    assert sum(r['reference']=='ignore' for r in joined)==931
    assert sum(not r['input_valid'] for r in joined)==2
    csvwrite(DATA/'joined_reference_scores_private.csv',joined)
    models=read(REPO/'src/research_results/phase93_20261012/two_model_lock.json')['models']
    rows=[];validation=[];cpu_deltas=[]
    for gid,rr in sorted(groups.items()):
        y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in rr]);p=int(y.sum());n=len(rr)-p
        row=dict(group_id=gid,tile=rr[0]['product_id'].split('_T')[1][:5],positive=p,negative=n,eligible=p>=5 and n>=20)
        independent_values={}
        for name in ('G6','G7'):
            s=np.array([r[name] for r in rr],np.float32)
            ap=curve(s,y)['ap'];ind=independent_ap(y,s) if p and n else None
            assert (ap is None)==(ind is None)
            if ap is not None:assert abs(ap-ind)<1e-12
            row[name+'_AP']=ap
            independent_values[name]=ind
            row.update({name+'_'+k:v for k,v in operating(rr,name,models[name]['threshold']).items()})
            validation.append(dict(group_id=gid,model=name,AP_independent_max_error=abs(ap-ind) if ap is not None else None))
        row['delta_AP_pp']=(row['G7_AP']-row['G6_AP'])*100 if p and n else None
        row['change']='NA' if row['delta_AP_pp'] is None else ('improved' if row['delta_AP_pp']>1 else 'declined' if row['delta_AP_pp']< -1 else 'stable')
        if row['eligible']:cpu_deltas.append((independent_values['G7']-independent_values['G6'])*100)
        rows.append(row)
    eligible=[r for r in rows if r['eligible']];assert len(eligible)==8
    delta=np.array([r['delta_AP_pp'] for r in eligible],np.float64)
    indices=np.random.default_rng(105013).integers(0,8,(10000,8))
    samples=delta[indices].mean(1);ci=np.quantile(samples,[.025,.975]);mean=float(delta.mean())
    # Independent scalar replay, recompute the delta from independent AP records above.
    cpu_samples=sorted(sum(cpu_deltas[i] for i in index)/8 for index in np.random.default_rng(105013).integers(0,8,(10000,8)))
    def quantile(q):
        index=(len(cpu_samples)-1)*q;lo=int(index);hi=min(lo+1,len(cpu_samples)-1)
        return cpu_samples[lo]+(index-lo)*(cpu_samples[hi]-cpu_samples[lo])
    cpu_ci=[quantile(.025),quantile(.975)]
    assert np.max(np.abs(ci-cpu_ci))<1e-12
    assert abs(mean-sum(float(r['delta_AP_pp']) for r in eligible)/8)<1e-12
    passed=bool(ci[0]>0 and ci[1]-ci[0]<=20)
    primary=dict(comparison='G7_153 minus G6_153',unit='related group; products merged before AP',n_groups=8,
        mean_delta_AP_pp=mean,paired_group_bootstrap95_pp=ci.tolist(),CI_width_pp=float(ci[1]-ci[0]),
        seed=105013,resamples=10000,rule='lower>0 AND total width<=20pp AND >=8 groups each>=5 positive/20 negative',
        passed=passed,conclusion='预定Source条件增量复现证据通过' if passed else '未取得预定清晰复现证据',
        improved=sum(r['change']=='improved' for r in eligible),stable=sum(r['change']=='stable' for r in eligible),
        declined=sum(r['change']=='declined' for r in eligible),worst_group=min(eligible,key=lambda r:r['delta_AP_pp'])['tile'],
        worst_delta_AP_pp=float(delta.min()),descriptive_interval=True,no_full_scene_safety_claim=True)
    totals=[]
    for name in ('G6','G7'):
        y=np.array([r['reference']=='positive_ground_cloud_shadow' for r in all_valid]);s=np.array([r[name] for r in all_valid],np.float32)
        pooled=curve(s,y)['ap'];assert abs(pooled-independent_ap(y,s))<1e-12
        total=dict(model=name,macro_sampled_AP_pp=float(np.mean([r[name+'_AP'] for r in eligible]))*100,
            pooled_sampled_AP_pp=pooled*100,frozen_original_threshold=models[name]['threshold'],
            macro_recall=float(np.mean([r[name+'_recall'] for r in eligible])),
            macro_selected_negative_false_positive_fraction=float(np.mean([r[name+'_selected_negative_false_positive_fraction'] for r in rows])),
            macro_precision=float(np.mean([r[name+'_precision'] for r in rows if r[name+'_precision'] is not None])),
            AP_groups=8,negative_groups=10,precision_defined_groups=sum(r[name+'_precision'] is not None for r in rows))
        total.update({'pooled_'+k:v for k,v in operating(all_valid,name,models[name]['threshold']).items()});totals.append(total)
    cpu=dict(status='passed',group_AP_replays=validation,pooled_AP_replays=2,paired_bootstrap_scalar_replay=True,
        group_mean_replay=True,score_ties_handled=True,NA_groups=2,shared20m_score_identity=True,
        original_points=9500,input_valid_points=9498,reference_ignored=931,evaluated_positive=379,evaluated_negative=8188,
        CPU_tree_acceptance_sha256=sha(OUT/'CPU_tree_acceptance.json'),no_extra_Source_forward=True)
    write(RESULT/'primary_endpoint.json',primary,True);csvwrite(RESULT/'two_model_summary.csv',totals);csvwrite(RESULT/'per_group_results.csv',rows)
    write(RESULT/'CPU_statistics_acceptance.json',cpu,True)
    forward=read(OUT/'forward_receipts.json')
    write(RESULT/'cost.json',dict(source_forwards=531,unique_windows=529,repeats=2,source_seconds=forward['seconds'],
        source_peak_cuda_bytes=forward['peak_cuda_bytes'],tree_cost=read(OUT/'CPU_tree_acceptance.json'),new_fits=0,new_download_bytes=0,
        score_file_sha256=sha(scorepath),private_local_join_sha256=sha(DATA/'joined_reference_scores_private.csv')),True)
    table='\n'.join(f"|{r['model']}|{r['macro_sampled_AP_pp']:.3f}|{r['pooled_sampled_AP_pp']:.3f}|{r['macro_recall']*100:.3f}|{r['macro_selected_negative_false_positive_fraction']*100:.3f}|" for r in totals)
    report=f'''# Phase105：冻结G7/G6的PixBox评价\n\n## 1 数据结果\n\n固定16产品/10相关组、9500原采样点；379正类、8188负类进入评价，931 ignore，2输入无效。Source执行529唯一窗口与2次预定重复，共531次；逐窗口回执、重复与树模型/指标独立CPU复核通过。无拟合、调阈值或新增影像下载。\n\n|模型|逐组平均AP(pp)|池化AP(pp)|逐组平均召回(%)|所选负类逐组平均误报比例(%)|\n|---|---:|---:|---:|---:|\n{table}\n\n唯一主要终点G7−G6逐组平均AP差：**{mean:.3f}pp**；配对相关组bootstrap描述95%区间 **[{ci[0]:.3f}, {ci[1]:.3f}]pp**，宽度{ci[1]-ci[0]:.3f}pp。8组等权、10000次、seed105013。改善/稳定/下降：{primary['improved']}/{primary['stable']}/{primary['declined']}，沿用±1pp。最差组{primary['worst_group']}：{primary['worst_delta_AP_pp']:.3f}pp。19GBQ与48NUG的AP为NA，继续参与负类误报描述；逐组及地表/云分层见per_group_results.csv。\n\n## 2 结论分析\n\n**{primary['conclusion']}**。判定同时要求区间下界严格>0且总宽度≤20pp，不能以池化指标或其他辅助结果替代。\n\n结论只针对保守映射及本批选择性采样点上，Source分数在纯光谱树上的条件增量；不归因为空间上下文、物理机制或预训练优势，不替代额外光谱增量方向。参考语义、选择性采样、未知基础预训练独立性限制保留。所选负类误报比例不等于全场景FPR，不能据此判断全场景1%预算、尾部安全或密集mIoU。没有新校准、oracle阈值扫描、重新挑选支持或恢复旧失败分支。\n\n原影像留远端；逐点无标签分数及带参考交集仅在受控data目录，Git只含汇总、锁、回执与SHA。\n'''
    (RESULT/'README.md').write_text(report,encoding='utf-8',newline='\n')
    for p in OUT.iterdir():
        if p.name in ['SCORES_SEALED.json','evaluation_lock.json','forward_receipts.json','CPU_tree_acceptance.json']:shutil.copyfile(p,RESULT/p.name)
    for p in RESULT.iterdir():
        if p.is_file():shutil.copyfile(p,OUT/p.name)
    print(json.dumps(primary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
