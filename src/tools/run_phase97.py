"""One fixed Source-monotonic fit; old100/53/dev32, no external pixel reads."""
import hashlib, importlib.metadata, json, pickle, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import run_phase94 as r
import run_phase96 as d96
from phase93_math import equal_weights, independent_probability, probes
from phase91_math import scan, cpu_validate
from phase96_math import describe

p=r.p; q=r.q; recovery=r.recovery; fit67=r.fit67
ROOT=p.old.SHARED/'phase97_20261012'; D=ROOT/'delivery'
OLD=p.old.SHARED/'phase94_20261012/delivery'
P96=p.old.SHARED/'phase96_20261012/delivery'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE97_SOURCE_MONOTONIC_TREE_20261012.md'
NAMES=('A_G7_100','B_SourceMonotonic','SourceS_100'); SEED=97012

def write(n,v):p.old.write(D/n,v)
def sha(a):return q.sha_array(a)

def predict(model,x,checks,meta):
    start=time.monotonic();s=model.predict_proba(x)[:,1].astype(np.float32)
    a=probes(x); other=independent_probability(model,a)
    err=float(np.max(np.abs(other-model.predict_proba(a)[:,1].astype(np.float32))))
    assert err<=6e-8 and np.isfinite(s).all()
    checks.append(dict(**meta,probe_count=len(a),max_error=err,seconds=time.monotonic()-start))
    return s

def metric(s,y,th,meta):
    row=q.row_metric(s,y,th,meta)
    if row['ap'] is not None:
        v=float(average_precision_score(y==2,s))
        assert abs(v-row['ap'])<1e-12
    return row

def summarize(rows,ss,ys,th):
    # Same definitions as Phase94, plus sklearn pooled AP cross-check.
    out=q.summarize(rows,ss,ys,th)
    all_s=np.concatenate(ss); all_y=np.concatenate(ys)
    if out['pooled']['ap'] is not None:
        assert abs(float(average_precision_score(all_y==2,all_s))-out['pooled']['ap'])<1e-12
    out['max_group_fpr']=max(a['fpr'] for a in rows if a['fpr'] is not None)
    return out

def main():
    started=time.monotonic(); q.SEED=SEED
    assert not ROOT.exists(),'Never overwrite/repeat a Phase97 fit attempt'
    assert p.old.digest(P96/'DELIVERY_SHA256.json')=='791d1abde6a591170bc596ccbeaeec28aa0c017b2d0744dbdf22957f60db1a66'
    for n,h in p.old.read(P96/'DELIVERY_SHA256.json').items():assert p.old.digest(P96/n)==h,n
    previous=p.old.read(P96/'input_lock.json'); sl=p.old.read(OLD/'split_input_lock.json')
    assert p.old.digest(OLD/'DELIVERY_SHA256.json')==previous['phase94_manifest_sha256']
    for n,h in p.old.read(OLD/'DELIVERY_SHA256.json').items():assert p.old.digest(OLD/n)==h,n
    assert previous['indices_before']==r.indices_audit()
    assert {n:importlib.metadata.version(n) for n in sl['dependency_lock']['versions']}==sl['dependency_lock']['versions']
    for n,m in previous['models'].items():assert p.old.digest(OLD/m['file'])==m['sha256'],n
    with (OLD/'G7_100.pkl').open('rb') as f:A=pickle.load(f)
    S=p.old.read(OLD/'SourceS_100.json'); oldth=p.old.read(OLD/'four_threshold_lock.json')['models']
    params=A.get_params(deep=False)
    assert params==r.PARAMS and params['monotonic_cst'] is None and params['early_stopping'] is False
    assert A.n_features_in_==7 and A.n_iter_==200 and np.array_equal(A.classes_,[0,1])
    audit=p.old.read(fit67.ROOT/'fit_input_audit.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==p.old.read(r.prior.OLD/'fit_input_lock.json')['original_fit_input_audit_sha256']
    am={a['product']:a for a in audit['rows']}; recs=recovery.records('fit_representatives'); dev=recovery.records('development_val')
    assert len(sl['fit'])==100 and len(sl['calibration'])==53 and len(dev)==32
    assert not {a['group_id'] for a in sl['fit']+sl['calibration']}&{a['group_id'] for a in dev.values()}
    base=fit67.base_verify(); ROOT.mkdir(); D.mkdir()
    files=['run_phase97.py','run_phase97.sh','verify_phase97_delivery.py','run_phase94.py','run_phase96.py','phase93_math.py','phase91_math.py','run_phase92.py','phase89_cpu_ap.py','phase65d_recovery_math.py']
    write('input_lock.json',dict(created_utc=p.old.now(),protocol_sha256=p.old.digest(PROTOCOL),
        phase96_manifest_sha256=p.old.digest(P96/'DELIVERY_SHA256.json'),base=base,
        phase94_split_sha256=p.old.digest(OLD/'split_input_lock.json'),phase94_fit_input_sha256=p.old.digest(OLD/'fit_input_lock.json'),
        models=previous['models'],baseline_thresholds=oldth,fit=sl['fit'],calibration=sl['calibration'],development=list(dev.values()),
        input_columns=['B04','B03','B02','B08','B11','B12','Source_shadow_logit'],source_column=6,
        source_clip=[1e-6,1-1e-6],input_rule='unchanged physical TOA; original float32 combined input then float64; no standardization for HGB',
        indices_before=r.indices_audit(),versions=sl['dependency_lock']['versions'],
        code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in files},
        fixed_parameters_A=params,fixed_parameters_B=dict(params,monotonic_cst=[0,0,0,0,0,0,1]),
        new_tree_fits_authorized=1,new_thresholds_authorized=1,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0))
    parts=[]; labels=[]; mono=[]; receipts=[]
    for i,row in enumerate(sl['fit']):
        product=row['product']; rec=recs[product]; recovery.prepare_verify(rec)
        truth=np.load(recovery.PREP/rec['mask_path']); full=fit67.full_fit_features(am[product])
        idx=np.random.default_rng(65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)).choice(len(full),min(8192,len(full)),replace=False)
        yy=(truth[truth!=255]==2)[idx]
        assert sha(idx)==row['sample_indices_sha256'] and sha(yy)==row['sample_truth_sha256']
        assert len(idx)==row['sample_count'] and int(yy.sum())==row['sample_positive']
        parts.append(full[idx]); labels.append(yy); mono.append(r.columns(full[idx[:64]],'G7_100'))
        receipts.append(dict(product=product,group_id=row['group_id'],indices_sha256=sha(idx),labels_sha256=sha(yy),count=len(idx)))
        if (i+1)%20==0:print('Phase97 original fit samples',i+1,100,flush=True)
    full=np.concatenate(parts).astype(np.float64); y=np.concatenate(labels).astype(float); del parts,labels
    reference=p.old.read(OLD/'fit_input_lock.json'); w=equal_weights([a['sample_count'] for a in sl['fit']])
    assert sha(full)==reference['sample_features_sha256'] and sha(y)==reference['sample_labels_sha256'] and sha(w)==reference['tree_weights_sha256']
    x=r.columns(full,'G7_100'); assert np.array_equal(x[:,6],full[:,1]); del full
    levels=np.quantile(x[:,6],[.01,.05,.5,.95,.99]); mx=np.concatenate(mono); del mono
    write('fit_input_lock.json',dict(rows=receipts,sample_count=len(x),sample_features_sha256=reference['sample_features_sha256'],
        selected_features_sha256=sha(x),labels_sha256=sha(y),weights_sha256=sha(w),weight_rule='N/(100*n_i)',
        source_probe_levels=levels.tolist(),probe_base_sha256=sha(mx),probe_points_per_group=64,probe_groups=100))
    B=HistGradientBoostingClassifier(**dict(params,monotonic_cst=[0,0,0,0,0,0,1]))
    write('FIT_ATTEMPT.json',dict(created_utc=p.old.now(),fit_attempt=1,model='B_SourceMonotonic',no_automatic_retry=True))
    t=time.monotonic(); B.fit(x,y,sample_weight=w); fit_seconds=time.monotonic()-t
    # Persist the sole fitted model before any downstream acceptance assertion.
    with (D/'B_SourceMonotonic.pkl').open('xb') as f:pickle.dump(B,f,protocol=5)
    assert B.n_iter_==200 and B.n_features_in_==7 and np.array_equal(B.classes_,[0,1])
    with (D/'B_SourceMonotonic.pkl').open('rb') as f:loaded=pickle.load(f)
    assert np.array_equal(B.predict_proba(x)[:,1].astype(np.float32),loaded.predict_proba(x)[:,1].astype(np.float32))
    assert {k:v for k,v in B.get_params().items() if k!='monotonic_cst'}=={k:v for k,v in params.items() if k!='monotonic_cst'}
    del x,y,w; B=loaded; monotonic={}
    for name,model in [('A_G7_100',A),('B_SourceMonotonic',B)]:
        seq=[]
        for value in levels:
            xx=mx.copy();xx[:,6]=value;seq.append(model.predict_proba(xx)[:,1].astype(np.float32))
        mat=np.column_stack(seq); violations=np.maximum(-np.diff(mat.astype(float),axis=1),0)
        monotonic[name]=dict(max_violation=float(violations.max()),violating_pairs=int((violations>1e-7).sum()),
            violating_points=int((violations>1e-7).any(1).sum()),point_count=len(mx),levels=levels.tolist(),scores_sha256=sha(mat))
    write('monotonicity_check.json',dict(models=monotonic,scope='100 groups x first64 original fixed samples x five fixed train quantiles',tolerance=1e-7))
    assert monotonic['B_SourceMonotonic']['max_violation']<=1e-7
    write('model_lock.json',dict(created_utc=p.old.now(),file='B_SourceMonotonic.pkl',sha256=p.old.digest(D/'B_SourceMonotonic.pkl'),
        params=B.get_params(),n_iter=int(B.n_iter_),classes=B.classes_.tolist(),training_reload_exact=True,
        fit_input_lock_sha256=p.old.digest(D/'fit_input_lock.json'),new_fits=1,fit_seconds=fit_seconds,process_peak_rss_bytes=r.rss()))
    del mx,xx,mat,seq
    checks=[]; negatives=[]; calmeta=[];calibration_started=time.monotonic()
    for i,row in enumerate(sl['calibration']):
        rec=recs[row['product']]; recovery.prepare_verify(rec); truth=np.load(recovery.PREP/rec['mask_path']);yy=truth[truth!=255]
        full=fit67.full_fit_features(am[row['product']]);s=predict(B,r.columns(full,'G7_100'),checks,dict(model='B_SourceMonotonic',domain='calibration',product=row['product']))
        negatives.append(s[yy!=2]);calmeta.append(d96.meta(rec))
        if (i+1)%10==0:print('Phase97 calibration',i+1,53,flush=True)
    eligible=[a for a in negatives if len(a)];assert len(eligible)==53
    point=scan(negatives,[np.zeros(len(a),np.uint8) for a in negatives],[.01])[0]
    cpu_validate(point,negatives,[np.zeros(len(a),np.uint8) for a in negatives]); th=point['threshold']
    rows=[dict(**m,negative_count=len(a),false_positive=int((a>=th).sum()),fpr=float((a>=th).mean()),negative_score_sha256=sha(a),
        boundary_tie_count=int((a==th).sum()),next_boundary_tie_count=int((a==point['next_lower_observed_threshold']).sum())) for m,a in zip(calmeta,negatives)]
    assert np.mean([a['fpr'] for a in rows])<=.01
    write('threshold_lock.json',dict(created_utc=p.old.now(),model_lock_sha256=p.old.digest(D/'model_lock.json'),
        threshold=th,rows=rows,exact_boundary=point,score_domain='float32 class1 probability >= threshold',
        calibration='original53 full original20m valid reference nonshadow; equal group weights',frozen_before_development=True,new_thresholds=1))
    calibration_seconds=time.monotonic()-calibration_started
    del negatives,eligible,s,full
    sources=[{a['product']:a for a in p.old.read(recovery.OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    scores={n:[] for n in NAMES};ys=[];policies={};seals=[];target={}
    baseline=p.old.read(OLD/'phase94_report.json')['domains']['development']['policies']
    oldseals={a['product']:a for a in p.old.read(OLD/'input_score_lock.json')['rows'] if a['domain']=='development'}
    thresholds={'A_G7_100':oldth['G7_100']['cal']['threshold'],'B_SourceMonotonic':th,'SourceS_100':oldth['SourceS_100']['cal']['threshold']}
    dev_started=time.monotonic()
    for i,rec in enumerate(dev.values()):
        truth,valid,raw,ss,full=d96.dev_load(rec,sources);yy=truth[valid];meta=d96.meta(rec);seal={}
        for name,model in [('A_G7_100',A),('B_SourceMonotonic',B)]:
            s=predict(model,r.columns(full,'G7_100'),checks,dict(model=name,domain='development',product=rec['product']));scores[name].append(s);seal[name]=sha(s)
        s=r.predict({'SourceS_100':S},'SourceS_100',r.columns(full,'SourceS_100'),checks,dict(domain='development',product=rec['product'])); scores['SourceS_100'].append(s);seal['SourceS_100']=sha(s)
        assert seal['A_G7_100']==oldseals[rec['product']]['score_sha256']['G7_100'] and seal['SourceS_100']==oldseals[rec['product']]['score_sha256']['SourceS_100']
        assert sha(yy)==oldseals[rec['product']]['truth_sha256'];ys.append(yy)
        seals.append(dict(**meta,truth_sha256=sha(yy),scores_sha256=seal))
        if rec['product']==d96.TARGET:
            a=scores['A_G7_100'][-1]>=thresholds['A_G7_100'];b=scores['B_SourceMonotonic'][-1]>=th;neg=yy!=2
            target=dict(**meta,reference_counts=np.bincount(yy,minlength=3).tolist(),source=describe(ss[valid]),
                A=describe(scores['A_G7_100'][-1]),B=describe(scores['B_SourceMonotonic'][-1]),
                false_positive_changes=dict(A_only=int((a&~b&neg).sum()),B_only=int((b&~a&neg).sum()),both=int((a&b&neg).sum()),neither=int((~a&~b&neg).sum())),
                no_manual_confirmation_of_reference=True)
        print('Phase97 development',i+1,32,flush=True)
    dev_seconds=time.monotonic()-dev_started;metric_started=time.monotonic()
    for name in NAMES:
        rr=[metric(s,y,thresholds[name],dict(**m,policy=name)) for s,y,m in zip(scores[name],ys,seals)]
        # Strip score receipts from metric metadata to keep per-group tables compact.
        for a in rr:a.pop('scores_sha256',None);a.pop('truth_sha256',None)
        policies[name]=summarize(rr,scores[name],ys,thresholds[name])
        if name!='B_SourceMonotonic':
            oldname='G7_100' if name=='A_G7_100' else name
            refs={a['product']:a for a in baseline[oldname+'_T_cal']['rows']}
            for a in rr:
                for k in ['ap','fpr','recall','precision','true_positive','false_positive','positive','negative','surface_fpr','cloud_fpr']:assert a[k]==refs[a['product']][k],(name,k)
        target[name+'_metrics']=next(a for a in rr if a['product']==d96.TARGET)
    pair=[]
    for a,b in zip(policies['A_G7_100']['rows'],policies['B_SourceMonotonic']['rows']):
        assert a['group_id']==b['group_id']
        pair.append(dict(product=a['product'],tile=a['tile'],group_id=a['group_id'],**{'delta_'+k+'_pp':100*(b[k]-a[k]) if a[k] is not None and b[k] is not None else None for k in ['ap','recall','fpr']}))
    comparisons={k:q.interval([a['delta_'+k+'_pp'] for a in pair if a['delta_'+k+'_pp'] is not None]) for k in ['ap','recall','fpr']}
    Bsum=policies['B_SourceMonotonic'];criteria=dict(AP_loss_at_most_1pp=comparisons['ap']['mean']>=-1,
        recall_loss_at_most_1pp=comparisons['recall']['mean']>=-1,macro_FPR_at_most_1percent=Bsum['macro']['fpr']['mean']<=.01,
        maximum_group_FPR_at_most_10percent=Bsum['max_group_fpr']<=.10)
    assert r.indices_audit()==previous['indices_before']
    write('input_score_lock.json',dict(rows=seals,indices_after=r.indices_audit(),threshold_lock_sha256=p.old.digest(D/'threshold_lock.json'),no_large_arrays_saved=True))
    write('T31TGF.json',target)
    write('CPU_VERIFIED.json',dict(rows=checks,independent_tree_walk_and_linear_probes=True,
        independent_AP_and_sklearn_per_group_and_pooled=True,integer_counts_direct_verified=True,original_baseline_scores_and_metrics_exact=True,
        calibration_exact_ties_direct_counts=True))
    report=dict(status='complete',created_utc=p.old.now(),policies=policies,paired_B_minus_A=dict(pairs=pair,intervals_pp=comparisons),
        screening=criteria,all_four_pass=all(criteria.values()),decision='eligible_for_separately_preregistered_stability_study' if all(criteria.values()) else 'close_this_fixed_monotonic_scheme',
        bootstrap_seed=SEED,resamples=10000,exploratory_only=True,no_mechanism_claim=True,
        new_fits=1,new_thresholds=1,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,
        limitations=['Exposed development; descriptive related-group intervals, not independent confirmation',
            'Human B not performed;54 reused Phase89 blocks lack pre-crash SHA; original failed gates unchanged',
            'Monotonicity does not require low probability at low Source score; no per-scene1percent guarantee',
            'Reference labels not newly human confirmed; no three-class mIoU fabricated'],
        cost=dict(seconds=time.monotonic()-started,fit_seconds=fit_seconds,calibration_scoring_and_threshold_seconds=calibration_seconds,
            development_loading_and_scoring_seconds=dev_seconds,development_metrics_seconds=time.monotonic()-metric_started,
            process_peak_rss_bytes=r.rss(),cpu_threads_max=4))
    write('phase97_report.json',report);render(report)
    write('DELIVERY_SHA256.json',{f.name:p.old.digest(f) for f in sorted(D.iterdir()) if f.is_file()})
    p.old.write(ROOT/'COMPLETED.json',dict(created_utc=p.old.now(),manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json')))
    print('Phase97 complete',report['decision'],flush=True)

def render(report):
    totals=[];groups=[];lines=['# Phase97：Source单调约束树，一次固定探索','','## 1．数据结果','',
        '原100组拟合、53组留出校准、32组development原20m评价；仅新增一次树拟合、一个冻结阈值。',
        '输入为B04/B03/B02/B08/B11/B12/Source logit；唯一模型参数变化是Source列单调递增约束。','',
        '|模型|宏AP %|池化AP %|宏召回 %|宏FPR %|FPR单侧95%上界 %|超1%组数|最大组FPR %|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda a:'NA' if a is None else f'{100*a:.4f}'
    for n,s in report['policies'].items():
        a=dict(model=n,threshold=s['threshold'],groups=s['groups'],shadow_groups=s['shadow_groups'],
            groups_above_1percent=s['groups_above_1percent'],maximum_group_fpr=s['max_group_fpr'])
        for k,v in s['macro'].items():a.update({f'macro_{k}':v['mean'],f'macro_{k}_ci95':json.dumps(v['ci95']),f'macro_{k}_upper95':v['upper95_one_sided']})
        a.update({'pooled_'+k:s['pooled'][k] for k in ['ap','recall','precision','fpr','surface_fpr','cloud_fpr']});totals.append(a);groups+=s['rows']
        lines.append('| '+' | '.join([n,fmt(s['macro']['ap']['mean']),fmt(s['pooled']['ap']),fmt(s['macro']['recall']['mean']),fmt(s['macro']['fpr']['mean']),fmt(s['macro']['fpr']['upper95_one_sided']),str(s['groups_above_1percent']),fmt(s['max_group_fpr'])])+' |')
    for k,v in report['paired_B_minus_A']['intervals_pp'].items():lines.append(f"\nB−A {k}：{v['mean']:.6f}个百分点，配对相关组95%描述区间{v['ci95']}，有效组数{v['n_groups']}。")
    lines+=['','四项固定筛选：',*[f'- {k}: {v}' for k,v in report['screening'].items()],
        '', 'T31TGF分数分位数及A-only/B-only/共同误报数量见T31TGF.json；全部组、Surface/Cloud分层见per_group.csv。',
        '', '## 2．结论探讨','',
        '四项全部满足，可另立稳定性与预算迁移协议；任一失败，关闭本次固定单调约束方案，不追加调参。',
        '本轮固定判定：'+report['decision'],
        '单调约束只规定其余六列固定时，Source升高不能使预测概率降低；低Source仍可能产生高预测值。无论本轮成败，都不能据此证明T31TGF原失效机制。',
        '置信区间使用10000次相关组配对bootstrap、seed97012；工程筛选使用预先约定点估计，不是非劣证明或部署保证。',
        *report['limitations'], '', '成本：'+json.dumps(report['cost'],ensure_ascii=False)]
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name,rows in [('three_strategy_totals.csv',totals),('per_group.csv',groups),('paired_changes.csv',report['paired_B_minus_A']['pairs'])]:p.old.csvwrite(D/name,rows)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
