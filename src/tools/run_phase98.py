"""Phase98: two frozen fusions, exactly two53-group thresholds, no estimator fits."""
import importlib.metadata,json,pickle,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score
import run_phase94 as r
import run_phase96 as d96
from phase91_math import scan,cpu_validate
from phase98_math import fuse,ap,self_test
from phase96_math import describe

p=r.p;q=r.q;recovery=r.recovery;fit67=r.fit67
ROOT=p.old.SHARED/'phase98_20261012';D=ROOT/'delivery';OLD=p.old.SHARED/'phase94_20261012/delivery'
P96=p.old.SHARED/'phase96_20261012/delivery'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE98_FROZEN_SCORE_FUSION_20261012.md'
NAMES=('G7_100','SourceS_100','AVG','MIN');BRANCHES=('AVG','MIN');SEED=98012
def write(n,v):p.old.write(D/n,v)
def sha(a):return q.sha_array(a)

def metric(s,y,threshold,meta):
    row=q.binary_row(s,y,threshold,meta);row['ap']=ap(s,y)
    if row['ap'] is not None:assert abs(float(average_precision_score(y==2,s))-row['ap'])<1e-12
    row['shadow_iou']=row['true_positive']/(row['positive']+row['false_positive']) if row['positive']+row['false_positive'] else None
    row['predicted_shadow_fraction']=(row['true_positive']+row['false_positive'])/len(y) if len(y) else None
    flags=s>=threshold
    assert row['true_positive']==int(flags[y==2].sum()) and row['false_positive']==int(flags[y!=2].sum())
    return row

def summarize(rows,scores,ys,threshold):
    pooled=metric(np.concatenate(scores),np.concatenate(ys),threshold,dict(product='pooled',group_id='pooled',tile='pooled'))
    for k in ['true_positive','false_positive','positive','negative','surface','cloud','surface_false_positive','cloud_false_positive']:assert pooled[k]==sum(a[k] for a in rows)
    return dict(rows=rows,pooled=pooled,groups=len(rows),shadow_groups=sum(a['positive']>0 for a in rows),threshold=threshold,
        macro={k:q.interval([a[k] for a in rows if a[k] is not None]) for k in ['ap','recall','precision','fpr','shadow_iou','surface_fpr','cloud_fpr']},
        no_shadow_macro_fpr=q.interval([a['fpr'] for a in rows if not a['positive'] and a['fpr'] is not None]),
        groups_above_1percent=sum(a['fpr']>.01 for a in rows if a['fpr'] is not None),
        max_group_fpr=max(a['fpr'] for a in rows if a['fpr'] is not None),zero_recall_groups=[a['product'] for a in rows if a['recall']==0])

def score(models,full,checks,meta):
    native={n:r.predict(models,n,r.columns(full,n),checks,dict(**meta)) for n in r.NAMES}
    assert all(a.dtype==np.float32 for a in native.values())
    fusion,receipt,ps=fuse(native['G7_100'],native['SourceS_100'])
    return dict(**native,**fusion),dict(**meta,**receipt),ps

def main():
    started=time.monotonic();q.SEED=SEED;tests=self_test()
    assert not ROOT.exists(),'Do not overwrite/restart Phase98 thresholds or evaluation'
    assert p.old.digest(P96/'DELIVERY_SHA256.json')=='791d1abde6a591170bc596ccbeaeec28aa0c017b2d0744dbdf22957f60db1a66'
    for n,h in p.old.read(P96/'DELIVERY_SHA256.json').items():assert p.old.digest(P96/n)==h,n
    previous=p.old.read(P96/'input_lock.json');sl=p.old.read(OLD/'split_input_lock.json')
    assert p.old.digest(OLD/'DELIVERY_SHA256.json')==previous['phase94_manifest_sha256']
    for n,h in p.old.read(OLD/'DELIVERY_SHA256.json').items():assert p.old.digest(OLD/n)==h,n
    assert {n:importlib.metadata.version(n) for n in sl['dependency_lock']['versions']}==sl['dependency_lock']['versions']
    assert r.indices_audit()==previous['indices_before']
    models={}
    for n,m in previous['models'].items():
        assert p.old.digest(OLD/m['file'])==m['sha256']
        if n=='G7_100':
            with (OLD/m['file']).open('rb') as f:models[n]=pickle.load(f)
        else:models[n]=p.old.read(OLD/m['file'])
    assert models['G7_100'].get_params()['monotonic_cst'] is None
    oldth=p.old.read(OLD/'four_threshold_lock.json')['models'];audit=p.old.read(fit67.ROOT/'fit_input_audit.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==p.old.read(r.prior.OLD/'fit_input_lock.json')['original_fit_input_audit_sha256']
    am={a['product']:a for a in audit['rows']};recs=recovery.records('fit_representatives');dev=recovery.records('development_val')
    assert len(sl['calibration'])==53 and len(dev)==32 and not {a['group_id'] for a in sl['calibration']}&{a['group_id'] for a in dev.values()}
    ROOT.mkdir();D.mkdir()
    files=['run_phase98.py','run_phase98.sh','verify_phase98_delivery.py','phase98_math.py','phase91_math.py','run_phase94.py','run_phase96.py','run_phase92.py','phase93_math.py','phase89_cpu_ap.py','phase65d_recovery_math.py']
    write('input_formula_lock.json',dict(created_utc=p.old.now(),protocol_sha256=p.old.digest(PROTOCOL),models=previous['models'],
        phase94_manifest_sha256=p.old.digest(OLD/'DELIVERY_SHA256.json'),phase94_split_sha256=p.old.digest(OLD/'split_input_lock.json'),
        phase96_manifest_sha256=p.old.digest(P96/'DELIVERY_SHA256.json'),baseline_thresholds=oldth,
        calibration=sl['calibration'],development=list(dev.values()),indices_before=r.indices_audit(),versions=sl['dependency_lock']['versions'],
        pG='G7_100 native float32 class1 probability cast float64; no sigmoid',
        pS='SourceS_100 native float32 logit cast float64 then exp(-logaddexp(0,-logit)); not original Source network score',
        formulas=dict(AVG='(pG+pS)/2',MIN='minimum(pG,pS)'),fusion_dtype='float64',comparison='score>=threshold; float64 exact ties',
        AP='non-interpolated average precision; retain input dtype; sklearn and independent score-bin cross-check',
        code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in files},self_tests=tests,
        new_fits=0,new_thresholds_authorized=2,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,Phase97_model_included=False))
    checks=[];fusion_checks=[];negatives={n:[] for n in BRANCHES};calmeta=[];calstarted=time.monotonic()
    oldcal={n:{a['product']:a for a in oldth[n]['cal']['rows']} for n in r.NAMES}
    for i,row in enumerate(sl['calibration']):
        rec=recs[row['product']];recovery.prepare_verify(rec);truth=np.load(recovery.PREP/rec['mask_path']);y=truth[truth!=255]
        full=fit67.full_fit_features(am[row['product']]);out,receipt,ps=score(models,full,checks,dict(domain='calibration',product=row['product']))
        fusion_checks.append(receipt);calmeta.append(d96.meta(rec))
        for n in r.NAMES:assert sha(out[n][y!=2])==oldcal[n][row['product']]['negative_score_sha256']
        for n in BRANCHES:negatives[n].append(out[n][y!=2])
        if (i+1)%10==0:print('Phase98 calibration',i+1,53,flush=True)
    locks={}
    for n in BRANCHES:
        arr=negatives[n];yy=[np.zeros(len(a),np.uint8) for a in arr]
        point=scan(arr,yy,[.01])[0];cpu_validate(point,arr,yy)
        point.update(status='heldout53_calibration_only',score_precision='float64 native20m fusion, >=, atomic exact ties')
        th=point['threshold'];assert th is not None
        locks[n]=dict(threshold=th,exact_boundary=point,score_dtype='float64',
            rows=[dict(**meta,negative_count=len(a),false_positive=int((a>=th).sum()),fpr=float((a>=th).mean()) if len(a) else None,
                negative_score_sha256=sha(a),threshold_tie_count=int((a==th).sum()),next_candidate_tie_count=int((a==point['next_lower_observed_threshold']).sum())) for a,meta in zip(arr,calmeta)])
        print('Phase98 threshold frozen',n,th,point['macro_fpr'],flush=True)
    write('two_threshold_lock.json',dict(created_utc=p.old.now(),input_formula_lock_sha256=p.old.digest(D/'input_formula_lock.json'),
        models=locks,all_frozen_before_development=True,new_thresholds=2,calibration_negative_only=True))
    cal_seconds=time.monotonic()-calstarted
    del negatives,arr,yy,out,full,ps
    sources=[{a['product']:a for a in p.old.read(recovery.OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    scores={n:[] for n in NAMES};ys=[];metas=[];seals=[];target={};devstarted=time.monotonic()
    oldseals={a['product']:a for a in p.old.read(OLD/'input_score_lock.json')['rows'] if a['domain']=='development'}
    thresholds={n:oldth[n]['cal']['threshold'] for n in r.NAMES};thresholds.update({n:locks[n]['threshold'] for n in BRANCHES})
    for i,rec in enumerate(dev.values()):
        truth,valid,raw,ss,full=d96.dev_load(rec,sources);y=truth[valid];meta=d96.meta(rec)
        out,receipt,ps=score(models,full,checks,dict(domain='development',product=rec['product']));fusion_checks.append(receipt)
        assert sha(y)==oldseals[rec['product']]['truth_sha256']
        for n in r.NAMES:assert sha(out[n])==oldseals[rec['product']]['score_sha256'][n]
        for n in NAMES:scores[n].append(out[n])
        ys.append(y);metas.append(meta);seals.append(dict(**meta,truth_sha256=sha(y),score_sha256={n:sha(out[n]) for n in NAMES},score_dtypes={n:str(out[n].dtype) for n in NAMES}))
        if rec['product']==d96.TARGET:
            target=dict(**meta,reference_counts=np.bincount(y,minlength=3).tolist(),scores={n:describe(a) for n,a in out.items()},
                SourceS_probability=describe(ps),no_manual_confirmation_of_no_true_shadow=True)
        print('Phase98 development',i+1,32,flush=True)
    devseconds=time.monotonic()-devstarted;metrics_started=time.monotonic();policies={}
    oldpol=p.old.read(OLD/'phase94_report.json')['domains']['development']['policies']
    for n in NAMES:
        rows=[metric(s,y,thresholds[n],dict(**meta,policy=n)) for s,y,meta in zip(scores[n],ys,metas)]
        policies[n]=summarize(rows,scores[n],ys,thresholds[n])
        if n in r.NAMES:
            refs={a['product']:a for a in oldpol[n+'_T_cal']['rows']}
            for a in rows:
                for k in ['ap','recall','fpr','precision','surface_fpr','cloud_fpr','true_positive','false_positive']:assert a[k]==refs[a['product']][k],(n,k)
        target[n+'_metrics']=next(a for a in rows if a['product']==d96.TARGET)
        print('Phase98 metrics complete',n,flush=True)
    comparisons={}
    for a,b in [('AVG','SourceS_100'),('MIN','SourceS_100'),('MIN','AVG')]:
        pairs=[]
        for ra,rb in zip(policies[a]['rows'],policies[b]['rows']):
            assert ra['group_id']==rb['group_id']
            pairs.append(dict(product=ra['product'],group_id=ra['group_id'],tile=ra['tile'],**{'delta_'+k+'_pp':100*(ra[k]-rb[k]) if ra[k] is not None and rb[k] is not None else None for k in ['ap','recall','fpr']}))
        comparisons[a+'_minus_'+b]=dict(pairs=pairs,intervals_pp={k:q.interval([v['delta_'+k+'_pp'] for v in pairs if v['delta_'+k+'_pp'] is not None]) for k in ['ap','recall','fpr']})
    screening={}
    for n in BRANCHES:
        v=comparisons[n+'_minus_SourceS_100']['intervals_pp'];s=policies[n]
        criteria=dict(AP_loss_at_most_1pp=v['ap']['mean']>=-1,recall_gain_at_least_5pp=v['recall']['mean']>=5,
            macro_FPR_at_most_1percent=s['macro']['fpr']['mean']<=.01,maximum_group_FPR_at_most_10percent=s['max_group_fpr']<=.1)
        screening[n]=dict(criteria=criteria,all_four_pass=all(criteria.values()))
    retained=[n for n in BRANCHES if screening[n]['all_four_pass']]
    assert r.indices_audit()==previous['indices_before']
    write('input_score_lock.json',dict(rows=seals,indices_after=r.indices_audit(),two_threshold_lock_sha256=p.old.digest(D/'two_threshold_lock.json'),no_large_arrays_saved=True))
    write('T31TGF.json',target)
    write('CPU_VERIFIED.json',dict(score_probes=checks,fusion_checks=fusion_checks,independent_AP_score_bins_and_sklearn=True,
        independent_integer_counts=True,original_baseline_scores_and_metrics_exact=True,calibration_baseline_negatives_exact=True,
        float64_AP_and_comparisons_preserved=True,threshold_direct_integer_counts=True))
    report=dict(status='complete',created_utc=p.old.now(),policies=policies,comparisons=comparisons,screening=screening,
        retained_branches=retained,decision='eligible_for_separate_protocol' if retained else 'close_both_fixed_fusion_branches',
        new_fits=0,new_thresholds=2,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,
        bootstrap_seed=SEED,resamples=10000,exploratory_only=True,not_calibrated_posterior_probabilities=True,
        limitations=['Repeatedly exposed development; descriptive related-group intervals only','Reference labels not newly human confirmed',
            'Human B not performed;54 reused Phase89 probability blocks lack pre-crash SHA; original failed gates unchanged',
            'Independent fusion thresholds may be lower; MIN score bound does not guarantee fewer false positives than SourceS'],
        cost=dict(seconds=time.monotonic()-started,calibration_scoring_and_threshold_seconds=cal_seconds,
            development_loading_and_scoring_seconds=devseconds,metrics_seconds=time.monotonic()-metrics_started,peak_rss_bytes=r.rss(),cpu_threads_max=4))
    write('phase98_report.json',report);render(report)
    write('DELIVERY_SHA256.json',{f.name:p.old.digest(f) for f in sorted(D.iterdir()) if f.is_file()})
    p.old.write(ROOT/'COMPLETED.json',dict(created_utc=p.old.now(),manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json')))
    print('Phase98 complete',report['decision'],retained,flush=True)

def render(report):
    totals=[];groups=[];changes=[];lines=['# Phase98：冻结分数AVG／MIN组合','','## 1．数据结果','',
        '原53组负类校准两个阈值，原32组development评价，零拟合／零网络前向／零下载。组合与组合AP全程float64。','',
        '|策略|宏AP %|池化AP %|宏召回 %|宏FPR %|FPR单侧95%上界 %|超1%组数|最大FPR %|','|---|---:|---:|---:|---:|---:|---:|---:|']
    f=lambda a:'NA' if a is None else f'{100*a:.4f}'
    for n,s in report['policies'].items():
        row=dict(policy=n,threshold=s['threshold'],groups=s['groups'],shadow_groups=s['shadow_groups'],groups_above_1percent=s['groups_above_1percent'],max_group_fpr=s['max_group_fpr'])
        for k,v in s['macro'].items():row.update({f'macro_{k}':v['mean'],f'macro_{k}_ci95':json.dumps(v['ci95']),f'macro_{k}_upper95':v['upper95_one_sided']})
        row.update({'pooled_'+k:s['pooled'][k] for k in ['ap','recall','precision','fpr','surface_fpr','cloud_fpr']});totals.append(row);groups+=s['rows']
        lines.append('| '+' | '.join([n,f(s['macro']['ap']['mean']),f(s['pooled']['ap']),f(s['macro']['recall']['mean']),f(s['macro']['fpr']['mean']),f(s['macro']['fpr']['upper95_one_sided']),str(s['groups_above_1percent']),f(s['max_group_fpr'])])+' |')
    for n,v in report['comparisons'].items():
        changes.extend(dict(comparison=n,**a) for a in v['pairs'])
        for k,it in v['intervals_pp'].items():lines.append(f"\n{n} {k}: {it['mean']:.6f}个百分点，95%描述区间{it['ci95']}，{it['n_groups']}组。")
    lines+=['','固定筛选结果：'+json.dumps(report['screening'],ensure_ascii=False),'','## 2．结论探讨','',
        '判定：'+report['decision']+'；通过分枝：'+str(report['retained_branches']),
        '每分枝四项全部满足才可另立协议；本轮不追加权重、公式、阈值或拟合。池化AP改善不替代宏AP／召回／误报条件。',
        '组合改变排序，已重新计算AP；组合评分不称已校准后验概率。MIN逐点小于两输入，不意味着使用自身校准阈值后误报必然更少。',
        'T31TGF单独分位数与误报见T31TGF.json；全部32组和Surface／Cloud分层保留。无参考云影组AP／召回为NA，继续参与误报评价。',
        *report['limitations'],'','成本：'+json.dumps(report['cost'],ensure_ascii=False)]
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name,rows in [('four_strategy_totals.csv',totals),('per_group.csv',groups),('paired_changes.csv',changes)]:p.old.csvwrite(D/name,rows)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
