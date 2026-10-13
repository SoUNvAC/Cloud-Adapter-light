"""Three calibration-only joint-tail thresholds, then one frozen dev evaluation."""
import importlib.metadata,json,pickle,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import run_phase99 as h
from phase99_math import oracles,self_test
from phase98_math import ap

r=h.r;p=h.p;q=h.q;recovery=h.recovery;fit67=h.fit67;d96=h.d96
ROOT=p.old.SHARED/'phase100_20261013';D=ROOT/'delivery';P99=p.old.SHARED/'phase99_20261012/delivery'
OLD=h.OLD;NAMES=h.NAMES;SEED=100013
PROTOCOL=p.old.REPO/'src/research_plans/PHASE100_TAIL_CALIBRATION_TRANSFER_20261013.md'
# Prior calibration-only console values, never development oracle thresholds.
CAL_REFERENCE=dict(G7_100=.870471477508545,SourceS_100=.5307986992836738,MIN=.2803397353855324)
def write(n,v):p.old.write(D/n,v)
def sha(a):return q.sha_array(a)

def prior_calibration_reference():
    found={}
    # Stop after six calibration lines; never consume development oracle lines.
    with (p.old.SHARED/'phase99_control_20261012/console.log').open() as f:
        for line in f:
            if line.startswith('Phase99 oracle calibration '):
                cols=line.split()
                if cols[4]=='O_tail':found[cols[3]]=float(cols[5])
                if len(found)==3:break
    assert found==CAL_REFERENCE
    return found

def estimate(values):return q.interval([v for v in values if v is not None])

def summarize(rows,t,pa,domain,model,kind):
    a=h.summary(rows,t,pa,domain,model,kind);a.update(oracle=False,deployable=None)
    a['intervals']={k:estimate([row[k] for row in rows]) for k in ['ap','recall','fpr','precision','shadow_iou','surface_fpr','cloud_fpr']}
    a['no_shadow_fpr']=estimate([row['fpr'] for row in rows if not row['positive']])
    return a

def checked_ap(s,y):
    value=ap(s,y)
    if value is not None:assert abs(value-float(average_precision_score(y==2,s)))<1e-12
    return value

def compare(a,b,name):
    pairs=[]
    for x,y in zip(a['rows'],b['rows']):
        assert x['group_id']==y['group_id']
        pairs.append(dict(comparison=name,product=x['product'],group_id=x['group_id'],tile=x['tile'],
            **{'delta_'+k+'_pp':100*(x[k]-y[k]) if x[k] is not None and y[k] is not None else None for k in ['ap','recall','fpr','precision','shadow_iou','surface_fpr','cloud_fpr']}))
    return dict(pairs=pairs,intervals_pp={k:estimate([row['delta_'+k+'_pp'] for row in pairs]) for k in ['ap','recall','fpr','precision','shadow_iou','surface_fpr','cloud_fpr']})

def runtime_self_test():
    q.SEED=SEED;tests=self_test()
    ss=[np.array([.9,.1,.2],np.float64),np.array([.1,.2],np.float64)]
    yy=[np.array([2,0,1]),np.array([0,1])]
    rows=[h.metric(s,y,.5,dict(product=str(i),group_id=str(i),tile=str(i)),checked_ap(s,y)) for i,(s,y) in enumerate(zip(ss,yy))]
    a=summarize(rows,.5,checked_ap(np.concatenate(ss),np.concatenate(yy)),'synthetic','synthetic','T_tail')
    assert a['macro_recall']==a['macro_precision']==1 and a['macro_fpr']==0
    assert rows[1]['ap'] is None and rows[1]['recall'] is None and rows[1]['precision'] is None
    z=estimate([.1,.9,None]);rng=np.random.default_rng(SEED)
    manual=np.array([.1,.9])[rng.integers(2,size=(10000,2))].mean(1)
    np.testing.assert_allclose(z['ci95'],np.quantile(manual,[.025,.975]),rtol=0,atol=1e-12)
    assert z['upper95_one_sided']==float(np.quantile(manual,.95)) and z['n_groups']==2
    tests.update(bootstrap_seed100013_and_group_resampling=True,NA_not_zero=True,pooled_and_macro_synthetic_counts=True)
    return tests

def main():
    started=time.monotonic();q.SEED=SEED;tests=runtime_self_test();assert not ROOT.exists(),'Preserve existing attempt'
    assert p.old.digest(P99/'DELIVERY_SHA256.json')=='ffa4582ebf943450f6cc6f1971a1293655430b72438616fa42b4c27ebd7adf79'
    # Hash-only verification of Phase99 artifacts is not reading oracle values.
    for n,v in p.old.read(P99/'DELIVERY_SHA256.json').items():assert p.old.digest(P99/n)==v,n
    previous=p.old.read(P99/'input_lock.json');scorelock=p.old.read(P99/'input_score_lock.json')
    assert p.old.digest(OLD/'DELIVERY_SHA256.json')==previous['phase94_manifest_sha256']
    for n,v in p.old.read(OLD/'DELIVERY_SHA256.json').items():assert p.old.digest(OLD/n)==v,n
    sl=p.old.read(OLD/'split_input_lock.json');indices=r.indices_audit()
    assert indices==previous['indices_before']
    assert {n:importlib.metadata.version(n) for n in sl['dependency_lock']['versions']}==sl['dependency_lock']['versions']
    calreference=prior_calibration_reference();models={}
    for n,m in previous['models'].items():
        assert p.old.digest(OLD/m['file'])==m['sha256']
        if n=='G7_100':
            with (OLD/m['file']).open('rb') as f:models[n]=pickle.load(f)
        else:models[n]=p.old.read(OLD/m['file'])
    oldth=previous['old_frozen_thresholds_probability64']
    audit=p.old.read(fit67.ROOT/'fit_input_audit.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==p.old.read(r.prior.OLD/'fit_input_lock.json')['original_fit_input_audit_sha256']
    am={a['product']:a for a in audit['rows']};recs=recovery.records('fit_representatives');dev=recovery.records('development_val')
    assert len(sl['fit'])==100 and len(sl['calibration'])==53 and len(dev)==32
    assert not {a['group_id'] for a in sl['calibration']}&{a['group_id'] for a in dev.values()}
    refs={(a['domain'],a['product']):a for a in scorelock['rows']}
    ROOT.mkdir();D.mkdir()
    code=['run_phase100.py','run_phase100.sh','verify_phase100_delivery.py','run_phase99.py','phase99_math.py','phase91_math.py','phase98_math.py','phase89_cpu_ap.py','run_phase94.py','run_phase96.py','run_phase92.py','phase93_math.py']
    write('input_lock.json',dict(created_utc=p.old.now(),protocol_sha256=p.old.digest(PROTOCOL),
        models=previous['models'],models_included=list(NAMES),phase99_manifest_sha256=p.old.digest(P99/'DELIVERY_SHA256.json'),
        phase94_manifest_sha256=p.old.digest(OLD/'DELIVERY_SHA256.json'),versions=sl['dependency_lock']['versions'],
        fit=sl['fit'],calibration=sl['calibration'],development=list(dev.values()),indices_before=indices,
        old_frozen_thresholds=oldth,calibration_only_reference=calreference,
        formula=previous['formula'],support='original20m full valid; not homogeneous Phase95',
        code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in code},self_tests=tests,
        new_fits=0,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,new_calibration_thresholds=3,
        bootstrap_seed=SEED,bootstrap_resamples=10000,development_oracle_used_for_thresholds=False,
        constraints=dict(mean_fpr=.01,max_group_fpr=.1)))
    checks=[];seals=[];scores={n:[] for n in NAMES};ys=[];metas=[];calstart=time.monotonic()
    for i,row in enumerate(sl['calibration']):
        rec=recs[row['product']];recovery.prepare_verify(rec);truth=np.load(recovery.PREP/rec['mask_path']);y=truth[truth!=255]
        full=fit67.full_fit_features(am[row['product']]);native,out=h.score(models,full,checks,dict(domain='calibration',product=rec['product']))
        ref=refs['calibration',rec['product']];assert sha(y)==ref['truth_sha256']
        assert {n:sha(a) for n,a in native.items()}==ref['native_score_sha256']
        assert {n:sha(a) for n,a in out.items()}==ref['probability64_sha256']
        assert np.array_equal(native['SourceS_100']>=previous['SourceS_old_logit_threshold'],out['SourceS_100']>=oldth['SourceS_100'])
        for n in NAMES:scores[n].append(out[n])
        ys.append(y);metas.append(d96.meta(rec));seals.append(ref)
        del full,native,out
        if (i+1)%10==0 or i+1==53:print('Phase100 calibration score recovery',i+1,53,flush=True)
    locks={};calpol={}
    for n in NAMES:
        # O_mean is only an internal component of the joint constraint algorithm.
        point=oracles(scores[n],ys)[1];t=point['threshold'];assert t==calreference[n]
        point.update(kind='T_tail',status='calibration_only_empirical_joint_budget',algorithm='Phase99 O_tail')
        aps=[checked_ap(s,y) for s,y in zip(scores[n],ys)]
        rows=[h.metric(s,y,t,dict(**m,domain='calibration',model=n,kind='T_tail'),a) for s,y,m,a in zip(scores[n],ys,metas,aps)]
        pooled_ap=checked_ap(np.concatenate(scores[n]),np.concatenate(ys))
        calpol[n]=summarize(rows,t,pooled_ap,'calibration',n,'T_tail')
        assert calpol[n]['macro_fpr']<=.01 and calpol[n]['max_group_fpr']<=.1
        locks[n]=dict(threshold=t,boundary=point,calibration=calpol[n],
            negative_score_sha256=[sha(s[y!=2]) for s,y in zip(scores[n],ys)],threshold_source='original53_calibration_only')
        print('Phase100 T_tail',n,t,calpol[n]['macro_fpr'],calpol[n]['max_group_fpr'],flush=True)
    write('three_T_tail_lock.json',dict(created_utc=p.old.now(),input_lock_sha256=p.old.digest(D/'input_lock.json'),
        models=locks,all_frozen_before_development=True,new_calibration_thresholds=3,
        development_oracle_read_before_lock=False,Phase99_calibration_O_tail_reproduced=True))
    thresholdhash=p.old.digest(D/'three_T_tail_lock.json');calseconds=time.monotonic()-calstart
    del scores,ys,aps
    # This is the first development metric file read; calibration lock is on disk.
    oldrefs=p.old.read(P99/'development_frozen_reproduction.json')['policies']
    sources=[{a['product']:a for a in p.old.read(recovery.OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    scores={n:[] for n in NAMES};ys=[];metas=[];devstart=time.monotonic()
    for i,rec in enumerate(dev.values()):
        assert p.old.digest(D/'three_T_tail_lock.json')==thresholdhash
        truth,valid,raw,ss,full=d96.dev_load(rec,sources);y=truth[valid]
        native,out=h.score(models,full,checks,dict(domain='development',product=rec['product']))
        ref=refs['development',rec['product']];assert sha(y)==ref['truth_sha256']
        assert {n:sha(a) for n,a in native.items()}==ref['native_score_sha256']
        assert {n:sha(a) for n,a in out.items()}==ref['probability64_sha256']
        assert np.array_equal(native['SourceS_100']>=previous['SourceS_old_logit_threshold'],out['SourceS_100']>=oldth['SourceS_100'])
        for n in NAMES:scores[n].append(out[n])
        ys.append(y);metas.append(d96.meta(rec));seals.append(ref)
        del full,native,out,raw,ss
        if (i+1)%10==0 or i+1==32:print('Phase100 development score recovery',i+1,32,flush=True)
    policies={};metricsstart=time.monotonic()
    for n in NAMES:
        aps=[checked_ap(s,y) for s,y in zip(scores[n],ys)]
        joined=np.concatenate(scores[n]);jy=np.concatenate(ys);pa=checked_ap(joined,jy)
        for kind,t in [('old',oldth[n]),('T_tail',locks[n]['threshold'])]:
            rows=[h.metric(s,y,t,dict(**m,domain='development',model=n,kind=kind),a) for s,y,m,a in zip(scores[n],ys,metas,aps)]
            if kind=='old':
                for row,ref in zip(rows,oldrefs[n]['rows']):
                    assert row['product']==ref['product']
                    for key in ['ap','recall','fpr','precision','true_positive','false_positive']:assert row[key]==ref[key],(n,key)
            policy=summarize(rows,t,pa,'development',n,kind)
            direct=h.metric(joined,jy,t,dict(product='pooled',group_id='pooled',tile='pooled'),pa)
            for key in ['true_positive','false_positive','positive','negative','surface','cloud','surface_false_positive','cloud_false_positive']:assert direct[key]==sum(row[key] for row in rows)
            for key in ['recall','fpr','precision','shadow_iou','surface_fpr','cloud_fpr']:assert policy['pooled_'+key]==direct[key]
            policies[n+'/'+kind]=policy
        assert policies[n+'/old']['macro_ap']==policies[n+'/T_tail']['macro_ap']
        assert policies[n+'/old']['pooled_ap']==policies[n+'/T_tail']['pooled_ap']
        del joined,jy
        print('Phase100 frozen evaluation complete',n,flush=True)
    definitions=[('MIN_T_tail_minus_SourceS_T_tail','MIN/T_tail','SourceS_100/T_tail'),
        ('G7_T_tail_minus_SourceS_T_tail','G7_100/T_tail','SourceS_100/T_tail')]
    definitions +=[(n+'_tail_minus_old',n+'/T_tail',n+'/old') for n in NAMES]
    comparisons={name:compare(policies[a],policies[b],name) for name,a,b in definitions}
    base=policies['SourceS_100/T_tail'];screening={}
    for n in ['G7_100','MIN']:
        a=policies[n+'/T_tail'];gain=a['macro_recall']-base['macro_recall'];apdelta=a['macro_ap']-base['macro_ap']
        criteria=dict(macro_fpr_at_most_1percent=a['macro_fpr']<=.01,max_group_fpr_at_most_10percent=a['max_group_fpr']<=.1,
            recall_gain_at_least_5pp=100*gain>=5,macro_AP_loss_at_most_1pp=100*apdelta>=-1)
        screening[n]=dict(criteria=criteria,all_four_pass=all(criteria.values()),delta_recall_pp=100*gain,delta_AP_pp=100*apdelta,
            decision='eligible_for_separate_stability_protocol' if all(criteria.values()) else 'close_fixed_tail_calibration_candidate')
    retained=[n for n in ['G7_100','MIN'] if screening[n]['all_four_pass']]
    # All six evaluations finished. Oracle values may now be read only for a
    # separate descriptive table, never to set or change any threshold.
    assert p.old.digest(D/'three_T_tail_lock.json')==thresholdhash
    oracle=p.old.read(P99/'phase99_report.json')['policies'];oraclecomparison=[]
    for n in NAMES:
        actual=policies[n+'/T_tail'];ref=oracle['development/'+n+'/O_tail']
        oraclecomparison.append(dict(model=n,calibration_T_tail=actual['threshold'],oracle_development_threshold=ref['threshold'],
            T_tail_macro_recall=actual['macro_recall'],oracle_macro_recall=ref['macro_recall'],
            T_tail_macro_fpr=actual['macro_fpr'],oracle_macro_fpr=ref['macro_fpr'],
            T_tail_max_group_fpr=actual['max_group_fpr'],oracle_max_group_fpr=ref['max_group_fpr'],
            labels_used_to_select_oracle=True,not_achievable_calibration_gain=True))
    assert r.indices_audit()==indices
    write('input_score_lock.json',dict(rows=seals,indices_after=indices,three_threshold_lock_sha256=thresholdhash))
    write('CPU_VERIFIED.json',dict(score_probes=checks,self_tests=tests,native_and_float64_scores_exact=True,
        calibration_thresholds_reproduce_Phase99=True,original_frozen_metrics_exact=True,AP_threshold_invariant=True,
        AP_independent_bins_and_sklearn=True,independent_pooled_integer_counts=True,
        development_oracle_read_only_after_all_six_evaluations=True,threshold_lock_unchanged_after_evaluation=True))
    report=dict(status='complete',created_utc=p.old.now(),policies=policies,comparisons=comparisons,screening=screening,retained_candidates=retained,
        decision='eligible_for_separate_stability_protocol' if retained else 'end_global_threshold_repair_route',
        SourceS_budget=dict(macro_point_at_most_1percent=base['macro_fpr']<=.01,max_group_at_most_10percent=base['max_group_fpr']<=.1,
            macro_upper95_at_most_1percent=base['intervals']['fpr']['upper95_one_sided']<=.01),
        new_calibration_thresholds=3,new_fits=0,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,
        bootstrap_seed=SEED,bootstrap_resamples=10000,Phase98_failure_unchanged=True,development_independent_confirmation=False,
        limitations=['Repeatedly exposed development;53 calibration maximum is only empirical',
            'Reference semantics not newly human confirmed; human B not performed;54 reused Phase89 blocks lack pre-crash SHA',
            'Actual false-positive rates differ across strategies; not equal actual FPR comparison',
            'Oracle gap is not achievable deployment calibration gain; no deployment winner selected'],
        cost=dict(seconds=time.monotonic()-started,calibration_seconds=calseconds,development_seconds=time.monotonic()-devstart,
            metrics_seconds=time.monotonic()-metricsstart,peak_rss_bytes=r.rss(),cpu_threads_max=4))
    write('phase100_report.json',report)
    write('oracle_comparison.json',dict(after_six_frozen_evaluations=True,rows=oraclecomparison))
    targetrows=[row for a in policies.values() for row in a['rows'] if row['tile'] in ['T31TGF','T02LKH']]
    write('T31TGF_T02LKH.json',dict(rows=targetrows,excluded_from_evaluation=False))
    render(report,oraclecomparison)
    write('DELIVERY_SHA256.json',{f.name:p.old.digest(f) for f in sorted(D.iterdir()) if f.is_file()})
    from verify_phase100_delivery import verify
    acceptance=verify(D);p.old.write(ROOT/'COMPLETED.json',dict(acceptance=acceptance,manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json')))
    print('Phase100 complete',report['decision'],retained,flush=True)

def render(report,oraclecomparison):
    totals=[];lines=['# Phase100：联合误报约束的校准迁移','','## 1．数据结果','',
        '仅原53组校准三个T_tail，落锁后原32组development一次评价六工作点。零拟合／网络前向／下载；非独立确认。','',
        '|评分|阈值|宏AP %|池化AP %|宏召回 %|宏FPR %|FPR单侧95%上界 %|最大FPR %|超1%组|零召回组|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,a in report['policies'].items():
        lines.append('|'+ '|'.join([a['model'],a['kind'],*[f'{100*a[k]:.4f}' for k in ['macro_ap','pooled_ap','macro_recall','macro_fpr']],
            f"{100*a['intervals']['fpr']['upper95_one_sided']:.4f}",f"{100*a['max_group_fpr']:.4f}",str(a['groups_above_budget']),str(len(a['zero_recall_groups']))])+'|')
        flat={k:v for k,v in a.items() if k not in ['rows','intervals','no_shadow_fpr']}
        for k,v in a['intervals'].items():
            flat[k+'_ci95']=v['ci95'];flat[k+'_upper95_one_sided']=v['upper95_one_sided']
        totals.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in flat.items()})
    for name,a in report['comparisons'].items():
        z=a['intervals_pp']['recall'];lines.append(f"\n{name}：宏召回差{z['mean']:.6f}个百分点，配对相关组95%描述区间{z['ci95']}，{z['n_groups']}组。")
    lines+=['','bootstrap10000次，seed100013；召回/AP仅有参考云影组，FPR全部有负类组；无参考云影组为NA。',
        '同评分新旧AP完全不变。实际误报率不同；不能称完全相同实际误报下比较。','',
        '## 2．结论探讨','',report['decision'],'筛选：'+json.dumps(report['screening'],ensure_ascii=False),
        'SourceS预算：'+json.dumps(report['SourceS_budget'],ensure_ascii=False),
        '点估计达标不代表稳定满足预算；单侧FPR上界完整报告。Phase98失败结论不变，未选择部署赢家。',
        'Phase99 development oracle只在全部六行评价完成后另表对照，不能将差额直接称可恢复校准收益。',
        *report['limitations'],'','成本：'+json.dumps(report['cost'])]
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    p.old.csvwrite(D/'six_strategy_totals.csv',totals)
    p.old.csvwrite(D/'per_group.csv',[row for a in report['policies'].values() for row in a['rows']])
    p.old.csvwrite(D/'paired_changes.csv',[row for a in report['comparisons'].values() for row in a['pairs']])
    p.old.csvwrite(D/'oracle_comparison.csv',oraclecomparison)

if __name__=='__main__':
    import sys
    with threadpool_limits(limits=4):
        if '--self-test' in sys.argv:print(runtime_self_test())
        else:main()
