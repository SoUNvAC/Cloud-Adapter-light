"""Exactly two fixed HGB fits after exact Phase92 baseline reproduction."""
import argparse
import copy
import importlib.metadata
import io
import json
import os
import pickle
import resource
import shutil
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_info, threadpool_limits
import run_phase92 as q
from phase93_math import COLUMNS, PARAMS, SEED, selected, equal_weights, independent_probability, probes
from verify_phase92_delivery import verify as verify92

p=q.p; fit67=q.fit67; recovery=q.recovery
ROOT=p.old.SHARED/'phase93_20261012'; D=ROOT/'delivery'
OLD=p.old.SHARED/'phase92_20261012/delivery'
OLD_SHA='abb2bea0612be87bc66ba78eef3ddad15b9099a7d5172f1d5df306fcf4a8f195'
ENV=p.old.SHARED/'phase93_environment_20261012'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE93_NONLINEAR_SPECTRAL_BASELINES_20261012.md'
NAMES=(*q.NAMES,'G6','G7')
FILES=['run_phase93.py','phase93_math.py','phase93_requirements.txt','run_phase93.sh',
       'setup_phase93_env.sh','run_phase92.py','run_phase90.py','run_phase67.py',
       'phase65d_recovery_math.py','phase65d_metrics.py','phase91_math.py','run_phase91.py',
       'phase89_math.py','phase89_cpu_ap.py','verify_phase93_delivery.py']


def write(name,value):p.old.write(D/name,value)
def sha(x):return q.sha_array(x)
def memory():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def preflight():
    assert sklearn.__version__=='1.5.2'
    assert not ROOT.exists(),'Never overwrite or repeat a fit attempt'
    assert p.old.digest(OLD/'DELIVERY_SHA256.json')==OLD_SHA
    acceptance=verify92(OLD)
    ROOT.mkdir();D.mkdir()
    wheels={f.name:p.old.digest(f) for f in sorted((ENV/'wheels').glob('*.whl'))}
    assert len(wheels)==5
    dependencies={n:importlib.metadata.version(n) for n in
                  ['numpy','scipy','scikit-learn','joblib','threadpoolctl']}
    assert dependencies==dict(numpy='1.26.4',scipy='1.13.1',**{'scikit-learn':'1.5.2'},joblib='1.4.2',threadpoolctl='3.5.0')
    write('dependency_lock.json',dict(versions=dependencies,wheel_sha256=wheels,
        python=sys.version,executable=sys.executable,environment=str(ENV),
        requirements_sha256=p.old.digest(Path(__file__).with_name('phase93_requirements.txt')),
        threadpools=threadpool_info(),thread_limit=4,original_conda_unchanged=True))
    # Read-only reuse of the prior evaluator with a NEW small-report destination.
    # No old model fitting; outputs are recomputed before either new tree is fit.
    saved=(q.ROOT,q.D)
    q.ROOT=ROOT/'baseline_reproduction';q.D=q.ROOT/'delivery';q.D.mkdir(parents=True)
    for n in ['fit_input_lock.json','two_model_lock.json','model_S3_lock.json','model_R6_lock.json']:
        shutil.copyfile(OLD/n,q.D/n)
    begin=time.monotonic()
    try:
        q.evaluate()
        fresh=p.old.read(q.D/'phase92_report.json');prior=p.old.read(OLD/'phase92_report.json')
        assert fresh['domains']==prior['domains'],'All old group, pooled, interval and oracle statistics must exactly reproduce'
    finally:q.ROOT,q.D=saved
    write('baseline_reproduction.json',dict(phase92_manifest_sha256=OLD_SHA,
        prior_acceptance=acceptance,all_four_policy_statistics_exact=True,
        domains=['development','original_main','original_common','revised_common'],
        frozen_rows=212,oracle_rows=212,pooled_summaries=32,
        original_bootstrap_seed=92012,reproduction_seconds=time.monotonic()-begin,
        newly_fit_models=0,completed_before_new_fits=True))
    print('Phase93 exact old baseline reproduction passed',flush=True)


def fit():
    begin=time.monotonic();base=fit67.base_verify()
    assert p.old.read(D/'baseline_reproduction.json')['all_four_policy_statistics_exact']
    audit=p.old.read(fit67.ROOT/'fit_input_audit.json');prior=p.old.read(OLD/'fit_input_lock.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==prior['original_fit_input_audit_sha256']
    fitrows=recovery.records('fit_representatives');samples=[];ys=[];receipts=[]
    assert len(fitrows)==len(audit['rows'])==153
    with zipfile.ZipFile(recovery.DATA/'subscenes.zip') as z:
        for i,row in enumerate(audit['rows']):
            r=fitrows[row['product']];recovery.prepare_verify(r)
            truth=np.load(recovery.PREP/r['mask_path']);v=truth!=255
            rawbytes=z.read('subscenes/'+row['product']+'.npy')
            assert p.old.hashlib.sha256(rawbytes).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(rawbytes),allow_pickle=False);x=fit67.full_fit_features(row)
            assert np.array_equal(x[:,2:5],raw[...,[7,11,12]][v].astype(np.float32))
            assert np.array_equal(x[:,5:8],raw[...,[3,2,1]][v].astype(np.float32))
            assert len(x)==int(v.sum())==row['valid_pixels']
            seed=65081+int(p.old.hashlib.sha256(row['product'].encode()).hexdigest()[:8],16)
            idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
            assert sha(idx)==row['sample_indices_sha256']
            yy=truth[v]==2;samples.append(x[idx]);ys.append(yy[idx])
            receipt=prior['rows'][i]
            assert receipt['product']==row['product'] and receipt['sample_truth_sha256']==sha(yy[idx])
            assert receipt['sample_count']==len(idx) and receipt['sample_positive']==int(yy[idx].sum())
            assert receipt['mask_sha256']==p.old.digest(recovery.PREP/r['mask_path'])
            receipts.append(receipt)
            print('Phase93 verify fit',i+1,153,flush=True)
    x=np.concatenate(samples).astype(np.float64);y=np.concatenate(ys).astype(float)
    del samples,ys
    assert sha(x)==audit['combined_sample_sha256']==prior['original_combined_sample_sha256']
    assert sha(y)==prior['sample_labels_sha256']
    assert len(x)==base['fit_samples']==prior['sample_count']==1253376
    w=equal_weights([r['sample_count'] for r in receipts])
    write('fit_input_lock.json',dict(protocol_sha256=p.old.digest(PROTOCOL),
        phase92_fit_input_sha256=p.old.digest(OLD/'fit_input_lock.json'),
        dependency_lock_sha256=p.old.digest(D/'dependency_lock.json'),
        baseline_reproduction_sha256=p.old.digest(D/'baseline_reproduction.json'),
        code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in FILES},
        sample_count=len(x),sample_labels_sha256=sha(y),combined_sample_sha256=sha(x),
        sample_weights_sha256=sha(w),sample_weight_sum=float(w.sum()),
        weight_rule='N/(153*n_i); sum N; no class weighting',rows=receipts,
        feature_columns=COLUMNS,standardization=False,fit_groups=153,
        Source_logit_clip=[1e-6,1-1e-6],new_segmentation_training=0,new_network_forwards=0))
    locks={};models={};costs={};reloaded=[]
    for n in COLUMNS:
        assert not (D/(n+'.pkl')).exists()
        xx=selected(x,n);m=HistGradientBoostingClassifier(**PARAMS)
        started=time.monotonic();m.fit(xx,y,sample_weight=w)
        cost=dict(fit_seconds=time.monotonic()-started,process_peak_rss_bytes=memory(),
                  peak_scope='cumulative process high-water, including preflight; not isolated estimator memory')
        assert np.array_equal(m.classes_,[0,1]) and m.n_iter_==200
        assert all(m.get_params()[k]==v for k,v in PARAMS.items())
        with (D/(n+'.pkl')).open('xb') as f:pickle.dump(m,f,protocol=5)
        with (D/(n+'.pkl')).open('rb') as f:reload=pickle.load(f)
        a=m.predict_proba(xx)[:,1].astype(np.float32);b=reload.predict_proba(xx)[:,1].astype(np.float32)
        assert np.array_equal(a,b)
        probe=probes(xx);other=independent_probability(reload,probe)
        np.testing.assert_allclose(other,reload.predict_proba(probe)[:,1].astype(np.float32),rtol=0,atol=6e-8)
        reloaded.append(dict(policy=n,all_training_samples_exact=True,samples=len(x),
            probability_sha256=sha(a),independent_tree_walk_probes=len(probe),
            independent_tree_walk_max_error=float(np.max(np.abs(other-reload.predict_proba(probe)[:,1].astype(np.float32))))))
        del xx,a,b,m
        started=time.monotonic();negatives=[];nr=[]
        for row in audit['rows']:
            xx=selected(fit67.full_fit_features(row),n);r=fitrows[row['product']]
            truth=np.load(recovery.PREP/r['mask_path']);yy=truth[truth!=255]==2
            prob=reload.predict_proba(xx)[:,1].astype(np.float32);negative=prob[~yy]
            assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all() and len(negative)
            negatives.append(negative);nr.append(dict(product=row['product'],negative_count=len(negative),
                                                     negative_probability_sha256=sha(negative)))
        threshold=q.group_threshold(negatives,.01)
        for a,r in zip(negatives,nr):r['false_positive']=int(np.count_nonzero(a>=threshold))
        rate=float(np.mean([r['false_positive']/r['negative_count'] for r in nr]));assert rate<=.01
        cost['full_fit_threshold_seconds']=time.monotonic()-started
        cost['process_peak_rss_bytes']=memory();costs[n]=cost
        lock=dict(model_file=n+'.pkl',model_sha256=p.old.digest(D/(n+'.pkl')),
            params=reload.get_params(),rounds=reload.n_iter_,classes=reload.classes_.tolist(),
            threshold=threshold,threshold_domain='probability; direct >=; NO sigmoid',fit_macro_fpr=rate,
            sample_features_sha256=sha(selected(x,n)),full_fit_negative_counts=nr,
            fit_input_lock_sha256=p.old.digest(D/'fit_input_lock.json'),
            frozen_before_evaluation=True,cost=cost,calibration_independent=False)
        write('model_'+n+'_lock.json',lock);locks[n]=p.old.digest(D/('model_'+n+'_lock.json'));models[n]=lock
        del negatives,xx,prob,negative,reload
        print('Phase93 frozen',n,rate,'rounds',200,flush=True)
    write('serialization_checks.json',dict(rows=reloaded,no_additional_estimator_fit=True))
    write('two_model_lock.json',dict(status='two_models_and_probability_thresholds_frozen',models=models,
        model_lock_sha256=locks,fit_input_lock_sha256=p.old.digest(D/'fit_input_lock.json'),
        new_tree_fits=2,new_segmentation_training=0,new_network_forwards=0,
        fit_costs=costs,fit_verification_seconds=time.monotonic()-begin,created_utc=p.old.now()))


def rebootstrap(summary):
    out=copy.deepcopy(summary)
    for k in out['macro']:out['macro'][k]=q.interval([r[k] for r in out['rows'] if r[k] is not None])
    out['no_shadow_macro_fpr']=q.interval([r['fpr'] for r in out['rows'] if not r['positive'] and r['fpr'] is not None])
    return out


def compare(policies):
    out={}
    for a,b in [('G7','G6'),('Source_spectral','G6'),('G6','R6')]:
        pairs=[]
        for x,y in zip(policies[a]['rows'],policies[b]['rows']):
            assert x['group_id']==y['group_id']
            pairs.append(dict(product=x['product'],tile=x['tile'],group_id=x['group_id'],
                **{'delta_'+k+'_pp':100*(x[k]-y[k]) if x[k] is not None and y[k] is not None else None
                   for k in ['ap','recall','fpr']}))
        supported=[r for r in pairs if r['delta_ap_pp'] is not None];v=[r['delta_ap_pp'] for r in supported]
        recalls=[r for r in pairs if r['delta_recall_pp'] is not None]
        out[a+'_minus_'+b]=dict(primary=a=='G7',pairs=pairs,delta_ap_pp=q.interval(v),
            improved=sum(d>1 for d in v),stable=sum(abs(d)<=1 for d in v),declined=sum(d< -1 for d in v),
            worst_ap=min(supported,key=lambda r:r['delta_ap_pp']) if supported else None,
            worst_recall=min(recalls,key=lambda r:r['delta_recall_pp']) if recalls else None,formal_confirmation=False)
    return out


def evaluate():
    started=time.monotonic();assert not (D/'phase93_report.json').exists()
    lock=p.old.read(D/'two_model_lock.json');fi=p.old.read(D/'fit_input_lock.json')
    assert p.old.digest(D/'fit_input_lock.json')==lock['fit_input_lock_sha256']
    assert p.old.digest(PROTOCOL)==fi['protocol_sha256']
    for n,h in fi['code_sha256'].items():assert p.old.digest(Path(__file__).with_name(n))==h,n
    assert p.old.digest(D/'dependency_lock.json')==fi['dependency_lock_sha256']
    models={}
    for n,h in lock['model_lock_sha256'].items():
        assert p.old.digest(D/('model_'+n+'_lock.json'))==h
        assert p.old.digest(D/(n+'.pkl'))==lock['models'][n]['model_sha256']
        with (D/(n+'.pkl')).open('rb') as f:models[n]=pickle.load(f)
    q.SEED=SEED;q.NAMES=tuple(COLUMNS);q.compare=compare
    prior=p.old.read(OLD/'phase92_report.json');results={};scans=[];costs={};inputs=[];checks=[]
    # Old baseline scans are recomputed in preflight and then carried unchanged;
    # only group-bootstrap intervals change to the predeclared Phase93 seed.
    old_scans=p.old.read(OLD/'oracle_threshold_receipts.json')['rows']
    scans+=copy.deepcopy(old_scans)
    for r in scans:r['score_units']='development: float32 logits; ALCD: float64 probability60m; original scan unchanged'
    def finish(domain,scores,ys,metas):
        v=dict(policies={},oracle_1percent={},oracle_only=True,no_oracle_threshold_deployment=True)
        for n in prior['domains'][domain]['policies']:
            for kind in ['policies','oracle_1percent']:v[kind][n]=rebootstrap(prior['domains'][domain][kind][n])
        for n in COLUMNS:
            threshold=lock['models'][n]['threshold']
            rr=[q.row_metric(s,y,threshold,dict(**m,policy=n)) for s,y,m in zip(scores[n],ys,metas)]
            v['policies'][n]=q.summarize(rr,scores[n],ys,threshold)
            for s,y,r in zip(scores[n],ys,rr):
                if r['ap'] is not None:
                    reference=float(average_precision_score(y==2,np.asarray(s,np.float32)))
                    assert abs(reference-r['ap'])<1e-12
            point=q.scan(scores[n],ys,[.01])[0];q.cpu_validate(point,scores[n],ys)
            point['score_precision']='float32 probability20m for development; float64 mean9 probability60m for ALCD; exact >= ties'
            rr=[q.row_metric(s,y,point['threshold'],dict(**m,policy=n)) for s,y,m in zip(scores[n],ys,metas)]
            v['oracle_1percent'][n]=q.summarize(rr,scores[n],ys,point['threshold'])
            scans.append(dict(domain=domain,policy=n,point=point))
        v['comparisons']=compare(v['policies']);results[domain]=v
        print('Phase93 domain complete',domain,flush=True)
    def predict(n,x,meta):
        t=time.monotonic();score=models[n].predict_proba(x)[:,1].astype(np.float32)
        costs[n]=costs.get(n,0.)+time.monotonic()-t
        a=probes(x);official=models[n].predict_proba(a)[:,1].astype(np.float32)
        other=independent_probability(models[n],a);err=float(np.max(np.abs(official-other)))
        assert err<=6e-8
        assert np.isfinite(score).all() and (score>=0).all() and (score<=1).all()
        checks.append(dict(**meta,policy=n,probe_count=len(a),independent_tree_max_error=err))
        return score
    dev=recovery.records('development_val')
    sources=[{r['product']:r for r in p.old.read(recovery.OLD/'development_val'/('index_'+n+'.json'))['rows']}
             for n in ['source','msre']]
    scores={n:[] for n in COLUMNS};ys=[];metas=[]
    with zipfile.ZipFile(recovery.DATA/'subscenes.zip') as z:
        for product,r in dev.items():
            recovery.prepare_verify(r);truth=np.load(recovery.PREP/r['mask_path']);v=truth!=255
            b=z.read('subscenes/'+product+'.npy');assert p.old.hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            ss=recovery.load(recovery.OLD,sources[0][product],'score');ms=recovery.load(recovery.OLD,sources[1][product],'score')
            full=p.combined(recovery.features(raw,ms,ss,v),raw[...,[3,2,1]][v])
            meta=dict(product=product,group_id=r['group_id'],tile=r.get('tile',product.split('_')[-2]))
            seals={}
            for n in COLUMNS:
                s=predict(n,selected(full,n),dict(domain='development',product=product));scores[n].append(s);seals[n]=sha(s)
            yy=truth[v];ys.append(yy);metas.append(meta)
            inputs.append(dict(domain='development',**meta,truth_sha256=sha(yy),score_sha256=seals))
            print('Phase93 development',len(ys),32,flush=True)
    finish('development',scores,ys,metas);del scores,ys,full
    p.old.ROOT=p.old.SHARED/'phase89_recovered_20261011';p.old.locked()
    rows=p.old.read(p.old.ROOT/'prepared/index.json')['rows']
    old_chain=p.old.read(OLD/'input_code_score_lock.json')['phase90_chain']
    assert rows==old_chain['prepared']['rows']
    scores={v:{n:[] for n in COLUMNS} for v,_,_ in p.VERSIONS}
    ys={v:[] for v,_,_ in p.VERSIONS};metas={v:[] for v,_,_ in p.VERSIONS}
    for i,row in enumerate(rows):
        prob={n:np.empty(row['shape60'],np.float64) for n in COLUMNS};coverage=np.zeros(row['shape60'],np.uint8)
        for y,x,h,w,ss,ms,spec in p.blocks(row):
            raw=np.ascontiguousarray(spec[...,[2,1,0,3,4,5]].reshape(-1,6),dtype=np.float64)
            from phase65d_recovery_math import logit_score
            # Original Phase67 combined input stores the Source logit as
            # float32. Preserve that quantization at both spatial resolutions.
            source=logit_score(ss[...,2]).astype(np.float32).ravel().astype(np.float64)
            for n in COLUMNS:
                xx=raw if n=='G6' else np.column_stack([raw,source])
                s=predict(n,xx,dict(domain='ALCD20m',product=row['product'],y60=y,x60=x)).reshape(h*3,w*3)
                a=s.astype(np.float64);value=p.area(a,3)
                independent=sum(a[dy::3,dx::3] for dy in range(3) for dx in range(3))/9
                error=float(np.max(np.abs(value-independent)));assert error<=2e-15
                threshold=lock['models'][n]['threshold']
                assert np.array_equal(value>=threshold,independent>=threshold)
                checks[-1].update(mean9_max_error=error,decisions_exact=True)
                prob[n][y:y+h,x:x+w]=value
            coverage[y:y+h,x:x+w]+=1
        assert (coverage==1).all()
        for version,tkey,mkey in p.VERSIONS:
            truth=p.old.array(row,tkey);mask=p.old.array(row,mkey)
            if not mask.any():continue
            yy=truth[mask];ys[version].append(yy);meta={k:row[k] for k in ['product','group_id','tile']};metas[version].append(meta)
            seals={}
            for n in COLUMNS:scores[version][n].append(prob[n][mask]);seals[n]=sha(prob[n][mask])
            inputs.append(dict(domain=version,**meta,truth_sha256=sha(yy),score_sha256=seals))
        print('Phase93 ALCD',i+1,9,flush=True)
    for v,_,_ in p.VERSIONS:finish(v,scores[v],ys[v],metas[v])
    write('input_score_lock.json',dict(fit_freeze_sha256=p.old.digest(D/'two_model_lock.json'),
        phase92_manifest_sha256=OLD_SHA,rows=inputs,no_large_arrays_saved=True,
        ALCD_rule='predict at20m; class1 float32 probability; float64 mean9 at60m; unchanged probability threshold'))
    write('CPU_VERIFIED.json',dict(independent_tree_walk_checks=checks,
        probe_scope='257 deterministic spaced pixels per full development group or ALCD20m block; not full-tree duplicate inference',
        independent_AP_score_bins_and_sklearn=True,independent_binary_counts=True,
        original_four_baseline_statistics_exact=True,oracle_direct_count_checks=24))
    write('oracle_threshold_receipts.json',dict(oracle_only=True,rows=scans))
    report=dict(status='completed_two_fixed_HGB_baselines',created_utc=p.old.now(),domains=results,
        primary_comparison='G7_minus_G6',secondary_comparisons=['Source_spectral_minus_G6','G6_minus_R6'],
        bootstrap_seed=SEED,resamples=10000,AP_boundary_pp=1,exploratory_only=True,
        new_tree_fits=2,new_segmentation_training=0,new_network_forwards=0,new_remote_sensing_downloads=0,
        no_deployment_winner_selected=True,original_failed_gates_unchanged=True,
        costs=dict(fit=lock['fit_costs'],predict_seconds=costs,evaluation_seconds=time.monotonic()-started,
                   process_peak_rss_bytes=memory(),CPU_threads_maximum=4))
    write('phase93_report.json',report);render(report)
    write('DELIVERY_SHA256.json',{f.name:p.old.digest(f) for f in sorted(D.iterdir()) if f.is_file()})
    p.old.write(ROOT/'COMPLETED.json',dict(created_utc=p.old.now(),delivery_manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json')))
    print('Phase93 completed',flush=True)


def render(report):
    saved=(q.D,q.NAMES);q.D=D;q.NAMES=NAMES
    # Reuse CSV rendering only, with the Phase93 comparison/summary wording below.
    totals=[];groups=[];pairs=[];oracles=[]
    for domain,v in report['domains'].items():
        for n,s in v['policies'].items():
            r=dict(domain=domain,policy=n,groups=s['groups'],shadow_groups=s['shadow_groups'],threshold=s['threshold'])
            for k,t in s['macro'].items():r.update({f'macro_{k}':t['mean'],f'macro_{k}_ci95':json.dumps(t['ci95']),f'macro_{k}_upper95':t['upper95_one_sided']})
            r.update({'pooled_'+k:s['pooled'][k] for k in ['ap','recall','precision','fpr','shadow_iou','surface_fpr','cloud_fpr']})
            r.update(groups_above_1percent=s['groups_above_1percent'],zero_recall_groups=len(s['zero_recall_groups']),no_shadow_fpr=s['no_shadow_macro_fpr']['mean'])
            totals.append(r);groups.extend(dict(domain=domain,**x) for x in s['rows'])
        for n,s in v['comparisons'].items():pairs.extend(dict(domain=domain,comparison=n,primary=s['primary'],**x) for x in s['pairs'])
        for n,s in v['oracle_1percent'].items():
            oracles.append(dict(domain=domain,policy=n,threshold=s['threshold'],oracle_only=True,
                macro_recall=s['macro']['recall']['mean'],macro_fpr=s['macro']['fpr']['mean'],
                pooled_recall=s['pooled']['recall'],pooled_fpr=s['pooled']['fpr']))
    for n,rows in [('six_strategy_totals.csv',totals),('per_group.csv',groups),('paired_changes.csv',pairs),('oracle_1percent.csv',oracles)]:p.old.csvwrite(D/n,rows)
    lines=['# Phase93: nonlinear raw spectral baselines','','## 1. Data results','',
        '| Domain | Model | MacroAP % | Recall % | FPR % | FPR upper95 % | PooledAP % |',
        '|---|---|---:|---:|---:|---:|---:|']
    f=lambda x:'NA' if x is None else f'{100*x:.4f}'
    for domain,v in report['domains'].items():
        for n,s in v['policies'].items():lines.append('| '+' | '.join([domain,n,*[f(s['macro'][k]['mean']) for k in ['ap','recall','fpr']],f(s['macro']['fpr']['upper95_one_sided']),f(s['pooled']['ap'])])+' |')
        c=v['comparisons']['G7_minus_G6'];t=c['delta_ap_pp']
        lines.append(f"\n{domain} G7-G6: {t['mean']:.4f} pp, CI95 {t['ci95']}, improved/stable/declined {c['improved']}/{c['stable']}/{c['declined']}.")
    lines+=['','## 2. Interpretation boundaries','',
        'Exactly two fixed 200-round HGB fits; no segmentation training or network forward. No parameter selection or deployment winner.',
        'All intervals are descriptive related-group bootstrap (10000, seed93012); exposed ALCD pools are not independent confirmation.',
        'Compare G6 with R6 to assess linear baseline weakness; G7-G6 measures practical Source score increment under this fixed learner only.',
        'G6/G7 thresholds are probabilities, used directly. Fit calibration is in-sample, not an independent FPR guarantee.',
        'Oracle thresholds use evaluation references and must not be deployed. No fabricated three-class mIoU.',
        'Phase89:54 reused probabilities lack pre-crash SHA. Human B was not performed; original empty fields and failed gates remain.']
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');q.D,q.NAMES=saved


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','fit','evaluate']);a=parser.parse_args()
    assert sklearn.__version__=='1.5.2'
    with threadpool_limits(limits=4):globals()[a.action]()
