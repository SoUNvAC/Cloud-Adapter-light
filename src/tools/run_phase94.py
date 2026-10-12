"""Phase94: exactly two fresh fits and four frozen thresholds; no network forwards."""
import argparse, copy, hashlib, importlib.metadata, io, json, pickle, resource, sys, time, zipfile
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import run_phase93 as prior
from phase93_math import PARAMS, equal_weights, independent_probability, probes
from phase65d_recovery_math import fit_model, predict_logit, logit_score, standardize, group_threshold
q=prior.q; p=prior.p; fit67=prior.fit67; recovery=prior.recovery
ROOT=p.old.SHARED/'phase94_20261012';D=ROOT/'delivery'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE94_HELDOUT_GROUP_CALIBRATION_20261012.md'
NAMES=('G7_100','SourceS_100');SEED=94012
OLD=p.old.SHARED/'phase93_20261012/delivery'
OLD_SHA='6ac8fd9c50cc94c19f5dd04d293cef2b28b048f881dd673e14b7e7b74323b8ec'

def split_rows(rows):
    assert len(rows)==153 and len({r['group_id'] for r in rows})==153
    ranked=sorted(rows,key=lambda r:(hashlib.sha256(('PHASE94_SPLIT_V1|'+r['group_id']).encode()).hexdigest(),r['group_id']))
    return ranked[:100],ranked[100:]

def write(n,v):p.old.write(D/n,v)
def sha(x):return q.sha_array(x)
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def columns(x,n):
    return np.ascontiguousarray(x[:,[5,6,7,2,3,4,1]],dtype=np.float64) if n=='G7_100' else x[:,[1,2,3,4]]
def sigmoid(s):return np.exp(-np.logaddexp(0.,-np.asarray(s,np.float64)))

def indices_audit():
    root=p.old.SHARED/'phase92_20261012/delivery';lock=p.old.read(root/'input_code_score_lock.json')
    chain=lock['phase90_chain'];out={}
    for n,h in lock['development_cache_index_sha256'].items():
        f=recovery.OLD/'development_val'/('index_'+n+'.json');assert p.old.digest(f)==h;out[str(f)]=h
    rec=p.old.SHARED/'phase89_recovered_20261011';runtime=rec/'delivery/input_and_runtime_index.json'
    assert p.old.digest(runtime)==chain['phase89_runtime_index_sha256'];original=p.old.read(runtime)
    for n in ['Source','MsRE']:
        f=rec/('scores_'+n)/'index.json';assert p.old.read(f)==original['inference'][n];out[str(f)]=p.old.digest(f)
    f=rec/'prepared/index.json';assert p.old.read(f)==original['prepared'];out[str(f)]=p.old.digest(f)
    return out

def fit():
    begin=time.monotonic();assert not ROOT.exists(),'Do not repeat or overwrite a fit attempt'
    assert sklearn.__version__=='1.5.2' and p.old.digest(OLD/'DELIVERY_SHA256.json')==OLD_SHA
    assert {n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','joblib','threadpoolctl']}==p.old.read(OLD/'dependency_lock.json')['versions']
    for n,h in p.old.read(OLD/'DELIVERY_SHA256.json').items():assert p.old.digest(OLD/n)==h,n
    records=recovery.records('fit_representatives');audit=p.old.read(fit67.ROOT/'fit_input_audit.json')
    oldfi=p.old.read(OLD/'fit_input_lock.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==p.old.read(prior.OLD/'fit_input_lock.json')['original_fit_input_audit_sha256']
    receipts=copy.deepcopy(oldfi['rows']);assert [r['product'] for r in receipts]==[r['product'] for r in audit['rows']]
    for r in receipts:assert r['group_id']==records[r['product']]['group_id']
    fitting,calibration=split_rows(receipts)
    assert not {r['group_id'] for r in receipts}&{r['group_id'] for r in recovery.records('development_val').values()}
    ROOT.mkdir();D.mkdir()
    files=['run_phase94.py','run_phase94.sh','verify_phase94_delivery.py','run_phase93.py','run_phase92.py','run_phase90.py','run_phase67.py','phase93_math.py','phase65d_recovery_math.py','phase65d_metrics.py','phase89_cpu_ap.py','phase91_math.py']
    # No truth arrays or sample labels have been opened: seal identities, indices and existing SHA first.
    write('split_input_lock.json',dict(created_utc=p.old.now(),rule='SHA256(PHASE94_SPLIT_V1|group_id), ascending; ties group_id',
        fit=fitting,calibration=calibration,fit_groups=100,calibration_groups=53,original_fit_groups=153,
        protocol_sha256=p.old.digest(PROTOCOL),phase93_manifest_sha256=OLD_SHA,
        code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in files},
        indices_before=indices_audit(),sealed_before_truth_arrays=True,zero_group_overlap=True,
        dependency_lock= p.old.read(OLD/'dependency_lock.json')))
    base=fit67.base_verify();am={r['product']:r for r in audit['rows']};samples=[];labels=[]
    with zipfile.ZipFile(recovery.DATA/'subscenes.zip') as z:
        for i,r in enumerate(fitting):
            row=am[r['product']];rec=records[r['product']];recovery.prepare_verify(rec)
            truth=np.load(recovery.PREP/rec['mask_path']);v=truth!=255
            b=z.read('subscenes/'+r['product']+'.npy');assert hashlib.sha256(b).hexdigest()==rec['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False);x=fit67.full_fit_features(row)
            assert np.array_equal(x[:,2:5],raw[...,[7,11,12]][v].astype(np.float32))
            assert np.array_equal(x[:,5:8],raw[...,[3,2,1]][v].astype(np.float32))
            seed=65081+int(hashlib.sha256(r['product'].encode()).hexdigest()[:8],16)
            idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False);y=truth[v]==2
            assert sha(idx)==r['sample_indices_sha256'] and sha(y[idx])==r['sample_truth_sha256']
            assert len(idx)==r['sample_count'] and int(y[idx].sum())==r['sample_positive']
            samples.append(x[idx]);labels.append(y[idx]);print('Phase94 input',i+1,100,flush=True)
    x=np.concatenate(samples).astype(np.float64);y=np.concatenate(labels).astype(float);del samples,labels
    assert np.unique(y).tolist()==[0.,1.],'Stop: absent binary support; do not repartition'
    w=equal_weights([r['sample_count'] for r in fitting]);logw=w/w.sum()
    write('fit_input_lock.json',dict(split_lock_sha256=p.old.digest(D/'split_input_lock.json'),
        sample_count=len(x),sample_features_sha256=sha(x),sample_labels_sha256=sha(y),
        tree_weights_sha256=sha(w),tree_weights_sum=float(w.sum()),tree_weight_rule='N/(100*n_i)',
        logistic_weights_sha256=sha(logw),logistic_weight_sum=float(logw.sum()),standardization_groups=[r['group_id'] for r in fitting],
        no_calibration_group_standardization=True,new_fits=2,new_network_forwards=0,new_segmentation_training=0))
    models={};cost={}
    t=time.monotonic();m=HistGradientBoostingClassifier(**PARAMS);m.fit(columns(x,'G7_100'),y,sample_weight=w)
    assert m.n_iter_==200 and np.array_equal(m.classes_,[0,1])
    with (D/'G7_100.pkl').open('xb') as f:pickle.dump(m,f,protocol=5)
    with (D/'G7_100.pkl').open('rb') as f:models['G7_100']=pickle.load(f)
    a=m.predict_proba(columns(x,'G7_100'))[:,1].astype(np.float32)
    assert np.array_equal(a,models['G7_100'].predict_proba(columns(x,'G7_100'))[:,1].astype(np.float32))
    cost['G7_100']=dict(fit_seconds=time.monotonic()-t,process_peak_rss_bytes=rss(),all_training_reload_exact=True)
    del m,a
    t=time.monotonic();m=fit_model(columns(x,'SourceS_100'),y,logw,1)
    mean,scale=standardize(columns(x,'SourceS_100'),logw)
    assert np.array_equal(mean,m['mean']) and np.array_equal(scale,m['scale']) and m['theta'][0]>=0
    write('SourceS_100.json',m);models['SourceS_100']=p.old.read(D/'SourceS_100.json')
    assert np.array_equal(predict_logit(columns(x,'SourceS_100'),m),predict_logit(columns(x,'SourceS_100'),models['SourceS_100']))
    cost['SourceS_100']=dict(fit_seconds=time.monotonic()-t,process_peak_rss_bytes=rss(),all_training_reload_exact=True)
    del x,y,w,logw
    write('two_model_lock.json',dict(created_utc=p.old.now(),fit_input_lock_sha256=p.old.digest(D/'fit_input_lock.json'),
        models={'G7_100':dict(file='G7_100.pkl',sha256=p.old.digest(D/'G7_100.pkl'),params=PARAMS),
                'SourceS_100':dict(file='SourceS_100.json',sha256=p.old.digest(D/'SourceS_100.json'),params=m)},
        new_fits=2,calibration_labels_not_used_in_fit=True,cost=cost))
    negatives={n:{'in':[],'cal':[]} for n in NAMES};metadata={'in':[],'cal':[]};checks=[]
    for subset,rows in [('in',fitting),('cal',calibration)]:
        for i,r in enumerate(rows):
            rec=records[r['product']];recovery.prepare_verify(rec);truth=np.load(recovery.PREP/rec['mask_path']);yy=truth[truth!=255]
            xx=fit67.full_fit_features(am[r['product']]);metadata[subset].append({k:r[k] for k in ['product','group_id']})
            for n in NAMES:
                s=predict(models,n,columns(xx,n),checks,dict(domain='fit_'+subset,product=r['product']))
                negatives[n][subset].append(s[yy!=2])
            print('Phase94 calibrate',subset,i+1,len(rows),flush=True)
    locks={};diagnostics=[]
    for n in NAMES:
        locks[n]={}
        for subset in ['in','cal']:
            arr=negatives[n][subset];eligible=[a for a in arr if len(a)]
            assert eligible,'Stop: no negative support'
            th=group_threshold(eligible,.01);rate=float(np.mean([(a>=th).mean() for a in eligible]));assert rate<=.01
            below=[a[a<th].max() for a in eligible if (a<th).any()];next_t=float(max(below)) if below else None
            next_rate=float(np.mean([(a>=next_t).mean() for a in eligible])) if next_t is not None else None
            assert next_rate is None or next_rate>.01,'Next observed exact-tie threshold must violate budget'
            locks[n][subset]=dict(threshold=th,probability_threshold=th if n=='G7_100' else float(sigmoid(th)),
                score_domain='float32 probability direct >=' if n=='G7_100' else 'float32 logit >=; ALCD float64 mean9 sigmoid',
                calibration_subset=subset,groups=len(arr),negative_groups=len(eligible),macro_fpr=rate,
                next_observed_threshold=next_t,next_macro_fpr=next_rate,
                rows=[dict(**r,negative_count=len(a),false_positive=int((a>=th).sum()),negative_score_sha256=sha(a),
                           fpr=float((a>=th).mean()) if len(a) else None) for a,r in zip(arr,metadata[subset])])
    # Four thresholds sealed before diagnostic aggregation or any external evaluation.
    write('four_threshold_lock.json',dict(created_utc=p.old.now(),two_model_lock_sha256=p.old.digest(D/'two_model_lock.json'),
        models=locks,all_frozen_before_external_evaluation=True,no_model_selection=True))
    for n in NAMES:
        for target in ['in','cal']:
            for threshold_source in ['in','cal']:
                th=locks[n][threshold_source]['threshold'];rs=[]
                for a,r in zip(negatives[n][target],metadata[target]):
                    rs.append(dict(**r,negative_count=len(a),false_positive=int((a>=th).sum()),fpr=float((a>=th).mean()) if len(a) else None))
                diagnostics.append(dict(model=n,data_subset=target,threshold_source=threshold_source,rows=rs,
                    macro_fpr=q.interval([r['fpr'] for r in rs if r['fpr'] is not None]),generalization=False))
    write('train_calibration_diagnostics.json',dict(rows=diagnostics));write('calibration_cpu_checks.json',dict(rows=checks))
    write('fit_cost.json',dict(seconds=time.monotonic()-begin,process_peak_rss_bytes=rss(),
        scope='one fit process, cumulative input verification + two fits + full 153-group threshold calibration'))
    print('Phase94 four thresholds locked',flush=True)

def predict(models,n,x,checks,meta):
    if n=='G7_100':
        s=models[n].predict_proba(x)[:,1].astype(np.float32);a=probes(x)
        err=float(np.max(np.abs(independent_probability(models[n],a)-models[n].predict_proba(a)[:,1].astype(np.float32))))
        assert err<=6e-8;checks.append(dict(**meta,model=n,probe_count=len(a),max_error=err))
    else:
        s=predict_logit(x,models[n]);a=probes(x);m=models[n]
        # Explicit per-feature accumulation, independent of BLAS matrix multiplication.
        z=(a-np.asarray(m['mean']))/np.asarray(m['scale']);v=np.full(len(a),m['theta'][-1],float)
        for j,c in enumerate(m['theta'][:-1]):v+=z[:,j]*c
        err=float(np.max(np.abs(v.astype(np.float32)-predict_logit(a,m))))
        assert err<1e-4;checks.append(dict(**meta,model=n,probe_count=len(a),max_error=err))
    assert np.isfinite(s).all();return s

def comparisons(policies):
    out={}
    for a,b in [('G7_100_T_cal','G7_100_T_in'),('SourceS_100_T_cal','SourceS_100_T_in'),('G7_100_T_cal','SourceS_100_T_cal')]:
        pairs=[]
        for x,y in zip(policies[a]['rows'],policies[b]['rows']):
            assert x['group_id']==y['group_id']
            pairs.append(dict(product=x['product'],group_id=x['group_id'],tile=x['tile'],
                **{'delta_'+k+'_pp':100*(x[k]-y[k]) if x[k] is not None and y[k] is not None else None for k in ['ap','fpr','recall','precision','shadow_iou']}))
        out[a+'_minus_'+b]=dict(primary=a=='G7_100_T_cal' and b=='G7_100_T_in',pairs=pairs,
            **{'delta_'+k+'_pp':q.interval([r['delta_'+k+'_pp'] for r in pairs if r['delta_'+k+'_pp'] is not None]) for k in ['ap','fpr','recall','precision','shadow_iou']},
            worst_recall=min((r for r in pairs if r['delta_recall_pp'] is not None),key=lambda r:r['delta_recall_pp'],default=None),
            largest_fpr_increase=max((r for r in pairs if r['delta_fpr_pp'] is not None),key=lambda r:r['delta_fpr_pp'],default=None))
    return out

def evaluate():
    started=time.monotonic();assert not (D/'phase94_report.json').exists()
    lock=p.old.read(D/'four_threshold_lock.json');ml=p.old.read(D/'two_model_lock.json');sl=p.old.read(D/'split_input_lock.json')
    assert lock['two_model_lock_sha256']==p.old.digest(D/'two_model_lock.json')
    assert ml['fit_input_lock_sha256']==p.old.digest(D/'fit_input_lock.json')
    assert p.old.read(D/'fit_input_lock.json')['split_lock_sha256']==p.old.digest(D/'split_input_lock.json')
    assert sl['protocol_sha256']==p.old.digest(PROTOCOL) and sl['indices_before']==indices_audit()
    for n,h in sl['code_sha256'].items():assert p.old.digest(Path(__file__).with_name(n))==h,n
    models={}
    for n,m in ml['models'].items():
        assert p.old.digest(D/m['file'])==m['sha256']
        if n=='G7_100':
            with (D/m['file']).open('rb') as f:models[n]=pickle.load(f)
        else:models[n]=p.old.read(D/m['file'])
    checks=[];inputs=[];results={};historical=p.old.read(OLD/'phase93_report.json')
    def finish(domain,scores,ys,metas):
        policies={}
        for n in NAMES:
            for subset in ['in','cal']:
                name=n+'_T_'+subset;th=lock['models'][n][subset]['threshold' if domain=='development' else 'probability_threshold']
                rr=[q.row_metric(s,y,th,dict(**m,policy=name)) for s,y,m in zip(scores[n],ys,metas)]
                policies[name]=q.summarize(rr,scores[n],ys,th)
            a=policies[n+'_T_in'];b=policies[n+'_T_cal']
            assert a['macro']['ap']==b['macro']['ap'] and a['pooled']['ap']==b['pooled']['ap']
            assert [r['ap'] for r in a['rows']]==[r['ap'] for r in b['rows']]
        results[domain]=dict(policies=policies,comparisons=comparisons(policies),
            historical_153_group_reference={n:historical['domains'][domain]['policies'][n] for n in ['G7','Source_spectral']},
            historical_not_causal_calibration_comparison=True,same_model_scores_AP_exact_invariant=True)
        print('Phase94 domain complete',domain,flush=True)
    sources=[{r['product']:r for r in p.old.read(recovery.OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    scores={n:[] for n in NAMES};ys=[];metas=[]
    with zipfile.ZipFile(recovery.DATA/'subscenes.zip') as z:
        for product,r in recovery.records('development_val').items():
            recovery.prepare_verify(r);truth=np.load(recovery.PREP/r['mask_path']);v=truth!=255
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image'];raw=np.load(io.BytesIO(b),allow_pickle=False)
            ss=recovery.load(recovery.OLD,sources[0][product],'score');ms=recovery.load(recovery.OLD,sources[1][product],'score')
            full=p.combined(recovery.features(raw,ms,ss,v),raw[...,[3,2,1]][v]);meta=dict(product=product,group_id=r['group_id'],tile=r.get('tile',product.split('_')[-2]))
            seals={}
            for n in NAMES:
                s=predict(models,n,columns(full,n),checks,dict(domain='development',product=product));scores[n].append(s);seals[n]=sha(s)
            ys.append(truth[v]);metas.append(meta);inputs.append(dict(domain='development',**meta,truth_sha256=sha(truth[v]),score_sha256=seals))
            print('Phase94 development',len(ys),32,flush=True)
    finish('development',scores,ys,metas);del scores,ys,full
    p.old.ROOT=p.old.SHARED/'phase89_recovered_20261011';p.old.locked();rows=p.old.read(p.old.ROOT/'prepared/index.json')['rows']
    assert rows==p.old.read(prior.OLD/'input_code_score_lock.json')['phase90_chain']['prepared']['rows']
    scores={v:{n:[] for n in NAMES} for v,_,_ in p.VERSIONS};ys={v:[] for v,_,_ in p.VERSIONS};metas={v:[] for v,_,_ in p.VERSIONS}
    for i,row in enumerate(rows):
        prob={n:np.empty(row['shape60'],np.float64) for n in NAMES};coverage=np.zeros(row['shape60'],np.uint8)
        for y,x,h,w,ss,ms,spec in p.blocks(row):
            source=logit_score(ss[...,2]).astype(np.float32).ravel()
            features={'G7_100':np.column_stack([spec[...,[2,1,0,3,4,5]].reshape(-1,6),source]).astype(np.float64),
                      'SourceS_100':np.asfortranarray(np.column_stack([source,spec[...,[3,4,5]].reshape(-1,3)]),dtype=np.float64)}
            for n in NAMES:
                s=predict(models,n,features[n],checks,dict(domain='ALCD20m',product=row['product'],y60=y,x60=x)).reshape(h*3,w*3)
                a=s.astype(np.float64) if n=='G7_100' else sigmoid(s)
                value=p.area(a,3);independent=sum(a[dy::3,dx::3] for dy in range(3) for dx in range(3))/9
                err=float(np.max(np.abs(value-independent)));assert err<=2e-15
                for subset in ['in','cal']:assert np.array_equal(value>=lock['models'][n][subset]['probability_threshold'],independent>=lock['models'][n][subset]['probability_threshold'])
                checks[-1].update(mean9_error=err,decisions_exact=True);prob[n][y:y+h,x:x+w]=value
            coverage[y:y+h,x:x+w]+=1
        assert (coverage==1).all()
        for version,tkey,mkey in p.VERSIONS:
            truth=p.old.array(row,tkey);mask=p.old.array(row,mkey)
            if not mask.any():continue
            yy=truth[mask];ys[version].append(yy);meta={k:row[k] for k in ['product','group_id','tile']};metas[version].append(meta);seals={}
            for n in NAMES:scores[version][n].append(prob[n][mask]);seals[n]=sha(prob[n][mask])
            inputs.append(dict(domain=version,**meta,truth_sha256=sha(yy),score_sha256=seals))
        print('Phase94 ALCD',i+1,9,flush=True)
    for v,_,_ in p.VERSIONS:finish(v,scores[v],ys[v],metas[v])
    assert sl['indices_before']==indices_audit()
    write('input_score_lock.json',dict(four_threshold_lock_sha256=p.old.digest(D/'four_threshold_lock.json'),rows=inputs,
        indices_after=indices_audit(),no_large_arrays_saved=True,ALCD_rule='predict20m; sigmoid only SourceS float32 logit; float64 mean9; frozen probability threshold'))
    write('CPU_VERIFIED.json',dict(rows=checks,independent_score_bin_AP_and_integer_binary_counts=True,
        all_AP_threshold_invariants=True,independent_tree_and_linear_probes_only=True))
    report=dict(status='complete',created_utc=p.old.now(),domains=results,new_fits=2,new_network_forwards=0,new_segmentation_training=0,new_downloads=0,
        bootstrap_seed=SEED,resamples=10000,exploratory_only=True,no_deployment_winner=True,original_failed_gates_unchanged=True,
        limits=['Human B not performed; original empty fields unchanged','54 Phase89 reused probability blocks lack pre-crash SHA','Exposed development and ALCD pools; no independent confirmation',
                'Threshold change cannot improve ranking; no synthetic three-class mIoU','100-group models never refit on 153 groups'],
        cost=dict(fit=p.old.read(D/'fit_cost.json'),evaluation_seconds=time.monotonic()-started,evaluation_peak_rss_bytes=rss(),cpu_threads_max=4))
    write('phase94_report.json',report);render(report)
    write('DELIVERY_SHA256.json',{f.name:p.old.digest(f) for f in sorted(D.iterdir()) if f.is_file()})
    p.old.write(ROOT/'COMPLETED.json',dict(created_utc=p.old.now(),delivery_manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json')))
    print('Phase94 complete',flush=True)

def render(report):
    totals=[];groups=[];changes=[];history=[]
    lines=['# Phase94：训练内与留出组校准','','## 1．数据结果','','100组拟合、53组校准；两个全新模型，四个冻结阈值。零网络训练、零网络前向、零下载。','',
        '|评价池|策略|宏AP %|宏召回 %|宏精确率 %|宏FPR %|FPR单侧95%上界 %|超预算组|Shadow IoU %|池化AP %|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    f=lambda x:'未定义' if x is None else f'{100*x:.4f}'
    for domain,v in report['domains'].items():
        for n,s in v['policies'].items():
            r=dict(domain=domain,policy=n,groups=s['groups'],shadow_groups=s['shadow_groups'],threshold=s['threshold'],groups_above_1percent=s['groups_above_1percent'],zero_recall_groups=len(s['zero_recall_groups']))
            for k,it in s['macro'].items():r.update({f'macro_{k}':it['mean'],f'macro_{k}_ci95':json.dumps(it['ci95']),f'macro_{k}_upper95':it['upper95_one_sided']})
            r.update({'pooled_'+k:s['pooled'][k] for k in ['ap','fpr','recall','precision','shadow_iou','surface_fpr','cloud_fpr']});r['no_shadow_macro_fpr']=s['no_shadow_macro_fpr']['mean']
            totals.append(r);groups.extend(dict(domain=domain,**x) for x in s['rows'])
            lines.append('| '+' | '.join([domain,n,*[f(s['macro'][k]['mean']) for k in ['ap','recall','precision','fpr']],f(s['macro']['fpr']['upper95_one_sided']),str(s['groups_above_1percent']),f(s['macro']['shadow_iou']['mean']),f(s['pooled']['ap'])])+' |')
        for n,c in v['comparisons'].items():changes.extend(dict(domain=domain,comparison=n,primary=c['primary'],**r) for r in c['pairs'])
        for n,s in v['historical_153_group_reference'].items():history.append(dict(domain=domain,model=n,macro_ap=s['macro']['ap']['mean'],macro_fpr=s['macro']['fpr']['mean'],macro_recall=s['macro']['recall']['mean'],historical_only=True))
    lines+=['','## 2．结论探讨','','同模型T_cal−T_in比较只涉及工作点，不是排序恢复；两阈值间逐组与池化AP完全一致。',
        '留出校准内≤1%不保证未见组或跨数据集≤1%。误报降低须同时查看召回代价。两模型FPR不同不能仅按召回选择赢家。',
        '原153组模型仅历史参考，不能用新旧差异隔离校准作用。拟合组数、校准组差异、跨域及60m聚合均可能影响。',
        '所有区间为相关组bootstrap10000次、seed94012的描述性区间；已暴露评价池不是独立确认。',*report['limits'],'',
        '完整同模型配对区间、最差召回损伤和最大FPR增加见phase94_report.json；全部逐组值见per_group.csv和paired_changes.csv。']
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for n,rs in [('four_strategy_totals.csv',totals),('per_group.csv',groups),('paired_changes.csv',changes),('historical_reference.csv',history)]:p.old.csvwrite(D/n,rs)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['fit','evaluate']);args=a.parse_args();q.SEED=SEED
    assert sklearn.__version__=='1.5.2'
    with threadpool_limits(limits=4):globals()[args.action]()
