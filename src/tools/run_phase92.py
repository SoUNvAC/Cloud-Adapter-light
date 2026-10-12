"""Two fixed raw-TOA logistic baselines; fit freeze before exploratory readout."""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import run_phase90 as p
import run_phase67 as fit67
import run_phase65d_recovery as recovery
from phase65d_recovery_math import fit_model, predict_logit, group_threshold
from phase91_math import scan, cpu_validate
from run_phase91 import binary_row

ROOT=p.old.SHARED/'phase92_20261012'; D=ROOT/'delivery'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE92_RAW_SPECTRAL_BASELINES_20261012.md'
NAMES=('S3','R6','Source_spectral','L3')
COLUMNS={'S3':[2,3,4],'R6':[5,6,7,2,3,4]}
BANDS={'S3':['B08','B11','B12'],'R6':['B04','B03','B02','B08','B11','B12']}
SEED=92012
P90=p.old.SHARED/'phase90_20261012_run02/delivery'
P90_SHA='1e994f31cf03b593e7fce7962e4b77057910de4072a9aee8635db77262ee2923'


def sha_array(x):return p.old.hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(name,x):p.old.write(D/name,x)


def interval(values):
    x=np.asarray(values,np.float64)
    r=dict(n_groups=len(x),mean=float(x.mean()) if len(x) else None,ci95=None,upper95_one_sided=None,
           seed=SEED,resamples=10000,descriptive_only=True)
    if len(x)>1:
        means=x[np.random.default_rng(SEED).integers(len(x),size=(10000,len(x)))].mean(1)
        r.update(ci95=np.quantile(means,[.025,.975]).tolist(),upper95_one_sided=float(np.quantile(means,.95)))
    return r


def raw_selected(x,name):return np.asarray(x)[:,COLUMNS[name]]


def fit():
    started=time.monotonic();assert not ROOT.exists(),'Never refit or overwrite an attempt'
    base=fit67.base_verify();lock67=p.old.read(fit67.ROOT/'control_lock.json')
    assert p.old.digest(fit67.ROOT/'control_lock.json')==p.CONTROL_SHA
    audit67=p.old.read(fit67.ROOT/'fit_input_audit.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==lock67['fit_input_audit_sha256']
    assert p.old.digest(recovery.PREP/'manifest.json')==base['prepared_manifest_sha256']
    fitrows=recovery.records('fit_representatives');devrows=recovery.records('development_val')
    assert not {r['group_id'] for r in fitrows.values()} & {r['group_id'] for r in devrows.values()}
    previous=p.old.read(fit67.RECOVERY/'fit_feature_index.json')
    assert len(audit67['rows'])==len(fitrows)==153
    assert [r['product'] for r in audit67['rows']]==list(fitrows)
    samples=[];ys=[];weights=[];receipts=[]
    # Verify all original spectral columns against official archived TOA, not
    # merely a cache filename. Retain the original indices, labels and weights.
    with zipfile.ZipFile(recovery.DATA/'subscenes.zip') as z:
        for i,row in enumerate(audit67['rows']):
            r=fitrows[row['product']];recovery.prepare_verify(r)
            truth=np.load(recovery.PREP/r['mask_path']);valid=truth!=255
            b=z.read('subscenes/'+row['product']+'.npy')
            assert p.old.hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            x=fit67.full_fit_features(row)
            assert np.array_equal(x[:,2:5],raw[...,[7,11,12]][valid].astype(np.float32))
            assert np.array_equal(x[:,5:8],raw[...,[3,2,1]][valid].astype(np.float32))
            assert len(x)==row['valid_pixels']==int(valid.sum())
            seed=65081+int(p.old.hashlib.sha256(row['product'].encode()).hexdigest()[:8],16)
            idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
            y=truth[valid]==2
            assert sha_array(idx)==row['sample_indices_sha256']
            assert len(idx)==row['sample_count'] and int(y[idx].sum())==row['sample_positive']
            samples.append(x[idx]);ys.append(y[idx]);weights.append(np.full(len(idx),1/(153*len(idx))))
            receipts.append(dict(product=row['product'],group_id=r['group_id'],
                image_member_sha256=r['member_sha256']['image'],mask_sha256=p.old.digest(recovery.PREP/r['mask_path']),
                old_feature_sha256=row['feature_sha256'],rgb_sha256=row['rgb_sha256'],
                sample_indices_sha256=sha_array(idx),sample_count=len(idx),sample_positive=int(y[idx].sum()),
                sample_truth_sha256=sha_array(y[idx]),valid_pixels=len(x),spectral_columns_exact=True))
            print('Phase92 verify fit',i+1,153,flush=True)
    x=np.concatenate(samples).astype(np.float64);y=np.concatenate(ys).astype(float);w=np.concatenate(weights)
    assert sha_array(x)==audit67['combined_sample_sha256']
    assert sha_array(x[:,:5])==previous['sample_x_sha256'] and sha_array(y)==previous['sample_y_sha256']
    assert len(x)==base['fit_samples']==audit67['sample_count']
    assert np.isfinite(x).all() and abs(w.sum()-1)<1e-12
    ROOT.mkdir();D.mkdir()
    write('fit_input_lock.json',dict(created_utc=p.old.now(),protocol_sha256=p.old.digest(PROTOCOL),
        original_fit_input_audit_sha256=p.old.digest(fit67.ROOT/'fit_input_audit.json'),
        recovery_lock_sha256=p.old.digest(fit67.RECOVERY/'recovery_lock.json'),
        phase67_control_lock_sha256=p.old.digest(fit67.ROOT/'control_lock.json'),
        sample_count=len(x),sample_positive=int(y.sum()),sample_labels_sha256=sha_array(y),
        sample_weights_sha256=sha_array(w),original_combined_sample_sha256=sha_array(x),
        feature_columns=COLUMNS,band_names=BANDS,rows=receipts,
        fit_groups=153,group_equal_weights=True,class_balancing=False,positive_features=0,
        penalty=.001,standardization='fit group-weighted mean and population scale only',
        optimizer='unchanged phase65d_recovery_math.fit_model L-BFGS-B; no retry on nonconvergence',
        code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in
                     ['run_phase92.py','phase65d_recovery_math.py','phase65d_metrics.py','phase91_math.py']},
        development_or_external_fitting=False,phase83_nonshadow_samples_used=False))
    models={};costs={};negative_receipts={}
    for name in COLUMNS:
        start=time.monotonic();xx=raw_selected(x,name);m=fit_model(xx,y,w,positive_features=0)
        costs[name]=dict(fit_seconds=time.monotonic()-start);negatives=[];nr=[]
        start=time.monotonic()
        for row in audit67['rows']:
            xx=raw_selected(fit67.full_fit_features(row),name)
            r=fitrows[row['product']];truth=np.load(recovery.PREP/r['mask_path']);yy=truth[truth!=255]==2
            score=predict_logit(xx,m);negative=score[~yy];assert len(negative)>0
            independent=independent_logit(xx,m);assert np.array_equal(score,independent)
            negatives.append(negative)
            nr.append(dict(product=row['product'],negative_count=len(negative),negative_logit_sha256=sha_array(negative)))
        m['threshold']=group_threshold(negatives,.01)
        for a,r in zip(negatives,nr):r['false_positive']=int(np.count_nonzero(a>=m['threshold']))
        m['fit_macro_fpr']=float(np.mean([r['false_positive']/r['negative_count'] for r in nr]))
        assert m['fit_macro_fpr']<=.01,'Stop if original conservative threshold fails exact count budget'
        m.update(bands=BANDS[name],columns=COLUMNS[name],sample_features_sha256=sha_array(raw_selected(x,name)))
        costs[name]['full_fit_threshold_seconds']=time.monotonic()-start
        models[name]=m;negative_receipts[name]=nr
        write('model_'+name+'_lock.json',dict(model=m,fit_input_lock_sha256=p.old.digest(D/'fit_input_lock.json'),
            frozen_before_evaluation=True,full_fit_negative_counts=nr,cost=costs[name],
            threshold_rule='unchanged conservative group_threshold, >=, full153 fit negatives, atomic float32 ties',
            development_or_ALCD_calibration=False))
        print('Phase92 frozen',name,'fitFPR',m['fit_macro_fpr'],'iterations',m['iterations'],flush=True)
    write('two_model_lock.json',dict(status='two_models_and_thresholds_frozen_before_evaluation',models=models,
        model_lock_sha256={n:p.old.digest(D/('model_'+n+'_lock.json')) for n in COLUMNS},
        fit_input_lock_sha256=p.old.digest(D/'fit_input_lock.json'),fit_costs=costs,
        prefit_verification_and_fit_seconds=time.monotonic()-started,fit_groups=153,
        new_logistic_fits=2,new_segmentation_training=0,new_network_forwards=0,new_downloads=0))


def independent_logit(x,m):
    # Separate implementation; same explicit layout preserves BLAS rounding.
    standardized=(np.asarray(x,np.float64)-np.array(m['mean']))/np.array(m['scale'])
    return (standardized.dot(np.array(m['theta'][:-1]))+m['theta'][-1]).astype(np.float32)


def row_metric(score,y,threshold,meta):
    r=binary_row(score,y,threshold,meta)
    r['ap']=p.old.ap(score,y==2)
    r['shadow_iou']=r['true_positive']/(r['positive']+r['false_positive']) if r['positive']+r['false_positive'] else None
    r['predicted_shadow_fraction']=(r['true_positive']+r['false_positive'])/len(y) if len(y) else None
    # Independent AP implementation and flat binary counts, no confusion helper.
    p.independent_ap(y,score,r['ap'])
    flags=np.asarray(score)>=threshold
    tp=int(np.sum(flags[y==2]));fp=int(np.sum(flags[y!=2]))
    assert tp==r['true_positive'] and fp==r['false_positive']
    return r


def summarize(rows,scores,ys,threshold):
    tp=sum(r['true_positive'] for r in rows);fp=sum(r['false_positive'] for r in rows)
    pos=sum(r['positive'] for r in rows);neg=sum(r['negative'] for r in rows)
    # Binary pooled metrics are derived directly; no synthetic Surface/Cloud predictions.
    score=np.concatenate(scores);y=np.concatenate(ys)
    pooled=row_metric(score,y,threshold,dict(product='pooled',group_id='pooled',tile='pooled'))
    assert pooled['true_positive']==tp and pooled['false_positive']==fp and pooled['positive']==pos and pooled['negative']==neg
    return dict(rows=rows,pooled=pooled,groups=len(rows),shadow_groups=sum(r['positive']>0 for r in rows),
        threshold=threshold,macro={k:interval([r[k] for r in rows if r[k] is not None]) for k in
          ['ap','recall','precision','fpr','shadow_iou','surface_fpr','cloud_fpr']},
        no_shadow_macro_fpr=interval([r['fpr'] for r in rows if not r['positive'] and r['fpr'] is not None]),
        groups_above_1percent=sum(r['fpr'] is not None and r['fpr']>.01 for r in rows),
        zero_recall_groups=[r['product'] for r in rows if r['recall']==0])


def compare(policies):
    result={}
    for a,b in [('Source_spectral','S3'),('Source_spectral','R6'),('L3','S3'),('L3','R6')]:
        pairs=[]
        for x,y in zip(policies[a]['rows'],policies[b]['rows']):
            assert x['group_id']==y['group_id']
            pairs.append(dict(product=x['product'],tile=x['tile'],group_id=x['group_id'],
                delta_ap_pp=(x['ap']-y['ap'])*100 if x['ap'] is not None and y['ap'] is not None else None,
                delta_recall_pp=(x['recall']-y['recall'])*100 if x['recall'] is not None and y['recall'] is not None else None,
                delta_fpr_pp=(x['fpr']-y['fpr'])*100 if x['fpr'] is not None and y['fpr'] is not None else None))
        eligible=[r for r in pairs if r['delta_ap_pp'] is not None];d=[r['delta_ap_pp'] for r in eligible]
        rr=[r for r in pairs if r['delta_recall_pp'] is not None]
        result[a+'_minus_'+b]=dict(primary=(a=='Source_spectral' and b=='S3'),pairs=pairs,
            delta_ap_pp=interval(d),improved=sum(v>1 for v in d),stable=sum(abs(v)<=1 for v in d),declined=sum(v< -1 for v in d),
            worst_ap=min(eligible,key=lambda r:r['delta_ap_pp']) if eligible else None,
            worst_recall=min(rr,key=lambda r:r['delta_recall_pp']) if rr else None,
            formal_confirmation=False)
    return result


def finish_domain(name,scores,ys,metas,thresholds):
    policies={};oracle={};receipts=[]
    for n in NAMES:
        rows=[row_metric(s,y,thresholds[n],dict(**m,policy=n)) for s,y,m in zip(scores[n],ys,metas)]
        policies[n]=summarize(rows,scores[n],ys,thresholds[n])
        point=scan(scores[n],ys,[.01])[0];cpu_validate(point,scores[n],ys)
        rows=[row_metric(s,y,point['threshold'],dict(**m,policy=n)) for s,y,m in zip(scores[n],ys,metas)]
        oracle[n]=summarize(rows,scores[n],ys,point['threshold'])
        receipts.append(dict(domain=name,policy=n,point=point))
        print('Phase92 summary',name,n,flush=True)
    return dict(policies=policies,comparisons=compare(policies),oracle_1percent=oracle,
        oracle_only=True,no_oracle_threshold_deployment=True),receipts


def check_reference(row,ref,apkey):
    assert row['ap']==ref[apkey]
    for key in ['true_positive','false_positive','positive','negative','recall','precision','fpr']:
        assert row[key]==ref[key],(row['product'],row['policy'],key,row[key],ref[key])
    cm=np.array(ref['confusion'])
    assert row['surface_false_positive']==int(cm[0,2]) and row['cloud_false_positive']==int(cm[1,2])


def evaluate():
    start=time.monotonic();assert (D/'two_model_lock.json').is_file() and not (D/'phase92_report.json').exists()
    p.old.ROOT=p.old.SHARED/'phase89_recovered_20261011'
    lock=p.old.read(D/'two_model_lock.json');fi=p.old.read(D/'fit_input_lock.json')
    assert p.old.digest(PROTOCOL)==fi['protocol_sha256']
    for n,h in fi['code_sha256'].items():assert p.old.digest(Path(__file__).with_name(n))==h,n
    assert p.old.digest(D/'fit_input_lock.json')==lock['fit_input_lock_sha256']
    for n,h in lock['model_lock_sha256'].items():assert p.old.digest(D/('model_'+n+'_lock.json'))==h
    base=fit67.base_verify();lock67=p.old.read(fit67.ROOT/'control_lock.json')
    assert p.old.digest(fit67.ROOT/'control_lock.json')==p.CONTROL_SHA
    assert p.old.digest(P90/'DELIVERY_SHA256.json')==P90_SHA
    for n,h in p.old.read(P90/'DELIVERY_SHA256.json').items():assert p.old.digest(P90/n)==h,n
    lock90=p.old.read(P90/'parameter_input_lock.json');p.old.locked()
    assert p.old.digest(p.old.ROOT/'recovery_lock.json')==lock90['phase89_recovery_lock_sha256']
    assert p.old.digest(p.old.ROOT/'delivery/DELIVERY_SHA256.json')==p.P89_MANIFEST_SHA
    models={**lock['models'],'Source_spectral':lock67['models']['Source_spectral'],
            'L3':base['models']['msre_source_spectral']}
    ref67=p.old.read(fit67.ROOT/'delivery/phase67_report.json');ref90=p.old.read(P90/'phase90_report.json')
    assert p.old.digest(fit67.ROOT/'delivery/phase67_report.json')==p.P67_REPORT_SHA
    dev=recovery.records('development_val');sources=[];cache_index={}
    for name in ['source','msre']:
        path=recovery.OLD/'development_val'/('index_'+name+'.json');cache=p.old.read(path)
        cache_index[name]=p.old.digest(path);sources.append({r['product']:r for r in cache['rows']})
        assert set(sources[-1])==set(dev) and cache['manifest_sha256']==p.old.digest(recovery.PREP/'manifest.json')
    assert cache_index==ref67['development_cache_index_sha256']
    reproduction=[];input_seals=[];cpu=[];all_results={};scan_receipts=[];costs={}
    scores={n:[] for n in NAMES};ys=[];metas=[];begin=time.monotonic()
    with zipfile.ZipFile(recovery.DATA/'subscenes.zip') as z:
        for product,r in dev.items():
            recovery.prepare_verify(r);truth=np.load(recovery.PREP/r['mask_path']);valid=truth!=255;y=truth[valid]
            b=z.read('subscenes/'+product+'.npy');assert p.old.hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            ss=recovery.load(recovery.OLD,sources[0][product],'score');ms=recovery.load(recovery.OLD,sources[1][product],'score')
            for prior in [sources[0][product],sources[1][product]]:
                assert p.old.digest(recovery.OLD/prior['truth_path'])==prior['truth_sha256']
                assert np.array_equal(np.load(recovery.OLD/prior['truth_path']),truth)
            full=p.combined(recovery.features(raw,ms,ss,valid),raw[...,[3,2,1]][valid])
            meta=dict(product=product,group_id=r['group_id'],tile=r.get('tile',product.split('_')[-2]))
            seals={}
            for n in NAMES:
                xx=raw_selected(full,n) if n in COLUMNS else p.selected(full,n) if n=='Source_spectral' else full[:,:5]
                score=predict_logit(xx,models[n]);independent=independent_logit(xx,models[n]);assert np.array_equal(score,independent)
                metric=row_metric(score,y,models[n]['threshold'],dict(**meta,policy=n))
                if n in ['Source_spectral','L3']:
                    ref=next(t for t in ref67['policies'][n]['rows'] if t['product']==product)
                    check_reference(metric,dict(ref,ap=ref['ranking']['ap']),'ap')
                    if n=='L3':assert np.array_equal(score,np.load(fit67.RECOVERY/'development_scores'/(product+'_msre_source_spectral.npy')))
                    reproduction.append(dict(domain='development',product=product,policy=n,AP_confusion_exact=True))
                scores[n].append(score);seals[n]=sha_array(score)
            ys.append(y);metas.append(meta);input_seals.append(dict(domain='development',**meta,score_sha256=seals,
                image_member_sha256=r['member_sha256']['image'],truth_sha256=sha_array(y)))
            print('Phase92 development',len(ys),32,flush=True)
    costs['development_scoring_and_reproduction_seconds']=time.monotonic()-begin
    all_results['development'],sc=finish_domain('development',scores,ys,metas,{n:models[n]['threshold'] for n in NAMES});scan_receipts+=sc
    del scores,ys
    prepared=p.old.read(p.old.ROOT/'prepared/index.json')['rows'];assert prepared==lock90['prepared']['rows']
    scores={v:{n:[] for n in NAMES} for v,_,_ in p.VERSIONS};ys={v:[] for v,_,_ in p.VERSIONS};metas={v:[] for v,_,_ in p.VERSIONS}
    begin=time.monotonic()
    for i,row in enumerate(prepared):
        prob=p.old.probabilities(row['product']);new={n:np.empty(row['shape60'],np.float64) for n in NAMES if n!='L3'}
        coverage=np.zeros(row['shape60'],np.uint8)
        for y,x,h,w,ss,ms,spec in p.blocks(row):
            for n in new:
                if n=='Source_spectral':xx=p.inputs(ms[...,2],ss[...,2],spec,n)
                else:xx=np.asfortranarray(spec[...,[3,4,5] if n=='S3' else [2,1,0,3,4,5]].reshape(-1,len(BANDS[n])))
                logits=predict_logit(xx,models[n]);other=independent_logit(xx,models[n]);assert np.array_equal(logits,other)
                value=p.area(p.sigmoid(logits).reshape(h*3,w*3),3)
                flat=np.exp(-np.logaddexp(0.,-other.astype(np.float64))).reshape(h*3,w*3)
                independent=sum(flat[a::3,b::3] for a in range(3) for b in range(3))/9
                error=float(np.max(np.abs(value-independent)));assert error<=2e-15
                assert np.array_equal(value>=p.sigmoid(models[n]['threshold']),independent>=p.sigmoid(models[n]['threshold']))
                new[n][y:y+h,x:x+w]=value
                cpu.append(dict(product=row['product'],policy=n,y60=y,x60=x,max_mean_error=error,decision_exact=True))
            coverage[y:y+h,x:x+w]+=1
        assert (coverage==1).all();prob.update(new)
        for version,tkey,mkey in p.VERSIONS:
            truth=p.old.array(row,tkey);mask=p.old.array(row,mkey)
            if not mask.any():continue
            yy=truth[mask];ys[version].append(yy)
            meta={k:row[k] for k in ['product','group_id','tile']};metas[version].append(meta)
            seals={}
            for n in NAMES:
                score=prob[n][mask];scores[version][n].append(score);seals[n]=sha_array(score)
                if n in ['Source_spectral','L3']:
                    metric=row_metric(score,yy,p.sigmoid(models[n]['threshold']),dict(**meta,policy=n))
                    ref=next(t for t in ref90['versions'][version]['policies'][n]['rows'] if t['product']==row['product'])
                    check_reference(metric,ref,'ap')
                    reproduction.append(dict(domain=version,product=row['product'],policy=n,AP_confusion_exact=True))
            input_seals.append(dict(domain=version,**meta,score_sha256=seals,truth_sha256=sha_array(yy)))
        print('Phase92 ALCD',i+1,9,flush=True)
    costs['ALCD_scoring_and_reproduction_seconds']=time.monotonic()-begin
    for v,_,_ in p.VERSIONS:
        all_results[v],sc=finish_domain(v,scores[v],ys[v],metas[v],{n:float(p.sigmoid(models[n]['threshold'])) for n in NAMES});scan_receipts+=sc
    assert len(reproduction)==106 and len(cpu)==972
    write('baseline_reproduction.json',dict(rows=reproduction,expected_rows=106,all_exact=True))
    write('input_code_score_lock.json',dict(fit_model_lock_sha256=p.old.digest(D/'two_model_lock.json'),
        phase90_manifest_sha256=P90_SHA,phase90_chain=lock90,development_cache_index_sha256=cache_index,
        source_spectral_model_sha256=p.old.digest(fit67.ROOT/'control_lock.json'),
        rows=input_seals,new_network_forwards=0,no_large_scores_saved=True,
        AP_precision='unchanged float32 tied noninterpolated AP',ALCD='float32 logit20 then sigmoid float64 and mean9 at60; native reference/masks unchanged',
        binary_only=True,no_Surface_Cloud_predictions_from_S3_R6=True))
    write('oracle_threshold_receipts.json',dict(oracle_only=True,rows=scan_receipts))
    write('CPU_VERIFIED.json',dict(independent_score_bin_AP=True,independent_binary_counts=True,
        logit_independent_array_equal=True,block_checks=cpu,reference_reproduction_rows=len(reproduction),
        scan_direct_count_checked=len(scan_receipts),not_independent_data_confirmation=True))
    report=dict(status='completed_two_raw_spectral_baselines',created_utc=p.old.now(),domains=all_results,
        primary_comparison='Source_spectral_minus_S3',secondary_comparison='Source_spectral_minus_R6',
        bootstrap_seed=SEED,resamples=10000,AP_boundary_pp=1,domains_not_pooled_for_confirmation=True,
        new_logistic_fits=2,new_segmentation_training=0,new_network_forwards=0,new_downloads=0,
        exploratory_only=True,no_new_deployment_winner_or_threshold=True,original_failed_gates_unchanged=True,
        costs=dict(**costs,evaluation_seconds=time.monotonic()-start,fit=lock['fit_costs']))
    write('phase92_report.json',report);render(report)
    write('DELIVERY_SHA256.json',{f.name:p.old.digest(f) for f in sorted(D.iterdir()) if f.is_file()})
    p.old.write(ROOT/'COMPLETED.json',dict(created_utc=p.old.now(),evaluation_seconds=time.monotonic()-start,
        delivery_manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json'),new_network_forwards=0,new_downloads=0))
    print('Phase92 complete',time.monotonic()-start,flush=True)


def render(report):
    totals=[];groups=[];pairs=[];oracles=[]
    for domain,v in report['domains'].items():
        for n,q in v['policies'].items():
            r=dict(domain=domain,policy=n,threshold=q['threshold'],groups=q['groups'],shadow_groups=q['shadow_groups'])
            for k,t in q['macro'].items():
                r['macro_'+k]=t['mean'];r['macro_'+k+'_ci95']=json.dumps(t['ci95']);r['macro_'+k+'_upper95']=t['upper95_one_sided']
            r.update({'pooled_'+k:q['pooled'][k] for k in ['ap','recall','precision','fpr','shadow_iou','surface_fpr','cloud_fpr']})
            r.update(groups_above_1percent=q['groups_above_1percent'],zero_recall_groups=len(q['zero_recall_groups']),no_shadow_fpr=q['no_shadow_macro_fpr']['mean'])
            totals.append(r);groups.extend(dict(domain=domain,**x) for x in q['rows'])
        for comparison,q in v['comparisons'].items():pairs.extend(dict(domain=domain,comparison=comparison,primary=q['primary'],**x) for x in q['pairs'])
        for n,q in v['oracle_1percent'].items():
            oracles.append(dict(domain=domain,policy=n,threshold=q['threshold'],oracle_only=True,
                macro_recall=q['macro']['recall']['mean'],macro_fpr=q['macro']['fpr']['mean'],
                pooled_recall=q['pooled']['recall'],pooled_fpr=q['pooled']['fpr'],groups_above_1percent=q['groups_above_1percent']))
    for n,rows in [('four_strategy_totals.csv',totals),('per_group.csv',groups),('paired_changes.csv',pairs),('oracle_1percent.csv',oracles)]:p.old.csvwrite(D/n,rows)
    lines=['# Phase92 data results and interpretation','','## 1. Numerical results','',
        'Exploratory only. All intervals resample related groups equally (10000, seed92012); domains are separate.',
        'S3/R6 are binary Shadow baselines, not three-class systems. Thresholds frozen on all153 fit representatives.',
        '', '| Domain | Strategy | Macro AP % | Macro recall % | Macro FPR % | FPR upper95 % | Pooled AP % |',
        '|---|---|---:|---:|---:|---:|---:|']
    f=lambda x:'NA' if x is None else f'{x*100:.4f}'
    for d,v in report['domains'].items():
        for n,q in v['policies'].items():lines.append('| '+' | '.join([d,n,f(q['macro']['ap']['mean']),f(q['macro']['recall']['mean']),f(q['macro']['fpr']['mean']),f(q['macro']['fpr']['upper95_one_sided']),f(q['pooled']['ap'])])+' |')
    lines+=['','Primary Source_spectral minus S3 macro AP (percentage points):']
    for d,v in report['domains'].items():
        q=v['comparisons']['Source_spectral_minus_S3'];r=q['delta_ap_pp']
        lines.append(f"- {d}: {r['mean']:.4f}, CI95 {r['ci95']}; improved/stable/declined {q['improved']}/{q['stable']}/{q['declined']}.")
    lines+=['','## 2. Interpretation limits','','Read each domain and both common-support reference versions separately. Cross-zero intervals are not equivalence.','Existing Source score plus the same three spectral bands tests practical incremental value; comparison against R6 is not capacity matched.',
        'No winner selected for deployment, no old failure conclusion changed. Oracle table uses evaluation labels and cannot supply deployed thresholds.',
        'No new network training, forward pass or download. Independent CPU recomputation is not independent-data confirmation.',
        'Phase89 recovery provenance limitation remains:54 reused probabilities received hashes after the disk failure, not before it.']
    (D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','evaluate']);args=parser.parse_args()
    fit() if args.action=='fit' else evaluate()
