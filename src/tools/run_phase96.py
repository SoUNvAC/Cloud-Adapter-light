"""Phase96 fixed tail diagnosis: no fits, thresholds, network forwards, or ALCD reads."""
import hashlib,importlib.metadata,io,json,pickle,time,zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_phase94 as r
from phase95_math import support,same_scores
from phase96_math import BANDS,describe,tail,extremes,bin_masks,self_test
from phase96_preview import create as preview
from phase93_math import independent_probability
from prepare_phase65a_explore import convert
from verify_phase95_delivery import verify as verify95

p=r.p;q=r.q;recovery=r.recovery;fit67=r.fit67
ROOT=p.old.SHARED/'phase96_20261012';D=ROOT/'delivery';OLD=p.old.SHARED/'phase94_20261012/delivery';P95=p.old.SHARED/'phase95_20261012/delivery'
PROTOCOL=p.old.REPO/'src/research_plans/PHASE96_TAIL_FAILURE_DIAGNOSIS_20261012.md'
TARGET='S2B_MSIL1C_20180420T103019_N0206_R108_T31TGF_20180420T114307'
TYPES=('T20_full','T20_same','T60_same');NAMES=r.NAMES

def write(n,v):p.old.write(D/n,v)
def sha(a):return q.sha_array(a)
def prob(n,a):return np.asarray(a,np.float64) if n=='G7_100' else r.sigmoid(a)
def meta(rec):return dict(product=rec['product'],group_id=rec['group_id'],tile=rec.get('tile',rec['product'].split('_')[-2]))


def setup():
    assert not ROOT.exists(),'Never repeat or overwrite Phase96'
    assert p.old.digest(P95/'DELIVERY_SHA256.json')=='5fe95b0fe496914a2e1861942eaeb76e4ad38b8ebf0134d6e0ffaf51a2e2fd68'
    acceptance=verify95(P95);lock=p.old.read(P95/'input_lock.json');sl=p.old.read(OLD/'split_input_lock.json')
    assert lock['indices_before']==r.indices_audit() and lock['split_lock_sha256']==p.old.digest(OLD/'split_input_lock.json')
    for n,h in lock['source_files_sha256'].items():assert p.old.digest(OLD/n)==h,n
    assert {n:importlib.metadata.version(n) for n in sl['dependency_lock']['versions']}==sl['dependency_lock']['versions']
    models={}
    for n,m in lock['model_lock']['models'].items():
        assert p.old.digest(OLD/m['file'])==m['sha256']
        if n=='G7_100':
            with (OLD/m['file']).open('rb') as f:models[n]=pickle.load(f)
        else:models[n]=p.old.read(OLD/m['file'])
    base=fit67.base_verify();audit=p.old.read(fit67.ROOT/'fit_input_audit.json')
    assert p.old.digest(fit67.ROOT/'fit_input_audit.json')==p.old.read(r.prior.OLD/'fit_input_lock.json')['original_fit_input_audit_sha256']
    ROOT.mkdir();D.mkdir()
    code=['run_phase96.py','phase96_math.py','phase96_preview.py','verify_phase96_delivery.py','run_phase96.sh','run_phase94.py','phase95_math.py','phase93_math.py']
    write('input_lock.json',dict(created_utc=p.old.now(),protocol_sha256=p.old.digest(PROTOCOL),phase95_acceptance=acceptance,
        phase95_manifest_sha256=p.old.digest(P95/'DELIVERY_SHA256.json'),phase94_manifest_sha256=p.old.digest(OLD/'DELIVERY_SHA256.json'),
        models=lock['model_lock']['models'],thresholds=p.old.read(P95/'four_new_threshold_lock.json')['models'],
        fit=sl['fit'],calibration=sl['calibration'],development=list(recovery.records('development_val').values()),
        indices_before=r.indices_audit(),base=base,code_sha256={n:p.old.digest(Path(__file__).with_name(n)) for n in code},
        primary_threshold='Phase94 T_cal = T20_full',sensitivities=['T20_same','T60_same'],
        original20_rule='T20 thresholds on native float32 score/logit; T60 threshold on float64 probability',
        synthetic60_rule='same Phase95 top-left1020, all9valid+sameclass, float64 mean9 probability',
        new_fits=0,new_thresholds=0,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,
        marginal_rule='strict below train P1 / above train P99; canonical7 input columns; only original100 fixed samples',
        source_bins=['[0,.1)','[.1,.5)','[.5,1]'],no_joint_OOD_or_causal_claim=True))
    return models,sl,{a['product']:a for a in audit['rows']},p.old.read(P95/'four_new_threshold_lock.json')['models']


def dev_load(rec,sources):
    recovery.prepare_verify(rec);truth=np.load(recovery.PREP/rec['mask_path']);valid=truth!=255
    with zipfile.ZipFile(recovery.DATA/'subscenes.zip') as z:b=z.read('subscenes/'+rec['product']+'.npy')
    assert hashlib.sha256(b).hexdigest()==rec['member_sha256']['image'];raw=np.load(io.BytesIO(b),allow_pickle=False)
    ss=recovery.load(recovery.OLD,sources[0][rec['product']],'score');ms=recovery.load(recovery.OLD,sources[1][rec['product']],'score')
    assert ss.shape==truth.shape and np.isfinite(ss).all() and ((ss>=0)&(ss<=1)).all()
    full=p.combined(recovery.features(raw,ms,ss,valid),raw[...,[3,2,1]][valid])
    assert np.array_equal(full[:,1],r.logit_score(ss[valid]).astype(np.float32))
    return truth,valid,raw,ss,full


def scores(models,full,truth,checks,domain,product):
    keep,y60,retention=support(truth);out={};cpu=[]
    for n in NAMES:
        s=r.predict(models,n,r.columns(full,n),checks,dict(domain=domain,product=product))
        fine,mean,ind,err=same_scores(s,truth,keep,lambda a:prob(n,a))
        out[n]=dict(original20=s,synthetic60=mean)
        cpu.append(dict(model=n,mean9_error=err,area20_same=len(fine),area60=len(mean),finite_scores=True))
        checks[-1].update(mean9_error=err,same_area_9to1=True)
    return out,keep,y60,retention,cpu


def threshold_score(n,t,scale,ss,thresholds):
    v=thresholds[n][t]
    if scale=='synthetic60':return ss,v['probability_threshold']
    if t=='T60_same':return prob(n,ss),v['probability_threshold']
    return ss,v['threshold']


def metric_rows(out,truth,y60,rec,cohort,thresholds):
    result=[]
    for scale,y in [('original20',truth[truth!=255]),('synthetic60',y60)]:
        for n in NAMES:
            for t in TYPES:
                s,th=threshold_score(n,t,scale,out[n][scale],thresholds)
                a=q.binary_row(s,y,th,dict(**meta(rec),cohort=cohort,scale=scale,model=n,threshold_name=t))
                # AP is not an endpoint in Phase96; no-positive target is explicitly undefined.
                a['ap']=None;a['ap_status']='undefined_no_reference_shadow' if not a['positive'] else 'not_computed_in_tail_diagnosis'
                assert a['false_positive']==int(np.sum((s>=th)[y!=2]));result.append(a)
    return result


def chain_target(models,rec,sources,thresholds,checks):
    start=time.monotonic();truth,valid,raw,source,full=dev_load(rec,sources)
    with zipfile.ZipFile(recovery.DATA/'masks.zip') as z:b=z.read('masks/'+TARGET+'.npy')
    assert hashlib.sha256(b).hexdigest()==rec['member_sha256']['mask']
    official=np.load(io.BytesIO(b),allow_pickle=False);rgb,tt,iv,_=convert(raw,official)
    assert np.array_equal(tt,truth) and np.array_equal(rgb,np.load(recovery.PREP/rec['rgb_path'])) and np.array_equal(iv,np.load(recovery.PREP/rec['image_valid_path']))
    assert np.array_equal(full[:,2:5],raw[...,[7,11,12]][valid].astype(np.float32))
    assert np.array_equal(full[:,5:8],raw[...,[3,2,1]][valid].astype(np.float32))
    out,keep,y60,retention,cpu=scores(models,full,truth,checks,'target_chain',TARGET)
    assert len(y60)==115600 and np.array_equal(np.unique(y60),[0]) and np.array_equal(np.unique(truth[valid]),[0])
    rows=metric_rows(out,truth,y60,rec,'development',thresholds)
    old94=p.old.read(OLD/'input_score_lock.json')['rows']; old95=p.old.read(P95/'input_score_lock.json')['rows']
    oldreport=p.old.read(P95/'phase95_report.json')['domains']['development_same60']['policies']
    independent=[];fp20={};fp60={}
    for n in NAMES:
        x=r.columns(full,n)
        if n=='G7_100':ind=independent_probability(models[n],x)
        else:
            # Independent explicit coefficient accumulation on every valid pixel.
            z=(x-np.asarray(models[n]['mean']))/np.asarray(models[n]['scale']);ind=np.full(len(x),models[n]['theta'][-1],np.float64)
            for j,c in enumerate(models[n]['theta'][:-1]):ind+=z[:,j]*c
            ind=ind.astype(np.float32)
        err=float(np.max(np.abs(ind-out[n]['original20'])));assert err<=(6e-8 if n=='G7_100' else 1e-4)
        _,ind60,_,_=same_scores(ind,truth,keep,lambda a:prob(n,a))
        for t in TYPES:
            for scale,indvalue in [('original20',ind),('synthetic60',ind60)]:
                original,th=threshold_score(n,t,scale,out[n][scale],thresholds);other,_=threshold_score(n,t,scale,indvalue,thresholds)
                assert np.array_equal(original>=th,other>=th),(n,t,scale,'independent decision mismatch')
                name=n+'_'+t;mask=original>=th
                if scale=='original20':a=np.zeros(truth.shape,bool);a[valid]=mask;fp20[name]=a
                else:a=np.zeros(keep.shape,bool);a[keep]=mask;fp60[name]=a
            rr=next(a for a in rows if a['model']==n and a['threshold_name']==t and a['scale']=='synthetic60')
            expected=next(a for a in oldreport[n+'_'+t]['rows'] if a['product']==TARGET)
            for k in ['negative','surface','cloud','positive','false_positive','fpr','recall']:assert rr[k]==expected[k]
        for old,domain,scale in [(old94,'development','original20'),(old95,'development_same60','synthetic60')]:
            row=next(a for a in old if a['domain']==domain and a['product']==TARGET)
            assert sha(out[n][scale])==row['score_sha256'][n]
        independent.append(dict(model=n,pixels=len(ind),max_score_error=err,all_frozen_decisions_exact=True,full_independent_scoring=True))
    rates=[next(a['fpr'] for a in rows if a['model']=='G7_100' and a['scale']=='synthetic60' and a['threshold_name']==t) for t in TYPES]
    np.testing.assert_allclose(rates,[.4337629757785467,.5696280276816609,.7063235294117646],rtol=0,atol=1e-14)
    native=recovery.load(recovery.OLD,sources[0][TARGET],'native_score');source_pred=recovery.load(recovery.OLD,sources[0][TARGET],'pred')
    stats=[]
    for j,name in enumerate(BANDS):stats.append(dict(name=name,scope='valid20',**describe(r.columns(full,'G7_100')[:,j])))
    for name,a in [('Source_probability',source[valid]),('Source_native_aggregate',native[valid]),('G7_probability20',prob('G7_100',out['G7_100']['original20'])),('SourceS_logit20',out['SourceS_100']['original20']),('SourceS_probability20',prob('SourceS_100',out['SourceS_100']['original20'])),('G7_probability60',out['G7_100']['synthetic60']),('SourceS_probability60',out['SourceS_100']['synthetic60'])]:stats.append(dict(name=name,scope='valid20' if len(a)==valid.sum() else 'same60',**describe(a)))
    sourceS_x=r.columns(full,'SourceS_100');standard=(sourceS_x-np.array(models['SourceS_100']['mean']))/np.array(models['SourceS_100']['scale'])
    for j,name in enumerate(['Source_logit','B08','B11','B12']):stats.append(dict(name='SourceS_standardized_'+name,scope='valid20',**describe(standard[:,j])))
    g7=np.zeros(truth.shape);g7[valid]=prob('G7_100',out['G7_100']['original20']);sp=np.zeros(truth.shape);sp[valid]=prob('SourceS_100',out['SourceS_100']['original20'])
    previews=preview(D/'T31TGF_preview',raw,source,g7,sp,truth,fp20,fp60)
    chain=dict(status='passed',product=TARGET,group_id=rec['group_id'],official_image_and_mask_SHA_and_conversion_exact=True,
        band_order=list(BANDS),TOA='official float32 physical reflectance; no rescale/clipping for shallow inputs; RGB network preparation historically TOA*255',
        source_definition=p.old.read(recovery.OLD/'development_val/index_source.json')['score_definition'],source_clip=[1e-6,1-1e-6],
        source_clip_below_count=int((source[valid]<1e-6).sum()),source_clip_above_count=int((source[valid]>1-1e-6).sum()),
        source_argmax_counts=np.bincount(source_pred[valid],minlength=3).tolist(),
        input_stats=stats,metrics=rows,independent=independent,mean9=cpu,preview=previews,
        reference20_counts=np.bincount(truth[valid],minlength=3).tolist(),reference60_counts=np.bincount(y60,minlength=3).tolist(),
        full20_valid=int(valid.sum()),same60_valid=len(y60),same60_mask_sha256=sha(keep),three_G7_FPR_reproduced=rates,
        no_numerical_or_identity_error_detected=True,no_manual_confirmation_of_no_shadow=True,seconds=time.monotonic()-start)
    write('T31TGF_chain_acceptance.json',chain);print('Phase96 T31TGF chain passed',flush=True)
    return (truth,valid,raw,source,full,out,keep,y60,retention),chain


def training_reference(sl,audit):
    samples=[];receipts=[]
    for i,row in enumerate(sl['fit']):
        product=row['product'];x=fit67.full_fit_features(audit[product])
        seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
        idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
        assert sha(idx)==row['sample_indices_sha256'] and len(idx)==row['sample_count']
        samples.append(x[idx]);receipts.append(dict(product=product,group_id=row['group_id'],sample_count=len(idx),sample_indices_sha256=sha(idx)))
        if (i+1)%20==0:print('Phase96 reference inputs',i+1,100,flush=True)
    x=np.concatenate(samples).astype(np.float64);assert sha(x)==p.old.read(OLD/'fit_input_lock.json')['sample_features_sha256']
    features=r.columns(x,'G7_100');refs={};errors={}
    for j,n in enumerate(BANDS):
        refs[n]=describe(features[:,j]);a=np.sort(features[:,j]);positions=(len(a)-1)*np.array([.01,.05,.5,.95,.99]);low=np.floor(positions).astype(int);high=np.ceil(positions).astype(int)
        independent=a[low]+(a[high]-a[low])*(positions-low);original=np.array([refs[n]['p'+str(k)] for k in [1,5,50,95,99]])
        np.testing.assert_allclose(original,independent,rtol=0,atol=1e-12);errors[n]=float(np.max(np.abs(original-independent)))
    result=dict(fit_groups=100,rows=receipts,sample_count=len(x),sample_features_sha256=sha(x),features=refs,
        quantile_method='numpy linear empirical quantiles over unchanged original fixed sample; no development/calibration boundary selection',independent_quantile_errors=errors)
    write('training_reference_lock.json',result);return refs


def finish(rows):
    summaries=[]
    for cohort in ['calibration','development']:
        for scale in ['original20','synthetic60']:
            for n in NAMES:
                for t in TYPES:
                    selected=[a for a in rows if a['cohort']==cohort and a['scale']==scale and a['model']==n and a['threshold_name']==t]
                    for stratum,rs in [('all',selected),('with_reference_shadow',[a for a in selected if a['positive']]),('no_reference_shadow',[a for a in selected if not a['positive']])]:
                        for key in ['fpr','surface_fpr','cloud_fpr']:summaries.append(dict(cohort=cohort,scale=scale,model=n,threshold_name=t,stratum=stratum,rate=key,**tail(rs,key)))
    return summaries


def main():
    started=time.monotonic();tests=self_test();models,sl,audit,thresholds=setup();checks=[]
    dev=recovery.records('development_val');assert TARGET in dev
    sources=[{a['product']:a for a in p.old.read(recovery.OLD/'development_val'/('index_'+n+'.json'))['rows']} for n in ['source','msre']]
    target,chain=chain_target(models,dev[TARGET],sources,thresholds,checks)
    references=training_reference(sl,audit);rows=[];retention=[];seals=[];strata=[];bins=[]
    original94=p.old.read(OLD/'phase94_report.json')['domains']['development']['policies'];original95=p.old.read(P95/'phase95_report.json')['domains']['development_same60']['policies']
    old94seals={a['product']:a for a in p.old.read(OLD/'input_score_lock.json')['rows'] if a['domain']=='development'}
    old95seals={a['product']:a for a in p.old.read(P95/'input_score_lock.json')['rows'] if a['domain']=='development_same60'}
    calrecords=recovery.records('fit_representatives');calreceipts={a['product']:a for a in p.old.read(P95/'calibration_support_lock.json')['rows']}
    for cohort,selected in [('calibration',sl['calibration']),('development',list(dev.values()))]:
        for i,row in enumerate(selected):
            product=row['product'];rec=calrecords[product] if cohort=='calibration' else dev[product]
            if cohort=='calibration':
                recovery.prepare_verify(rec);truth=np.load(recovery.PREP/rec['mask_path']);valid=truth!=255;full=fit67.full_fit_features(audit[product])
                out,keep,y60,counts,_=scores(models,full,truth,checks,cohort,product)
                assert sha(keep)==calreceipts[product]['blocks_sha256'] and sha(y60)==calreceipts[product]['truth60_sha256']
                for n in NAMES:assert sha(out[n]['original20'])==calreceipts[product]['models'][n]['full_score_sha256']
            else:
                if product==TARGET:truth,valid,raw,source,full,out,keep,y60,counts=target
                else:
                    truth,valid,raw,source,full=dev_load(rec,sources);out,keep,y60,counts,_=scores(models,full,truth,checks,cohort,product)
                assert sha(truth[valid])==old94seals[product]['truth_sha256'] and sha(y60)==old95seals[product]['truth_sha256']
                for n in NAMES:
                    assert sha(out[n]['original20'])==old94seals[product]['score_sha256'][n]
                    assert sha(out[n]['synthetic60'])==old95seals[product]['score_sha256'][n]
                x=r.columns(full,'G7_100');y=truth[valid];negative=y!=2
                gscore,gth=threshold_score('G7_100','T20_full','original20',out['G7_100']['original20'],thresholds);fp=(gscore>=gth)&negative;tn=(gscore<gth)&negative
                for subset,mask in [('all_valid',np.ones(len(y),bool)),('nonshadow_G7_false_positive',fp),('nonshadow_G7_correct_rejection',tn)]:
                    for j,n in enumerate(BANDS):strata.append(dict(**meta(rec),subset=subset,feature=n,**extremes(x[mask,j],references[n])))
                for label,bmask in zip(['[0,.1)','[.1,.5)','[.5,1]'],bin_masks(source[valid])):
                    mask=bmask&negative;a=dict(**meta(rec),bin=label,negative_count=int(mask.sum()))
                    for n in NAMES:
                        s,th=threshold_score(n,'T20_full','original20',out[n]['original20'],thresholds);count=int(((s>=th)&mask).sum());a[n+'_false_positive']=count;a[n+'_fpr']=count/a['negative_count'] if a['negative_count'] else None
                    bins.append(a)
            rr=metric_rows(out,truth,y60,rec,cohort,thresholds)
            for a in rr:
                n=a['model'];t=a['threshold_name'];scale=a['scale']
                if cohort=='development':
                    old=original94[n+'_T_cal'] if scale=='original20' else original95[n+'_'+t]
                    if scale=='synthetic60' or t=='T20_full':
                        oldrow=next(v for v in old['rows'] if v['product']==product)
                        for k in ['positive','negative','false_positive','surface_false_positive','cloud_false_positive','fpr']:assert a[k]==oldrow[k]
                elif scale=='original20' and t=='T20_full':
                    oldrow=next(v for v in thresholds[n][t]['rows'] if v['product']==product)
                    assert a['false_positive']==oldrow['false_positive'] and a['negative']==oldrow['negative_count']
                rows.append(a)
            retention.extend(dict(cohort=cohort,**meta(rec),**c) for c in counts)
            seals.append(dict(cohort=cohort,**meta(rec),truth20_sha256=sha(truth[valid]),truth60_sha256=sha(y60),same60_mask_sha256=sha(keep),scores={n:{s:sha(v) for s,v in out[n].items()} for n in NAMES}))
            print('Phase96 diagnostic',cohort,i+1,len(selected),flush=True)
    assert sl['indices_before']==r.indices_audit()
    summaries=finish(rows);write('tail_report.json',dict(status='complete',summaries=summaries,rows=rows,new_fits=0,new_thresholds=0,new_network_forwards=0,new_downloads=0,new_ALCD_pixel_reads=0,
        primary='T20_full',calibration_is_in_sample_for_threshold=True,no_removal_based_pass=True,no_manual_semantic_or_causal_claim=True))
    write('input_score_lock.json',dict(rows=seals,indices_after=r.indices_audit(),training_reference_lock_sha256=p.old.digest(D/'training_reference_lock.json'),input_lock_sha256=p.old.digest(D/'input_lock.json')))
    write('CPU_VERIFIED.json',dict(tests=tests,rows=checks,target_independent=chain['independent'],all_original_development_Tcal_and_Phase95_same60_counts_exact=True,
        all_calibration_Tcal_counts_exact=True,training_sample_features_SHA_exact=True,no_large_arrays_saved=True))
    write('cost.json',dict(seconds=time.monotonic()-started,process_peak_rss_bytes=r.rss(),peak_scope='single process cumulative full target independent scoring/reference verification/diagnosis',cpu_threads_max=4,new_fits=0,new_thresholds=0,new_network_forwards=0,new_downloads=0))
    for name,rs in [('per_group_tail.csv',rows),('tail_totals.csv',summaries),('fixed_input_strata.csv',strata),('source_probability_bins.csv',bins),('support_retention.csv',retention),('T31TGF_input_quantiles.csv',chain['input_stats'])]:p.old.csvwrite(D/name,rs)
    write('fixed_strata.json',dict(input_strata=strata,source_bins=bins))
    (D/'README.md').write_text('# Phase96：尾部误报诊断\n\n仅诊断；零拟合、零阈值、零网络前向、零下载。\n\n## 1．数据结果\n\nT31TGF链路通过；所有115600个合成60m块参考为Surface。三G7误报率精确复现43.3763%/56.9628%/70.6324%。详见tail_totals.csv、固定分层表和T31TGF_preview/index.html。\n\n## 2．结论探讨\n\n链路通过不代表参考语义经人工确认；分箱与边际越界仅关联，不证明联合OOD或原因。完整均值始终为主；不删景后判通过。校准为选阈值数据，不是独立泛化。人工作业B未完成及54块恢复缓存缺故障前SHA限制保留。\n',encoding='utf-8')
    write('DELIVERY_SHA256.json',{f.relative_to(D).as_posix():p.old.digest(f) for f in sorted(D.rglob('*')) if f.is_file()})
    from verify_phase96_delivery import verify
    accepted=verify(D);p.old.write(ROOT/'COMPLETED.json',dict(created_utc=p.old.now(),acceptance=accepted,manifest_sha256=p.old.digest(D/'DELIVERY_SHA256.json')))
    print('Phase96 complete',flush=True)


if __name__=='__main__':
    try:
        with threadpool_limits(limits=4):main()
    except Exception as error:
        if ROOT.exists() and not (ROOT/'FAILURE.json').exists():p.old.write(ROOT/'FAILURE.json',dict(created_utc=p.old.now(),error=repr(error),preserve_all_outputs=True,no_silent_repair=True))
        raise
