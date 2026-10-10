"""One fixed Phase68 fit, lock four policies, reproduce A, then development only."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from prepare_phase65a_explore import digest,SPLIT_SHA,SOURCE_SHA
from run_phase65d_recovery import DATA,PREP,OLD,REPO,ADAPTED_SHA,records,prepare_verify,load,features,brief
from run_phase67 import base_verify,old_features,RECOVERY,RECOVERY_SHA,read,write
from phase65d_recovery_math import predict_logit,group_threshold
from phase65d_metrics import curve,confusion,measures,corrected
from phase68_math import conditional_weights,fit_fixed_scaler,dual_threshold,interval,continue_gate

ROOT=DATA/'phase68_20261010'
PROTOCOL=REPO/'src/research_plans/PHASE68_20261010.md'
EXECUTION=REPO/'src/research_plans/PHASE68_EXECUTION_20261010.md'
CONTROL=DATA/'phase67_20261010/control_lock.json'
CONTROL_SHA='c2a83964073918015460e0ad89a9857b60b6a764e0f8562652472b224915f5ad'
CODE_NAMES=['run_phase68.py','phase68_math.py','test_phase68.py','launch_phase68.sh',
    'run_phase67.py','run_phase65d_recovery.py','phase65d_recovery_math.py',
    'phase65d_metrics.py','prepare_phase65a_explore.py','phase67_controls.py']


def code_hashes():return {n:digest(Path(__file__).with_name(n)) for n in CODE_NAMES}


def verify():
    base=base_verify()
    assert digest(CONTROL)==CONTROL_SHA
    assert read(CONTROL)['recovery_lock_sha256']==RECOVERY_SHA
    for path,sha in [(REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',SOURCE_SHA),
        (REPO/'src/work_dirs/phase65a_explore_20261008/msre_seed65/best_mIoU_iter_2000.pth',ADAPTED_SHA)]:
        assert digest(path)==sha
    return base


def fit():
    if datetime.now(timezone.utc)>=datetime(2026,10,19,16,tzinfo=timezone.utc):raise RuntimeError('Research budget expired')
    base=verify();original=base['models']['msre_source_spectral']
    prior=read(RECOVERY/'fit_feature_index.json');fitrows=records('fit_representatives')
    assert set(fitrows)=={r['product'] for r in prior['rows']}
    # This lock precedes any optimizer call; original masks/cache hashes are transitively bound.
    write(ROOT/'input_lock.json',dict(original_split_sha256=SPLIT_SHA,recovery_lock_sha256=RECOVERY_SHA,
        phase67_control_lock_sha256=CONTROL_SHA,source_checkpoint_sha256=SOURCE_SHA,
        msre_checkpoint_sha256=ADAPTED_SHA,prepared_manifest_sha256=digest(PREP/'manifest.json'),
        fit_input_sha256=base['fit_input_sha256'],original_l3=original,
        original_development_report_sha256=digest(RECOVERY/'recovery_report.json'),
        protocol_sha256=digest(PROTOCOL),execution_sha256=digest(EXECUTION),code_sha256=code_hashes(),
        new_shallow_fits=1,policies=['A','B','C','D'],bootstrap_seed=68010,bootstrap_replicates=10000,
        data_scope='153 original fit representatives and 32 original development representatives only',
        no_confirmation_access=True,phase65c_authorized=False))
    samples=[];ys=[];audit=[]
    for i,row in enumerate(prior['rows']):
        product=row['product'];r=fitrows[product];assert row['group_id']==r['group_id'];prepare_verify(r)
        truth=np.load(PREP/r['mask_path']);yy=truth[truth!=255]==2;x=old_features(row)
        assert len(x)==row['valid_pixels']==len(yy)
        seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
        idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
        assert hashlib.sha256(idx.tobytes()).hexdigest()==row['sample_indices_sha256']
        assert len(idx)==row['sample_count'] and int(yy[idx].sum())==row['sample_positive']
        samples.append(x[idx]);ys.append(yy[idx])
        audit.append(dict(**row,mask_sha256=digest(PREP/r['mask_path']),
            full_positive=int(yy.sum()),full_negative=int((~yy).sum()),
            sample_negative=int((~yy[idx]).sum()),sample_missed_positive_support=bool(yy.any() and not yy[idx].any())))
        print('Phase68 fit audit',i+1,153,flush=True)
    x=np.concatenate(samples).astype(np.float64);y=np.concatenate(ys).astype(float)
    assert hashlib.sha256(x.tobytes()).hexdigest()==prior['sample_x_sha256']
    assert hashlib.sha256(y.tobytes()).hexdigest()==prior['sample_y_sha256']
    w=conditional_weights(ys)
    assert np.isclose(w[y==1].sum(),.5) and np.isclose(w[y==0].sum(),.5)
    support=dict(full_positive_groups=sum(r['full_positive']>0 for r in audit),
        full_negative_groups=sum(r['full_negative']>0 for r in audit),
        sample_positive_groups=sum(r['sample_positive']>0 for r in audit),
        sample_negative_groups=sum(r['sample_negative']>0 for r in audit),
        sample_missed_positive_groups=sum(r['sample_missed_positive_support'] for r in audit))
    write(ROOT/'fit_input_audit.json',dict(rows=audit,support=support,sample_count=len(y),
        sample_positive=int(y.sum()),sample_x_sha256=prior['sample_x_sha256'],sample_y_sha256=prior['sample_y_sha256'],
        weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(),positive_weight=float(w[y==1].sum()),
        negative_weight=float(w[y==0].sum()),full_positive_pixels=sum(r['full_positive'] for r in audit)))
    print('Phase68 support',support,flush=True)
    # Exactly one optimizer invocation, preserving original scaler and optimizer settings.
    model=fit_fixed_scaler(x,y,w,original);del x,y,w,samples,ys
    print('Phase68 single fit converged',model['iterations'],flush=True)
    oldneg=[];newneg=[];supports=[]
    for i,row in enumerate(audit):
        xx=old_features(row);r=fitrows[row['product']];truth=np.load(PREP/r['mask_path']);yy=truth[truth!=255]==2
        oldneg.append(predict_logit(xx,original)[~yy]);newneg.append(predict_logit(xx,model)[~yy]);supports.append(bool(yy.any()))
        print('Phase68 full-negative calibration inputs',i+1,153,flush=True)
    tb,oldparts=dual_threshold(oldneg,supports);print('Phase68 B threshold locked',tb,flush=True)
    tc=group_threshold(newneg,.01);print('Phase68 C threshold locked',tc,flush=True)
    td,newparts=dual_threshold(newneg,supports);print('Phase68 D threshold locked',td,flush=True)
    assert tb>=original['threshold'] and td>=tc
    thresholds=dict(A=original['threshold'],B=tb,C=tc,D=td)
    calibration={}
    for name,t in thresholds.items():
        neg=oldneg if name in ['A','B'] else newneg
        rates=[float((a>=t).mean()) for a in neg]
        calibration[name]=dict(threshold=t,all_group_mean_fpr=float(np.mean(rates)),
            shadow_group_mean_fpr=float(np.mean([v for v,s in zip(rates,supports) if s])),
            no_shadow_group_mean_fpr=float(np.mean([v for v,s in zip(rates,supports) if not s])),
            rows=[dict(product=r['product'],group_id=r['group_id'],has_shadow=s,fpr=v) for r,s,v in zip(audit,supports,rates)])
    assert calibration['A']['all_group_mean_fpr']<=.01 and calibration['C']['all_group_mean_fpr']<=.01
    for n in ['B','D']:
        assert calibration[n]['shadow_group_mean_fpr']<=.01 and calibration[n]['no_shadow_group_mean_fpr']<=.01
    write(ROOT/'policy_lock.json',dict(status='one_model_and_four_thresholds_frozen_before_development',
        input_lock_sha256=digest(ROOT/'input_lock.json'),fit_input_audit_sha256=digest(ROOT/'fit_input_audit.json'),
        models={'original':original,'conditional':model},thresholds=thresholds,
        dual_stratum_thresholds={'B':oldparts,'D':newparts},fit_calibration=calibration,
        policy_model={'A':'original','B':'original','C':'conditional','D':'conditional'},
        nonshadow_fallback='same frozen MsRE Surface/Cloud winner for every policy',
        continue_rule='all seven prespecified point-estimate conditions in execution protocol; no tolerance changes',
        development_used_for_fitting_or_calibration=False,phase65c_authorized=False))
    print('Phase68 policies frozen before development',flush=True)


def make_row(product,r,truth,score,threshold,ns,ranking,name):
    pred=corrected(score,threshold,ns);cm=confusion(truth,pred)
    return dict(product=product,group_id=r['group_id'],policy=name,threshold=threshold,
        ranking=ranking,**measures(cm),predicted_shadow_fraction=float((pred==2).mean()))


def evaluate():
    verify();lock=read(ROOT/'policy_lock.json');inputs=read(ROOT/'input_lock.json')
    assert digest(ROOT/'input_lock.json')==lock['input_lock_sha256']
    assert digest(ROOT/'fit_input_audit.json')==lock['fit_input_audit_sha256']
    assert digest(PROTOCOL)==inputs['protocol_sha256'] and digest(EXECUTION)==inputs['execution_sha256']
    assert code_hashes()==inputs['code_sha256']
    md=records('development_val');sources=[];index_sha={}
    reference_report=read(RECOVERY/'recovery_report.json')
    assert digest(RECOVERY/'recovery_report.json')==inputs['original_development_report_sha256']
    reference={r['product']:r for r in reference_report['policies']['msre_source_spectral']['rows']}
    for name in ['source','msre']:
        path=OLD/'development_val'/('index_'+name+'.json');d=read(path);index_sha[name]=digest(path)
        assert d['checkpoint_sha256']==(SOURCE_SHA if name=='source' else ADAPTED_SHA)
        assert d['manifest_sha256']==inputs['prepared_manifest_sha256']
        rows={r['product']:r for r in d['rows']};assert set(rows)==set(md);sources.append(rows)
    assert index_sha==reference_report['development_cache_index_sha256']
    assert set(reference)==set(md)
    rows={n:[] for n in ['A','B','C','D']};cache=[];dest=ROOT/'development_features';dest.mkdir(exist_ok=False)
    # Finish ALL original-A checks before evaluating any B/C/D policy.
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in md.items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;yy=truth[valid]==2
            sr=sources[0][product];mr=sources[1][product];assert sr['group_id']==mr['group_id']==r['group_id']
            for prior in [sr,mr]:
                assert digest(OLD/prior['truth_path'])==prior['truth_sha256']
                assert np.array_equal(np.load(OLD/prior['truth_path']),truth)
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            xx=features(raw,load(OLD,mr,'score'),load(OLD,sr,'score'),valid)
            score=predict_logit(xx,lock['models']['original'])
            assert np.array_equal(score,np.load(RECOVERY/'development_scores'/(product+'_msre_source_spectral.npy')))
            ranking=brief(curve(score,yy));ns=load(OLD,mr,'nonshadow')[valid]
            row=make_row(product,r,truth[valid],score,lock['thresholds']['A'],ns,ranking,'A')
            assert np.array_equal(row['confusion'],reference[product]['confusion'])
            assert ranking['ap']==reference[product]['ranking']['ap'];rows['A'].append(row)
            path=dest/(product+'.npy');np.save(path,xx,allow_pickle=False)
            cache.append(dict(product=product,feature_path=str(path.relative_to(ROOT)),feature_sha256=digest(path)))
            print('Phase68 A exact reproduction',len(cache),32,flush=True)
    write(ROOT/'A_reproduction.json',dict(all_32_exact=True,score_and_confusion_and_ap=True,feature_index=cache))
    print('Phase68 all A reproduced; evaluating B C D',flush=True)
    pool={'original':[],'conditional':[]};labels=[]
    for entry in cache:
        product=entry['product'];r=md[product];truth=np.load(PREP/r['mask_path']);valid=truth!=255;yy=truth[valid]==2
        path=ROOT/entry['feature_path'];assert digest(path)==entry['feature_sha256'];xx=np.load(path,mmap_mode='r')
        scores={n:predict_logit(xx,m) for n,m in lock['models'].items()}
        ranks={'original':rows['A'][len(labels)]['ranking'],'conditional':brief(curve(scores['conditional'],yy))}
        ns=load(OLD,sources[1][product],'nonshadow')[valid]
        for name in ['B','C','D']:
            key=lock['policy_model'][name]
            rows[name].append(make_row(product,r,truth[valid],scores[key],lock['thresholds'][name],ns,ranks[key],name))
        assert rows['B'][-1]['true_positive']<=rows['A'][len(labels)]['true_positive']
        assert rows['D'][-1]['true_positive']<=rows['C'][-1]['true_positive']
        for key in pool:pool[key].append(scores[key])
        labels.append(yy);print('Phase68 four-policy development',len(labels),32,flush=True)
    pooled_ranks={};all_y=np.concatenate(labels)
    for key in list(pool):
        pooled_ranks[key]=brief(curve(np.concatenate(pool.pop(key)),all_y));print('Phase68 pooled ranking',key,flush=True)
    policies={}
    for name,rr in rows.items():
        positive=[r for r in rr if r['positive']>0];ap=[r for r in rr if r['ranking']['ap'] is not None]
        assert len(positive)==len(ap)==7 and len(rr)==32
        strata={k:interval([r['fpr'] for r in subset if r['fpr'] is not None]) for k,subset in
            [('all',rr),('shadow',positive),('no_shadow',[r for r in rr if not r['positive']])]}
        policies[name]=dict(**measures(np.sum([r['confusion'] for r in rr],axis=0)),rows=rr,
            ranking=pooled_ranks[lock['policy_model'][name]],n_groups=32,shadow_groups=len(positive),
            macro_ap=float(np.mean([r['ranking']['ap'] for r in ap])),macro_ap_groups=len(ap),
            macro_recall=float(np.mean([r['recall'] for r in positive])),macro_recall_groups=len(positive),
            macro_precision=float(np.mean([r['precision'] for r in rr if r['precision'] is not None])),
            macro_precision_groups=sum(r['precision'] is not None for r in rr),
            macro_miou_percent=float(np.mean([r['miou_percent'] for r in rr])),
            macro_iou_percent=[float(np.mean([r['iou_percent'][i] for r in rr if r['iou_percent'][i] is not None])) for i in range(3)],
            macro_iou_groups=[sum(r['iou_percent'][i] is not None for r in rr) for i in range(3)],
            macro_predicted_shadow_fraction=float(np.mean([r['predicted_shadow_fraction'] for r in rr])),
            pooled_predicted_shadow_fraction=float(sum(r['true_positive']+r['false_positive'] for r in rr)/sum(r['positive']+r['negative'] for r in rr)),
            fpr_strata=strata,groups_fpr_above_1pct=sum(r['fpr']>.01 for r in rr if r['fpr'] is not None),
            zero_recall_groups=sum(r['true_positive']==0 for r in positive))
    assert policies['A']['macro_ap']==policies['B']['macro_ap']
    assert policies['C']['macro_ap']==policies['D']['macro_ap']
    for key in ['macro_ap','macro_recall']:
        assert policies['A'][key]==reference_report['policies']['msre_source_spectral'][key]
    comparisons={}
    for left,right in [('D','B'),('D','A'),('A','B'),('A','C'),('C','D')]:
        pairs=[]
        for l,r in zip(rows[left],rows[right]):
            assert l['group_id']==r['group_id']
            if l['positive']:
                pairs.append(dict(product=l['product'],group_id=l['group_id'],delta_ap=l['ranking']['ap']-r['ranking']['ap'],
                    delta_recall=l['recall']-r['recall'],delta_fpr=l['fpr']-r['fpr']))
        da=np.array([r['delta_ap'] for r in pairs])
        comparisons[left+'_minus_'+right]=dict(pairs=pairs,recall=interval([r['delta_recall'] for r in pairs]),
            ap=interval(da),improved_ap=int((da>.01).sum()),stable_ap=int((np.abs(da)<=.01).sum()),
            declined_ap=int((da<-.01).sum()),worst_ap=min(pairs,key=lambda r:r['delta_ap']),
            worst_recall=min(pairs,key=lambda r:r['delta_recall']))
    report=dict(status='completed_one_fixed_fit_four_development_policies',policies=policies,comparisons=comparisons,
        continue_gate=continue_gate(policies),fit_groups=153,development_groups=32,shadow_development_groups=7,
        new_shallow_fits=1,new_segmenter_forwards=0,original_l3_reproduced_exactly=True,
        policy_lock_sha256=digest(ROOT/'policy_lock.json'),development_cache_index_sha256=index_sha,
        confirmation_pixels_scores_metrics_read=False,original_h1_failure_unchanged=True,phase65c_authorized=False,
        statistics='10000 paired related-group resamples seed68010; development descriptive only; conditional on frozen models',
        unfinished=['No new independent validation authorized or performed; no Phase65c access'])
    write(ROOT/'delivery/phase68_report.json',report)
    table=[]
    for name,p in policies.items():
        for r in p['rows']:
            table.append(dict(policy=name,product=r['product'],group_id=r['group_id'],threshold=r['threshold'],ap=r['ranking']['ap'],
                **{k:r[k] for k in ['positive','negative','true_positive','false_positive','recall','precision','fpr','miou_percent',
                    'shadow_to_surface','shadow_to_cloud','predicted_shadow_fraction']},
                **{'iou_'+str(i):v for i,v in enumerate(r['iou_percent'])},
                oracle_recall_fpr01=r['ranking']['matched_fpr'].get('0.01',{}).get('recall'),
                oracle_recall_fpr05=r['ranking']['matched_fpr'].get('0.05',{}).get('recall')))
    with (ROOT/'delivery/development_groups.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    for name in ['input_lock.json','policy_lock.json','fit_input_audit.json','A_reproduction.json']:
        write(ROOT/'delivery'/name,read(ROOT/name))
    print('Phase68 completed:',report['continue_gate'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fit','evaluate']);args=parser.parse_args()
    if args.action=='fit':fit()
    else:evaluate()
