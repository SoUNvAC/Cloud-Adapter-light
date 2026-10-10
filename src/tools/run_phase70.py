"""Phase70: three fixed independent branches and matched refitted nulls."""
import argparse,csv,hashlib,io,json,os,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from prepare_phase65a_explore import digest,SPLIT_SHA,SOURCE_SHA
from run_phase65d_recovery import DATA,PREP,OLD,REPO,ADAPTED_SHA,records,prepare_verify,indices,load,features,brief
from run_phase67 import base_verify,old_features,RECOVERY,RECOVERY_SHA,read,write
from phase65d_recovery_math import fit_model,predict_logit,group_threshold
from phase65d_metrics import curve,confusion,measures,corrected
from phase70_math import BRANCHES,interaction,local_contrast,normalized_differences,checked,permutation,interval,promotion
from phase70_metadata import audit as metadata_audit

ROOT=DATA/'phase70_20261010'
PROTOCOL=REPO/'src/research_plans/PHASE70_20261010.md'
CODE_NAMES=['run_phase70.py','phase70_math.py','phase70_metadata.py','test_phase70.py','launch_phase70.sh',
    'run_phase67.py','run_phase65d_recovery.py','phase65d_recovery_math.py','phase65d_metrics.py',
    'prepare_phase65a_explore.py','phase67_controls.py']


def hashes():return {n:digest(Path(__file__).with_name(n)) for n in CODE_NAMES}


def verify():
    base=base_verify()
    for path,sha in [(REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',SOURCE_SHA),
        (REPO/'src/work_dirs/phase65a_explore_20261008/msre_seed65/best_mIoU_iter_2000.pth',ADAPTED_SHA)]:assert digest(path)==sha
    return base


def development_indices():
    result=[];sha={}
    for name in ['source','msre']:
        path=OLD/'development_val'/('index_'+name+'.json');d=read(path);sha[name]=digest(path)
        assert d['checkpoint_sha256']==(SOURCE_SHA if name=='source' else ADAPTED_SHA)
        assert d['manifest_sha256']==digest(PREP/'manifest.json')
        rows={r['product']:r for r in d['rows']};assert set(rows)==set(records('development_val'));result.append(rows)
    return result,sha


def prepare():
    base=verify();original=base['models']['msre_source_spectral'];fitrows=records('fit_representatives');dev=records('development_val')
    stat=os.statvfs(ROOT);assert stat.f_bavail*stat.f_frsize>16*1024**3,'Less than 16 GiB free; no automatic cleanup'
    fit_source,fit_msre=indices(RECOVERY);development,devsha=development_indices()
    old_report=read(RECOVERY/'recovery_report.json');assert old_report['development_cache_index_sha256']==devsha
    write(ROOT/'input_lock.json',dict(original_split_sha256=SPLIT_SHA,recovery_lock_sha256=RECOVERY_SHA,
        source_checkpoint_sha256=SOURCE_SHA,msre_checkpoint_sha256=ADAPTED_SHA,
        prepared_manifest_sha256=digest(PREP/'manifest.json'),fit_input_sha256=base['fit_input_sha256'],
        development_cache_index_sha256=devsha,original_development_report_sha256=digest(RECOVERY/'recovery_report.json'),
        original_l3=original,protocol_sha256=digest(PROTOCOL),execution_addendum_sha256=digest(PROTOCOL.with_name('PHASE70_EXECUTION_20261010.md')),code_sha256=hashes(),
        policies=['L3','A','A_shuffle','B','B_shuffle','C','C_shuffle'],max_new_shallow_fits=6,new_segmenter_forwards=0,
        B_window=61,center_excluded=True,observable_mask='existing image_valid only; no truth input',
        A_standardization='both logits and spectral columns use original frozen L3 fit scaler before interaction',
        C_native_channels=[2,7,11],C_denominator_epsilon=1e-6,no_index_clipping=True,
        shuffle='joint new columns within all image-valid pixels, per product and branch, before truth-valid selection',
        shuffle_seed='70010+branch_index*1000+int(SHA256(product_id)[:8],16)',
        fit_sampler='original uniform max8192; seed65081+product hash',
        optimizer='original group equal BCE ridge .001 LBFGSB maxiter500 gtol1e-7 ftol1e-12 first two coefficients>=0',
        calibration='full fit negatives, equal groups, exact ties >=, mean FPR<=1%',
        bootstrap_seed=70010,bootstrap_replicates=10000,group_improvement_rule='strict delta AP>.01; user confirmed',
        no_confirmation_access=True,phase65c_authorized=False))
    original_dir=ROOT/'development_original_features';original_dir.mkdir(exist_ok=False)
    reference={r['product']:r for r in old_report['policies']['msre_source_spectral']['rows']};assert set(reference)==set(dev)
    a_rows=[];original_index={}
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in dev.items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;yy=truth[valid]==2
            sr,mr=development[0][product],development[1][product]
            assert sr['group_id']==mr['group_id']==r['group_id']
            for prior in [sr,mr]:
                assert digest(OLD/prior['truth_path'])==prior['truth_sha256']
                assert np.array_equal(np.load(OLD/prior['truth_path']),truth)
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            xx=features(raw,load(OLD,mr,'score'),load(OLD,sr,'score'),valid)
            score=predict_logit(xx,original);ranking=brief(curve(score,yy))
            assert np.array_equal(score,np.load(RECOVERY/'development_scores'/(product+'_msre_source_spectral.npy')))
            ns=load(OLD,mr,'nonshadow')[valid];pred=corrected(score,original['threshold'],ns)
            cm=confusion(truth[valid],pred);assert np.array_equal(cm,reference[product]['confusion'])
            assert ranking['ap']==reference[product]['ranking']['ap']
            a_rows.append(dict(product=product,group_id=r['group_id'],policy='L3',ranking=ranking,**measures(cm),
                threshold=original['threshold'],predicted_shadow_fraction=float((pred==2).mean())))
            path=original_dir/(product+'.npy');np.save(path,xx,allow_pickle=False)
            original_index[product]=dict(feature_path=str(path.relative_to(ROOT)),feature_sha256=digest(path))
            print('Phase70 original L3 exact reproduction',len(a_rows),32,flush=True)
    write(ROOT/'A_reproduction.json',dict(all_32_exact=True,score_confusion_ap=True,rows=a_rows))
    prior=read(RECOVERY/'fit_feature_index.json');oldfit={r['product']:r for r in prior['rows']};assert set(oldfit)==set(fitrows)
    metadata_audit(DATA,PREP,{**fitrows,**dev},ROOT,write)
    dest=ROOT/'observable_features';dest.mkdir(exist_ok=False);feature_index={};status={n:dict(status='prepared') for n in BRANCHES}
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for cohort,selected,sources,msres,predroot in [('fit',fitrows,fit_source,fit_msre,RECOVERY),('development',dev,*development,OLD)]:
            rows=[]
            for product,r in selected.items():
                prepare_verify(r);iv=np.load(PREP/r['image_valid_path']);truth=np.load(PREP/r['mask_path']);valid=truth!=255
                b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
                raw=np.load(io.BytesIO(b),allow_pickle=False)
                original=features(raw,load(predroot,msres[product],'score'),load(predroot,sources[product],'score'),iv)
                old=oldfit[product] if cohort=='fit' else original_index[product]
                previous=old_features(old) if cohort=='fit' else np.load(ROOT/old['feature_path'])
                assert np.array_equal(original[truth[iv]!=255],previous),'Original five feature drift'
                row=dict(product=product,group_id=r['group_id'],cohort=cohort,original_features=old,
                    image_valid_path=r['image_valid_path'],image_valid_sha256=digest(PREP/r['image_valid_path']),
                    mask_path=r['mask_path'],mask_sha256=digest(PREP/r['mask_path']),cube_member_sha256=r['member_sha256']['image'],branches={})
                for branch in BRANCHES:
                    if status[branch]['status']=='failed':continue
                    try:
                        extra={}
                        if branch=='A':values=interaction(original,base['models']['msre_source_spectral'])
                        elif branch=='B':values=local_contrast(raw,iv)
                        else:values,extra=normalized_differences(raw,iv)
                        values=checked(values);assert len(values)==int(iv.sum())
                        _,shuffle=permutation(product,branch,len(values));path=dest/(product+'_'+branch+'.npy');np.save(path,values,allow_pickle=False)
                        row['branches'][branch]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),permutation=shuffle,
                            feature_min=values.min(axis=0).tolist(),feature_max=values.max(axis=0).tolist(),**extra)
                    except (ValueError,FloatingPointError) as e:
                        status[branch]=dict(status='failed',stage='features',product=product,error=str(e));print('BRANCH FAILED',branch,status[branch],flush=True)
                rows.append(row);print('Phase70 observable features',cohort,len(rows),len(selected),flush=True)
            feature_index[cohort]=rows
    feature_index['branch_status']=status;write(ROOT/'feature_index.json',feature_index)


def full_inputs(row,branch,shuffle=False):
    ivpath=PREP/row['image_valid_path'];maskpath=PREP/row['mask_path']
    assert digest(ivpath)==row['image_valid_sha256'] and digest(maskpath)==row['mask_sha256']
    iv=np.load(ivpath);truth=np.load(maskpath);valid=truth!=255;assert np.all(iv[valid])
    entry=row['branches'][branch];path=ROOT/entry['path'];assert digest(path)==entry['sha256']
    values=np.load(path,mmap_mode='r');assert len(values)==int(iv.sum());selection=np.flatnonzero(truth[iv]!=255)
    if shuffle:
        perm,audit=permutation(row['product'],branch,len(values));assert audit==entry['permutation'];selection=perm[selection]
    if row['cohort']=='fit':original=old_features(row['original_features'])
    else:
        old=row['original_features'];assert digest(ROOT/old['feature_path'])==old['feature_sha256'];original=np.load(ROOT/old['feature_path'],mmap_mode='r')
    return np.column_stack([original,values[selection]]),truth[valid]==2


def fit():
    if datetime.now(timezone.utc)>=datetime(2026,10,19,16,tzinfo=timezone.utc):raise RuntimeError('Research budget expired')
    base=verify();initial=read(ROOT/'input_lock.json');assert hashes()==initial['code_sha256'] and digest(PROTOCOL)==initial['protocol_sha256']
    assert read(ROOT/'A_reproduction.json')['all_32_exact'];index=read(ROOT/'feature_index.json');rows=index['fit'];prior=read(RECOVERY/'fit_feature_index.json')
    status=index['branch_status'];models={'L3':base['models']['msre_source_spectral']};audits={};attempts=0
    for branch in BRANCHES:
        if status[branch]['status']=='failed':continue
        for shuffled in [False,True]:
            if datetime.now(timezone.utc)>=datetime(2026,10,19,16,tzinfo=timezone.utc):raise RuntimeError('Research budget expired before next fit')
            name=branch+('_shuffle' if shuffled else '');samples=[];ys=[];weights=[];audit=[]
            for row in rows:
                x,y=full_inputs(row,branch,shuffled);old=row['original_features'];product=row['product']
                seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
                idx=np.random.default_rng(seed).choice(len(y),min(8192,len(y)),replace=False)
                assert hashlib.sha256(idx.tobytes()).hexdigest()==old['sample_indices_sha256']
                assert int(y[idx].sum())==old['sample_positive'] and len(idx)==old['sample_count']
                samples.append(x[idx]);ys.append(y[idx]);weights.append(np.full(len(idx),1/(153*len(idx))))
                audit.append(dict(product=product,group_id=row['group_id'],sample_count=len(idx),sample_positive=int(y[idx].sum()),sample_indices_sha256=old['sample_indices_sha256']))
            x=np.concatenate(samples).astype(np.float64);y=np.concatenate(ys).astype(float);w=np.concatenate(weights);del samples
            assert hashlib.sha256(x[:,:5].tobytes()).hexdigest()==prior['sample_x_sha256']
            assert hashlib.sha256(y.tobytes()).hexdigest()==prior['sample_y_sha256']
            audits[name]=dict(rows=audit,sample_count=len(y),sample_positive=int(y.sum()),sample_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                original_five_sample_sha256=prior['sample_x_sha256'],sample_y_sha256=prior['sample_y_sha256'],weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())
            attempts+=1;assert attempts<=6;write(ROOT/('fit_attempt_'+name+'.json'),dict(policy=name,attempt=attempts,started=datetime.now(timezone.utc).isoformat(),sample_sha256=audits[name]['sample_x_sha256']))
            try:model=fit_model(x,y,w,2)
            except RuntimeError as e:
                status[branch]=dict(status='failed',stage='optimizer',policy=name,error=str(e));print('BRANCH FAILED',branch,status[branch],flush=True);break
            del x,y,w;negatives=[];print('Phase70 fitted',name,model['iterations'],flush=True)
            for row in rows:
                xx,yy=full_inputs(row,branch,shuffled);score=predict_logit(xx,model);negatives.append(score[~yy])
            model['threshold']=group_threshold(negatives,.01);model['fit_macro_fpr']=float(np.mean([(a>=model['threshold']).mean() for a in negatives]))
            assert model['fit_macro_fpr']<=.01;model['n_features']=5+BRANCHES[branch];model['nonshadow']='frozen MsRE Surface/Cloud winner';models[name]=model
            write(ROOT/('fitted_'+name+'.json'),model);del negatives;print('Phase70 calibrated',name,model['threshold'],model['fit_macro_fpr'],flush=True)
        else:status[branch]=dict(status='completed')
    write(ROOT/'fit_input_audit.json',dict(policies=audits,attempts=attempts,sample_count=1253376,original_five_sample_sha256=prior['sample_x_sha256'],sample_y_sha256=prior['sample_y_sha256']))
    write(ROOT/'policy_lock.json',dict(status='all_attempted_models_and_thresholds_frozen_before_new_development_scores',models=models,branch_status=status,fit_attempts=attempts,
        input_lock_sha256=digest(ROOT/'input_lock.json'),feature_index_sha256=digest(ROOT/'feature_index.json'),fit_input_audit_sha256=digest(ROOT/'fit_input_audit.json'),
        A_reproduction_sha256=digest(ROOT/'A_reproduction.json'),development_used_for_fitting_or_calibration=False,phase65c_authorized=False))
    print('Phase70 ALL policies frozen',status,flush=True)


def evaluate():
    verify();lock=read(ROOT/'policy_lock.json');initial=read(ROOT/'input_lock.json')
    assert digest(ROOT/'input_lock.json')==lock['input_lock_sha256'] and hashes()==initial['code_sha256']
    assert digest(PROTOCOL)==initial['protocol_sha256']
    for name in ['feature_index','fit_input_audit','A_reproduction']:assert digest(ROOT/(name+'.json'))==lock[name+'_sha256']
    index=read(ROOT/'feature_index.json');dev=records('development_val');sources,sha=development_indices()
    assert sha==initial['development_cache_index_sha256']
    a=read(ROOT/'A_reproduction.json');assert a['all_32_exact']
    names=['L3']+[n+s for n in BRANCHES if lock['branch_status'][n]['status']=='completed' for s in ['', '_shuffle']];rows={n:[] for n in names};pool={n:[] for n in rows};labels=[]
    for entry in index['development']:
        product=entry['product'];r=dev[product];old=entry['original_features'];assert digest(ROOT/old['feature_path'])==old['feature_sha256'];original=np.load(ROOT/old['feature_path']);truth=np.load(PREP/r['mask_path']);y=truth[truth!=255]==2
        truth=np.load(PREP/r['mask_path']);valid=truth!=255;ns=load(OLD,sources[1][product],'nonshadow')[valid]
        for name in rows:
            model=lock['models'][name];xx=original if name=='L3' else full_inputs(entry,name[0],name.endswith('_shuffle'))[0];score=predict_logit(xx,model)
            pred=corrected(score,model['threshold'],ns);ranking=brief(curve(score,y));cm=confusion(truth[valid],pred)
            if name=='L3':
                reference=a['rows'][len(labels)];assert product==reference['product']
                assert ranking['ap']==reference['ranking']['ap'] and np.array_equal(cm,reference['confusion'])
            rows[name].append(dict(product=product,group_id=r['group_id'],policy=name,threshold=model['threshold'],
                ranking=ranking,**measures(cm),predicted_shadow_fraction=float((pred==2).mean())))
            pool[name].append(score)
        labels.append(y);print('Phase70 development policies',len(labels),32,flush=True)
    policies={};all_y=np.concatenate(labels)
    for name,rr in rows.items():
        positive=[r for r in rr if r['positive']];aps=[r for r in rr if r['ranking']['ap'] is not None]
        assert len(positive)==len(aps)==7 and len(rr)==32
        strata={k:interval([r['fpr'] for r in subset if r['fpr'] is not None]) for k,subset in
            [('all',rr),('shadow',positive),('no_shadow',[r for r in rr if not r['positive']])]}
        policies[name]=dict(**measures(np.sum([r['confusion'] for r in rr],axis=0)),rows=rr,
            ranking=brief(curve(np.concatenate(pool.pop(name)),all_y)),n_groups=32,shadow_groups=7,
            macro_ap=float(np.mean([r['ranking']['ap'] for r in aps])),macro_recall=float(np.mean([r['recall'] for r in positive])),
            macro_precision=float(np.mean([r['precision'] for r in rr if r['precision'] is not None])),
            macro_precision_groups=sum(r['precision'] is not None for r in rr),
            macro_miou_percent=float(np.mean([r['miou_percent'] for r in rr])),
            macro_iou_percent=[float(np.mean([r['iou_percent'][i] for r in rr if r['iou_percent'][i] is not None])) for i in range(3)],
            fpr_strata=strata,groups_fpr_above_1pct=sum(r['fpr']>.01 for r in rr if r['fpr'] is not None),
            zero_recall_groups=sum(r['true_positive']==0 for r in positive),
            macro_predicted_shadow_fraction=float(np.mean([r['predicted_shadow_fraction'] for r in rr])),
            pooled_predicted_shadow_fraction=float(sum(r['true_positive']+r['false_positive'] for r in rr)/sum(r['positive']+r['negative'] for r in rr)))
        print('Phase70 pooled ranking',name,flush=True)
    comparisons={}
    for left,right in [(n,r) for n in BRANCHES if lock['branch_status'][n]['status']=='completed' for r in ['L3',n+'_shuffle']]:
        pairs=[]
        for l,r in zip(rows[left],rows[right]):
            assert l['group_id']==r['group_id']
            if l['positive']:pairs.append(dict(product=l['product'],group_id=l['group_id'],
                delta_ap=l['ranking']['ap']-r['ranking']['ap'],delta_recall=l['recall']-r['recall'],delta_fpr=l['fpr']-r['fpr']))
        delta=np.array([r['delta_ap'] for r in pairs])
        comparisons[left+'_minus_'+right]=dict(pairs=pairs,ap=interval(delta),recall=interval([r['delta_recall'] for r in pairs]),
            improved_ap=int((delta>.01).sum()),stable_ap=int((np.abs(delta)<=.01).sum()),declined_ap=int((delta<-.01).sum()),
            worst_ap=min(pairs,key=lambda r:r['delta_ap']),worst_recall=min(pairs,key=lambda r:r['delta_recall']))
    report=dict(status='completed_fixed_tree_exploration',policies=policies,comparisons=comparisons,
        promotion=promotion(policies,comparisons,lock['branch_status']),branch_status=lock['branch_status'],fit_groups=153,development_groups=32,shadow_development_groups=7,
        new_shallow_fits=lock['fit_attempts'],new_segmenter_forwards=0,original_l3_reproduced_exactly=True,
        policy_lock_sha256=digest(ROOT/'policy_lock.json'),confirmation_pixels_scores_metrics_read=False,original_h1_failure_unchanged=True,phase65c_authorized=False,
        statistics='10000 paired related-group bootstrap seed70010; descriptive seven-positive-group multi-branch development exploration only',
        metadata_readiness=read(ROOT/'metadata_readiness.json')['status'])
    write(ROOT/'delivery/phase70_report.json',report)
    table=[]
    for name,p in policies.items():
        for r in p['rows']:
            table.append(dict(policy=name,product=r['product'],group_id=r['group_id'],threshold=r['threshold'],ap=r['ranking']['ap'],
                **{k:r[k] for k in ['positive','negative','true_positive','false_positive','recall','precision','fpr','miou_percent',
                    'shadow_to_surface','shadow_to_cloud','predicted_shadow_fraction']},**{'iou_'+str(i):v for i,v in enumerate(r['iou_percent'])},
                oracle_recall_fpr01=r['ranking']['matched_fpr'].get('0.01',{}).get('recall'),
                oracle_recall_fpr05=r['ranking']['matched_fpr'].get('0.05',{}).get('recall')))
    with (ROOT/'delivery/development_groups.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    for name in ['input_lock','policy_lock','feature_index','fit_input_audit','A_reproduction','metadata_readiness']:write(ROOT/'delivery'/(name+'.json'),read(ROOT/(name+'.json')))
    import shutil
    shutil.copyfile(ROOT/'metadata_readiness.csv',ROOT/'delivery/metadata_readiness.csv')
    print('Phase70 completed',report['promotion'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','fit','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'fit':fit,'evaluate':evaluate}[args.action]()
