"""Phase71: four fixed B specificity controls; no new segmentation inference."""
import argparse,csv,hashlib,io,json,os,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from prepare_phase65a_explore import digest,SPLIT_SHA,SOURCE_SHA
from run_phase65d_recovery import DATA,PREP,OLD,REPO,ADAPTED_SHA,records,prepare_verify,indices,load,features,brief
from run_phase67 import base_verify,old_features,RECOVERY,RECOVERY_SHA,read,write
from phase65d_recovery_math import fit_model,predict_logit,group_threshold
from phase65d_metrics import curve,confusion,measures,corrected
from phase71_math import CONTROLS,rgb_local,inputs
import run_phase70 as prior70
from phase71_report import interval

ROOT=DATA/'phase71_20261010'
PROTOCOL=REPO/'src/research_plans/PHASE71_B_SPECIFICITY_20261010.md'
CODE_NAMES=['run_phase71.py','phase71_math.py','phase71_report.py','test_phase71.py','launch_phase71.sh','run_phase70.py','phase70_math.py','phase70_metadata.py',
    'run_phase67.py','run_phase65d_recovery.py','phase65d_recovery_math.py','phase65d_metrics.py',
    'prepare_phase65a_explore.py','phase67_controls.py']


def hashes():return {n:digest(Path(__file__).with_name(n)) for n in CODE_NAMES}


def verify():
    base=base_verify()
    for path,sha in [(REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth',SOURCE_SHA),
        (REPO/'src/work_dirs/phase65a_explore_20261008/msre_seed65/best_mIoU_iter_2000.pth',ADAPTED_SHA)]:assert digest(path)==sha
    assert digest(prior70.ROOT/'policy_lock.json')=='5bc0b8911b3010a5276546f72d445888d96f9e0935cdafbadac986192b570df9'
    assert digest(prior70.ROOT/'feature_index.json')=='2d1a20225539226ae8ad5b4e732f75a6811c2edb67622f8c8684b057051c8b94'
    assert digest(prior70.ROOT/'delivery/phase70_report.json')=='6a22606d12a0de69a20ed06625934401262214db52f3e667f656ec3da624bbd2'
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
    stat=os.statvfs(ROOT);assert stat.f_bavail*stat.f_frsize>4*1024**3,'Less than 4 GiB free; no automatic cleanup'
    verify();index=read(prior70.ROOT/'feature_index.json');oldlock=read(prior70.ROOT/'policy_lock.json');oldreport=read(prior70.ROOT/'delivery/phase70_report.json')
    dev=records('development_val');sources,devsha=development_indices();models={n:oldlock['models'][n] for n in ['L3','B']}
    write(ROOT/'input_lock.json',dict(original_split_sha256=SPLIT_SHA,recovery_lock_sha256=RECOVERY_SHA,
        phase70_policy_lock_sha256=digest(prior70.ROOT/'policy_lock.json'),phase70_feature_index_sha256=digest(prior70.ROOT/'feature_index.json'),
        phase70_report_sha256=digest(prior70.ROOT/'delivery/phase70_report.json'),development_cache_index_sha256=devsha,
        protocol_sha256=digest(PROTOCOL),code_sha256=hashes(),controls=list(CONTROLS),max_new_fits=4,
        native_RGB_channels=[3,2,1],window=61,formula='(center-mean)/(abs(mean)+1e-6)',center_excluded=True,
        image_valid_only=True,original_five_retained=True,drop_only_local_contrast=True,bootstrap_seed=71010,bootstrap_replicates=10000,
        development_used_for_fitting_or_calibration=False,new_segmenter_forwards=0,confirmation_access=False))
    references={n:{r['product']:r for r in oldreport['policies'][n]['rows']} for n in models};reproduced={n:[] for n in models}
    for row in index['development']:
        x,y=prior70.full_inputs(row,'B');product=row['product'];r=dev[product];truth=np.load(PREP/r['mask_path']);valid=truth!=255;ns=load(OLD,sources[1][product],'nonshadow')[valid]
        for n,model in models.items():
            score=predict_logit(x[:,:5] if n=='L3' else x,model);ranking=brief(curve(score,y));cm=confusion(truth[valid],corrected(score,model['threshold'],ns));ref=references[n][product]
            assert ranking['ap']==ref['ranking']['ap'] and np.array_equal(cm,ref['confusion'])
            if n=='L3':assert np.array_equal(score,np.load(RECOVERY/'development_scores'/(product+'_msre_source_spectral.npy')))
            reproduced[n].append(dict(product=product,ap=ranking['ap'],confusion=cm.tolist(),score_sha256=hashlib.sha256(score.tobytes()).hexdigest()))
        print('Phase71 exact L3/B reproduction',len(reproduced['B']),32,flush=True)
    write(ROOT/'baseline_reproduction.json',dict(all_32_exact=True,rows=reproduced))
    dest=ROOT/'rgb_local';dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for cohort in ['fit','development']:
            selected=records('fit_representatives' if cohort=='fit' else 'development_val')
            assert set(selected)=={r['product'] for r in index[cohort]}
            for i,row in enumerate(index[cohort]):
                product=row['product'];r=selected[product];prepare_verify(r);iv=np.load(PREP/r['image_valid_path'])
                b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image'];raw=np.load(io.BytesIO(b),allow_pickle=False)
                values=rgb_local(raw,iv);path=dest/(product+'.npy');np.save(path,values,allow_pickle=False)
                row['rgb']=dict(path=str(path.relative_to(ROOT)),sha256=digest(path),minimum=values.min(0).tolist(),maximum=values.max(0).tolist(),observable_count=len(values))
                print('Phase71 RGB local',cohort,i+1,len(selected),flush=True)
    write(ROOT/'feature_index.json',index)


def full_inputs(row,name):
    xx,y=prior70.full_inputs(row,'B');rgb=None
    if name=='RGB_local':
        e=row['rgb'];assert digest(ROOT/e['path'])==e['sha256'];iv=np.load(PREP/row['image_valid_path']);truth=np.load(PREP/row['mask_path'])
        values=np.load(ROOT/e['path'],mmap_mode='r');assert len(values)==int(iv.sum())==e['observable_count'];rgb=values[truth[iv]!=255]
    return inputs(xx[:,:5],xx[:,5:],rgb,name),y


def fit():
    verify();initial=read(ROOT/'input_lock.json');assert hashes()==initial['code_sha256'] and digest(PROTOCOL)==initial['protocol_sha256']
    assert read(ROOT/'baseline_reproduction.json')['all_32_exact'];rows=read(ROOT/'feature_index.json')['fit'];prior=read(RECOVERY/'fit_feature_index.json')
    oldlock=read(prior70.ROOT/'policy_lock.json');models={n:oldlock['models'][n] for n in ['L3','B']};audits={}
    for attempt,name in enumerate(CONTROLS,1):
        if datetime.now(timezone.utc)>=datetime(2026,10,19,16,tzinfo=timezone.utc):raise RuntimeError('Research budget expired')
        samples=[];ys=[];weights=[];audit=[]
        for row in rows:
            x,y=full_inputs(row,name);old=row['original_features'];product=row['product'];seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
            idx=np.random.default_rng(seed).choice(len(y),min(8192,len(y)),replace=False)
            assert hashlib.sha256(idx.tobytes()).hexdigest()==old['sample_indices_sha256'] and int(y[idx].sum())==old['sample_positive']
            samples.append(x[idx]);ys.append(y[idx]);weights.append(np.full(len(idx),1/(153*len(idx))))
            audit.append(dict(product=product,group_id=row['group_id'],sample_indices_sha256=old['sample_indices_sha256']))
        x=np.concatenate(samples).astype(float);y=np.concatenate(ys).astype(float);w=np.concatenate(weights);del samples
        assert hashlib.sha256(x[:,:5].tobytes()).hexdigest()==prior['sample_x_sha256'] and hashlib.sha256(y.tobytes()).hexdigest()==prior['sample_y_sha256']
        audits[name]=dict(rows=audit,sample_count=len(y),sample_positive=int(y.sum()),sample_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            original_five_sample_sha256=prior['sample_x_sha256'],sample_y_sha256=prior['sample_y_sha256'],weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())
        write(ROOT/('fit_attempt_'+name+'.json'),dict(policy=name,attempt=attempt,started=datetime.now(timezone.utc).isoformat()))
        try:model=fit_model(x,y,w,2)
        except RuntimeError as e:
            write(ROOT/('failed_'+name+'.json'),dict(error=str(e),no_retry=True));raise
        del x,y,w;negatives=[];print('Phase71 fitted',name,model['iterations'],flush=True)
        for row in rows:
            xx,yy=full_inputs(row,name);negatives.append(predict_logit(xx,model)[~yy])
        model['threshold']=group_threshold(negatives,.01);model['fit_macro_fpr']=float(np.mean([(a>=model['threshold']).mean() for a in negatives]));assert model['fit_macro_fpr']<=.01
        model['n_features']=8 if name=='RGB_local' else 7;models[name]=model;write(ROOT/('fitted_'+name+'.json'),model);del negatives
        print('Phase71 calibrated',name,model['threshold'],model['fit_macro_fpr'],flush=True)
    write(ROOT/'fit_input_audit.json',dict(policies=audits,attempts=4,original_five_sample_sha256=prior['sample_x_sha256'],sample_y_sha256=prior['sample_y_sha256']))
    write(ROOT/'policy_lock.json',dict(models=models,fit_attempts=4,status='all_four_controls_frozen_before_new_development_evaluation',
        input_lock_sha256=digest(ROOT/'input_lock.json'),feature_index_sha256=digest(ROOT/'feature_index.json'),fit_input_audit_sha256=digest(ROOT/'fit_input_audit.json'),
        baseline_reproduction_sha256=digest(ROOT/'baseline_reproduction.json'),no_new_promotion_rule=True,phase65c_authorized=False))
    print('Phase71 ALL four controls frozen',flush=True)


def evaluate():
    verify();lock=read(ROOT/'policy_lock.json');initial=read(ROOT/'input_lock.json')
    assert digest(ROOT/'input_lock.json')==lock['input_lock_sha256'] and hashes()==initial['code_sha256']
    assert digest(PROTOCOL)==initial['protocol_sha256']
    for name in ['feature_index','fit_input_audit','baseline_reproduction']:assert digest(ROOT/(name+'.json'))==lock[name+'_sha256']
    index=read(ROOT/'feature_index.json');dev=records('development_val');sources,sha=development_indices()
    assert sha==initial['development_cache_index_sha256']
    a=read(ROOT/'baseline_reproduction.json');assert a['all_32_exact']
    names=['L3','B']+list(CONTROLS);rows={n:[] for n in names};pool={n:[] for n in rows};labels=[]
    for entry in index['development']:
        product=entry['product'];r=dev[product];truth=np.load(PREP/r['mask_path']);valid=truth!=255;ns=load(OLD,sources[1][product],'nonshadow')[valid]
        for name in rows:
            xx,y=full_inputs(entry,name);model=lock['models'][name];score=predict_logit(xx,model)
            pred=corrected(score,model['threshold'],ns);ranking=brief(curve(score,y));cm=confusion(truth[valid],pred)
            if name in ['L3','B']:
                reference=a['rows'][name][len(labels)];assert product==reference['product']
                assert ranking['ap']==reference['ap'] and np.array_equal(cm,reference['confusion']) and hashlib.sha256(score.tobytes()).hexdigest()==reference['score_sha256']
            rows[name].append(dict(product=product,group_id=r['group_id'],policy=name,threshold=model['threshold'],
                ranking=ranking,**measures(cm),predicted_shadow_fraction=float((pred==2).mean())))
            pool[name].append(score)
        labels.append(y);print('Phase71 development policies',len(labels),32,flush=True)
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
        print('Phase71 pooled ranking',name,flush=True)
    comparisons={}
    for left,right in [('B',n) for n in CONTROLS]+[(n,'L3') for n in ['B']+list(CONTROLS)]:
        pairs=[]
        for l,r in zip(rows[left],rows[right]):
            assert l['group_id']==r['group_id']
            if l['positive']:pairs.append(dict(product=l['product'],group_id=l['group_id'],
                delta_ap=l['ranking']['ap']-r['ranking']['ap'],delta_recall=l['recall']-r['recall'],delta_fpr=l['fpr']-r['fpr']))
        delta=np.array([r['delta_ap'] for r in pairs])
        comparisons[left+'_minus_'+right]=dict(pairs=pairs,ap=interval(delta),recall=interval([r['delta_recall'] for r in pairs]),
            improved_ap=int((delta>.01).sum()),stable_ap=int((np.abs(delta)<=.01).sum()),declined_ap=int((delta<-.01).sum()),
            worst_ap=min(pairs,key=lambda r:r['delta_ap']),worst_recall=min(pairs,key=lambda r:r['delta_recall']))
    report=dict(status='completed_four_fixed_B_controls',policies=policies,comparisons=comparisons,fit_groups=153,development_groups=32,shadow_development_groups=7,
        new_shallow_fits=4,new_segmenter_forwards=0,baseline_L3_B_exact=True,policy_lock_sha256=digest(ROOT/'policy_lock.json'),
        confirmation_pixels_scores_metrics_read=False,original_h1_failure_unchanged=True,phase70_B_not_promoted_unchanged=True,phase65c_authorized=False,
        statistics='10000 paired related-group bootstrap seed71010; descriptive seven-positive-group development controls only',
        prespecified_damage_tiles=['T35RPM','T32TML','T40XDR'])
    write(ROOT/'delivery/phase71_report.json',report)
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
    for name in ['input_lock','policy_lock','feature_index','fit_input_audit','baseline_reproduction']:write(ROOT/'delivery'/(name+'.json'),read(ROOT/(name+'.json')))
    print('Phase71 completed four controls; no additional experiments',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','fit','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'fit':fit,'evaluate':evaluate}[args.action]()
