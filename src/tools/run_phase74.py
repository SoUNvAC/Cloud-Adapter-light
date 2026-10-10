"""Phase74: exactly two fit-only shallow fits, original development only."""
import csv,hashlib,io,json,os,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import run_phase70 as p
from run_phase67 import read,write,RECOVERY,base_verify
from run_phase65d_recovery import DATA,PREP,OLD,REPO,records,prepare_verify,indices,load,brief
from prepare_phase65a_explore import digest,SPLIT_SHA
from phase65d_recovery_math import fit_model,predict_logit,group_threshold
from phase65d_metrics import curve,confusion,measures,corrected
from phase70_math import checked,local_contrast
from phase74_math import contrast,shuffled_mask,interval,changes,distribution,continue_conditions

ROOT=DATA/'phase74_20261010'
PROTOCOL=REPO/'src/research_plans/PHASE74_CLOUD_MIXING_20261010.md'
NAMES=['L3','B','B_E_fixed','B_R_fixed','B_E_fit','B_R_fit']
CODE=['run_phase74.py','phase74_math.py','test_phase74.py','run_phase74.sh',
      'run_phase70.py','phase70_math.py','run_phase67.py','run_phase65d_recovery.py',
      'phase65d_recovery_math.py','phase65d_metrics.py','prepare_phase65a_explore.py']


def hashes():return {n:digest(Path(__file__).with_name(n)) for n in CODE}


def csvfile(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def budget():
    if datetime.now(timezone.utc)>=datetime(2026,10,19,16,tzinfo=timezone.utc):raise RuntimeError('Budget expired')


def baseline():
    base_verify()
    expected={'policy_lock.json':'5bc0b8911b3010a5276546f72d445888d96f9e0935cdafbadac986192b570df9',
        'feature_index.json':'2d1a20225539226ae8ad5b4e732f75a6811c2edb67622f8c8684b057051c8b94',
        'delivery/phase70_report.json':'6a22606d12a0de69a20ed06625934401262214db52f3e667f656ec3da624bbd2'}
    for n,h in expected.items():assert digest(p.ROOT/n)==h,(n,'baseline SHA drift')
    lock=read(p.ROOT/'policy_lock.json');index=read(p.ROOT/'feature_index.json')
    return lock['models'],index,read(p.ROOT/'delivery/phase70_report.json'),expected


def reproduce(models,index,reference):
    dev=records('development_val');sources,sha=p.development_indices();rows=[]
    for row in index['development']:
        product=row['product'];r=dev[product];prepare_verify(r)
        truth=np.load(PREP/r['mask_path']);v=truth!=255;y=truth[v]==2
        ns=load(OLD,sources[1][product],'nonshadow')[v];x,yy=p.full_inputs(row,'B');assert np.array_equal(y,yy)
        for name,n in [('L3',5),('B',8)]:
            score=predict_logit(x[:,:n],models[name]);cm=confusion(truth[v],corrected(score,models[name]['threshold'],ns))
            ap=brief(curve(score,y))['ap'];ref=next(a for a in reference['policies'][name]['rows'] if a['product']==product)
            assert ap==ref['ranking']['ap'] and np.array_equal(cm,ref['confusion']),('reproduction',product,name)
            rows.append(dict(product=product,group_id=r['group_id'],policy=name,ap=ap,confusion=cm.tolist(),score_sha256=hashlib.sha256(score.tobytes()).hexdigest()))
        print('Phase74 baseline reproduced',len(rows)//2,32,flush=True)
    write(ROOT/'baseline_reproduction.json',dict(all_32_L3_and_B_exact=True,rows=rows,development_index_sha256=sha))


def prepare(models,index,expected):
    fit_source,fit_msre=indices(RECOVERY);dev_sources,devsha=p.development_indices()
    missing=[];cache_hashes={}
    for cohort,root,source in [('fit',RECOVERY,fit_source),('development',OLD,dev_sources[0])]:
        for product,row in source.items():
            entry=row.get('files',{}).get('pred')
            if not entry or not (root/entry['path']).is_file():missing.append([cohort,product,'Source argmax absent']);continue
            if digest(root/entry['path'])!=entry['sha256']:missing.append([cohort,product,'Source argmax SHA mismatch'])
            cache_hashes[cohort+'/'+product]=entry
    write(ROOT/'source_cache_preflight.json',dict(missing=missing,files=cache_hashes))
    if missing:raise RuntimeError('Missing Source argmax; no automatic forward')
    write(ROOT/'input_lock.json',dict(protocol_sha256=digest(PROTOCOL),code_sha256=hashes(),baseline_sha256=expected,
        original_split_sha256=SPLIT_SHA,prepared_manifest_sha256=digest(PREP/'manifest.json'),
        source_cache_preflight_sha256=digest(ROOT/'source_cache_preflight.json'),development_index_sha256=devsha,
        policies=NAMES,max_fits=2,window=61,min_neighbors=16,channels=[7,11,12],epsilon=1e-6,
        E='Source cached three-class argmax ==1; image-valid neighbors only; center never removed',
        R='joint image-valid E permutation seed74010+SHA256(product)[:8]; whole-image counts only',
        gates=['E_fit all macroFPR < B','E_fit macroRecall >= B-.01','E_fit macroAP >= B',
               'E_fit macroAP > R_fit','E_fit all macroFPR <= R_fit'],
        fitting='original uniform sample indices; equal-group BCE; ridge .001; original LBFGSB; first two coefficients>=0',
        calibration='all fit negatives; exact ties >=; macro related-group FPR <=.01',
        bootstrap_seed=74010,bootstrap_replicates=10000,new_segmenter_forwards=0,confirmation_access=False))
    dest=ROOT/'features';dest.mkdir();audits=[];new_index={};interventions=[];strata=[];fit_baselines=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for cohort,selected,source,msre,cache in [('fit',records('fit_representatives'),fit_source,fit_msre,RECOVERY),
            ('development',records('development_val'),dev_sources[0],dev_sources[1],OLD)]:
            new_index[cohort]=[]
            for number,row in enumerate(index[cohort],1):
                product=row['product'];r=selected[product];prepare_verify(r)
                iv=np.load(PREP/r['image_valid_path']);truth=np.load(PREP/r['mask_path']);v=truth!=255
                selection=truth[iv]!=255;y=truth[v]==2;x,yy=p.full_inputs(row,'B');assert np.array_equal(y,yy)
                b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
                raw=np.load(io.BytesIO(b));assert np.array_equal(checked(local_contrast(raw,iv))[selection],x[:,5:])
                pred=load(cache,source[product],'pred');assert pred.shape==iv.shape and np.isin(pred,[0,1,2]).all()
                e=pred==1;random,perm=shuffled_mask(product,iv,e);ns=load(cache,msre[product],'nonshadow')[v]
                l3=predict_logit(x[:,:5],models['L3']);bs=predict_logit(x,models['B'])
                bpos=bs>=models['B']['threshold'];lpos=l3>=models['L3']['threshold']
                masks={'B_vs_L3_new_FP':~y & ~lpos & bpos,'B_vs_L3_rescued_TP':y & ~lpos & bpos}
                fit_baselines.append(dict(cohort=cohort,product=product,group_id=r['group_id'],
                    L3_ap=brief(curve(l3,y))['ap'] if cohort=='fit' else None,B_ap=brief(curve(bs,y))['ap'] if cohort=='fit' else None,
                    L3_confusion=confusion(truth[v],corrected(l3,models['L3']['threshold'],ns)).tolist(),
                    B_confusion=confusion(truth[v],corrected(bs,models['B']['threshold'],ns)).tolist()))
                outrow=dict(product=product,group_id=r['group_id'],cohort=cohort,original=row,variants={})
                for variant,excluded in [('E',e),('R',random)]:
                    values,means,oldmeans,a=contrast(raw,iv,excluded)
                    path=dest/(product+'_'+variant+'.npy');np.save(path,values[selection],allow_pickle=False)
                    outrow['variants'][variant]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
                    newx=np.column_stack([x[:,:5],values[selection]]);score=predict_logit(newx,models['B'])
                    cross={str(c):int((iv & excluded & (truth==c)).sum()) for c in [0,1,2,255]}
                    audits.append(dict(cohort=cohort,product=product,group_id=r['group_id'],variant=variant,
                        permutation=perm if variant=='R' else None,excluded_gt_crosscount_offline_only=cross,
                        excluded_image_valid=int(excluded[iv].sum()),image_valid=int(iv.sum()),evaluation_valid=int(v.sum()),
                        center_evaluation_unchanged=True,remaining_image_valid=distribution(a['remaining']),
                        remaining_evaluation_valid=distribution(a['remaining'][selection]),
                        excluded_neighbor_fraction=distribution(1-a['remaining']/a['original']),
                        fallback_image_valid_fraction=float(a['fallback'].mean()),fallback_eval_fraction=float(a['fallback'][selection].mean())))
                    interventions.append(dict(cohort=cohort,product=product,group_id=r['group_id'],variant=variant,
                        local_mean_delta=[distribution(means[selection,i]-oldmeans[selection,i]) for i in range(3)],
                        local_feature_delta=[distribution(values[selection,i]-x[:,5+i]) for i in range(3)],
                        logit_delta=distribution(score-bs),**changes(bpos,score>=models['B']['threshold'],y)))
                    for label,mask in masks.items():strata.append(dict(cohort=cohort,product=product,group_id=r['group_id'],
                        variant=variant,stratum=label,**distribution((score-bs)[mask])))
                    del values,means,oldmeans,newx
                new_index[cohort].append(outrow)
                print('Phase74 features',cohort,number,len(selected),flush=True)
    write(ROOT/'feature_index.json',new_index);write(ROOT/'neighbor_audit.json',audits)
    write(ROOT/'fixed_interventions.json',interventions);write(ROOT/'original_B_L3_strata.json',strata)
    write(ROOT/'fit_baseline_results.json',fit_baselines)


def inputs(row,variant):
    x,y=p.full_inputs(row['original'],'B');e=row['variants'][variant];path=ROOT/e['path'];assert digest(path)==e['sha256']
    return np.column_stack([x[:,:5],np.load(path,mmap_mode='r')]),y


def fit(models):
    budget();lock=read(ROOT/'input_lock.json');assert hashes()==lock['code_sha256']
    assert read(ROOT/'baseline_reproduction.json')['all_32_L3_and_B_exact']
    index=read(ROOT/'feature_index.json');prior=read(RECOVERY/'fit_feature_index.json');audits={}
    for attempt,variant in enumerate(['E','R'],1):
        budget();samples=[];ys=[];weights=[];rows=[]
        for row in index['fit']:
            x,y=inputs(row,variant);old=row['original']['original_features'];product=row['product']
            seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
            idx=np.random.default_rng(seed).choice(len(y),min(8192,len(y)),replace=False)
            assert hashlib.sha256(idx.tobytes()).hexdigest()==old['sample_indices_sha256'] and int(y[idx].sum())==old['sample_positive']
            samples.append(x[idx]);ys.append(y[idx]);weights.append(np.full(len(idx),1/(153*len(idx))))
            rows.append(dict(product=product,group_id=row['group_id'],indices_sha256=old['sample_indices_sha256'],count=len(idx)))
        x=np.concatenate(samples).astype(np.float64);y=np.concatenate(ys).astype(float);w=np.concatenate(weights);del samples
        assert hashlib.sha256(x[:,:5].tobytes()).hexdigest()==prior['sample_x_sha256']
        assert hashlib.sha256(y.tobytes()).hexdigest()==prior['sample_y_sha256']
        audits[variant]=dict(rows=rows,x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),y_sha256=prior['sample_y_sha256'],
            weight_sha256=hashlib.sha256(w.tobytes()).hexdigest())
        write(ROOT/('fit_attempt_'+variant+'.json'),dict(attempt=attempt,variant=variant,started=datetime.now(timezone.utc).isoformat()))
        model=fit_model(x,y,w,2);del x,y,w;negative=[]
        print('Phase74 fitted',variant,model['iterations'],flush=True)
        for row in index['fit']:
            x,y=inputs(row,variant);negative.append(predict_logit(x,model)[~y])
        model['threshold']=group_threshold(negative,.01);model['fit_macro_fpr']=float(np.mean([(a>=model['threshold']).mean() for a in negative]))
        assert model['fit_macro_fpr']<=.01;models['B_'+variant+'_fit']=model;del negative
        write(ROOT/('fitted_'+variant+'.json'),model);print('Phase74 calibrated',variant,model['threshold'],flush=True)
    models['B_E_fixed']=models['B'];models['B_R_fixed']=models['B'];write(ROOT/'fit_audit.json',audits)
    write(ROOT/'policy_lock.json',dict(models=models,fit_attempts=2,development_used_for_selection_or_calibration=False,
        frozen_before_new_development_evaluation=True,input_lock_sha256=digest(ROOT/'input_lock.json'),
        feature_index_sha256=digest(ROOT/'feature_index.json'),fit_audit_sha256=digest(ROOT/'fit_audit.json')))


def evaluate():
    lock=read(ROOT/'policy_lock.json');initial=read(ROOT/'input_lock.json')
    assert hashes()==initial['code_sha256'] and digest(PROTOCOL)==initial['protocol_sha256']
    for n in ['input_lock','feature_index','fit_audit']:assert digest(ROOT/(n+'.json'))==lock[n+'_sha256']
    models=lock['models'];index=read(ROOT/'feature_index.json');dev=records('development_val');sources,h=p.development_indices()
    rows={n:[] for n in NAMES};pool={n:[] for n in NAMES};labels=[]
    for number,row in enumerate(index['development'],1):
        product=row['product'];r=dev[product];truth=np.load(PREP/r['mask_path']);v=truth!=255;y=truth[v]==2
        ns=load(OLD,sources[1][product],'nonshadow')[v];x,yy=p.full_inputs(row['original'],'B')
        scores={n:predict_logit(x[:,:5] if n=='L3' else x,models[n]) for n in ['L3','B']}
        for variant in ['E','R']:
            xx,yy=inputs(row,variant)
            for mode in ['fixed','fit']:
                name='B_'+variant+'_'+mode;scores[name]=predict_logit(xx,models[name])
        for name in NAMES:
            score=scores[name];pred=corrected(score,models[name]['threshold'],ns);cm=confusion(truth[v],pred)
            rows[name].append(dict(product=product,group_id=r['group_id'],policy=name,threshold=models[name]['threshold'],
                ranking=brief(curve(score,y)),**measures(cm),flows_vs_B=changes(scores['B']>=models['B']['threshold'],pred==2,y),
                flows_vs_L3=changes(scores['L3']>=models['L3']['threshold'],pred==2,y)))
            pool[name].append(score)
        labels.append(y);print('Phase74 development',number,32,flush=True)
    all_y=np.concatenate(labels);policies={}
    for name,rr in rows.items():
        pos=[r for r in rr if r['positive']];assert len(pos)==7 and len(rr)==32
        policies[name]=dict(rows=rr,**measures(np.sum([r['confusion'] for r in rr],axis=0)),
            ranking=brief(curve(np.concatenate(pool.pop(name)),all_y)),macro_ap=float(np.mean([r['ranking']['ap'] for r in pos])),
            macro_recall=float(np.mean([r['recall'] for r in pos])),
            fpr_strata={s:interval([r['fpr'] for r in subset if r['fpr'] is not None]) for s,subset in
                [('all',rr),('shadow',pos),('no_shadow',[r for r in rr if not r['positive']])]},
            zero_recall_groups=sum(r['true_positive']==0 for r in pos),groups_fpr_above_1pct=sum(r['fpr']>.01 for r in rr),
            flows_vs_B={k:sum(r['flows_vs_B'][k] for r in rr) for k in rr[0]['flows_vs_B']})
        print('Phase74 pooled',name,flush=True)
    comparisons={};pairs_table=[]
    for left,right in [('B_E_fit','B_R_fit')]+[(n,b) for n in NAMES[2:] for b in ['B','L3']]:
        pairs=[]
        for a,b in zip(rows[left],rows[right]):
            assert a['group_id']==b['group_id']
            pair=dict(left=left,right=right,product=a['product'],group_id=a['group_id'],
                delta_ap=None if not a['positive'] else a['ranking']['ap']-b['ranking']['ap'],
                delta_recall=None if not a['positive'] else a['recall']-b['recall'],delta_fpr=a['fpr']-b['fpr'],
                delta_miou_pp=a['miou_percent']-b['miou_percent'])
            pairs.append(pair);pairs_table.append(pair)
        valid=[r for r in pairs if r['delta_ap'] is not None];delta=np.array([r['delta_ap'] for r in valid])
        comparisons[left+'_minus_'+right]=dict(pairs=pairs,ap=interval(delta),recall=interval([r['delta_recall'] for r in valid]),
            fpr=interval([r['delta_fpr'] for r in pairs]),improved=int((delta>.01).sum()),stable=int((abs(delta)<=.01).sum()),declined=int((delta<-.01).sum()))
    conditions=continue_conditions(policies);strata=read(ROOT/'original_B_L3_strata.json');aggregated=[]
    for cohort in ['fit','development']:
        for variant in ['E','R']:
            for s in ['B_vs_L3_new_FP','B_vs_L3_rescued_TP']:
                q=[r for r in strata if r['cohort']==cohort and r['variant']==variant and r['stratum']==s and r['n']]
                aggregated.append(dict(cohort=cohort,variant=variant,stratum=s,support_groups=len(q),pixels=sum(r['n'] for r in q),
                    equal_group_mean_logit_change=float(np.mean([r['mean'] for r in q])) if q else None,
                    pooled_mean_logit_change=sum(r['mean']*r['n'] for r in q)/sum(r['n'] for r in q) if q else None))
    report=dict(status='completed_exploratory',policies=policies,comparisons=comparisons,continue_conditions=conditions,
        continue_branch=all(conditions.values()),stratum_summary=aggregated,fit_groups=153,development_groups=32,positive_development_groups=7,
        fits=2,new_network_forwards=0,confirmation_access=False,original_h1_failure_unchanged=True,
        geometry='not_ready; no terrain branch',policy_lock_sha256=digest(ROOT/'policy_lock.json'))
    delivery=ROOT/'delivery';delivery.mkdir();write(delivery/'phase74_report.json',report)
    csvfile(delivery/'comparisons_groups.csv',pairs_table)
    csvfile(delivery/'development_groups.csv',[dict(policy=n,product=r['product'],group_id=r['group_id'],ap=r['ranking']['ap'],
        **{k:r[k] for k in ['threshold','positive','negative','recall','precision','fpr','miou_percent','true_positive','false_positive','shadow_to_surface','shadow_to_cloud']},
        **{'iou_'+str(i):v for i,v in enumerate(r['iou_percent'])},**r['flows_vs_B']) for n in NAMES for r in rows[n]])
    csvfile(delivery/'summary.csv',[dict(policy=n,macro_ap=q['macro_ap'],pooled_ap=q['ranking']['ap'],macro_recall=q['macro_recall'],
        macro_fpr=q['fpr_strata']['all']['mean'],fpr_upper95=q['fpr_strata']['all']['upper95'],
        shadow_macro_fpr=q['fpr_strata']['shadow']['mean'],no_shadow_macro_fpr=q['fpr_strata']['no_shadow']['mean'],
        pooled_miou_percent=q['miou_percent'],zero_recall_groups=q['zero_recall_groups'],groups_fpr_above_1pct=q['groups_fpr_above_1pct']) for n,q in policies.items()])
    for name in ['input_lock','source_cache_preflight','policy_lock','feature_index','fit_audit','baseline_reproduction',
                 'neighbor_audit','fixed_interventions','original_B_L3_strata','fit_baseline_results','fitted_E','fitted_R','fit_attempt_E','fit_attempt_R']:
        shutil.copyfile(ROOT/(name+'.json'),delivery/(name+'.json'))
    print('Phase74 complete',conditions,'continue',all(conditions.values()),flush=True)


def main():
    budget();ROOT.mkdir(exist_ok=False)
    models,index,report,expected=baseline()
    # Reproduce the old results before constructing any new development scores.
    reproduce(models,index,report);prepare(models,index,expected);fit(models);evaluate()

if __name__=='__main__':main()
