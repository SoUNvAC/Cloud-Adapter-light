"""Fit exactly three controls, freeze, then evaluate only original development."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from prepare_phase65a_explore import digest,SPLIT_SHA,SOURCE_SHA
from run_phase65d_recovery import (DATA,PREP,OLD,REPO,ADAPTED_SHA,records,prepare_verify,
                                   indices,load,features,brief)
from phase65d_recovery_math import fit_model,predict_logit,group_threshold,paired_interval
from phase65d_metrics import curve,confusion,measures,corrected
from phase67_controls import CONTROLS,combined,selected,choose

ROOT=DATA/'phase67_20261010'
RECOVERY=DATA/'phase65d_recovery_20261008'
RECOVERY_SHA='bcb91cf56d3b796c22bbf60891fb8d045dd8e6af80f5c42a00818fa72501f31f'
PROTOCOL=REPO/'src/research_plans/PHASE67_20261010.md'


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2,allow_nan=False)


def base_verify():
    assert digest(REPO/'src/work_dirs/phase65a/split_lock.json')==SPLIT_SHA
    assert digest(RECOVERY/'recovery_lock.json')==RECOVERY_SHA
    lock=read(RECOVERY/'recovery_lock.json')
    assert digest(PREP/'manifest.json')==lock['prepared_manifest_sha256']
    for rel,sha in lock['fit_input_sha256'].items():assert digest(RECOVERY/rel)==sha
    for name,sha in lock['code_sha256'].items():assert digest(Path(__file__).with_name(name))==sha
    assert lock['source_checkpoint_sha256']==SOURCE_SHA and lock['msre_checkpoint_sha256']==ADAPTED_SHA
    assert not {r['group_id'] for r in records('fit_representatives').values()} & {r['group_id'] for r in records('development_val').values()}
    return lock


def old_features(row):
    path=RECOVERY/row['feature_path'];assert digest(path)==row['feature_sha256']
    return np.load(path,mmap_mode='r')


def full_fit_features(row):
    rgb=ROOT/row['rgb_path'];assert digest(rgb)==row['rgb_sha256']
    return combined(old_features(row),np.load(rgb,mmap_mode='r'))


def fit():
    base=base_verify();assert ROOT.is_dir()
    assert not (ROOT/'control_lock.json').exists()
    dest=ROOT/'fit_rgb';dest.mkdir(exist_ok=False)
    previous=read(RECOVERY/'fit_feature_index.json')
    oldrows={r['product']:r for r in previous['rows']}
    fitrows=records('fit_representatives');assert set(oldrows)==set(fitrows)
    samples=[];ys=[];weights=[];rows=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in fitrows.items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            rgb=raw[...,[3,2,1]][valid].astype(np.float32)
            original=oldrows[product];x=combined(old_features(original),rgb)
            assert len(x)==original['valid_pixels']==int(valid.sum())
            path=dest/(product+'.npy');np.save(path,rgb,allow_pickle=False)
            seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
            idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
            assert hashlib.sha256(idx.tobytes()).hexdigest()==original['sample_indices_sha256']
            y=truth[valid]==2
            assert len(idx)==original['sample_count'] and int(y[idx].sum())==original['sample_positive']
            samples.append(x[idx]);ys.append(y[idx]);weights.append(np.full(len(idx),1/(153*len(idx))))
            rows.append(dict(**original,rgb_path=str(path.relative_to(ROOT)),rgb_sha256=digest(path)))
            print('Phase67 fit inputs',len(rows),153,product,flush=True)
    x=np.concatenate(samples).astype(np.float64);y=np.concatenate(ys).astype(float);w=np.concatenate(weights)
    assert len(rows)==153 and len(x)==base['fit_samples']==previous['sample_count']
    assert hashlib.sha256(x[:,:5].tobytes()).hexdigest()==previous['sample_x_sha256']
    assert hashlib.sha256(y.tobytes()).hexdigest()==previous['sample_y_sha256']
    audit=dict(rows=rows,combined_sample_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
               original_five_sample_sha256=previous['sample_x_sha256'],sample_y_sha256=previous['sample_y_sha256'],
               sample_count=len(x),sample_positive=int(y.sum()),equal_group_weights=True)
    write(ROOT/'fit_input_audit.json',audit)
    models={}
    for name,spec in CONTROLS.items():
        model=fit_model(selected(x,name),y,w,spec['positive_features']);negatives=[]
        for row in rows:
            xx=full_fit_features(row);r=fitrows[row['product']];truth=np.load(PREP/r['mask_path']);yy=truth[truth!=255]==2
            score=predict_logit(selected(xx,name),model);negatives.append(score[~yy])
        model['threshold']=group_threshold(negatives,.01)
        model['fit_macro_fpr']=float(np.mean([(a>=model['threshold']).mean() for a in negatives]))
        assert model['fit_macro_fpr']<=.01+1e-10
        model['nonshadow']=spec['nonshadow'];model['columns']=spec['columns'];models[name]=model
        del negatives
        print('Phase67 frozen control',name,model['fit_macro_fpr'],model['iterations'],flush=True)
    code={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('phase67_controls.py'),
          Path(__file__).with_name('phase65d_recovery_math.py'),Path(__file__).with_name('phase65d_metrics.py'),
          Path(__file__).with_name('run_phase65d_recovery.py'),Path(__file__).with_name('prepare_phase65a_explore.py'),
          Path(__file__).with_name('phase66_statistics.py')]}
    write(ROOT/'control_lock.json',dict(status='three_controls_frozen_before_development',models=models,
        recovery_lock_sha256=RECOVERY_SHA,original_split_sha256=SPLIT_SHA,protocol_sha256=digest(PROTOCOL),
        code_sha256=code,fit_input_audit_sha256=digest(ROOT/'fit_input_audit.json'),
        reference_report_sha256=digest(RECOVERY/'recovery_report.json'),
        prepared_manifest_sha256=digest(PREP/'manifest.json'),fit_input_sha256=base['fit_input_sha256'],
        source_checkpoint_sha256=SOURCE_SHA,msre_checkpoint_sha256=ADAPTED_SHA,
        scope='fit and development only; no independent validation authorized',development_fitting=False,
        no_phase66_or_other_confirmation_access=True,phase65c_authorized=False,
        selection_rule='development macro AP maximum among candidates with actual macro FPR<=1%; tie <=1e-12: fewer inputs then lexicographic name'))


def evaluate():
    base=base_verify();lock=read(ROOT/'control_lock.json')
    assert lock['status']=='three_controls_frozen_before_development'
    assert digest(PROTOCOL)==lock['protocol_sha256']
    for name,sha in lock['code_sha256'].items():assert digest(Path(__file__).with_name(name))==sha
    assert digest(ROOT/'fit_input_audit.json')==lock['fit_input_audit_sha256']
    md=records('development_val');sources=[];index_sha={}
    for name in ['source','msre']:
        path=OLD/'development_val'/('index_'+name+'.json');d=read(path);index_sha[name]=digest(path)
        assert d['checkpoint_sha256']==(SOURCE_SHA if name=='source' else ADAPTED_SHA)
        assert d['manifest_sha256']==digest(PREP/'manifest.json')
        rows={r['product']:r for r in d['rows']};assert set(rows)==set(md);sources.append(rows)
    old_report=read(RECOVERY/'recovery_report.json')
    assert digest(RECOVERY/'recovery_report.json')==lock['reference_report_sha256']
    assert old_report['recovery_lock_sha256']==RECOVERY_SHA
    assert index_sha==old_report['development_cache_index_sha256']
    reference={r['product']:r for r in old_report['policies']['msre_source_spectral']['rows']}
    assert set(reference)==set(md)
    names=['L3',*CONTROLS];rows={n:[] for n in names};pool={n:[] for n in names};ys=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in md.items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;yy=truth[valid]==2
            sr=sources[0][product];mr=sources[1][product]
            assert sr['group_id']==mr['group_id']==r['group_id']
            for prior in [sr,mr]:
                assert digest(OLD/prior['truth_path'])==prior['truth_sha256']
                assert np.array_equal(np.load(OLD/prior['truth_path']),truth)
            ss=load(OLD,sr,'score');ms=load(OLD,mr,'score')
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            xx=combined(features(raw,ms,ss,valid),raw[...,[3,2,1]][valid])
            nonshadow={'source':load(OLD,sr,'nonshadow')[valid],'msre':load(OLD,mr,'nonshadow')[valid]}
            for name in names:
                if name=='L3':model=base['models']['msre_source_spectral'];x=xx[:,:5];ns=nonshadow['msre']
                else:model=lock['models'][name];x=selected(xx,name);ns=nonshadow[model['nonshadow']]
                score=predict_logit(x,model);pred=corrected(score,model['threshold'],ns)
                cm=confusion(truth[valid],pred);ranking=brief(curve(score,yy))
                if name=='L3':
                    assert np.array_equal(score,np.load(RECOVERY/'development_scores'/(product+'_msre_source_spectral.npy')))
                    assert np.array_equal(cm,np.asarray(reference[product]['confusion']))
                    assert ranking['ap']==reference[product]['ranking']['ap']
                row=dict(product=product,group_id=r['group_id'],policy=name,ranking=ranking,**measures(cm),
                         predicted_shadow_fraction=float((pred==2).mean()))
                rows[name].append(row);pool[name].append(score)
            ys.append(yy);print('Phase67 development',len(ys),32,product,flush=True)
    labels=np.concatenate(ys);policies={}
    for name in names:
        rr=rows[name];supported=[r for r in rr if r['ranking']['ap'] is not None]
        nons=[r for r in rr if not r['positive']]
        policies[name]=dict(**measures(np.sum([r['confusion'] for r in rr],axis=0)),
            ranking=brief(curve(np.concatenate(pool.pop(name)),labels)),rows=rr,n_groups=len(rr),
            shadow_supported_groups=len(supported),n_features=5 if name=='L3' else len(CONTROLS[name]['columns']),
            macro_ap=float(np.mean([r['ranking']['ap'] for r in supported])) if supported else None,
            macro_recall=float(np.mean([r['recall'] for r in supported])) if supported else None,
            macro_fpr=float(np.mean([r['fpr'] for r in rr if r['fpr'] is not None])),
            macro_precision=float(np.mean([r['precision'] for r in rr if r['precision'] is not None])),
            precision_defined_groups=sum(r['precision'] is not None for r in rr),
            macro_miou_percent=float(np.mean([r['miou_percent'] for r in rr])),
            no_shadow_groups=len(nons),no_shadow_macro_fpr=float(np.mean([r['fpr'] for r in nons])) if nons else None)
        print('Phase67 pooled',name,flush=True)
    from phase66_statistics import fpr_constraint
    # This imports a generic statistics function only, never any Phase66 data or lock.
    for p in policies.values():p['fpr_constraint_descriptive']=fpr_constraint([r['fpr'] for r in p['rows'] if r['fpr'] is not None])
    comparisons={}
    for name in CONTROLS:
        pairs=[]
        for l3,control in zip(rows['L3'],rows[name]):
            assert l3['group_id']==control['group_id']
            if l3['ranking']['ap'] is not None:
                pairs.append(dict(product=l3['product'],group_id=l3['group_id'],delta_ap=l3['ranking']['ap']-control['ranking']['ap'],
                                  delta_recall=l3['recall']-control['recall'],delta_fpr=l3['fpr']-control['fpr']))
        delta=np.asarray([r['delta_ap'] for r in pairs])
        comparisons['L3_minus_'+name]=dict(pairs=pairs,ap=paired_interval(delta),
            improved=int((delta>.01).sum()),stable=int((np.abs(delta)<=.01).sum()),declined=int((delta<-.01).sum()),
            worst_ap=min(pairs,key=lambda r:r['delta_ap']),worst_recall=min(pairs,key=lambda r:r['delta_recall']),
            scope='post-hoc development descriptive, not independent confirmation')
    report=dict(status='completed_exactly_three_fit_development_controls',policies=policies,comparisons=comparisons,
        selected_on_development=choose(policies),fit_groups=153,development_groups=32,shadow_development_groups=7,
        new_segmenter_forwards=0,new_shallow_fits=3,control_lock_sha256=digest(ROOT/'control_lock.json'),
        original_reference_sha256=digest(RECOVERY/'recovery_report.json'),development_cache_index_sha256=index_sha,
        baseline_reference={n:p for n,p in old_report['policies'].items() if n!='msre_source_spectral'},
        original_l3_reproduced_exactly=True,confirmation_pixels_scores_metrics_read=False,
        original_h1_failure_unchanged=True,phase65c_authorized=False)
    write(ROOT/'delivery/phase67_report.json',report)
    table=[]
    csv_policies={**report['baseline_reference'],**policies}
    for name,p in csv_policies.items():
        for r in p['rows']:
            table.append(dict(policy=name,product=r['product'],group_id=r['group_id'],ap=r['ranking']['ap'],
                **{k:r[k] for k in ['positive','negative','recall','precision','fpr','miou_percent','shadow_to_surface','shadow_to_cloud','predicted_shadow_fraction']},
                **{'iou_'+str(i):v for i,v in enumerate(r['iou_percent'])},
                oracle_recall_fpr01=r['ranking']['matched_fpr'].get('0.01',{}).get('recall'),
                oracle_recall_fpr05=r['ranking']['matched_fpr'].get('0.05',{}).get('recall')))
    with (ROOT/'delivery/development_groups.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write(ROOT/'delivery/control_lock.json',lock)
    write(ROOT/'delivery/fit_input_audit.json',read(ROOT/'fit_input_audit.json'))
    print('Phase67 completed; development-selected:',report['selected_on_development'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','evaluate']);a=p.parse_args()
    if a.action=='fit':fit()
    else:evaluate()


if __name__=='__main__':main()
