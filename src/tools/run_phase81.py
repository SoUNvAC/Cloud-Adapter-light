"""Frozen crossed readouts; development caches only, no fit or inference."""
import csv
import hashlib
import io
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from phase80_common import DATA, ROOT as PREVIOUS, PREP, OLD, CONTROL, REPO, MANIFEST_SHA, read, write, digest
from run_phase65d_recovery import records, prepare_verify, load
from phase65d_recovery_math import logit_score, predict_logit
from phase65d_metrics import curve, confusion, measures, corrected
from verify_phase78_local import independent_AP

ROOT=DATA/'phase81_20261011'
PROTOCOL='src/research_plans/PHASE81_CROSSED_READOUT_20261011.md'
NAMES=('MM','LL','LM','ML')

def interval(values):
    x=np.asarray(values,np.float64)
    assert len(x) and np.isfinite(x).all()
    means=x[np.random.default_rng(81011).integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                upper95_one_sided=float(np.quantile(means,.95)),n_groups=len(x),seed=81011,resamples=10000,descriptive_only=True)

def cross(scores,thresholds,winners):
    assert set(scores)==set(thresholds)==set(winners)=={'M','L'}
    for v in winners.values():assert np.isin(v,[0,1]).all()
    return {n:corrected(scores[n[0]],thresholds[n[0]],winners[n[1]]) for n in NAMES}

def counts(truth,pred):
    # Independent explicit boolean counts, separate from bincount implementation.
    return np.array([[np.count_nonzero((truth==i)&(pred==j)) for j in range(3)] for i in range(3)])

def checks(a,b):
    for k in ('positive','negative','true_positive','false_positive','recall','fpr','precision'):
        assert a[k]==b[k],k
    assert a['iou_percent'][2]==b['iou_percent'][2]
    assert a['ap']==b['ap']

def group_row(n,r,t,p,ap):
    cm=confusion(t,p);assert np.array_equal(cm,counts(t,p))
    out=dict(policy=n,product=r['product'],group_id=r['group_id'],ap=ap,**measures(cm))
    out.update(valid_pixels=len(t),reference_support=np.bincount(t,minlength=3).tolist(),
               defined_iou_classes=sum(v is not None for v in out['iou_percent']),
               surface_to_cloud=int(cm[0,1]),cloud_to_surface=int(cm[1,0]),
               predicted_shadow_fraction=float(np.mean(p==2)))
    return out

def csvwrite(name,rows):
    with (ROOT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    assert not ROOT.exists(),'Never overwrite an existing run'
    bindings=read(REPO/'src/research_plans/PHASE81_INPUT_BINDINGS_20261011.json')
    for rel,h in bindings['phase80'].items():assert digest(PREVIOUS/rel)==h,rel
    assert digest(CONTROL/'control_lock.json')==bindings['phase67_lock_sha256']
    train=read(PREVIOUS/'training_completed.json');lock=read(PREVIOUS/'calibration_lock.json')
    trainlock=read(PREVIOUS/'training_lock.json');ref=read(PREVIOUS/'phase80_report.json')
    assert train['best_iteration']==3500
    assert train['best_checkpoint_sha256']=='e6106b765feee929f98e3c1daa026feb7061384f4abdb7a604bb3b52782fe5e1'
    assert digest(REPO/'src/tools/run_phase80_analysis.py')==lock['analysis_code_sha256']
    for rel,h in trainlock['code_sha256'].items():assert digest(REPO/rel)==h,rel
    models={'M':read(CONTROL/'control_lock.json')['models']['MsRE_spectral'],'L':lock['models']['L+S']}
    thresholds={k:v['threshold'] for k,v in models.items()}
    dev=records('development_val');assert len(dev)==len({r['group_id'] for r in dev.values()})==32
    li=read(PREVIOUS/'development_val/index.json');mi=read(OLD/'development_val/index_msre.json')
    assert li['checkpoint_sha256']==train['best_checkpoint_sha256']
    assert mi['checkpoint_sha256']=='82f88d235948ef0a9202cebaa026e7f9fb1e1ad5c0aa361d9045c3e6991ee16d'
    assert li['manifest_sha256']==mi['manifest_sha256']==MANIFEST_SHA==digest(PREP/'manifest.json')
    indices={'L':{r['product']:r for r in li['rows']},'M':{r['product']:r for r in mi['rows']}}
    assert set(indices['M'])==set(indices['L'])==set(dev)
    audit={r['product']:r for r in read(PREVIOUS/'development_input_audit.json')['rows']}
    ROOT.mkdir()
    code=[PROTOCOL,'src/research_plans/PHASE81_INPUT_BINDINGS_20261011.json','src/tools/run_phase81.py',
          'src/tools/test_phase81.py','src/tools/verify_phase81.py','src/tools/report_phase81.py',
          'src/tools/phase65d_metrics.py','src/tools/phase65d_recovery_math.py','src/tools/run_phase65d_recovery.py']
    write(ROOT/'input_configuration_lock.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          code_sha256={p:digest(REPO/p) for p in code},previous=bindings,models=models,threshold_operator='>=',score_dtype='float32',
          selected=list(dev.values()),training=False,fitting=False,threshold_search=False,new_inference=False,fit_pixels=False,
          confirmation_access=False,work_dirs_transfer=False))
    cache=[];baseline={n:[] for n in ('MM','LL')};inputs=[];verified=[]
    # First reproduce ALL original rows and full aggregates before any crossing.
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in dev.items():
            prepare_verify(r);t=np.load(PREP/r['mask_path']);valid=t!=255;t=t[valid]
            a=audit[product];assert a['prepared_sha256']==r['prepared_sha256']
            assert a['image_member_sha256']==r['member_sha256']['image']
            assert a['msre_cache']==indices['M'][product] and a['lora_cache']==indices['L'][product]
            blob=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(blob).hexdigest()==a['image_member_sha256']
            raw=np.load(io.BytesIO(blob),allow_pickle=False)
            scores={};winners={}
            for k,root in [('M',OLD),('L',PREVIOUS)]:
                network=load(root,indices[k][product],'score');winner=load(root,indices[k][product],'nonshadow')
                assert network.shape==winner.shape==valid.shape and network.dtype==np.float32
                x=np.column_stack([logit_score(network[valid]),raw[...,7][valid],raw[...,11][valid],raw[...,12][valid]]).astype(np.float32)
                scores[k]=predict_logit(x,models[k]);winners[k]=np.array(winner[valid]);assert np.isin(winners[k],[0,1]).all()
                ap=curve(scores[k],t==2)['ap'];other=independent_AP(t==2,scores[k])
                assert (ap is None and other is None) or abs(ap-other)<1e-12
                n=k+k;row=group_row(n,r,t,corrected(scores[k],thresholds[k],winners[k]),ap)
                prior=next(v for v in ref['policies'][{'M':'M+S','L':'L+S'}[k]]['rows'] if v['product']==product)
                assert row['confusion']==prior['confusion'] and ap==prior['ranking']['ap']
                for metric in ('fpr','miou_percent','recall','precision','iou_percent'):assert row[metric]==prior[metric],metric
                baseline[n].append(row)
            cache.append((r,t,scores,winners))
            inputs.append(dict(product=product,prepared_sha256=r['prepared_sha256'],image_member_sha256=a['image_member_sha256'],
                               msre_cache=indices['M'][product],lora_cache=indices['L'][product]))
            print('Phase81 reproduce',len(cache),32,product,flush=True)
    pool_ap={}
    labels=np.concatenate([t==2 for r,t,s,w in cache])
    for n,k in [('MM','M'),('LL','L')]:
        prior=ref['policies'][{'M':'M+S','L':'L+S'}[k]];rows=baseline[n]
        for metric,key in [('ap','macro_ap'),('fpr','macro_fpr'),('miou_percent','macro_miou_percent')]:
            assert float(np.mean([r[metric] for r in rows if r[metric] is not None]))==prior[key]['mean']
        assert measures(np.sum([r['confusion'] for r in rows],axis=0))['miou_percent']==prior['miou_percent']
        pooled=np.concatenate([s[k] for r,t,s,w in cache]);pool_ap[k]=curve(pooled,labels)['ap']
        assert pool_ap[k]==prior['ranking']['ap'] and abs(pool_ap[k]-independent_AP(labels,pooled))<1e-12
    write(ROOT/'reproduction_checks.json',dict(status='all_64_scene_confusions_AP_and_metrics_exact',
          aggregate_macro_AP_FPR_mIoU_and_pooled_AP_mIoU_exact=True,independent_boolean_confusions=True,independent_AP=True))
    rows={n:[] for n in NAMES};decisions=[]
    for r,t,s,w in cache:
        preds=cross(s,thresholds,w)
        assert np.array_equal(preds['LL']==2,preds['LM']==2) and np.array_equal(preds['MM']==2,preds['ML']==2)
        for n in NAMES:
            ap=next(v['ap'] for v in baseline[n[0]*2] if v['product']==r['product'])
            rows[n].append(group_row(n,r,t,preds[n],ap))
        checks(rows['LL'][-1],rows['LM'][-1]);checks(rows['MM'][-1],rows['ML'][-1])
        code=(preds['LL']==2).astype(np.uint8)*2+(preds['MM']==2).astype(np.uint8)
        table=np.array([[np.count_nonzero((code==v)&(t==c)) for c in range(3)] for v in (3,2,1,0)])
        assert table.sum()==len(t)
        decisions.append(dict(product=r['product'],group_id=r['group_id'],counts=table.tolist()))
        verified.append(dict(product=r['product'],binary_masks_identical=True,AP_recall_FPR_precision_shadow_IoU_identical=True))
    policies={}
    for n in NAMES:
        rr=rows[n];shadow=[r for r in rr if r['positive']];zero=[r for r in rr if not r['positive']]
        pooled=measures(np.sum([r['confusion'] for r in rr],axis=0))
        policies[n]=dict(rows=rr,**pooled,pooled_ap=pool_ap[n[0]],macro_ap=interval([r['ap'] for r in shadow]),
             macro_recall=interval([r['recall'] for r in shadow]),macro_fpr=interval([r['fpr'] for r in rr]),
             macro_miou_percent=interval([r['miou_percent'] for r in rr]),
             shadow_fpr=interval([r['fpr'] for r in shadow]),no_shadow_fpr=interval([r['fpr'] for r in zero]),
             fpr_above_one_percent_groups=sum(r['fpr']>.01 for r in rr),
             defined_iou_class_counts={str(c):sum(r['defined_iou_classes']==c for r in rr) for c in (1,2,3)})
    comparisons={};pairs=[]
    for a,b in [('LM','LL'),('ML','MM'),('LM','MM')]:
        key=a+' minus '+b;d=[]
        for x,y in zip(rows[a],rows[b]):
            delta=x['miou_percent']-y['miou_percent'];d.append(delta)
            pairs.append(dict(comparison=key,product=x['product'],group_id=x['group_id'],delta_miou_pp=delta,
                delta_surface_to_cloud=x['surface_to_cloud']-y['surface_to_cloud'],delta_cloud_to_surface=x['cloud_to_surface']-y['cloud_to_surface'],
                delta_shadow_to_surface=x['shadow_to_surface']-y['shadow_to_surface'],delta_shadow_to_cloud=x['shadow_to_cloud']-y['shadow_to_cloud']))
        comparisons[key]=dict(macro_miou_pp=interval(d),pooled_miou_pp=policies[a]['miou_percent']-policies[b]['miou_percent'],
                              worst_group_delta_miou_pp=min(d),groups_declining=sum(v<0 for v in d),groups_improving=sum(v>0 for v in d))
    mm,ll,lm=[policies[n]['miou_percent'] for n in ('MM','LL','LM')]
    outcome='pooled_gap_eliminated' if lm>=mm else ('pooled_gap_partly_recovered' if lm>ll else 'pooled_gap_not_recovered')
    write(ROOT/'input_audit.json',dict(rows=inputs))
    write(ROOT/'invariant_checks.json',dict(status='passed',rows=verified,scope='same_reference_pixels_and_same_predicted_nonshadow_region_within_each_scorer'))
    write(ROOT/'decision_combinations.json',dict(order=['11','10','01','00'],bit_order=['L_shadow','M_shadow'],
          reference_order=['Surface','Cloud','Shadow'],rows=decisions,pooled_counts=np.sum([r['counts'] for r in decisions],axis=0).tolist()))
    report=dict(scope='development_only_crossed_readout_diagnostic',formal_confirmation=False,policies=policies,
                comparisons=comparisons,outcome=outcome,bootstrap_unit='32 related-group representatives',seed=81011)
    write(ROOT/'phase81_report.json',report)
    write(ROOT/'confusion_matrices.json',{n:dict(pooled=p['confusion'],groups=[dict(product=r['product'],confusion=r['confusion']) for r in p['rows']]) for n,p in policies.items()})
    flat=[]
    for n in NAMES:
        for r in rows[n]:
            item={k:v for k,v in r.items() if k not in ('confusion','iou_percent','reference_support')}
            for i,c in enumerate(('Surface','Cloud','Shadow')):
                item[c+'_reference_pixels']=r['reference_support'][i];item[c+'_IoU_percent']=r['iou_percent'][i]
                for j,p in enumerate(('Surface','Cloud','Shadow')):item[c+'_to_'+p]=r['confusion'][i][j]
            flat.append(item)
    csvwrite('development_groups.csv',flat);csvwrite('paired_differences.csv',pairs)
    write(ROOT/'COMPLETED.json',dict(status='completed',report_sha256=digest(ROOT/'phase81_report.json'),
          training=0,fitting=0,new_inference=0,threshold_search=0,development_groups=32,all_invariants_passed=True))
    print('Phase81 completed',outcome,flush=True)

if __name__=='__main__':main()
