"""Restore original samples and fixed coefficients; no fitting or neural imports."""
import os
import sys
import traceback
import hashlib
import numpy as np
import auc_phase306 as ranking
from phase308_common import *
from math_phase308 import *
INPUTS={}
def pin(p,expected=None):
    h=sha(p);assert expected is None or h==expected,str(p)
    INPUTS[str(p)]=h
    return h
def locked(p,manifest,base):pin(p,manifest[p.relative_to(base).as_posix()]['sha256'])
def main():
    assert 'torch' not in sys.modules
    acceptance=tests();lock=load(ROOT/'EXECUTION_LOCK.json')
    for n,h in lock['code'].items():pin(R/n,h)
    pin(R/'src/research_plans/phase308.md',lock['protocol_sha256'])
    # Pin all archived parent artifacts, not only the subset used below.
    manifests={}
    for base,phase in ((P,306),(Q,307)):
        manifest=load(base/'BACKUP_FILES.json');manifests[phase]=manifest;pin(base/'BACKUP_FILES.json')
        receipt=load(base/'ARCHIVE_RECEIPT.json');pin(base/'delivery.tar.gz',receipt['sha256'])
        assert load(R/f'src/research_results/phase{phase}_20261025/BACKUP_RECEIPT.json')['sha256']==receipt['sha256']
        for n,v in manifest.items():pin(base/n,v['sha256'])
    m306=manifests[306];m307=manifests[307]
    channel=load(P/'CHANNEL_LOCK.json');assert channel['bands']==list(BANDS) and channel['indices']==list(INDICES)
    samples=read(P/'SUPPORT_AND_SAMPLE_SHA.csv');old=read(Q/'FOLD_METRICS.csv')
    mapped={(r['product'],r['field']):r['value'] for r in read(P/'SCENE_MAPPING.csv')}
    original={r['product']:r for r in read(Q/'GROUP_SAMPLING.csv')}
    top=__import__('json').loads(next(r['products'] for r in read(P/'TOP_1_3_5_CONCENTRATION.csv') if int(r['top_n'])==3))
    assert all(any(t in p for p in top) for t in TOP)
    groups=[]
    for fold,r in enumerate(sorted(samples,key=lambda r:r['group_id'])):
        with np.load(P/r['samples_file']) as z:
            ii=z['flat_indices'];ev=z['event'];raw=z['TOA'];common=z['common_packbits'];six=z['six_common_packbits']
        assert ah(ii)==r['selected_flat_indices_sha256']
        cr=np.flatnonzero(ev!=1);flat=ii[cr];codes=ev[cr];x=raw[cr]
        with np.load(Q/'sampling'/(r['group_id']+'.npz')) as z:
            selected=z['candidate_rows'];assert np.array_equal(z['source_sample_rows'],cr[selected])
            assert np.array_equal(z['flat_indices'],flat[selected]) and np.array_equal(z['y'],(codes[selected]==2).astype(np.uint8))
        sr=original[r['product']];assert ah(flat)==sr['test_C_support_sha256'] and ah(flat[selected])==sr['selected_flat_sha256']
        assert len(np.unique(selected))==len(selected) and np.isin(codes[selected],[0,2]).all()
        groups.append(dict(fold=fold,product=r['product'],group_id=r['group_id'],snow=mapped[r['product'],'snow/ice'],raw=x,flat=flat,codes=codes,selected=selected))
    reproduction=[];composition=[];ranges=[];bounds=[];scores=[];contributions=[];intervals=[]
    maxerr=0.;maxsum=0.;toprecords=[]
    for test in groups:
        fold=test['fold'];train=[g for g in groups if g['group_id']!=test['group_id'] and len(g['selected'])]
        support=load(Q/'models'/f'fold{fold:02d}_support.json')
        assert [g['group_id'] for g in train]==support['train_group_ids'] and test['group_id'] not in support['train_group_ids']
        x=np.concatenate([g['raw'][g['selected']] for g in train]).astype(np.float64)
        y=np.concatenate([(g['codes'][g['selected']]==2).astype(np.float64) for g in train])
        w=np.concatenate([np.full(len(g['selected']),1/(len(g['selected'])*len(train)),np.float64) for g in train]);w/=w.sum()
        h=hashlib.sha256();start=0
        for g,orig in zip(train,support['group_weights']):
            n=len(g['selected']);codes=g['codes'][g['selected']];ww=w[start:start+n];start+=n
            h.update(g['group_id'].encode());h.update(g['product'].encode());h.update(g['flat'][g['selected']].tobytes())
            assert orig['n']==n and orig['D']==int((codes==0).sum()) and orig['K']==int((codes==2).sum())
            for name,code in [('D',0),('K',2)]:
                choose=codes==code;total=float(w[y==(code==2)].sum())
                composition.append(dict(fold=fold,heldout=test['product'],training_group=g['group_id'],training_product=g['product'],snow_ice=g['snow'],label=name,n=int(choose.sum()),original_weight=float(ww[choose].sum()),conditional_weight=float(ww[choose].sum()/total) if total else None))
        assert h.hexdigest()==support['train_support_sha256'] and ah(y)==support['train_labels_sha256'] and ah(w)==support['train_weights_sha256']
        assert len(y)==support['train_samples'] and int((y==0).sum())==support['train_D']
        for model in MODELS:
            prefix=dict(fold=fold,product=test['product'],group_id=test['group_id'],model=model,detailed=test['product'] in top)
            saved=load(Q/'models'/f'fold{fold:02d}_{model}.json');receipt=saved['receipt']
            row=next(r for r in old if int(r['fold'])==fold and r['model']==model)
            with np.load(Q/'oof'/f'fold{fold:02d}_{model}.npz') as z:op=z['p_K'];assert np.array_equal(z['flat_indices'],test['flat'])
            if receipt is None:
                assert not len(op) and saved['status']=='empty_test';error=0.;brier=None;repaired=revoked=0;auc=None
            else:
                xt=features(x,model);xx=features(test['raw'],model)
                mu=np.sum(xt*w[:,None],axis=0);sc=np.sqrt(np.sum((xt-mu)**2*w[:,None],axis=0));constant=np.all(xt==xt[0],axis=0)
                sc[sc==0]=1;mu[constant]=xt[0,constant];sc[constant]=1
                assert np.allclose(mu,receipt['mean'],atol=1e-12,rtol=1e-12) and np.allclose(sc,receipt['scale'],atol=1e-12,rtol=1e-12)
                zt,lt,pt,ct,e1=fixed(xt,receipt);zz,ll,pp,cc,e2=fixed(xx,receipt)
                error=float(np.max(np.abs(pp-op)));assert error<=1e-12
                maxerr=max(maxerr,error);maxsum=max(maxsum,e1,e2)
                d=test['codes']==0;k=test['codes']==2;binary=d|k
                repaired=int((d&(pp<.5)).sum());revoked=int((k&(pp<.5)).sum())
                brier=float(np.mean((pp[binary]-k[binary].astype(np.float64))**2)) if binary.any() else None
                auc=ranking.auc(k[binary],pp[binary]);assert abs(brier-float(row['Brier']))<=1e-12
                assert repaired==int(row['repaired_D']) and revoked==int(row['revoked_K'])
                if row['AUC']=='NA':assert auc is None
                else:assert abs(auc-float(row['AUC']))<=1e-12
                names=list(BANDS[:3]) if model=='RGB3' else list(BANDS[:3])+[b+'^2' for b in BANDS[:3]] if model=='RGB6Q' else list(BANDS)
                # Distinct original TOA, actual model features, and standardized coordinates.
                for space,tx,ex,fnames in [('raw6',x,test['raw'].astype(np.float64),BANDS),('model_raw',xt,xx,names),('standardized',zt,zz,names)]:
                    trstats={}
                    for label,flag in [('D',False),('K',True)]:
                        choose=y==flag
                        trstats[label]=[quant(tx[choose,j],w[choose]) for j in range(tx.shape[1])]
                        for j,fname in enumerate(fnames):
                            ranges.append(dict(**prefix,space=space,population='training_in',label=label,feature=fname,denominator='original fitting samples; conditional group weights',**describe(tx[choose,j],w[choose])))
                    for label,code in [('D',0),('K',2)]:
                        choose=test['codes']==code;ev=ex[choose]
                        for j,fname in enumerate(fnames):ranges.append(dict(**prefix,space=space,population='heldout',label=label,feature=fname,denominator='all heldout class pixels; unweighted',**describe(ev[:,j])))
                        for trainlabel in ('D','K'):
                            low=np.array([s['min'] for s in trstats[trainlabel]]);high=np.array([s['max'] for s in trstats[trainlabel]])
                            l1=np.array([s['p1'] for s in trstats[trainlabel]]);h99=np.array([s['p99'] for s in trstats[trainlabel]])
                            assert low.dtype!=object,'Unexpected empty training class; preserve failure'
                            outside=(ev<low)|(ev>high);tail=(ev<l1)|(ev>h99)
                            for j,fname in enumerate(fnames):
                                a=int(outside[:,j].sum());b=int(tail[:,j].sum())
                                assert a==len(ev)-int(((ev[:,j]>=low[j])&(ev[:,j]<=high[j])).sum())
                                bounds.append(dict(**prefix,space=space,heldout_label=label,training_label=trainlabel,feature=fname,n=len(ev),outside_minmax=a,frac_minmax=a/len(ev) if len(ev) else None,outside_p1p99=b,frac_p1p99=b/len(ev) if len(ev) else None,train_min=float(low[j]),train_p1=float(l1[j]),train_p99=float(h99[j]),train_max=float(high[j])))
                            bounds.append(dict(**prefix,space=space,heldout_label=label,training_label=trainlabel,feature='ANY',n=len(ev),outside_minmax=int(outside.any(1).sum()),frac_minmax=float(outside.any(1).mean()) if len(ev) else None,outside_p1p99=int(tail.any(1).sum()),frac_p1p99=float(tail.any(1).mean()) if len(ev) else None,train_min=None,train_p1=None,train_p99=None,train_max=None))
                    for j,fname in enumerate(fnames):
                        ds=trstats['D'][j];ks=trstats['K'][j]
                        intervals.append(dict(**prefix,space=space,feature=fname,D_min=ds['min'],D_max=ds['max'],K_min=ks['min'],K_max=ks['max'],overlap_min=max(ds['min'],ks['min']),overlap_max=min(ds['max'],ks['max']),intervals_overlap=max(ds['min'],ks['min'])<=min(ds['max'],ks['max'])))
                for pop,labels,l,p,c,weight in [('training_in',y,lt,pt,ct,w),('heldout',test['codes'],ll,pp,cc,None)]:
                    classlist=[('D',0),('K',1)] if pop=='training_in' else [('D',0),('K',2),('C_Shadow',3)]
                    for label,code in classlist:
                        choose=labels==code;ww=weight[choose] if weight is not None else None
                        for quantity,v in [('logit',l),('p_K',p)]:scores.append(dict(**prefix,population=pop,label=label,quantity=quantity,intercept=receipt['theta'][-1],fixed_below_half=int((p[choose]<.5).sum()),Brier=float(np.mean((p[choose]-(code==2))**2)) if pop=='heldout' and label!='C_Shadow' and choose.any() else None,**describe(v[choose],ww)))
                        for j,fname in enumerate(names):contributions.append(dict(**prefix,population=pop,label=label,feature=fname,beta=receipt['theta'][j],intercept=receipt['theta'][-1],**describe(c[choose,j],ww)))
                if test['product'] in top:
                    toprecords.append(dict(**prefix,Brier=brier,repaired_D=repaired,revoked_K=revoked,D=int(d.sum()),K=int(k.sum()),p_D=quant(pp[d]),train_D_groups=support['train_D_groups'],train_K_groups=support['train_K_groups']))
            reproduction.append(dict(**prefix,C=len(op),D=int((test['codes']==0).sum()),K=int((test['codes']==2).sum()),C_Shadow=int((test['codes']==3).sum()),train_D=support['train_D'],train_K=support['train_K'],train_D_groups=support['train_D_groups'],train_K_groups=support['train_K_groups'],original_repaired_D=int(row['repaired_D']),recomputed_repaired_D=repaired,original_revoked_K=int(row['revoked_K']),recomputed_revoked_K=revoked,original_Brier=None if row['Brier']=='NA' else float(row['Brier']),recomputed_Brier=brier,max_probability_error=error,AUC=auc,status=saved['status']))
        write(ROOT/'PROGRESS.json',dict(fold=fold,complete_folds=fold+1))
    for name,rows in [('REPRODUCTION',reproduction),('TRAIN_COMPOSITION',composition),('FEATURE_DISTRIBUTIONS',ranges),('HELDOUT_BOUNDS',bounds),('TRAIN_D_K_OVERLAP',intervals),('SCORE_DISTRIBUTIONS',scores),('SIGNED_CONTRIBUTIONS',contributions)]:table(ROOT/(name+'.csv'),rows)
    write(ROOT/'DETAILED_RESULTS.json',toprecords)
    from preview_phase308 import previews
    previews(groups,top,INPUTS,pin)
    assert 'torch' not in sys.modules
    write(ROOT/'CPU_ACCEPTANCE.json',dict(tests=acceptance,folds=32,rows=len(reproduction),probability_max_error=maxerr,contribution_sum_max_error=maxsum,all_original_counts_Brier_AUC_reproduced=True,samples_weights_labels_SHA_restored=True,train_only_scaler_checked=True,bounds_boolean_complement_checked=True,torch_imported=False,classifier_fits=0,training_updates=0,neural_calls=0,downloads=0))
    write(ROOT/'INPUT_SHA256.json',INPUTS);write(ROOT/'FAILURES.json',[])
    from report_phase308 import report
    report()
if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'FAILURES.json',dict(traceback=traceback.format_exc(),no_retry=True));raise
