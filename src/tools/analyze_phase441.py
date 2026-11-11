"""Shared weighted PCA, scene bootstrap geometry, grouped out-of-fold probes."""
import csv,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from math_phase441 import *
R=Path('/home/scv/shared/phase441_20261103')
ARM=sys.argv[1];OUT=R/ARM/'analysis';OUT.mkdir(parents=True,exist_ok=True)
def save(n,x):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def ci(a):return np.quantile(a,[.025,.975],axis=0).tolist()
def shared_moments(arrays,g,groups):
    result=None
    for x in arrays:
        m,s,n=scene_moments(x,g,groups)
        if result is None:result=[m,s,n]
        else:result[0]+=m;result[1]+=s
    result[0]/=len(arrays);result[1]/=len(arrays)
    return result
def mean_ratio(a,b):
    gap=np.linalg.norm(a[...,None,:,:]-a[...,:,None,:],axis=-1)+np.eye(3)*1e100
    return np.linalg.norm(b-a,axis=-1)/np.maximum(gap.min(-1),1e-15)

def main():
    save('MATH_TESTS.json',self_test());start=time.monotonic()
    rows=json.loads((R/'INPUTS.json').read_text());group_names=sorted({x['group'] for x in rows});groups=np.arange(len(group_names));gmap={s:i for i,s in enumerate(group_names)}
    domains=np.array([0 if s.startswith('A:') else 1 for s in group_names]);meta=[];features={}
    for row in rows:
        with np.load(R/ARM/'features'/f"{row['index']:03d}.npz") as z:
            for key in z.files:features.setdefault(key,[]).append(z[key])
        meta.extend(dict(group=gmap[row['group']],domain=0 if row['domain']=='A' else 1,category=s['category'],pure=s['kind']=='pure') for s in row['selected'])
    features={k:np.concatenate(v).astype(np.float64) for k,v in features.items()}
    g=np.array([v['group'] for v in meta]);y=np.array([v['domain'] for v in meta]);c=np.array([v['category'] for v in meta]);pure=np.array([v['pure'] for v in meta])
    assert len(meta)>0 and all(len(x)==len(meta) for x in features.values())
    gp=g[pure];yp=y[pure];cp=c[pure]
    represented=np.stack([np.bincount(gp[cp==cl],minlength=len(groups))>0 for cl in range(3)])
    rng=np.random.default_rng(44102);boot=[]
    while len(boot)<256:
        w=np.zeros(len(groups))
        for d in [0,1]:
            ids=groups[domains==d];w[ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
        if all((w*represented[cl]*(domains==d)).sum()>0 for cl in range(3) for d in [0,1]):boot.append(w)
    boot=np.stack(boot);folds={};nulls={}
    for cl in range(3):
        ids=[groups[represented[cl]&(domains==d)] for d in [0,1]];k=min(5,*map(len,ids));assert k>=2
        rng=np.random.default_rng(44110+cl);splits=[]
        for v in ids:splits.append(np.array_split(rng.permutation(v),k))
        folds[cl]=[np.r_[splits[0][i],splits[1][i]].tolist() for i in range(k)]
        m=min(32,len(ids[0])//2,len(ids[1])//2);assert m>=1
        arrays={n:np.zeros((100,len(groups))) for n in ['A_left','A_right','B_left','B_right','cross_A','cross_B']}
        rng=np.random.default_rng(44120+cl)
        for i in range(100):
            for d,dd in enumerate(['A','B']):
                selected=rng.choice(ids[d],2*m,replace=False)
                arrays[dd+'_left'][i,selected[:m]]=1;arrays[dd+'_right'][i,selected[m:]]=1
                arrays['cross_'+dd][i,rng.choice(ids[d],m,replace=False)]=1
        nulls[cl]=(m,arrays)
    save('RESAMPLING.json',dict(group_names=group_names,bootstrap_counts=boot.astype(int).tolist(),folds=folds,
        nulls={cl:dict(per_side=m,draws={k:v.astype(int).tolist() for k,v in a.items()}) for cl,(m,a) in nulls.items()}))
    stages={f's{i}':[f's{i}_block'] if ARM=='official' else [f's{i}_pre',f's{i}_attention',f's{i}_post'] for i in range(1,5)}
    stages['final']=['final']
    if ARM!='official':stages.update(spatial_context=['spatial_context'],spatial_detail=['spatial_detail'])
    result=[];boundary=[]
    for stage,locations in stages.items():
        raw=[features[k][pure] for k in locations]
        class_shared=[shared_moments([a[cp==cl] for a in raw],gp[cp==cl],groups) for cl in range(3)]
        total_n=sum(z[2] for z in class_shared);den=np.maximum(total_n,1)
        pm=sum(z[0]*z[2][:,None] for z in class_shared)/den[:,None]
        ps=sum(z[1]*z[2][:,None,None] for z in class_shared)/den[:,None,None]
        center,basis,explained,eigenvalues=pca_from_moments(pm,ps,total_n,domains)
        np.savez_compressed(OUT/(stage+'_PCA.npz'),center=center,basis=basis,eigenvalues=eigenvalues)
        save(stage+'_PCA.json',dict(model=ARM,locations=locations,dimensions=basis.shape[1],explained_variance=explained,whiten=False,domain_scene_location_balanced=True))
        # Per-class folds share a PCA across locations but never use a held-out scene.
        fold_pca={}
        for cl in range(3):
            all_ids=groups[represented[cl]]
            for f,test in enumerate(folds[cl]):
                train=np.setdiff1d(all_ids,test);cm,cb,ce,_=pca_from_moments(*class_shared[cl],domains,chosen=np.isin(groups,train))
                fold_pca[cl,f]=(train,np.array(test),cm,cb,ce)
        del class_shared,ps,pm
        for loc,x in zip(locations,raw):
            z=(x-center)@basis;dimension=z.shape[1]
            moments=[scene_moments(z[cp==cl],gp[cp==cl],groups) for cl in range(3)]
            rawmeans=[]
            for cl in range(3):
                xx=x[cp==cl];gg=gp[cp==cl]
                rawmeans.append(np.stack([xx[gg==s].mean(0) if (gg==s).any() else np.zeros(x.shape[1]) for s in groups]))
            means=[];covs=[];bmeans=[];bcovs=[];rmeans=[];rbmeans=[]
            for d in [0,1]:
                simple=[combine(*moments[cl],weights=(domains==d).astype(float)) for cl in range(3)]
                bs=[combine(*moments[cl],weights=boot*(domains==d)) for cl in range(3)]
                means.append(np.stack([v[0] for v in simple]));covs.append(np.stack([v[1] for v in simple]))
                bmeans.append(np.stack([v[0] for v in bs],1));bcovs.append(np.stack([v[1] for v in bs],1))
                rm=[];rb=[]
                for cl in range(3):
                    w=(domains==d)*represented[cl];rm.append(w@rawmeans[cl]/w.sum())
                    w=boot*(domains==d)*represented[cl];rb.append((w@rawmeans[cl])/w.sum(1,keepdims=True))
                rmeans.append(np.stack(rm));rbmeans.append(np.stack(rb,1))
            geo=geometry(means[0],covs[0],means[1],covs[1]);bg=geometry(bmeans[0],bcovs[0],bmeans[1],bcovs[1])
            rr=mean_ratio(*rmeans);brr=mean_ratio(*rbmeans)
            np.savez_compressed(OUT/(loc+'_BOOTSTRAP.npz'),F=bg['F'],R=bg['R'],P=bg['P'],R_raw=brr,source_gap=bg['source_gap'],target_gap=bg['target_gap'])
            rng=np.random.default_rng(44130);directions=rng.normal(size=(dimension,64));directions/=np.linalg.norm(directions,axis=0)
            np.save(OUT/(loc+'_SW_DIRECTIONS.npy'),directions)
            for cl in range(3):
                row=dict(model=ARM,stage=stage,position=loc,category=cl,source_scenes=int((represented[cl]&(domains==0)).sum()),target_scenes=int((represented[cl]&(domains==1)).sum()),pca_explained=explained)
                for key in ['F','R','source_gap','target_gap','shift']:row[key]=float(geo[key][cl]);row[key+'_CI']=ci(bg[key][:,cl])
                row.update(R_raw=float(rr[cl]),R_raw_CI=ci(brr[:,cl]),P=geo['P'][cl].tolist(),P_CI=ci(bg['P'][:,cl]),
                    small_source_gap=bool(geo['source_gap'][cl]<.1*np.sqrt(np.trace(covs[0],axis1=-2,axis2=-1).mean())),
                    gap_flag_rule='nearest source centroid gap < 0.1 * pooled within-class RMS radius')
                row.update(sliced(z[cp==cl],gp[cp==cl],groups,domains,boot,directions))
                m,draws=nulls[cl];null_result={}
                for label,left,right in [('source_null','A_left','A_right'),('target_null','B_left','B_right'),('matched_cross','cross_A','cross_B')]:
                    a,ca=combine(*moments[cl],weights=draws[left]);b,cb=combine(*moments[cl],weights=draws[right]);dist=gaussian(a,ca,b,cb)
                    null_result[label]=dict(median=float(np.median(dist)),q025_q975=ci(dist),values=dist.tolist())
                row['null']=dict(per_side_scenes=m,**null_result)
                oof=np.full(len(groups),np.nan);fold_records=[]
                for f in range(len(folds[cl])):
                    train,test,cm,cb,ce=fold_pca[cl,f]
                    scores,params,fit=probe_fold(x[cp==cl],yp[cp==cl],gp[cp==cl],train,test,cm,cb)
                    for s,acc in scores.items():oof[s]=acc
                    np.savez_compressed(OUT/f'{loc}_class{cl}_fold{f}.npz',**params)
                    fold_records.append(dict(fold=f,train_groups=train.tolist(),test_groups=test.tolist(),pca_explained=ce,**fit))
                acc=[];bacc=[]
                for d in [0,1]:
                    ok=np.isfinite(oof)&(domains==d);acc.append(oof[ok].mean());w=boot[:,ok];bacc.append(w@oof[ok]/w.sum(1))
                row.update(domain_accuracy=float(np.mean(acc)),domain_accuracy_CI=ci(np.mean(bacc,axis=0)),domain_accuracy_by_domain=acc,
                    OOF_scene_accuracy={group_names[s]:float(oof[s]) for s in groups if np.isfinite(oof[s])},folds=fold_records,
                    probe_CI='scene bootstrap of fixed OOF predictions; does not refit each bootstrap')
                result.append(row)
                save('STATUS.json',dict(stage=stage,position=loc,category=cl,rows=len(result),seconds=time.monotonic()-start))
            # Boundary/mixed tokens remain a separate diagnostic, projected with pure-token PCA.
            mixed=(~pure);zz=(features[loc][mixed]-center)@basis;gm=g[mixed];cmix=c[mixed]
            mm=[scene_moments(zz[cmix==cl],gm[cmix==cl],groups) for cl in range(3)]
            if all(((v[2]>0)&(domains==d)).any() for v in mm for d in [0,1]):
                pairs=[[combine(*v,weights=(domains==d).astype(float)) for v in mm] for d in [0,1]]
                a=np.stack([v[0] for v in pairs[0]]);ca=np.stack([v[1] for v in pairs[0]]);b=np.stack([v[0] for v in pairs[1]]);cb=np.stack([v[1] for v in pairs[1]])
                geom=geometry(a,ca,b,cb)
                for cl in range(3):boundary.append(dict(model=ARM,position=loc,category=cl,F=float(geom['F'][cl]),R=float(geom['R'][cl]),P=geom['P'][cl].tolist(),source_scenes=int(((mm[cl][2]>0)&(domains==0)).sum()),target_scenes=int(((mm[cl][2]>0)&(domains==1)).sum()),scope='mixed-patch descriptive supplement, no probe or CI'))
            save('RESULTS.json',result);save('BOUNDARY.json',boundary)
            del moments,rawmeans,bmeans,bcovs,z
        del raw,fold_pca
    save('COMPLETE.json',dict(passed=True,rows=len(result),seconds=time.monotonic()-start,bootstrap=256,training='linear logistic probes only',network_updates=0))
if __name__=='__main__':
    try:main()
    except BaseException:save('FAILURE.json',dict(error=traceback.format_exc()));raise
