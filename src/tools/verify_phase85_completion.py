"""Phase85 CPU audit without training, inference, fitting or threshold updates."""
import hashlib,io,zipfile
import numpy as np
from phase85_common import *
from run_phase65d_recovery import records,load,prepare_verify
from phase65d_recovery_math import objective,logit_score
from phase65d_metrics import measures
from verify_phase78_local import independent_AP
from run_phase81 import counts

def main():
    report=read(PARENT/'phase85_report.json');ref=read(PHASE81/'phase81_report.json');audits=[]
    for n,old in [('MM','MM'),('LL65','LL'),('LM65','LM')]:
        for x,y in zip(report['policies'][n]['rows'],ref['policies'][old]['rows']):assert dict(x,policy=old)==y
    oldfit={r['product']:r for r in read(RECOVERY/'fit_feature_index.json')['rows']}
    for seed in [66,67]:
        root=PARENT/('seed'+str(seed));done=read(root/'training_completed.json');lock=read(root/'calibration_lock.json');model=lock['models']['L+S']
        assert set(lock['models'])=={'L+S'} and model['columns']==[0,1,2,3] and model['optimizer_success'] and model['theta'][0]>=0
        curve=done['curve'];assert [r['iteration'] for r in curve]==list(range(500,4001,500))
        assert done['best_iteration']==max(curve,key=lambda r:(r['metrics']['mIoU'],-r['iteration']))['iteration']
        for f in done['checkpoints']:assert digest(f['path'])==f['sha256']
        for rel,h in read(root/'training_lock.json')['code_sha256'].items():assert digest(REPO/rel)==h
        random=read(root/'randomness_lock.json');assert random['seed']==random['runner_seed']==random['sampler_seed']==seed
        assert random['deterministic_algorithms'] and random['warn_only'] and random['num_workers']==2
        fa=read(root/'fit_input_audit.json');fi={r['product']:r for r in fa['rows']};li={r['product']:r for r in read(root/'fit_representatives/index.json')['rows']}
        xx=[];yy=[];ww=[];negative=[]
        for product,r in records('fit_representatives').items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);v=truth!=255;y=truth[v]==2;a=fi[product];prior=oldfit[product]
            path=root/a['feature_path'];assert digest(path)==a['feature_sha256'];x=np.load(path)
            old=RECOVERY/prior['feature_path'];assert digest(old)==prior['feature_sha256']
            np.testing.assert_array_equal(x[:,1:],np.load(old,mmap_mode='r')[:,2:5])
            np.testing.assert_array_equal(x[:,0],logit_score(load(root,li[product],'score')[v]).astype(np.float32))
            rng=np.random.default_rng(65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16));idx=rng.choice(len(x),min(8192,len(x)),replace=False)
            assert hashlib.sha256(idx.tobytes()).hexdigest()==a['sample_indices_sha256']==prior['sample_indices_sha256']
            xx.append(x[idx]);yy.append(y[idx]);ww.append(np.full(len(idx),1/(153*len(idx))))
            score=(((x-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['theta'][:-1])+model['theta'][-1]).astype(np.float32)
            negative.append(score[~y])
        x=np.concatenate(xx).astype(float);y=np.concatenate(yy).astype(float);w=np.concatenate(ww);w/=w.sum()
        assert hashlib.sha256(x.tobytes()).hexdigest()==fa['sample_x_sha256'];assert hashlib.sha256(y.tobytes()).hexdigest()==fa['sample_y_sha256']
        wn=w/w.sum()  # Original standardize renormalizes already normalized fit weights.
        mu=np.array([np.sum(x[:,i]*wn) for i in range(x.shape[1])])
        sc=np.sqrt(np.array([np.sum((x[:,i]-mu[i])**2*wn) for i in range(x.shape[1])]))
        sc[sc==0]=1
        np.testing.assert_allclose(mu,model['mean'],rtol=1e-13,atol=1e-14);np.testing.assert_allclose(sc,model['scale'],rtol=1e-13,atol=1e-14)
        val,grad=objective(np.array(model['theta']),(x-mu)/sc,y,w);assert np.isfinite(grad).all() and abs(val-model['objective'])<1e-12
        rate=np.mean([(a>=model['threshold']).mean() for a in negative]);assert rate<=.01+1e-10 and rate==model['fit_macro_fpr']
        below=[a[a<model['threshold']] for a in negative];lower=max(float(a.max()) for a in below if len(a))
        assert np.mean([(a>=lower).mean() for a in negative])>.01-1e-10,'Threshold must respect exact-ties maximal acceptance'
        del x,y,ww,xx,yy,w,negative,below
        da=read(root/'development_input_audit.json');li={r['product']:r for r in read(root/'development_val/index.json')['rows']};pools=[];labels=[]
        by={n:{r['product']:r for r in report['policies'][n]['rows']} for n in ['LL'+str(seed),'LM'+str(seed)]}
        with zipfile.ZipFile(DATA/'subscenes.zip') as z:
            for a in da['rows']:
                product=a['product'];r=records('development_val')[product];truth=np.load(PREP/r['mask_path']);v=truth!=255;t=truth[v]
                b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==a['image_sha256'];raw=np.load(io.BytesIO(b),allow_pickle=False)
                x=np.column_stack([logit_score(load(root,li[product],'score')[v]),raw[...,7][v],raw[...,11][v],raw[...,12][v]]).astype(np.float32)
                score=(((x-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['theta'][:-1])+model['theta'][-1]).astype(np.float32)
                assert digest(root/a['path'])==a['sha256'];np.testing.assert_array_equal(score,np.load(root/a['path']))
                ap=independent_AP(t==2,score);shadow=score>=model['threshold']
                for n,parent,index in [('LL'+str(seed),root,a['lora_cache']),('LM'+str(seed),OLD,a['msre_cache'])]:
                    winner=load(parent,index,'nonshadow')[v];pred=winner.copy();pred[shadow]=2;cm=counts(t,pred);refrow=by[n][product]
                    assert cm.tolist()==refrow['confusion'];assert ap is None and refrow['ap'] is None or ap is not None and abs(ap-refrow['ap'])<1e-12
                pools.append(score);labels.append(t==2)
        pool=independent_AP(np.concatenate(labels),np.concatenate(pools));assert abs(pool-report['policies']['LL'+str(seed)]['pooled_ap'])<1e-12
        audits.append(dict(seed=seed,fit_groups=153,development_groups=32,independent_confusions=64,independent_AP=True,exact_ties_threshold_verified=True))
        print('Phase85 CPU verification seed passed',seed,flush=True)
    for n,p in report['policies'].items():
        assert measures(np.sum([r['confusion'] for r in p['rows']],axis=0))['miou_percent']==p['miou_percent']
        assert len(p['rows'])==32 and len([r for r in p['rows'] if r['positive']])==7
    ds=[]
    mm=[r for r in report['policies']['MM']['rows'] if r['positive']]
    for seed in [66,67]:
        a=[r for r in report['policies']['LL'+str(seed)]['rows'] if r['positive']];d=np.array([100*(x['ap']-y['ap']) for x,y in zip(a,mm)]);ds.append(d)
        np.testing.assert_allclose(interval(d)['ci95'],report['comparisons']['LL'+str(seed)+' minus MM']['AP_pp']['ci95'],rtol=0,atol=1e-13)
    mean=np.mean(ds,axis=0);np.testing.assert_allclose(mean,report['new_seed_group_mean_AP_gains_pp'],rtol=0,atol=1e-13)
    np.testing.assert_allclose(interval(mean)['ci95'],report['new_seed_mean_AP_gain_pp']['ci95'],rtol=0,atol=1e-13)
    rules=gate(report['policies']);rules['all_crossed_shadow_invariants']=True;assert rules==report['gate'] and all(rules.values())==report['retain_branch']
    write(PARENT/'INDEPENDENT_VERIFIED.json',dict(status='passed',audits=audits,paired_group_seed_average_verified=True,gate_verified=True,
        verification_code_sha256=digest(Path(__file__)),verification_code_path='src/tools/verify_phase85_completion.py',
        normalization_repair='Original fit uses F-order column selection and pairwise column reductions plus internal weight renormalization; independently reduce each column at unchanged tolerance; no training or fitting rerun'))
    paths=[p for p in PARENT.rglob('*') if p.is_file() and p.suffix not in ('.npy',)]
    write(PARENT/'artifact_hashes.json',{str(p.relative_to(PARENT)):digest(p) for p in paths})
    print('Phase85 independent CPU verification passed',flush=True)
if __name__=='__main__':main()
