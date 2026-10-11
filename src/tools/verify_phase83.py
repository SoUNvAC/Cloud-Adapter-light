"""CPU verification: independently rebuild sampling/features, scores and boolean counts."""
import hashlib,io,zipfile
import numpy as np
from run_phase83 import ROOT,PREVIOUS,PHASE81,DATA,PREP,read,write,digest,records,prepare_verify,load
from phase65d_recovery_math import objective
from verify_phase78_local import independent_AP

def main():
    idx=read(ROOT/'probability_index.json');assert idx['forwards']==185 and idx['state_before_sha256']==idx['state_after_sha256']
    lock=read(ROOT/'readout_lock.json');report=read(ROOT/'phase83_report.json');audit=read(ROOT/'fit_input_audit.json')
    refs=read(PHASE81/'phase81_report.json');entries={r['product']:r for r in idx['rows']}
    fit={r['product']:r for r in audit['rows']};rng=np.random.default_rng(83011);xx=[];yy=[];ww=[];counts=[]
    def unpack(r,z):
        prepare_verify(r);truth=np.load(PREP/r['mask_path']);v=truth!=255;t=truth[v]
        b=z.read('subscenes/'+r['product']+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image'];raw=np.load(io.BytesIO(b),allow_pickle=False)
        e=entries[r['product']];path=ROOT/e['path'];assert digest(path)==e['sha256'];p=np.load(path,mmap_mode='r')
        assert np.array_equal(p[2],load(PREVIOUS,e['old_cache'],'score'))
        u=np.log(np.clip(p[1].astype(float),1e-6,None))-np.log(np.clip(p[0].astype(float),1e-6,None))
        x=np.column_stack((u[v],raw[...,7][v],raw[...,11][v],raw[...,12][v]));assert np.isfinite(x).all()
        return t,p,v,x
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in records('fit_representatives').items():
            t,p,v,x=unpack(r,z);eligible=np.where(t<2)[0];a=fit[product]
            if not len(eligible):assert a['excluded'];continue
            wanted=rng.choice(eligible,min(4096,len(eligible)),replace=False);path=ROOT/a['index_path'];assert digest(path)==a['index_sha256']
            np.testing.assert_array_equal(wanted,np.load(path));assert len(np.unique(wanted))==len(wanted)
            xx.append(x[wanted]);yy.append(t[wanted]);counts.append(len(wanted))
        x=np.concatenate(xx);y=np.concatenate(yy).astype(float);w=np.concatenate([np.full(n,1/(len(counts)*n)) for n in counts]);w/=w.sum()
        assert hashlib.sha256(x.tobytes()).hexdigest()==audit['sample_x_sha256'];assert hashlib.sha256(y.tobytes()).hexdigest()==audit['sample_y_sha256']
        for n,m in lock['models'].items():
            a=x[:,m['columns']];mu=(a*w[:,None]).sum(0);s=np.sqrt(((a-mu)**2*w[:,None]).sum(0));s[s==0]=1
            np.testing.assert_allclose(mu,m['mean'],rtol=1e-13,atol=1e-14);np.testing.assert_allclose(s,m['scale'],rtol=1e-13,atol=1e-14)
            val,g=objective(np.array(m['theta']),(a-mu)/s,y,w);assert abs(val-m['objective'])<1e-12 and m['success'] and m['theta'][0]>=0
        pools=[];ys=[]
        by={n:{r['product']:r for r in report['policies'][n]['rows']} for n in ('R1','R4','LL')}
        for product,r in records('development_val').items():
            t,p,v,x=unpack(r,z);m=lock['old_shadow_model'];a=np.column_stack((np.log(np.clip(p[2][v].astype(float),1e-6,1-1e-6))-np.log1p(-np.clip(p[2][v].astype(float),1e-6,1-1e-6)),x[:,1:])).astype(np.float32)
            s=(((a-np.array(m['mean']))/np.array(m['scale']))@np.array(m['theta'][:-1])+m['theta'][-1]).astype(np.float32);shadow=s>=m['threshold']
            for n in ('R1','R4'):
                m=lock['models'][n];zval=((x[:,m['columns']]-np.array(m['mean']))/np.array(m['scale']))@np.array(m['theta'][:-1])+m['theta'][-1]
                cloud=zval>=0;pred=np.select([shadow,cloud],[2,1],default=0)
                cm=np.array([[np.count_nonzero((t==i)&(pred==j)) for j in range(3)] for i in range(3)])
                assert cm.tolist()==by[n][product]['confusion'];assert np.array_equal(cm[:,2],np.array(by['LL'][product]['confusion'])[:,2])
            ap=independent_AP(t==2,s);assert ap==by['LL'][product]['ap'] or (ap is not None and abs(ap-by['LL'][product]['ap'])<1e-12)
            pools.append(s);ys.append(t==2)
    assert abs(independent_AP(np.concatenate(ys),np.concatenate(pools))-report['policies']['LL']['pooled_ap'])<1e-12
    for n,p in report['policies'].items():
        cm=np.sum([r['confusion'] for r in p['rows']],0);assert cm.tolist()==p['confusion']
        union=cm.sum(0)+cm.sum(1)-cm.diagonal();iou=cm.diagonal()/union*100
        np.testing.assert_allclose(iou,p['iou_percent'],rtol=0,atol=1e-12);assert abs(iou.mean()-p['miou_percent'])<1e-12
    for key,c in report['comparisons'].items():
        a,b=key.split(' minus ');br={r['product']:r for r in report['policies'][b]['rows']};d=np.array([r['miou_percent']-br[r['product']]['miou_percent'] for r in report['policies'][a]['rows']])
        bs=np.random.default_rng(83011).choice(d,(10000,32)).mean(1)
        np.testing.assert_allclose(np.percentile(bs,[2.5,97.5]),c['macro_mIoU_pp']['ci95'])
    write(ROOT/'INDEPENDENT_VERIFIED.json',dict(status='passed',fit_groups=len(counts),sampling_and_weighted_standardization=True,
        sample_features_sha_exact=True,fixed_fit_objectives_exact=True,independent_boolean_confusions=64,independent_AP_groups=32,
        independent_AP_pooled=True,macro_bootstrap=True,network_state_unchanged=True,no_inference_or_fitting=True,verifier_sha256=digest(__file__)))
    names=['README.md','summary.csv','development_groups.csv','paired_differences.csv','confusion_matrices.json','probability_index.json',
        'fit_input_audit.json','R1_fit.json','R4_fit.json','readout_lock.json','input_configuration_lock.json','invariant_checks.json','phase83_report.json','COMPLETED.json','INDEPENDENT_VERIFIED.json']
    write(ROOT/'artifact_hashes.json',{n:digest(ROOT/n) for n in names})
    print('Phase83 independent CPU verification passed',flush=True)
if __name__=='__main__':main()
