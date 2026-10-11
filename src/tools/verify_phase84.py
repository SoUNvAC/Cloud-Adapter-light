"""No inference or refitting: independently verify frozen features and full decisions."""
import hashlib,zipfile
import numpy as np
from run_phase84 import ROOT,CONTROL,PROTOCOL,P80,REPO,DATA,PREP,read,write,digest,records,p83,NAMES
from phase65d_recovery_math import objective
from verify_phase78_local import independent_AP

def checked(rel,expected):
    p=ROOT/rel;assert digest(p)==expected,rel
    return np.load(p,mmap_mode='r')

def main():
    lock=read(ROOT/'readout_lock.json');feature=read(ROOT/'feature_lock.json');extraction=read(ROOT/'extraction_audit.json');report=read(ROOT/'phase84_report.json')
    assert extraction['forwards']==185 and extraction['state_before_sha256']==extraction['state_after_sha256']
    assert extraction['no_shortcut_interpolation'] and all(r['probabilities_exact'] and r['semantic_postprocess_exact'] for r in extraction['rows'])
    assert extraction['feature']==feature['feature'] and feature['feature']['channels']==feature['feature']['shape'][1]
    oldlock=read(p83.ROOT/'readout_lock.json');oldfit=read(p83.ROOT/'fit_input_audit.json');audit={r['product']:r for r in feature['rows']};old={r['product']:r for r in oldfit['rows']}
    idx={r['product']:r for r in read(p83.ROOT/'probability_index.json')['rows']};rng=np.random.default_rng(84011);xs=[];ys=[];fs=[];perms=[];sizes=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in records('fit_representatives').items():
            a=audit[product];prior=old[product]
            if 'excluded' in a:assert a==prior;continue
            assert a['sample_index_sha256']==prior['index_sha256']==digest(a['sample_index_path'])
            sample=np.load(a['sample_index_path']);f=checked(a['feature_path'],a['feature_sha256']);perm=checked(a['permutation_path'],a['permutation_sha256'])
            assert f.shape==(len(sample),feature['feature']['channels']) and np.isfinite(f).all()
            np.testing.assert_array_equal(perm,rng.permutation(len(sample)))
            truth,valid,x,_=p83.full_features(r,idx[product],z);y=truth[valid][sample];assert np.isin(y,[0,1]).all()
            xs.append(x[sample]);ys.append(y);fs.append(np.array(f));perms.append(np.array(perm));sizes.append(len(sample))
        x=np.concatenate(xs);y=np.concatenate(ys).astype(float);f=np.concatenate(fs);del xs,ys,fs
        assert hashlib.sha256(x.tobytes()).hexdigest()==feature['base_sample_x_sha256'] and hashlib.sha256(y.tobytes()).hexdigest()==feature['sample_y_sha256']
        w=np.concatenate([np.full(n,1/(len(sizes)*n)) for n in sizes]);w/=w.sum()
        for name,m in lock['models'].items():
            assert m['mean'][:4]==oldlock['models']['R4']['mean'] and m['scale'][:4]==oldlock['models']['R4']['scale']
            ff=f if name=='F_real' else np.concatenate([part[p] for part,p in zip(np.split(f,np.cumsum(sizes)[:-1]),perms)])
            mu=np.average(ff.astype(float),axis=0,weights=w);var=np.average((ff.astype(float)-mu)**2,axis=0,weights=w);sc=np.sqrt(var);sc[sc==0]=1
            np.testing.assert_allclose(mu,m['mean'][4:],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(sc,m['scale'][4:],rtol=1e-12,atol=1e-12)
            zz=(np.column_stack([x,ff])-m['mean'])/m['scale'];val,grad=objective(np.array(m['theta']),zz,y,w)
            assert m['success'] and m['theta'][0]>=0 and abs(val-m['objective'])<1e-11 and np.isfinite(grad).all()
            del zz,ff
        del x,y,f,w
        audits={r['product']:r for r in read(ROOT/'development_audit.json')['rows']};rng=np.random.default_rng(84011);ref=read(p83.ROOT/'phase83_report.json');ll={r['product']:r for r in ref['policies']['LL']['rows']}
        rows={n:{r['product']:r for r in report['policies'][n]['rows']} for n in NAMES};pool=[];ys=[];max_error=0
        for product,r in records('development_val').items():
            a=audits[product];truth,valid,base,oldx=p83.full_features(r,idx[product],z);t=truth[valid];perm=checked(a['permutation_path'],a['permutation_sha256'])
            np.testing.assert_array_equal(perm,rng.permutation(len(t)))
            assert not np.isin(perm,[-1,len(t)]).any()
            oldm=lock['original_shadow_model'];s=(((oldx-np.array(oldm['mean']))/np.array(oldm['scale']))@np.array(oldm['theta'][:-1])+oldm['theta'][-1]).astype(np.float32);shadow=s>=oldm['threshold']
            points=np.array(a['check_indices']);cb=checked(a['files']['check_base']['path'],a['files']['check_base']['sha256']);np.testing.assert_array_equal(cb,base[points])
            for n in NAMES:
                e=a['files'][n];m=lock['models'][n];logits=checked(e['logits_path'],e['logits_sha256']);ff=checked(e['check_F_path'],e['check_F_sha256'])
                expected=((np.column_stack([cb,ff])-np.array(m['mean']))/np.array(m['scale']))@np.array(m['theta'][:-1])+m['theta'][-1]
                error=float(np.max(np.abs(logits[points]-expected)));assert error<=1e-6;max_error=max(max_error,error)
                if n=='F_shuffle':
                    proj=checked(e['unpermuted_F_projection_path'],e['unpermuted_F_projection_sha256'])
                    bp=((base-np.array(m['mean'][:4]))/np.array(m['scale'][:4]))@np.array(m['theta'][:4])+m['theta'][-1]
                    error=float(np.max(np.abs(logits-(bp+proj[perm]))));assert error<=1e-6;max_error=max(max_error,error)
                cloud=np.exp(-np.logaddexp(0,-logits))>=.5;pred=np.select([shadow,cloud],[2,1],default=0)
                cm=np.array([[np.count_nonzero((t==i)&(pred==j)) for j in range(3)] for i in range(3)]);assert cm.tolist()==rows[n][product]['confusion']
                assert np.array_equal(cm[:,2],np.array(ll[product]['confusion'])[:,2]);assert cm[2,:2].sum()==sum(ll[product]['confusion'][2][:2])
            ap=independent_AP(t==2,s);assert ap==ll[product]['ap'] or (ap is not None and abs(ap-ll[product]['ap'])<1e-12);pool.append(s);ys.append(t==2)
        assert abs(independent_AP(np.concatenate(ys),np.concatenate(pool))-ref['policies']['LL']['pooled_ap'])<1e-12
    for n,p in report['policies'].items():
        cm=np.sum([r['confusion'] for r in p['rows']],0);assert cm.tolist()==p['confusion'];u=cm.sum(0)+cm.sum(1)-cm.diagonal()
        np.testing.assert_allclose(cm.diagonal()/u*100,p['iou_percent'],rtol=0,atol=1e-12)
    for key,c in report['comparisons'].items():
        a,b=key.split(' minus ');by={r['product']:r for r in report['policies'][b]['rows']};d=np.array([r['miou_percent']-by[r['product']]['miou_percent'] for r in report['policies'][a]['rows']])
        bs=np.random.default_rng(84011).choice(d,(10000,32)).mean(1);np.testing.assert_allclose(np.percentile(bs,[2.5,97.5]),c['macro_mIoU_pp']['ci95'])
    write(ROOT/'INDEPENDENT_VERIFIED.json',dict(status='passed',verifier_sha256=digest(__file__),fit_groups=len(sizes),original_sample_indices_and_features_exact=True,
        weighted_standardization_and_objectives_exact=True,joint_permutations_exact=True,all_pixel_shuffle_projections=True,independent_boolean_confusions=64,independent_AP_groups=32,
        independent_AP_pool=True,numerical_check_max_abs_error=max_error,no_new_inference_or_fitting=True))
    names=['README.md','summary.csv','development_groups.csv','paired_differences.csv','confusion_matrices.json','feature_lock.json','extraction_audit.json',
        'F_real_fit.json','F_shuffle_fit.json','readout_lock.json','development_audit.json','input_configuration_lock.json','invariant_checks.json','phase84_report.json','COMPLETED.json','INDEPENDENT_VERIFIED.json']
    write(ROOT/'artifact_hashes.json',{n:digest(ROOT/n) for n in names})
    print('Phase84 independent CPU verification passed',flush=True)
if __name__=='__main__':main()
