"""Independent artifact and algebra checks, no additional network forwards."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
from math_phase441 import gaussian
R=Path('/home/scv/shared/phase441_20261103')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rows=js(R/'INPUTS.json');assert len(rows)==567
    counts=collections.Counter();puregroups={c:set() for c in range(3)}
    for row in rows:
        ids=[]
        for s in row['selected']:
            counts[row['group'],s['category'],s['kind']]+=1;ids.append(s['token'])
            assert s['valid_fraction']>=.9 and s['category']==int(np.argmax(s['fractions']))
            assert (max(s['fractions'])>=.9)==(s['kind']=='pure')
            if s['kind']=='pure':puregroups[s['category']].add(row['group'])
        assert len(ids)==len(set(ids))
    assert max(counts.values())<=128
    parity=[]
    for row in rows:
        with np.load(R/'source/features'/f"{row['index']:03d}.npz") as a,np.load(R/'official/features'/f"{row['index']:03d}.npz") as b:
            assert a['s1_pre'].shape==b['s1_block'].shape
            delta=float(np.max(np.abs(a['s1_pre']-b['s1_block']))) if a['s1_pre'].size else 0.
            assert delta==0,delta;parity.append(delta)
    checks=0
    for model in ['source','adapted','official']:
        signatures=js(R/model/'SIGNATURES.json');assert len(signatures)==567
        for sig in signatures:assert sha(R/model/'features'/f"{sig['index']:03d}.npz")==sig['feature_sha']
        res=js(R/model/'analysis/RESAMPLING.json');gn=res['group_names'];results=js(R/model/'analysis/RESULTS.json')
        for r in results:
            cl=r['category'];alltest=[]
            assert set(r['OOF_scene_accuracy'])==puregroups[cl]
            for f in r['folds']:
                tr={gn[i] for i in f['train_groups']};te={gn[i] for i in f['test_groups']}
                assert not tr&te and tr|te==puregroups[cl];alltest.extend(te);assert f['success']
                p=R/model/'analysis'/f"{r['position']}_class{cl}_fold{f['fold']}.npz"
                with np.load(p) as z:
                    assert np.allclose(z['basis'].T@z['basis'],np.eye(z['basis'].shape[1]),atol=1e-8)
                    assert (z['std']>0).all()
                checks+=1
            assert len(alltest)==len(set(alltest))==len(puregroups[cl])
            assert abs(r['SW1']-r['SW1_exact'])<=r['SW1_bin_error_bound']+1e-8
        for p in (R/model/'analysis').glob('*_PCA.npz'):
            with np.load(p) as z:
                assert z['basis'].shape[1]==64 and np.allclose(z['basis'].T@z['basis'],np.eye(64),atol=1e-8)
    # Independent Bures cross-term via singular values rather than nested eigensquare-root.
    rng=np.random.default_rng(44199);errors=[]
    for rank in [3,8]:
        a=rng.normal(size=(8,rank));b=rng.normal(size=(8,rank));ca=a@a.T;cb=b@b.T;ma=rng.normal(size=8);mb=rng.normal(size=8)
        expected=np.square(ma-mb).sum()+np.trace(ca)+np.trace(cb)-2*np.linalg.svd(a.T@b,compute_uv=False).sum()
        actual=gaussian(ma,ca,mb,cb);errors.append(abs(float(actual)-expected));assert errors[-1]<1e-5
    source={(r['position'],r['category']):r for r in js(R/'source/analysis/RESULTS.json')}
    official={(r['position'],r['category']):r for r in js(R/'official/analysis/RESULTS.json')}
    assert all(abs(source['s1_pre',c]['R_raw']-official['s1_block',c]['R_raw'])<1e-12 for c in range(3))
    result=dict(passed=True,first_stage_raw_feature_parity_all567_exact=True,feature_archives_verified=1701,
        class_group_fold_checks=checks,scene_sampling_cap=128,purity_and_indices_verified=True,
        weighted_SW_exact_check=True,gaussian_SVD_errors=errors,official_source_first_R_raw_equal=True,extra_forwards=0)
    (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    (R/'ANALYSIS_FILE_SHA.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for model in ['source','adapted','official'] for p in (R/model/'analysis').iterdir() if p.is_file()},indent=2))
    assert js(R/'REPORT_COMPLETE.json')['passed']
    (R/'COMPLETE.json').write_text(json.dumps(dict(passed=True,forwards=1701,network_updates=0,main_rows=105)))
if __name__=='__main__':main()
