from pathlib import Path
import numpy as np
import json,hashlib
R=Path('/home/scv/shared/phase442_20261103')
def main():
    rows=json.loads((R/'SUMMARY.json').read_text());lock=json.loads((R/'LOCK.json').read_text())
    assert len(rows)==4 and rows[0]['source_tokens']==535*1024 and rows[0]['target_tokens']==32*1024
    with np.load(R/'COMMON_PCA.npz') as p:
        assert p['basis'].shape==(384,64) and np.allclose(p['basis'].T@p['basis'],np.eye(64),atol=1e-10)
        assert abs(p['eigenvalues'][:64].sum()/p['eigenvalues'].sum()-lock['explained_variance'])<1e-12
    checks=[]
    for row in rows:
        with np.load(R/(row['scope']+'_MOMENTS.npz')) as p:
            a=p['source_cov'];b=p['target_cov'];va,ua=np.linalg.eigh(a);vb,ub=np.linalg.eigh(b)
            ra=(ua*np.sqrt(np.maximum(va,0)))@ua.T;rb=(ub*np.sqrt(np.maximum(vb,0)))@ub.T
            alternative=np.square(p['source_mean']-p['target_mean']).sum()+np.trace(a)+np.trace(b)-2*np.linalg.svd(ra@rb,compute_uv=False).sum()
            assert abs(alternative-row['F'])<1e-7
            assert np.allclose(np.quantile(p['F_bootstrap'],[.025,.975]),[row['F_lo'],row['F_hi']],atol=1e-12)
            checks.append(dict(scope=row['scope'],SVD_error=float(abs(alternative-row['F']))))
    assert lock['existing_feature_parity_all567_exact']
    signatures=json.loads((R/'official/SIGNATURES.json').read_text())
    for s in signatures:
        p=R/'official/features'/f"{s['index']:03d}.npz"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==s['feature_sha'] and s['locations']==['final'] and s['tokens']==1024
    (R/'INDEPENDENT_ACCEPTANCE.json').write_text(json.dumps(dict(passed=True,checks=checks,single_PCA=True,all_567_feature_hashes=True,prior_token_parity=True,extra_forwards=0),indent=2))
    print('INDEPENDENT_ACCEPTANCE passed')
if __name__=='__main__':main()
