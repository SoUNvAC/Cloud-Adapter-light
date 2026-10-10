"""Independent local report/AP verification; exactly three authorized fit masks."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_AP(y,score):
    if not y.any() or y.all():return None
    order=np.argsort(-score,kind='stable');y=y[order];score=score[order]
    ends=np.r_[np.flatnonzero(score[:-1]!=score[1:]),len(score)-1]
    tp=np.cumsum(y,dtype='int64')[ends];precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0,tp])*precision)/tp[-1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);a=p.parse_args();out=a.reports
    hashes=json.loads((out/'artifact_hashes.json').read_text())
    for rel,h in hashes.items():
        path=out/rel;assert path.resolve().is_relative_to(out.resolve()) and sha(path)==h,rel
    lock=json.loads((out/'input_formula_lock.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert (out/'EXIT_CODE').read_text().strip()=='0' and summary['status']=='completed'
    commit=lock['git_commit']
    for rel,key in [('src/tools/run_phase78.py','code_sha256'),('src/tools/run_phase76.py','helper_sha256'),('src/research_plans/PHASE78_INPUT_BINDINGS_20261011.json','bindings_sha256')]:
        assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+rel])).hexdigest()==lock[key],rel
    previous=Path('outputs/phase76/phase76_20261010')
    for rel,h in lock['previous'].items():assert sha(previous/rel)==h,rel
    names=('sun','orthogonal90','reverse','orthogonal270','all','dark');verified=[]
    with zipfile.ZipFile('data/sentinel2_cloud_mask_catalogue_4172871/masks.zip') as masks:
        for r in lock['selected']:
            product=r['product'];root=out/product
            b=masks.read('masks/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['mask']
            mask=np.load(io.BytesIO(b),allow_pickle=False);truth=mask.argmax(-1);labels=mask.sum(-1)==1
            common=np.load(root/'common_mask.npy',allow_pickle=False);ev=np.load(root/'evaluation_mask.npy',allow_pickle=False)
            assert np.array_equal(ev,common&labels)
            with np.load(root/'native_scores.npz',allow_pickle=False) as scores:
                assert set(scores.files)==set(names)
                combined=np.max(np.stack([scores[n] for n in names[:4]]),axis=0)
                np.testing.assert_array_equal(combined,scores['all'])
                assert np.array_equal(common,np.isfinite(combined)&np.isfinite(scores['dark']))
                for name in names:
                    ap=independent_AP(truth[ev]==2,scores[name][ev]);reported=next(x for x in summary['metrics'] if x['product']==product and x['score']==name)['AP']
                    assert (ap is None and reported is None) or abs(ap-reported)<1e-12
                    verified.append(dict(product=product,score=name,AP=ap))
            cov=next(x for x in summary['coverage'] if x['product']==product)
            assert cov['evaluation_centers']==int(ev.sum()) and cov['common_centers']==int(common.sum())
            assert [cov[n] for n in ['Surface','Cloud','Shadow']]==np.bincount(truth[ev],minlength=3).tolist()
    for row in summary['macro']:
        other=row['comparison'].split('-',1)[1];diff=[]
        for r in lock['selected']:
            product=r['product'];v={m['score']:m['AP'] for m in verified if m['product']==product}
            diff.append((v['sun']-v[other])*100)
        assert abs(float(np.mean(diff))-row['mean_delta_AP_pp'])<1e-12
        assert sum(x>0 for x in diff)==row['positive_scenes']
    passed=all(x['mean_delta_AP_pp']>0 and (x['comparison']=='sun-dark' or x['positive_scenes']>=2) for x in summary['macro'] if x['comparison']!='sun-all')
    assert passed==summary['screen_passed']
    result=dict(remote_artifacts_verified=len(hashes),git_blob_verified=True,previous_inputs_verified=True,mask_sha_verified=3,independently_recomputed_AP=verified,
        all_three_common_support_verified=True,all_macro_differences_and_stop_rule_verified=True,EXIT_CODE=0)
    with (out/'LOCAL_VERIFIED.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(verified_artifacts=len(hashes),AP_values=len(verified),screen_passed=passed)))


if __name__=='__main__':main()
