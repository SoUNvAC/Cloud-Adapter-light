"""Independent frozen-score, AP/confusion and maximal full-fit threshold audit."""
import hashlib,io,subprocess,zipfile
import numpy as np
from phase82_common import *
from run_phase65d_recovery import records,prepare_verify,load
from phase65d_recovery_math import logit_score
from verify_phase78_local import independent_AP

def calc(x,m):
    return (((np.asarray(x)-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['theta'][:-1])+m['theta'][-1]).astype(np.float32)

def main():
    rep=read(ROOT/'phase82_report.json');cl=read(ROOT/'calibration_lock.json');model=cl['models']['D+S']
    audit=read(ROOT/'development_input_audit.json');idx={r['product']:r for r in read(ROOT/'development_val/index.json')['rows']}
    refs={r['product']:r for r in rep['policies']['D+S']['rows']};dev=records('development_val')
    assert set(refs)==set(dev)==set(idx);pools=[];labels=[];checks=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for row in audit['rows']:
            product=row['product'];r=dev[product];prepare_verify(r)
            t=np.load(PREP/r['mask_path']);v=t!=255;t=t[v]
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            x=np.column_stack([logit_score(load(ROOT,idx[product],'score')[v]),raw[...,7][v],raw[...,11][v],raw[...,12][v]]).astype(np.float32)
            score=calc(x,model);path=ROOT/'development_scores'/(product+'.npy')
            assert digest(path)==row['score_sha256'] and np.array_equal(score,np.load(path))
            pred=np.where(score>=model['threshold'],2,load(ROOT,idx[product],'nonshadow')[v])
            cm=np.array([[np.count_nonzero((t==a)&(pred==b)) for b in range(3)] for a in range(3)])
            assert cm.tolist()==refs[product]['confusion'];ap=independent_AP(t==2,score)
            assert (ap is None and refs[product]['ap'] is None) or abs(ap-refs[product]['ap'])<1e-12
            checks.append(dict(product=product,AP=ap,confusion_exact=True,score_exact=True));pools.append(score);labels.append(t==2)
    ap=independent_AP(np.concatenate(labels),np.concatenate(pools));assert abs(ap-rep['policies']['D+S']['pooled_ap'])<1e-12
    fitrows=records('fit_representatives');fi=read(ROOT/'fit_input_audit.json');rates=[];below=[];counts=[]
    idx={r['product']:r for r in read(ROOT/'fit_representatives/index.json')['rows']}
    old={r['product']:r for r in read(RECOVERY/'fit_feature_index.json')['rows']}
    for row in fi['rows']:
        product=row['product'];path=ROOT/row['feature_path'];assert digest(path)==row['feature_sha256']
        x=np.load(path,mmap_mode='r');truth=np.load(PREP/fitrows[product]['mask_path']);valid=truth!=255
        assert np.array_equal(x[:,0],logit_score(load(ROOT,idx[product],'score')[valid]).astype(np.float32))
        original=RECOVERY/old[product]['feature_path'];assert digest(original)==old[product]['feature_sha256']==row['old_feature_sha256']
        assert np.array_equal(x[:,1:],np.load(original,mmap_mode='r')[:,2:5])
        yy=truth[valid]==2;neg=calc(x,model)[~yy];rates.append(float(np.mean(neg>=model['threshold'])))
        low=neg[neg<model['threshold']];below.append(float(low.max()) if len(low) else -np.inf)
        counts.append(dict(product=product,negative=len(neg),false_positive=int(np.sum(neg>=model['threshold']))))
    macro=float(np.mean(rates));assert abs(macro-model['fit_macro_fpr'])<1e-12 and macro<=.01
    candidate=max(below);candidate_rates=[]
    assert np.isfinite(candidate)
    for row in fi['rows']:
        x=np.load(ROOT/row['feature_path'],mmap_mode='r');truth=np.load(PREP/fitrows[row['product']]['mask_path']);yy=truth[truth!=255]==2
        neg=calc(x,model)[~yy];candidate_rates.append(float(np.mean(neg>=candidate)))
    next_rate=float(np.mean(candidate_rates));assert next_rate>.01
    write(ROOT/'INDEPENDENT_SHARED_VERIFIED.json',dict(status='passed',rows=checks,pooled_AP=ap,fit_full_negative_rows=counts,
        fit_macro_FPR=macro,threshold=model['threshold'],next_lower_distinct_score=candidate,next_lower_macro_FPR=next_rate,
        all_153_fit_features_reconstructed_from_student_cache_and_frozen_spectral=True,no_new_inference_or_fitting=True,
        code_sha256=digest(Path(__file__)),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
    print('Phase82 independent shared pixel and full-fit threshold checks passed',flush=True)

if __name__=='__main__':main()
