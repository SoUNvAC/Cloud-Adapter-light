"""One SourceS retry ONLY after explicit user authorization; never refit G7."""
import argparse, hashlib, io, pickle, time, zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import run_phase94 as r

def main(authorized):
    assert authorized,'Explicit user authorization is required for this additional fit'
    d=r.D;read=r.p.old.read;digest=r.p.old.digest
    assert (d/'G7_100.pkl').is_file() and not (d/'SourceS_100.json').exists()
    assert not (d/'two_model_lock.json').exists() and not (d/'repair_authorization_lock.json').exists()
    assert (r.p.old.SHARED/'phase94_control_20261012/EXIT_CODE').read_text().strip()=='1'
    sl=read(d/'split_input_lock.json');fi=read(d/'fit_input_lock.json');assert digest(d/'split_input_lock.json')==fi['split_lock_sha256']
    assert sl['indices_before']==r.indices_audit();assert sl['protocol_sha256']==digest(r.PROTOCOL)
    fitting=sl['fit'];calibration=sl['calibration'];audit=read(r.fit67.ROOT/'fit_input_audit.json')
    assert r.split_rows(fitting+calibration)==(fitting,calibration)
    tree_sha=digest(d/'G7_100.pkl')
    with (d/'G7_100.pkl').open('rb') as f:tree=pickle.load(f)
    assert tree.n_iter_==200 and all(tree.get_params()[k]==v for k,v in r.PARAMS.items())
    delta={}
    for n,h in sl['code_sha256'].items():
        now=digest(Path(__file__).with_name(n))
        if now!=h:
            assert n in ['run_phase94.py','verify_phase94_delivery.py'];delta[n]=dict(before=h,after=now)
    r.write('repair_authorization_lock.json',dict(created_utc=r.p.old.now(),additional_SourceS_fit_authorized=True,
        authorization=authorized,G7_reused_without_refit=True,G7_preserved_sha256=tree_sha,
        failed_attempt='SourceS optimization finished but strict roundoff assertion stopped before coefficients were saved; no calibration/external evaluation',
        original_fit_attempts=2,additional_fit_attempts=1,final_models=2,code_delta=delta,
        resume_code_sha256=digest(Path(__file__)),protocol_unchanged=True))
    begin=time.monotonic();r.fit67.base_verify();records=r.recovery.records('fit_representatives');am={x['product']:x for x in audit['rows']};samples=[];labels=[]
    with zipfile.ZipFile(r.recovery.DATA/'subscenes.zip') as z:
        for i,a in enumerate(fitting):
            rec=records[a['product']];r.recovery.prepare_verify(rec);truth=np.load(r.recovery.PREP/rec['mask_path']);valid=truth!=255
            rawbytes=z.read('subscenes/'+a['product']+'.npy');assert hashlib.sha256(rawbytes).hexdigest()==rec['member_sha256']['image']
            raw=np.load(io.BytesIO(rawbytes),allow_pickle=False);x=r.fit67.full_fit_features(am[a['product']])
            assert np.array_equal(x[:,2:5],raw[...,[7,11,12]][valid].astype(np.float32)) and np.array_equal(x[:,5:8],raw[...,[3,2,1]][valid].astype(np.float32))
            seed=65081+int(hashlib.sha256(a['product'].encode()).hexdigest()[:8],16);idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False);yy=truth[valid]==2
            assert r.sha(idx)==a['sample_indices_sha256'] and r.sha(yy[idx])==a['sample_truth_sha256'];samples.append(x[idx]);labels.append(yy[idx])
            print('Phase94 recovery input',i+1,100,flush=True)
    x=np.concatenate(samples).astype(float);y=np.concatenate(labels).astype(float);del samples,labels
    assert r.sha(x)==fi['sample_features_sha256'] and r.sha(y)==fi['sample_labels_sha256'] and np.unique(y).tolist()==[0.,1.]
    w=r.equal_weights([a['sample_count'] for a in fitting]);w=w/w.sum();assert r.sha(w)==fi['logistic_weights_sha256']
    t=time.monotonic();m=r.fit_model(r.columns(x,'SourceS_100'),y,w,1)
    # Save before checking, so a diagnostic failure cannot lose fitted coefficients again.
    r.write('SourceS_100.json',m);reload=read(d/'SourceS_100.json')
    mean,scale=r.standardize(r.columns(x,'SourceS_100'),w/w.sum())
    assert np.array_equal(mean,m['mean']) and np.array_equal(scale,m['scale']) and m['theta'][0]>=0
    assert np.array_equal(r.predict_logit(r.columns(x,'SourceS_100'),m),r.predict_logit(r.columns(x,'SourceS_100'),reload))
    assert digest(d/'G7_100.pkl')==tree_sha
    cost={'G7_100':dict(reused=True,fit_seconds=None,reason='first process exited before cost receipt; not fabricated'),
          'SourceS_100':dict(fit_seconds=time.monotonic()-t,process_peak_rss_bytes=r.rss(),all_training_reload_exact=True,additional_authorized_fit=True)}
    r.write('two_model_lock.json',dict(created_utc=r.p.old.now(),fit_input_lock_sha256=digest(d/'fit_input_lock.json'),
        models={'G7_100':dict(file='G7_100.pkl',sha256=tree_sha,params=r.PARAMS),
                'SourceS_100':dict(file='SourceS_100.json',sha256=digest(d/'SourceS_100.json'),params=m)},
        new_fits=3,final_models=2,calibration_labels_not_used_in_fit=True,cost=cost,
        repair_authorization_lock_sha256=digest(d/'repair_authorization_lock.json')))
    del x,y,w
    r.calibrate({'G7_100':tree,'SourceS_100':reload},records,audit,fitting,calibration,begin)
    r.evaluate()
    from verify_phase94_delivery import verify
    print(verify(d),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',required=True);args=parser.parse_args();r.q.SEED=r.SEED
    with threadpool_limits(limits=4):main(args.authorization)
