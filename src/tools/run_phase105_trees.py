"""Original Phase93 environment, zero fits; independent CPU tree replay all points."""
import pickle,sys,time,importlib.metadata,resource,math
import numpy as np
from phase105_common import *
from phase65d_recovery_math import logit_score
from phase93_math import independent_probability

def main():
    started=time.monotonic();env=read(MODELS/'dependency_lock.json')['versions']
    assert {n:importlib.metadata.version(n) for n in env}==env
    source=read(ROOT/'source_points_no_labels.json')['rows'];sr={r['point_id']:r for r in source}
    receipt=read(ROOT/'forward_receipts.json');assert sha(ROOT/'source_points_no_labels.json')==receipt['source_points_sha256']
    assert receipt['actual_forwards']==531 and all(r['passed'] for r in receipt['repeats'])
    original=read(INPUT/'point_inputs_private.json')['rows'];valid=[r for r in original if r['input_valid']]
    assert len(original)==9500 and len(valid)==9498 and set(sr)=={r['point_id'] for r in original}
    features=[]
    for r in valid:
        p=sr[r['point_id']];assert all(p[k]==r[k] for k in ('product_id','row20','col20'))
        x=[np.float32(r['bands'][b]['TOA']) for b in ('B04','B03','B02','B08','B11','B12')]
        x.append(np.float32(logit_score(p['Source'])))
        features.append(x)
    x=np.asarray(features,dtype=np.float64);assert np.isfinite(x).all()
    replay=[]
    for r in valid:
        physical=[]
        for b in ('B04','B03','B02','B08','B11','B12'):
            dn=[v for line in r['bands'][b]['DN'] for v in line]
            physical.append(np.float32(sum(float(np.float32(v)/np.float32(10000)) for v in dn)/len(dn)))
        probability=max(1e-6,min(1-1e-6,sr[r['point_id']]['Source']))
        physical.append(np.float32(math.log(probability)-math.log1p(-probability)))
        replay.append(physical)
    replay=np.asarray(replay,np.float64);assert np.array_equal(x,replay),'Independent point input replay mismatch'
    predictions={};verification=[]
    for name,width in [('G6',6),('G7',7)]:
        path=MODELS/(name+'.pkl');assert sha(path)==MODEL_SHA[name]
        with path.open('rb') as f:model=pickle.load(f)
        assert np.array_equal(model.classes_,[0,1]) and model.n_features_in_==width
        score=model.predict_proba(x[:,:width])[:,1].astype(np.float32)
        with path.open('rb') as f:reload=pickle.load(f)
        assert np.array_equal(score,reload.predict_proba(x[:,:width])[:,1].astype(np.float32))
        independent=independent_probability(reload,replay[:,:width]);err=float(np.max(np.abs(independent-score)))
        assert err<=6e-8 and np.isfinite(score).all() and (score>=0).all() and (score<=1).all()
        predictions[name]=dict(zip((r['point_id'] for r in valid),map(float,score)))
        verification.append(dict(model=name,points=len(valid),reload_bitwise_equal=True,independent_CPU_tree_max_abs_error=err,
                                  model_sha256=sha(path),feature_columns=width))
    scores=[];shared={}
    for r in original:
        q=dict(point_id=r['point_id'],product_id=r['product_id'],group_id=r['group_id'],row20=r['row20'],col20=r['col20'],
            input_valid=r['input_valid'],Source=sr[r['point_id']]['Source'],
            G6=predictions['G6'].get(r['point_id']),G7=predictions['G7'].get(r['point_id']))
        cell=(r['product_id'],r['row20'],r['col20']);value=(q['Source'],q['G6'],q['G7'])
        if cell in shared:assert shared[cell]==value
        shared[cell]=value;scores.append(q)
    write(ROOT/'point_scores_no_labels.json',dict(rows=scores),True)
    verification=dict(status='passed',rows=verification,environment_versions=env,
        zero_fit=True,all_original_points=9500,valid_input_points=9498,shared20m_cells_consistent=True,independent_point_inputs_bitwise_equal=True,
        distinct20m_cells=len(shared),score_sha256=sha(ROOT/'point_scores_no_labels.json'),
        point_inputs_sha256=sha(INPUT/'point_inputs_private.json'),feature_matrix_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        seconds=time.monotonic()-started,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(ROOT/'CPU_tree_acceptance.json',verification,True)
    write(ROOT/'SCORES_SEALED.json',dict(status='no_label_scores_complete_before_local_reference_join',
        utc=now(),score_sha256=sha(ROOT/'point_scores_no_labels.json'),
        evaluation_lock_sha256=sha(ROOT/'evaluation_lock.json'),forward_receipts_sha256=sha(ROOT/'forward_receipts.json'),
        CPU_tree_acceptance_sha256=sha(ROOT/'CPU_tree_acceptance.json'),model_sha256=MODEL_SHA),True)
    print('Phase105 no-label point scores sealed; CPU tree replay passed',flush=True)

if __name__=='__main__':main()
