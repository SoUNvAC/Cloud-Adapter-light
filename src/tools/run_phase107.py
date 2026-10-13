"""One RGB readout fit on original fit samples; cached-score evaluation only."""
import io,os,sys,time,resource,zipfile,importlib.metadata
from phase107_common import *
from phase65d_recovery_math import fit_model,standardize

def fit():
    lock=read(RESULT/'pre_fit_lock.json')
    assert all(sha(REPO/p)==h for p,h in lock['code_sha256'].items())
    assert not (ROOT/'FIT_STARTED.json').exists(),'Never repeat an attempted fit'
    assert sha(CAT/'phase67_20261010/control_lock.json')==CONTROL_SHA
    for name,h in lock['historical_numeric_code_sha256'].items():assert sha(REPO/'src/tools'/name)==h
    p=CAT/'phase67_20261010';recovery=CAT/'phase65d_recovery_20261008'
    assert sha(p/'fit_input_audit.json')==lock['fit_input_audit_sha256']
    audit=read(p/'fit_input_audit.json')
    from run_phase65d_recovery import records,PREP
    metadata=records('fit_representatives');assert len(metadata)==153
    samples=[];labels=[];weights=[];receipts=[]
    for r in audit['rows']:
        product=r['product'];m=metadata[product];mask=PREP/m['mask_path']
        assert sha(recovery/r['feature_path'])==r['feature_sha256'] and sha(p/r['rgb_path'])==r['rgb_sha256']
        # Pin the already existing mask via its original prepared manifest SHA.
        suffix=next(k for k in m['prepared_sha256'] if k.endswith('_mask.npy'))
        assert sha(mask)==m['prepared_sha256'][suffix]
        truth=np.load(mask);v=truth!=255;y=truth[v]==2
        old=np.load(recovery/r['feature_path'],mmap_mode='r');rgb=np.load(p/r['rgb_path'],mmap_mode='r')
        x=np.column_stack([old,rgb]).astype(np.float32);assert len(x)==r['valid_pixels']==len(y)
        seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
        idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
        assert array_sha(idx)==r['sample_indices_sha256'] and len(idx)==r['sample_count']
        assert int(y[idx].sum())==r['sample_positive']
        samples.append(x[idx]);labels.append(y[idx]);weights.append(np.full(len(idx),1/(153*len(idx))))
        receipts.append(dict(product=product,group_id=r['group_id'],sample_count=len(idx),sample_indices_sha256=array_sha(idx),
            labels_sha256=array_sha(y[idx]),mask_sha256=sha(mask)))
        print('fit input',len(receipts),153,flush=True)
    x=np.concatenate(samples).astype(np.float64);y=np.concatenate(labels).astype(float);w=np.concatenate(weights)
    assert array_sha(x)==audit['combined_sample_sha256'] and array_sha(y)==audit['sample_y_sha256']
    assert len(x)==audit['sample_count']==1253376 and len({r['group_id'] for r in receipts})==153
    S=lock['original_S'];sx=x[:,[1,2,3,4]];mean,scale=standardize(sx,w/w.sum())
    np.testing.assert_allclose(mean,S['mean'],rtol=0,atol=1e-12);np.testing.assert_allclose(scale,S['scale'],rtol=0,atol=1e-12)
    rx=x[:,[1,5,6,7]];assert rx.flags.f_contiguous
    write(ROOT/'fit_input_lock.json',dict(rows=receipts,combined_sample_sha256=array_sha(x),label_sha256=array_sha(y),
        weights_sha256=array_sha(w),R_input_sha256=array_sha(rx),sample_count=len(x),original_S_standardization_reproduced=True,
        dependencies={n:importlib.metadata.version(n) for n in ['numpy','scipy']},columns=[1,5,6,7],
        weight_rule='1/(153*n_i) then original normalizations',pre_fit_lock_sha256=sha(RESULT/'pre_fit_lock.json')),True)
    write(ROOT/'FIT_STARTED.json',dict(utc=now(),attempt=1,limit=1),True)
    start=time.monotonic();R=fit_model(rx,y,w,1);seconds=time.monotonic()-start
    assert R['penalty']==S['penalty']==.001 and R['positive_features']==S['positive_features']==1
    assert len(R['theta'])==len(S['theta'])==5 and R['theta'][0]>=0 and 'threshold' not in R
    normalized=w/w.sum();normalized=normalized/normalized.sum()
    m2=np.array([np.sum(rx[:,j]*normalized) for j in range(4)])
    s2=np.array([np.sqrt(np.sum((rx[:,j]-m2[j])**2*normalized)) for j in range(4)]);s2[s2==0]=1
    np.testing.assert_allclose(m2,R['mean'],rtol=0,atol=1e-12);np.testing.assert_allclose(s2,R['scale'],rtol=0,atol=1e-12)
    write(ROOT/'R_model.json',R,True)
    write(ROOT/'model_lock.json',dict(status='sole_R_fit_complete_before_evaluation',R_model_sha256=sha(ROOT/'R_model.json'),
        original_S=S,fit_input_lock_sha256=sha(ROOT/'fit_input_lock.json'),fit_attempts=1,fit_seconds=seconds,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        independent_standardization_max_abs_error=float(max(np.max(np.abs(m2-R['mean'])),np.max(np.abs(s2-R['scale'])))),
        new_forwards=0,new_downloads=0,new_thresholds=0),True)
    print('Only R fit frozen; no evaluation observed',flush=True)

def development():
    lock=read(RESULT/'pre_fit_lock.json');R=read(ROOT/'R_model.json');S=lock['original_S']
    assert sha(ROOT/'R_model.json')==read(ROOT/'model_lock.json')['R_model_sha256']
    from run_phase65d_recovery import records,PREP,OLD,load
    index=read(OLD/'development_val/index_source.json');source={r['product']:r for r in index['rows']}
    reference=read(CAT/'phase67_20261010/phase67_report.json')
    Aold={r['product']:r['ranking']['ap'] for r in reference['baseline_reference']['source_original']['rows']}
    Sold={r['product']:r['ranking']['ap'] for r in reference['policies']['Source_spectral']['rows']}
    rows=[];pool={n:[] for n in ('A','R','S')};ys=[];checks=[]
    with zipfile.ZipFile(CAT/'subscenes.zip') as z:
        for product,m in records('development_val').items():
            mask=PREP/m['mask_path'];suffix=next(k for k in m['prepared_sha256'] if k.endswith('_mask.npy'))
            assert sha(mask)==m['prepared_sha256'][suffix]
            truth=np.load(mask);v=truth!=255;y=truth[v]==2
            rawbytes=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(rawbytes).hexdigest()==m['member_sha256']['image']
            raw=np.load(io.BytesIO(rawbytes),allow_pickle=False);ss=load(OLD,source[product],'score')[v].astype(np.float32)
            assert source[product]['group_id']==m['group_id']
            # Reproduce original S/A before computing R for this group.
            sx=np.asfortranarray(np.column_stack([logit_score(ss),raw[...,7][v],raw[...,11][v],raw[...,12][v]]).astype(np.float32))
            a=ap(y,ss);s=predict_logit(sx,S);s_ap=ap(y,s)
            assert a==Aold[product] and s_ap==Sold[product]
            rx=np.asfortranarray(np.column_stack([logit_score(ss),raw[...,3][v],raw[...,2][v],raw[...,1][v]]).astype(np.float32))
            r=predict_logit(rx,R);checkR=scalar_replay(rx,R,r);checkS=scalar_replay(sx,S,s)
            row=dict(group_id=m['group_id'],product=product,tile=product.split('_T')[1][:5],positive=int(y.sum()),negative=int((~y).sum()),
                A_AP=a,R_AP=None,S_AP=s_ap,S_minus_R_pp=None,R_minus_A_pp=None)
            rows.append(row);ys.append(y)
            for n,value in [('A',ss),('R',r),('S',s)]:pool[n].append(value)
            checks.append(dict(product=product,original_A_S_AP_exact=True,R_score_sha256=array_sha(r),A_score_sha256=array_sha(ss),
                               S_score_sha256=array_sha(s),R_CPU=checkR,S_CPU=checkS))
            print('development',len(rows),32,flush=True)
    y=np.concatenate(ys);totals=[]
    # Complete all old A/S group and pooled reproduction before inspecting any R AP.
    pooled_values={name:ap(y,np.concatenate(pool[name])) for name in ('A','S')}
    assert pooled_values['A']==reference['baseline_reference']['source_original']['ranking']['ap']
    assert pooled_values['S']==reference['policies']['Source_spectral']['ranking']['ap']
    for row,yy,rr in zip(rows,ys,pool['R']):
        value=ap(yy,rr);row['R_AP']=value
        if value is not None:
            row['S_minus_R_pp']=(row['S_AP']-value)*100;row['R_minus_A_pp']=(value-row['A_AP'])*100
    for name in ('A','R','S'):
        value=ap(y,np.concatenate(pool[name])) if name=='R' else pooled_values[name]
        if name=='A':assert value==reference['baseline_reference']['source_original']['ranking']['ap']
        if name=='S':assert value==reference['policies']['Source_spectral']['ranking']['ap']
        totals.append(dict(domain='development',score=name,macro_AP_percent=float(np.mean([r[name+'_AP'] for r in rows if r[name+'_AP'] is not None]))*100,
                           pooled_AP_percent=value*100,AP_groups=7,total_groups=32))
    write(ROOT/'development_aggregate.json',dict(rows=rows,totals=totals,comparisons=comparisons(rows,'development'),
        CPU=dict(status='passed',groups=checks,original_A_S_pooled_AP_exact=True,original_A_S_group_AP_exact=True,
                 reference_sha256=sha(CAT/'phase67_20261010/phase67_report.json')),no_point_labels_saved=True),True)

def pixbox():
    lock=read(RESULT/'pre_fit_lock.json');base=SHARED/'data/pixbox_s2_cmix_5036991'
    path=base/'phase106_20261013/point_scores_no_labels.json';meta_path=base/'phase104_20261013/point_inputs_private.json'
    assert sha(path)==lock['PixBox_score_sha256'] and sha(meta_path)==lock['PixBox_input_sha256']
    old=read(path)['rows'];meta={r['point_id']:r for r in read(meta_path)['rows']};valid=[r for r in old if r['input_valid']]
    assert len(old)==9500 and len(valid)==9498
    R=read(ROOT/'R_model.json');S=lock['original_S']
    def inputs(bands):return np.asfortranarray(np.array([[logit_score(r['A']),*[meta[r['point_id']]['bands'][b]['TOA'] for b in bands]] for r in valid],np.float32))
    rx=inputs(('B04','B03','B02'));sx=inputs(('B08','B11','B12'));r=predict_logit(rx,R);s=predict_logit(sx,S)
    assert np.array_equal(s,np.array([q['B'] for q in valid],np.float32))
    checks=dict(R=scalar_replay(rx,R,r),S=scalar_replay(sx,S,s),S_Phase106_scores_bitwise_equal=True)
    rs=dict(zip((q['point_id'] for q in valid),map(float,r)));scores=[];shared={}
    for q in old:
        row={k:q[k] for k in ('point_id','product_id','group_id','row20','col20','input_valid')};row.update(A=q['A'],R=rs.get(q['point_id']),S=q['B'])
        cell=(q['product_id'],q['row20'],q['col20']);value=(row['A'],row['R'],row['S'])
        if cell in shared:assert shared[cell]==value
        shared[cell]=value;scores.append(row)
    write(ROOT/'PixBox_scores_no_labels.json',dict(rows=scores),True)
    write(ROOT/'PixBox_score_seal.json',dict(utc=now(),score_sha256=sha(ROOT/'PixBox_scores_no_labels.json'),
        CPU=checks,all_point_ids=9500,shared20m_score_identity=True,no_label_access=True,
        model_lock_sha256=sha(ROOT/'model_lock.json'),development_aggregate_sha256=sha(ROOT/'development_aggregate.json')),True)

if __name__=='__main__':
    assert not ROOT.exists(),'Preserve prior attempts; never rerun'
    ROOT.mkdir();fit();development();pixbox();write(ROOT/'EXIT_CODE.json',dict(code=0,utc=now()),True)
