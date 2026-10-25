"""Separated image-only pseudo-label fit, independent prediction, and frozen student readout."""
import sys
import traceback
import numpy as np
from phase312_common import *
from images_phase312 import guard,image,source
from math_phase312 import features,fit,predict,topk,tests

def fit_models():
    rows=sorted(load(ROOT/'FIT_IMAGES.json')['rows'],key=lambda r:r['product']);dev=load(ROOT/'EVAL_IMAGES.json')['rows']
    assert len(rows)==100 and len(dev)==32 and not ({r['product'] for r in rows}&{r['product'] for r in dev}) and not ({r['group_id'] for r in rows}&{r['group_id'] for r in dev})
    guard('fit',rows);numerical=tests();old={r['product']:r for r in load(PL/'PSEUDO_LOCK.json')['rows']};rng=np.random.Generator(np.random.PCG64(31201))
    support=[];xs=[];ys=[];flats=[];groups=[];sizes=[];inputs={};pending=[]
    for i,row in enumerate(rows):
        pid=row['product'];entry=old[pid];score=Path(entry['scores_path']);teacher=PL/'teacher'/(pid+'.npz')
        assert sha(score)==entry['scores_sha256'] and sha(teacher)==entry['sha256'];inputs[str(score)]=sha(score);inputs[str(teacher)]=sha(teacher)
        s=np.load(score);assert ah(s)==entry['scores_array_sha256'];six,rgb,V6=image(row,'fit');A,P,pseudo,q=source(s,rgb)
        with np.load(teacher) as z:assert np.array_equal(z['valid'],rgb) and np.array_equal(z['pseudo'],np.where(A,pseudo,255).astype(np.uint8))
        candidate=A&(pseudo<2);eligible=np.flatnonzero(candidate&V6);selected=np.sort(rng.choice(eligible,min(8192,len(eligible)),replace=False))
        assert np.all(np.diff(selected)>0) and (candidate&V6).ravel()[selected].all()
        y=(pseudo.ravel()[selected]==0).astype(np.uint8);x=six.reshape(-1,6)[selected].copy();n=len(selected)
        if n:xs.append(x);ys.append(y);flats.append(selected);groups.append(np.full(n,i,np.int16));sizes.append(n)
        record=dict(product=pid,group_id=row['group_id'],index=i,A_Surface=int((A&(pseudo==0)).sum()),A_Cloud=int((A&(pseudo==1)).sum()),A_Shadow=int((A&(pseudo==2)).sum()),candidate_binary=int(candidate.sum()),V6_binary=int(len(eligible)),V6_extra_excluded_binary=int((candidate&~V6).sum()),RGB_valid=int(rgb.sum()),V6_valid=int(V6.sum()),extra_excluded_RGB=int((rgb&~V6).sum()),sampled=n,sampled_Surface=int(y.sum()),sampled_Cloud=n-int(y.sum()),product_weight=0.,sample_flat_SHA=ah(selected),sample_label_SHA=ah(y),image_SHA=row['image_member_sha256'],scores_path=str(score),scores_SHA=entry['scores_sha256'],truth_reads=False)
        support.append(record);print('fit sampling',i+1,n,flush=True)
    assert sum(r['A_Surface'] for r in support)==40622307 and sum(r['A_Cloud'] for r in support)==33554491 and sum(r['A_Shadow'] for r in support)==1292360
    nonempty=len(sizes)
    for row in support:row['product_weight']=1/nonempty if row['sampled'] else 0.
    table(ROOT/'FIT_SUPPORT.csv',support)
    assert nonempty>0,'Empty pseudo-label fit pool; support preserved'
    x=np.concatenate(xs);y=np.concatenate(ys);flat=np.concatenate(flats);group=np.concatenate(groups);w=np.concatenate([np.full(n,1/(nonempty*n),np.float64) for n in sizes])
    assert abs(w.sum()-1)<1e-12 and all(abs(w[group==r['index']].sum()-r['product_weight'])<1e-12 for r in support)
    np.savez_compressed(ROOT/'sampling.npz',TOA=x,y_Surface=y,flat_indices=flat,product_index=group,weight=w)
    models=[];fit_attempts=0
    for name in MODELS:
        dest=ROOT/'models'/(name+'.json');result=None;failure=None
        if len(np.unique(y))<2:failure='Total sampled pseudo-label pool is single class; no fit attempted'
        else:
            try:
                append('FIT_ATTEMPTS.jsonl',dict(model=name,attempt=1,initial='all zero',epoch=time.time()));fit_attempts+=1
                result=fit(features(x,name),y,w,sizes)
                if not result['optimizer_success']:failure='Optimizer did not converge: '+result['optimizer_message']
            except Exception:failure=traceback.format_exc()
        record=dict(model=name,status='passed' if not failure else 'failed',features=list(BANDS[:3]) if name=='RGB3' else ['R','G','B','R_squared','G_squared','B_squared'] if name=='RGB6Q' else list(BANDS),parameter_count=4 if name=='RGB3' else 7,fit_products=nonempty,samples=len(y),sample_Surface=int(y.sum()),sample_Cloud=int((y==0).sum()),sampling_SHA=sha(ROOT/'sampling.npz'),sample_ids_SHA=ah(np.c_[group.astype(np.int64),flat]),labels_SHA=ah(y),weights_SHA=ah(w),feature_order=name,result=result,failure=failure,truth_reads=False)
        write(dest,record)
        models.append(dict(model=name,status=record['status'],features=record['features'],parameters=record['parameter_count'],fit_products=nonempty,samples=len(y),sampling_SHA=record['sampling_SHA'],parameter_scaler_SHA=sha(dest),objective=result['objective'] if result else None,iterations=result['iterations'] if result else None,gradient_inf=result['final_gradient_inf'] if result else None,converged=not bool(failure),probability_reproduction_error=None,failure=failure))
        if failure:append('MODEL_FAILURES.jsonl',dict(model=name,failure=failure,retained_parameter_file=str(dest),retry=False))
        print('fit model',name,record['status'],flush=True)
    table(ROOT/'MODEL_FIT.csv',models)
    write(ROOT/'FIT_INPUT_SHA256.json',inputs)
    write(ROOT/'FIT_LOCK.json',dict(status='sealed',sealed_epoch=time.time(),code_lock_SHA=sha(ROOT/'EXECUTION_LOCK.json'),sampling_SHA=sha(ROOT/'sampling.npz'),support_SHA=sha(ROOT/'FIT_SUPPORT.csv'),models={n:sha(ROOT/'models'/(n+'.json')) for n in MODELS},numerical_tests=numerical,successful_models=[r['model'] for r in models if r['status']=='passed'],fit_attempts=fit_attempts,nonempty_products=nonempty,samples=len(y),seed=31201,PCG64=True,positive_class='Source-Surface',no_test_scaler_or_fit=True,no_reference_reads=True,neural_calls=0))

def predict_models():
    lock=load(ROOT/'FIT_LOCK.json');assert lock['status']=='sealed'
    rows=sorted(load(ROOT/'EVAL_IMAGES.json')['rows'],key=lambda r:r['product']);guard('predict',rows)
    models={n:load(ROOT/'models'/(n+'.json')) for n in lock['successful_models']}
    for n in MODELS:assert sha(ROOT/'models'/(n+'.json'))==lock['models'][n]
    old={(r['arm'],r['product']):r for r in load(PL/'EVALUATION_LOCK.json')['rows']};out=[];errors={n:0. for n in models}
    for row in rows:
        pid=row['product'];path=OLD/'evaluation'/('Source_'+pid+'.npz');assert sha(path)==old['Source',pid]['sha256']
        six,rgb,V6=image(row,'predict')
        with np.load(path) as z:s=z['scores'];assert np.array_equal(z['valid'],rgb) and np.array_equal(z['flat_indices'],np.flatnonzero(rgb))
        A,P,pred,q=source(s,rgb);ids=np.flatnonzero(P&V6);raw=six.reshape(-1,6)[ids]
        values=dict(flat_indices=ids,qSurface=q.ravel()[ids],P_pack=np.packbits(P),V6_pack=np.packbits(V6),RGB_valid_pack=np.packbits(rgb))
        for name,model in models.items():
            x=features(raw,name);r=predict(x,model['result']);values['r_'+name]=r;values['protect_'+name]=np.packbits(r>=.9)
            again=load(ROOT/'models'/(name+'.json'))['result'];z=(x-np.asarray(again['mean']))/np.asarray(again['scale']);theta=np.asarray(again['theta'])
            independent=np.exp(-np.logaddexp(0,-(np.sum(z*theta[:-1],axis=1)+theta[-1])))
            err=float(np.max(np.abs(r-independent))) if len(r) else 0.;assert err<=1e-12;errors[name]=max(errors[name],err)
        dest=ROOT/'predictions'/(pid+'.npz');np.savez_compressed(dest,**values)
        with np.load(dest) as z:
            assert np.array_equal(z['flat_indices'],ids)
            for name in models:assert np.array_equal(z['r_'+name],values['r_'+name])
        out.append(dict(product=pid,group_id=row['group_id'],prediction_file=str(dest),sha256=sha(dest),flat_SHA=ah(ids),P=int(P.sum()),P_V6=len(ids),RGB_valid=int(rgb.sum()),V6=int(V6.sum()),extra_excluded_RGB=int((rgb&~V6).sum()),source_file=str(path),source_SHA=sha(path),image_SHA=row['image_member_sha256']))
        print('predict',len(out),flush=True)
    fit_table=read(ROOT/'MODEL_FIT.csv')
    for record in fit_table:record['probability_reproduction_error']=errors.get(record['model'])
    table(ROOT/'MODEL_FIT.csv',fit_table)
    write(ROOT/'PREDICTION_LOCK.json',dict(status='sealed',rows=out,fit_lock_SHA=sha(ROOT/'FIT_LOCK.json'),model_fit_table_SHA=sha(ROOT/'MODEL_FIT.csv'),sealed_epoch=time.time(),probability_reproduction_max_error=errors,threshold=.9,positive='pseudo_Surface',student_or_reference_used=False,neural_calls=0))

def readout():
    fitlock=load(ROOT/'FIT_LOCK.json');prediction=load(ROOT/'PREDICTION_LOCK.json');assert prediction['status']=='sealed'
    rows=load(ROOT/'EVAL_IMAGES.json')['rows'];guard('readout',rows)
    old={(r['arm'],r['product']):r for r in load(PL/'EVALUATION_LOCK.json')['rows']};out=[];counts=[]
    for row in prediction['rows']:
        pid=row['product'];assert sha(row['prediction_file'])==row['sha256'];path=PL/'evaluation'/('LoRA_'+pid+'.npz');assert sha(path)==old['LoRA',pid]['sha256']
        with np.load(path) as z:pred=z['pred'];valid=z['valid'];shadow=z['shadow']
        with np.load(row['prediction_file']) as z:
            ids=z['flat_indices'];sourceq=z['qSurface'];imagevalid=np.unpackbits(z['RGB_valid_pack'],count=1022**2).reshape(1022,1022).astype(bool)
            assert np.array_equal(valid,imagevalid);candidate=pred.ravel()[ids]==1;rules={'Q90-PL':np.zeros(len(ids),bool)};protect={}
            for name in fitlock['successful_models']:
                protect[name]=np.unpackbits(z['protect_'+name],count=len(ids)).astype(bool);assert np.array_equal(protect[name],z['r_'+name]>=.9)
                rules[name+'_90']=protect[name]&candidate
            if 'MS6' in protect:
                k=int(rules['MS6_90'].sum());candidate_ids=np.flatnonzero(candidate)
                for name in ('RGB3','RGB6Q','Source_qSurface'):
                    if name!='Source_qSurface' and name not in protect:continue
                    scores=sourceq if name=='Source_qSurface' else z['r_'+name];chosen=topk(scores[candidate],ids[candidate],k)
                    select=np.zeros(len(ids),bool);select[candidate_ids[chosen]]=True;assert int(select.sum())==k
                    rules[name+'_equalMS6']=select
            values=dict(flat_indices=ids,C_pack=np.packbits(candidate))
            for name,select in rules.items():
                assert not (select&~candidate).any();result=pred.copy();result.ravel()[ids[select]]=0
                assert np.array_equal(result==2,pred==2) and np.all((result==pred)|((pred==1)&(result==0)))
                values[name]=np.packbits(select)
                counts.append(dict(rule=name,product=pid,group_id=row['group_id'],P=row['P'],P_V6=row['P_V6'],C_image=int(candidate.sum()),selected_image=int(select.sum()),own_threshold_P_protected=int(protect[name[:-3]].sum()) if name.endswith('_90') else None,Shadow_binary_unchanged=True,Shadow_score_reused_unchanged=True))
        dest=ROOT/'readouts'/(pid+'.npz');np.savez_compressed(dest,**values)
        out.append(dict(**row,readout_file=str(dest),readout_SHA=sha(dest),student_file=str(path),student_SHA=sha(path),rules=sorted(rules),Shadow_score_SHA=ah(shadow)))
    table(ROOT/'READOUT_IMAGE_COUNTS.csv',counts)
    write(ROOT/'READOUT_LOCK.json',dict(status='sealed',sealed_epoch=time.time(),rows=out,fit_lock_SHA=sha(ROOT/'FIT_LOCK.json'),prediction_lock_SHA=sha(ROOT/'PREDICTION_LOCK.json'),counts_SHA=sha(ROOT/'READOUT_IMAGE_COUNTS.csv'),reference_access=False,source_reference='PL EVALUATION_LOCK',unchanged_Shadow_binary_and_score=True,neural_calls=0))

if __name__=='__main__':
    try:
        assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python' and 'torch' not in sys.modules
        assert all(1<=int(os.environ[n])<=4 for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
        for n,h in load(ROOT/'EXECUTION_LOCK.json')['code'].items():assert sha(R/n)==h,n
        {'fit':fit_models,'predict':predict_models,'readout':readout}[sys.argv[1]]()
        assert 'torch' not in sys.modules
    except BaseException:append('FAILURES.jsonl',dict(stage=sys.argv[1],traceback=traceback.format_exc()));raise
