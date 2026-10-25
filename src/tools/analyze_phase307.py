"""Supervised offline information probes: fixed LOGO, no segmentation training or neural calls."""
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import auc_phase306 as ranking
from math_phase307 import features,fit,predict,numerical_tests
from phase307_common import *

INPUTS={}
COMPARISONS=(('MS6','RGB6Q'),('MS6','RGB3'),('RGB6Q','RGB3'))
BINS=(('q[0,0.5)',0.,.5),('q[0.5,0.7)',.5,.7),('q[0.7,0.9)',.7,.9))
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def pin(p):INPUTS[str(p)]=sha(p);return load(p)
def psha(p):INPUTS[str(p)]=sha(p);return INPUTS[str(p)]
def append(name,v):
    with (ROOT/name).open('a',encoding='utf-8') as f:f.write(__import__('json').dumps(v,allow_nan=False)+'\n')
def average(v):
    v=[x for x in v if x is not None];return sum(v)/len(v) if v else None

def measurements(codes,p,choose):
    codes=codes[choose];p=p[choose]
    d=codes==0;k=codes==2;shadow=codes==3;binary=d|k;y=k[binary];pp=p[binary]
    cloud=p>=.50
    nD=int(d.sum());nK=int(k.sum());nS=int(shadow.sum())
    repaired=int((d&~cloud).sum());revoked=int((k&~cloud).sum());remaining=int((d&cloud).sum());retained=int((k&cloud).sum())
    assert repaired+remaining==nD and revoked+retained==nK and nD+nK+nS==len(codes)
    mat=np.array([[repaired,remaining],[revoked,retained]],np.int64)
    if len(pp):
        independently=np.histogram2d(y.astype(np.uint8),(pp>=.5).astype(np.uint8),bins=([-0.5,.5,1.5],[-.5,.5,1.5]))[0].astype(np.int64)
        assert np.array_equal(mat,independently)
    return dict(C=len(codes),D=nD,K=nK,C_Shadow=nS,AUC=ranking.auc(y,pp),
        Brier=float(np.mean((pp-y.astype(np.float64))**2)) if len(pp) else None,
        repaired_D=repaired,revoked_K=revoked,remaining_D=remaining,retained_K=retained,
        Shadow_Cloud_to_Surface=int((shadow&~cloud).sum()),threshold_Cloud=.5)

def aggregate(model,predictions,choose_fn=lambda r:np.ones(len(r['codes']),bool)):
    parts=[];codes=[];scores=[]
    for row in predictions:
        if row['model']!=model:continue
        choose=choose_fn(row);parts.append(measurements(row['codes'],row['p'],choose))
        codes.append(row['codes'][choose]);scores.append(row['p'][choose])
    cc=np.concatenate(codes) if codes else np.empty(0,np.uint8);pp=np.concatenate(scores) if scores else np.empty(0,np.float64)
    pooled=measurements(cc,pp,np.ones(len(cc),bool))
    return dict(model=model,successful_nonempty_C_groups=sum(r['C']>0 for r in parts),
        AUC_groups=sum(r['AUC'] is not None for r in parts),Brier_groups=sum(r['Brier'] is not None for r in parts),
        group_mean_AUC=average(r['AUC'] for r in parts),pooled_AUC=pooled['AUC'],
        group_mean_Brier=average(r['Brier'] for r in parts),pooled_Brier=pooled['Brier'],
        **{k:pooled[k] for k in ('C','D','K','C_Shadow','repaired_D','revoked_K','remaining_D','retained_K','Shadow_Cloud_to_Surface')})

def main():
    started=time.monotonic();assert 'torch' not in sys.modules
    assert sys.executable=='/home/scv/miniconda3/envs/cloud-lite-pt210/bin/python'
    assert all(1<=int(os.environ[k])<=4 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=str(args[0]).replace('\\','/').lower()
            if any(x in name for x in ('masks.zip','subscenes.zip','classification_tags.csv','/fit_images','/teacher/','/pixbox','/l8')):
                raise PermissionError('Phase307 only fixed 32-group cached information-probe inputs: '+name)
    sys.addaudithook(guard)
    execution=pin(ROOT/'EXECUTION_LOCK.json')
    for n,h in execution['code'].items():assert sha(R/n)==h,n
    assert sha(R/'src/research_plans/phase307.md')==execution['protocol_sha256']
    math_tests=numerical_tests();auc_tests=ranking.numerical_tests()
    old_summary=R/'src/research_results/phase306_20261025';parent=pin(old_summary/'SUMMARY_SHA256.json')
    for n,h in parent['files'].items():assert sha(old_summary/n)==h,n
    backup=pin(P/'BACKUP_FILES.json');receipt=pin(P/'ARCHIVE_RECEIPT.json')
    assert pin(old_summary/'BACKUP_RECEIPT.json')['sha256']==receipt['sha256']
    assert psha(P/'delivery.tar.gz')==receipt['sha256']
    for n in ('SUPPORT_AND_SAMPLE_SHA.csv','CHANNEL_LOCK.json','EVENTS_PER_GROUP.csv','SCENE_MAPPING.csv','TOP_1_3_5_CONCENTRATION.csv','INPUT_SHA256.json'):
        assert psha(P/n)==parent['files'][n],n
    channels=pin(P/'CHANNEL_LOCK.json');assert channels['bands']==list(BANDS) and channels['indices']==list(INDICES)
    assert channels['event_codes']==dict(D=0,U=1,K=2,C_Shadow=3)
    manifests=read(P/'SUPPORT_AND_SAMPLE_SHA.csv');assert len(manifests)==32 and len({r['group_id'] for r in manifests})==32
    mapping={(r['product'],r['field']):r['value'] for r in read(P/'SCENE_MAPPING.csv')}
    expected={r['product']:r for r in read(P/'EVENTS_PER_GROUP.csv')}
    previous_inputs=pin(P/'INPUT_SHA256.json');top=read(P/'TOP_1_3_5_CONCENTRATION.csv')
    groups=[];sample_lock=[];rng=np.random.default_rng(30701)
    for name in ('sampling','models','oof','failures'):(ROOT/name).mkdir()
    for index,row in enumerate(sorted(manifests,key=lambda r:r['group_id'])):
        path=P/row['samples_file'];assert psha(path)==row['samples_sha256']==backup[row['samples_file']]['sha256']
        with np.load(path) as z:
            original_flat=z['flat_indices'];event=z['event'];values=z['TOA'];q=z['Source_qSurface']
            common=np.unpackbits(z['common_packbits'],count=1022*1022).astype(bool).reshape(1022,1022)
            six=np.unpackbits(z['six_common_packbits'],count=1022*1022).astype(bool).reshape(1022,1022)
        assert ah(common)==row['common_sha256'] and ah(six)==row['six_common_sha256'] and not (six&~common).any()
        assert ah(original_flat)==row['selected_flat_indices_sha256'] and np.all(np.diff(original_flat)>0)
        assert values.shape==(len(event),6) and values.dtype==np.float32 and np.isfinite(values).all()
        assert len(q)==len(event) and ((q>=0)&(q<np.float32(.9))).all() and np.isin(event,[0,1,2,3]).all()
        candidate_rows=np.flatnonzero(event!=1);codes=event[candidate_rows].copy();flat=original_flat[candidate_rows].copy()
        raw=values[candidate_rows].copy();qq=q[candidate_rows].copy()
        assert np.isin(codes,[0,2,3]).all() and six.ravel()[flat].all()
        ex=expected[row['product']]
        assert [int((codes==c).sum()) for c in (0,2,3)]==[int(ex[k]) for k in ('D','K','C_Shadow')]
        eligible=np.flatnonzero(codes!=3)
        selected=np.sort(rng.choice(eligible,size=min(8192,len(eligible)),replace=False))
        assert len(np.unique(selected))==len(selected) and np.isin(selected,eligible).all()
        sample_path=ROOT/'sampling'/(row['group_id']+'.npz')
        np.savez_compressed(sample_path,candidate_rows=selected,source_sample_rows=candidate_rows[selected],flat_indices=flat[selected],y=(codes[selected]==2).astype(np.uint8))
        sample_lock.append(dict(fold=index,group_id=row['group_id'],product=row['product'],D=int((codes==0).sum()),K=int((codes==2).sum()),
            C_Shadow=int((codes==3).sum()),C=len(codes),selected=len(selected),selected_D=int((codes[selected]==0).sum()),selected_K=int((codes[selected]==2).sum()),
            selected_flat_sha256=ah(flat[selected]),source_npz_sha256=row['samples_sha256'],sample_file=sample_path.relative_to(ROOT).as_posix(),sample_sha256=sha(sample_path),
            test_C_support_sha256=ah(flat),sampling='one uniform without-replacement draw from all D/K; no balancing'))
        groups.append(dict(fold=index,group_id=row['group_id'],product=row['product'],snow=mapping[row['product'],'snow/ice'],
            raw=raw,flat=flat,codes=codes,q=qq,selected=selected,sample=sample_lock[-1]))
    assert sum(len(r['codes']) for r in groups)==546400
    assert [sum(int((r['codes']==c).sum()) for r in groups) for c in (0,2,3)]==[450058,84005,12337]
    table(ROOT/'GROUP_SAMPLING.csv',sample_lock)
    write(ROOT/'SAMPLING_LOCK.json',dict(seed=30701,cap_per_group=8192,group_order=[g['group_id'] for g in groups],
        no_class_balancing=True,no_feature_or_score_selection=True,before_any_fit=True,table_sha256=sha(ROOT/'GROUP_SAMPLING.csv'),rows=sample_lock))
    folds=[dict(fold=g['fold'],heldout_group=g['group_id'],product=g['product'],training_groups=[r['group_id'] for r in groups if r['group_id']!=g['group_id']],
        models=list(MODELS),test_C=len(g['codes']),sample_lock_sha256=sha(ROOT/'SAMPLING_LOCK.json')) for g in groups]
    write(ROOT/'FOLD_LOCK.json',dict(rows=folds,groups=32,models=list(MODELS),main_comparison='MS6 minus RGB6Q',
        supervised_target_reference_labels=True,not_SFDA_result=True,not_new_blind_test=True,threshold_Cloud=.50))
    rows=[];predictions=[];prediction_lock=[];fit_count=0;failures=[];plans=[]
    for test in groups:
        train=[g for g in groups if g['group_id']!=test['group_id'] and len(g['selected'])]
        assert test['group_id'] not in {g['group_id'] for g in train} and len(train)==len({g['group_id'] for g in train})
        sizes=[len(g['selected']) for g in train]
        x=np.concatenate([g['raw'][g['selected']] for g in train]).astype(np.float64)
        y=np.concatenate([(g['codes'][g['selected']]==2).astype(np.float64) for g in train])
        w=np.concatenate([np.full(n,1/(n*len(train)),np.float64) for n in sizes]);w/=w.sum()
        offsets=np.r_[0,np.cumsum(sizes)]
        assert abs(w.sum()-1)<1e-14 and all(abs(w[offsets[i]:offsets[i+1]].sum()-1/len(train))<1e-14 for i in range(len(train)))
        support_hash=hashlib.sha256()
        group_weights=[]
        for g,n in zip(train,sizes):
            support_hash.update(g['group_id'].encode());support_hash.update(g['product'].encode());support_hash.update(g['flat'][g['selected']].tobytes())
            group_weights.append(dict(group_id=g['group_id'],product=g['product'],n=n,total_weight=1/len(train),per_sample_weight=1/(n*len(train)),
                D=int((g['codes'][g['selected']]==0).sum()),K=int((g['codes'][g['selected']]==2).sum())))
        train_support_sha=support_hash.hexdigest();labels_sha=ah(y);weights_sha=ah(w)
        plan=dict(fold=test['fold'],heldout_group=test['group_id'],product=test['product'],train_group_ids=[g['group_id'] for g in train],
            group_weights=group_weights,train_samples=len(y),train_D=int((y==0).sum()),train_K=int((y==1).sum()),train_nonempty_groups=len(train),
            train_D_groups=sum(r['D']>0 for r in group_weights),train_K_groups=sum(r['K']>0 for r in group_weights),
            train_support_sha256=train_support_sha,train_labels_sha256=labels_sha,train_weights_sha256=weights_sha,
            test_support_sha256=ah(test['flat']),test_C=len(test['flat']),train_test_groups_disjoint=True)
        plans.append(plan);write(ROOT/'models'/f"fold{test['fold']:02d}_support.json",plan)
        # Baseline cached student is only read for exact fixed-readout invariance checks.
        baseline_path=B/'evaluation'/('LoRA_'+test['product']+'.npz')
        assert psha(baseline_path)==previous_inputs[str(baseline_path)]
        with np.load(baseline_path) as z:baseline=z['pred'];shadow=z['shadow']
        assert np.all(baseline.ravel()[test['flat']]==1)
        shadow_sha=ah(shadow)
        for model in MODELS:
            prefix=dict(fold=test['fold'],group_id=test['group_id'],product=test['product'],snow_ice=test['snow'],model=model,
                features=3 if model=='RGB3' else 6,parameters=4 if model=='RGB3' else 7,train_samples=len(y),train_D=plan['train_D'],train_K=plan['train_K'],
                train_nonempty_groups=len(train),train_D_groups=plan['train_D_groups'],train_K_groups=plan['train_K_groups'],
                train_support_sha256=train_support_sha,train_labels_sha256=labels_sha,train_weights_sha256=weights_sha,test_support_sha256=ah(test['flat']))
            status='empty_test' if not len(test['flat']) else 'unidentifiable_training' if len(np.unique(y))<2 else 'fit'
            fitted=None;model_path=ROOT/'models'/f"fold{test['fold']:02d}_{model}.json"
            if status=='fit':
                assert fit_count<96;fit_count+=1
                append('FIT_STARTS.jsonl',dict(number=fit_count,fold=test['fold'],model=model,epoch=time.time()))
                try:
                    fitted=fit(features(x,model),y,w,sizes)
                    status='success' if fitted['optimizer_success'] else 'optimizer_failed'
                except BaseException:
                    fitted=dict(optimizer_success=False,error=traceback.format_exc());status='optimizer_exception'
                if status!='success':
                    failure=dict(**prefix,status=status,receipt=fitted);failures.append(failure)
                    write(ROOT/'failures'/f"fold{test['fold']:02d}_{model}.json",failure)
                append('FIT_LEDGER.jsonl',dict(number=fit_count,fold=test['fold'],model=model,status=status))
            if status=='unidentifiable_training':failures.append(dict(**prefix,status=status))
            # Save model/scaler first, then full held-out probabilities, before label-based evaluation.
            write(model_path,dict(**prefix,status=status,receipt=fitted,fold_support_file=f"fold{test['fold']:02d}_support.json"))
            usable=status in ('success','empty_test')
            if usable:
                prob=predict(features(test['raw'],model),fitted) if status=='success' else np.empty(0,np.float64)
                prob_path=ROOT/'oof'/f"fold{test['fold']:02d}_{model}.npz"
                np.savez_compressed(prob_path,flat_indices=test['flat'],p_K=prob)
                saved=dict(**prefix,status=status,model_sha256=sha(model_path),file=prob_path.relative_to(ROOT).as_posix(),sha256=sha(prob_path),
                    probabilities_sha256=ah(prob),Shadow_score_sha256=shadow_sha,model_saved_before_probability=True,probability_saved_before_metrics=True)
                prediction_lock.append(saved)
                assert ah(test['flat'])==prefix['test_support_sha256']
                changed=test['flat'][prob<.5];readout=baseline.copy();readout.ravel()[changed]=0
                change_mask=np.zeros(baseline.shape,bool);change_mask.ravel()[changed]=True
                assert np.array_equal(readout!=baseline,change_mask) and np.array_equal(readout==2,baseline==2)
                assert np.array_equal(readout[~change_mask],baseline[~change_mask]) and ah(shadow)==shadow_sha
                m=measurements(test['codes'],prob,np.ones(len(prob),bool))
                assert len(changed)==m['repaired_D']+m['revoked_K']+m['Shadow_Cloud_to_Surface']
                rows.append(dict(**prefix,status=status,iterations=fitted['iterations'] if fitted else 0,objective=fitted['objective'] if fitted else None,
                    **m,readout_C_only=True,Shadow_mask_and_score_unchanged=True))
                predictions.append(dict(model=model,group_id=test['group_id'],product=test['product'],snow=test['snow'],
                    codes=test['codes'],q=test['q'],p=prob,flat=test['flat'],status=status))
            else:
                rows.append(dict(**prefix,status=status,iterations=fitted.get('iterations',0) if fitted else 0,
                    objective=fitted.get('objective') if fitted else None,C=len(test['codes']),D=int((test['codes']==0).sum()),K=int((test['codes']==2).sum()),
                    C_Shadow=int((test['codes']==3).sum()),AUC=None,Brier=None,repaired_D=None,revoked_K=None,remaining_D=None,retained_K=None,
                    Shadow_Cloud_to_Surface=None,threshold_Cloud=.5,readout_C_only=None,Shadow_mask_and_score_unchanged=None))
        write(ROOT/'PROGRESS.json',dict(completed_folds=test['fold']+1,folds=32,classifier_fits=fit_count,failures=len(failures)))
        print('fold',test['fold']+1,'C',len(test['codes']),'fits',fit_count,'failures',len(failures),flush=True)
    assert len(rows)==96 and fit_count<=96
    for test in groups:
        subset=[r for r in prediction_lock if r['group_id']==test['group_id']]
        assert all(r['test_support_sha256']==ah(test['flat']) for r in subset)
    table(ROOT/'FOLD_SUPPORT.csv',[{k:v for k,v in r.items() if k!='group_weights'} for r in plans])
    table(ROOT/'FOLD_METRICS.csv',rows)
    write(ROOT/'PREDICTION_LOCK.json',dict(rows=prediction_lock,models=list(MODELS),expected_C=546400,failed_models=failures,
        all_groups_reported=True,models_and_scalers_saved_before_predictions=True,predictions_saved_before_metrics=True))
    write(ROOT/'FAILURE_LEDGER.json',dict(fit_attempts=fit_count,failures=failures,no_retries=True))
    summaries=[aggregate(model,predictions) for model in MODELS]
    table(ROOT/'FIXED_RESULTS.csv',summaries)
    paired=[];paired_summary=[]
    for left,right in COMPARISONS:
        combined=[]
        for group in groups:
            a=next(r for r in rows if r['group_id']==group['group_id'] and r['model']==left)
            b=next(r for r in rows if r['group_id']==group['group_id'] and r['model']==right)
            valid=a['status']=='success' and b['status']=='success'
            paired.append(dict(comparison=left+' minus '+right,group_id=group['group_id'],product=group['product'],both_success=valid,
                D=a['D'],K=a['K'],C_Shadow=a['C_Shadow'],
                AUC_delta=a['AUC']-b['AUC'] if valid and a['AUC'] is not None and b['AUC'] is not None else None,
                Brier_delta=a['Brier']-b['Brier'] if valid and a['Brier'] is not None and b['Brier'] is not None else None,
                repaired_D_delta=a['repaired_D']-b['repaired_D'] if valid else None,revoked_K_delta=a['revoked_K']-b['revoked_K'] if valid else None,
                left_status=a['status'],right_status=b['status']))
            if valid:combined.append(paired[-1])
        paired_summary.append(dict(comparison=left+' minus '+right,primary=left=='MS6' and right=='RGB6Q',
            both_success_C_groups=len(combined),AUC_groups=sum(r['AUC_delta'] is not None for r in combined),
            Brier_groups=sum(r['Brier_delta'] is not None for r in combined),D_coverage=sum(r['D'] for r in combined),K_coverage=sum(r['K'] for r in combined),
            group_mean_AUC_delta=average(r['AUC_delta'] for r in combined),group_mean_Brier_delta=average(r['Brier_delta'] for r in combined),
            repaired_D_delta=sum(r['repaired_D_delta'] for r in combined),revoked_K_delta=sum(r['revoked_K_delta'] for r in combined),
            AUC_improved_groups=[r['group_id'] for r in combined if r['AUC_delta'] is not None and r['AUC_delta']>0],
            AUC_worsened_groups=[r['group_id'] for r in combined if r['AUC_delta'] is not None and r['AUC_delta']<0],
            Brier_improved_groups=[r['group_id'] for r in combined if r['Brier_delta'] is not None and r['Brier_delta']<0],
            Brier_worsened_groups=[r['group_id'] for r in combined if r['Brier_delta'] is not None and r['Brier_delta']>0]))
    table(ROOT/'PAIRED_FOLDS.csv',paired);table(ROOT/'PAIRED_COMPARISONS.csv',paired_summary)
    strata=[];strata_groups=[]
    specs=[('q',name,lambda r,lo=lo,hi=hi:(r['q']>=np.float32(lo))&(r['q']<np.float32(hi))) for name,lo,hi in BINS]
    specs += [('snow/ice',v,lambda r,v=v:np.full(len(r['codes']),r['snow']==v,bool)) for v in ('0','1','missing')]
    for tr in top:
        products=set(__import__('json').loads(tr['products']))
        specs.append(('D_top','top'+tr['top_n'],lambda r,products=products:np.full(len(r['codes']),r['product'] in products,bool)))
    for dimension,value,selector in specs:
        for model in MODELS:
            strata.append(dict(dimension=dimension,value=value,**aggregate(model,predictions,selector)))
            for pred in (r for r in predictions if r['model']==model):
                strata_groups.append(dict(dimension=dimension,value=value,model=model,group_id=pred['group_id'],product=pred['product'],
                    **measurements(pred['codes'],pred['p'],selector(pred))))
    table(ROOT/'STRATIFIED_RESULTS.csv',strata);table(ROOT/'STRATIFIED_GROUPS.csv',strata_groups)
    complete=all(r['C']==546400 for r in summaries)
    unique_checks=[]
    for model in MODELS:
        used=[r for r in predictions if r['model']==model]
        assert len({r['group_id'] for r in used})==len(used)
        assert all(np.all(np.diff(r['flat'])>0) for r in used)
        h=hashlib.sha256()
        for r in sorted(used,key=lambda r:r['group_id']):h.update(r['group_id'].encode());h.update(r['product'].encode());h.update(r['flat'].tobytes())
        unique_checks.append(dict(model=model,C=sum(len(r['flat']) for r in used),key_sha256=h.hexdigest(),all_keys_unique=True))
    if complete:assert len({r['key_sha256'] for r in unique_checks})==1
    write(ROOT/'OOF_KEY_ACCEPTANCE.json',dict(rows=unique_checks,complete_C_coverage=complete,each_covered_key_exactly_once=True))
    write(ROOT/'INPUT_SHA256.json',INPUTS)
    write(ROOT/'CPU_ACCEPTANCE.json',dict(status='passed',products=32,folds=32,fit_attempts=fit_count,fit_failures=len(failures),
        neural_calls=0,segmentation_updates=0,downloads=0,supervised_target_reference_information_diagnosis=True,
        all32fold_rows_preserved=True,group_sorted_seed30701_uniform_sampling=True,training_groups_exclude_heldout=True,
        shared_samples_labels_weights_support_all_models=True,train_only_scaler_independently_checked=True,
        weights_equal_groups_sum_one=True,features_and_parameters_fixed=True,no_sign_constraints=True,no_hyperparameter_search=True,
        parameters_saved_before_probability_and_metrics=True,all_C_coverage=complete,covered_OOF_keys_exactly_once=True,
        fixed_half_threshold_counts_independent=True,readout_only_C_Cloud_to_Surface=True,Shadow_masks_and_scores_unchanged=True,
        objective_tests=math_tests,AUC_tests=auc_tests,independent_AUC_checks=ranking.CHECKS,
        torch_imported='torch' in sys.modules,BLAS_thread_limits={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        seconds=time.monotonic()-started,PID=os.getpid()))
    report(summaries,paired_summary,rows,strata,fit_count,failures)
    print('Phase307 completed; fits',fit_count,'failures',len(failures),'zero neural calls',flush=True)

def report(summaries,paired,rows,strata,fit_count,failures):
    def f(v):return 'NA' if v is None else f'{v:.4f}'
    lines=['# Phase307：按组留出的RGB与多光谱信息对照','','## 数据结果','',
        '**监督式目标参考标签离线信息诊断，不是SFDA训练或无标签性能。** 原32个相关组按完整group_id字典序留一组；每组seed30701一次均匀无放回最多8192个D/K样本，组等权；本折训练样本独立标准化，L2=0.001、固定L-BFGS-B，无正号约束/调参。RGB6Q只匹配MS6参数数，不等于所有RGB非线性方法。',
        f'固定C=546400，D/K/Shadow=450058/84005/12337；U未参与。三种模型全部32折保留（含空测试/单类）；实际拟合{fit_count}次、失败/不可识别{len(failures)}，不重试。所有训练组/样本/权重与留出组互斥，概率先封存再算测试指标。','',
        '|模型|AUC有效组数|组均/池化AUC|Brier有效组数|组均/池化Brier|修复D/撤销K|Shadow改动|剩余D|','|---|---|---|---|---|---|---|---|']
    for r in summaries:lines.append(f"|{r['model']}|{r['AUC_groups']}|{f(r['group_mean_AUC'])}/{f(r['pooled_AUC'])}|{r['Brier_groups']}|{f(r['group_mean_Brier'])}/{f(r['pooled_Brier'])}|{r['repaired_D']}/{r['revoked_K']}|{r['Shadow_Cloud_to_Surface']}|{r['remaining_D']}|")
    lines+=['', 'AUC仅双类组可算；Brier覆盖所有非空D/K组，包括最大纯D组。池化不替代组均；所有数值仅使用成功拟合折，失败折及覆盖另表。p(K)≥0.50保留Cloud，否则在C内改Surface；C外与Shadow决策及原连续评分逐像元不变。','',
        '|比较|双方成功组/AUC组|D/K覆盖|组均AUC差|组均Brier差（负为好）|修复D增量/撤销K增量|','|---|---|---|---|---|---|']
    for r in paired:lines.append(f"|{r['comparison']}|{r['both_success_C_groups']}/{r['AUC_groups']}|{r['D_coverage']}/{r['K_coverage']}|{f(r['group_mean_AUC_delta'])}|{f(r['group_mean_Brier_delta'])}|{r['repaired_D_delta']}/{r['revoked_K_delta']}|")
    lines+=['', '|模型|三个q层的组均AUC|三个q层的组均Brier|snow/ice=0/1组均Brier|','|---|---|---|---|']
    for model in MODELS:
        q=[next(r for r in strata if r['model']==model and r['dimension']=='q' and r['value']==name) for name,_,_ in BINS]
        snow=[next(r for r in strata if r['model']==model and r['dimension']=='snow/ice' and r['value']==v) for v in ('0','1')]
        lines.append('|'+ '|'.join([model,'/'.join(f(r['group_mean_AUC']) for r in q),'/'.join(f(r['group_mean_Brier']) for r in q),'/'.join(f(r['group_mean_Brier']) for r in snow)])+'|')
    lines+=['', '全部q层、snow/ice=0/1/缺失、D贡献前1/3/5组，以及各组正反向增量另表，不按这些层重新拟合。原最大损伤纯D组T25XEG：']
    for r in rows:
        if '_T25XEG_' in r['product']:lines.append(f"- {r['model']}：{r['status']}，D={r['D']}，修复{r['repaired_D']}，剩余{r['remaining_D']}，Brier={f(r['Brier'])}，AUC=NA。")
    lines+=['', '## 结论分析','',
        '主要比较固定MS6−RGB6Q，另报MS6−RGB3和RGB6Q−RGB3；需同时看双类组AUC、所有非空组Brier、单类大损伤组与撤销K代价，不能仅凭池化AUC选择赢家。实际改善/恶化组及同q分层完整保留。',
        '即使额外波段有折外增量，也仅表明该固定监督探针利用信息的能力，不证明条件信息独立性、物理因果或无监督SFDA收益；不能将这些目标标签拟合的折外读出混入Phase304主结果，也不部署或回写其预测。',
        '历史development多次暴露，单seed；组留出只避免本轮拟合直接使用测试组，不是新盲测。RGB6Q不是RGB非线性能力上界；原cube重采样和Source/DINO预训练独立性限制继承。不加阈值/seed/分类器或启动下一phase。']
    (ROOT/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc(),neural_calls=0));raise
