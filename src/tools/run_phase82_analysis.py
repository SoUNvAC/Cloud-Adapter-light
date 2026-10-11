"""Phase82 one student extraction, one fit-only spectral model and fixed evaluation."""
import argparse,hashlib,io,sys,time,zipfile
import numpy as np
from phase82_common import *
from run_phase65d_recovery import records,prepare_verify,load,features,brief
from phase65d_recovery_math import logit_score,fit_model,predict_logit,group_threshold
from phase65d_metrics import curve,confusion,measures,corrected
from verify_phase78_local import independent_AP
from run_phase81 import cross,counts,group_row
from run_phase80_analysis import csvwrite

def rows_index(root,rel):
    d=read(root/rel);assert d['manifest_sha256']==MANIFEST_SHA
    return {r['product']:r for r in d['rows']}

def baseline():
    bindings=read(REPO/'src/research_plans/PHASE82_INPUT_BINDINGS_20261011.json')
    for parent,key in [(PREVIOUS,'phase80'),(PHASE81,'phase81')]:
        for rel,h in bindings[key].items():assert digest(parent/rel)==h,rel
    base_check();report=read(PHASE81/'phase81_report.json')
    for rel,h in read(PREVIOUS/'training_lock.json')['code_sha256'].items():assert digest(REPO/rel)==h,rel
    cm=read(CONTROL/'control_lock.json');ll=read(PREVIOUS/'calibration_lock.json')
    assert digest(REPO/'src/tools/run_phase80_analysis.py')==ll['analysis_code_sha256']
    models={'M':cm['models']['MsRE_spectral'],'L':ll['models']['L+S']}
    indexes={'M':rows_index(OLD,'development_val/index_msre.json'),'L':rows_index(PREVIOUS,'development_val/index.json')}
    refs={n:{r['product']:r for r in report['policies'][n]['rows']} for n in ('MM','LL','LM')}
    audit={r['product']:r for r in read(PREVIOUS/'development_input_audit.json')['rows']}
    dev=records('development_val');assert set(dev)==set(indexes['M'])==set(indexes['L'])
    pools={'M':[],'L':[]};labels=[];rows={n:[] for n in refs}
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in dev.items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;t=truth[valid]
            assert r['prepared_sha256']==audit[product]['prepared_sha256']
            assert indexes['M'][product]==audit[product]['msre_cache'] and indexes['L'][product]==audit[product]['lora_cache']
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False);scores={};winners={}
            for k,parent in [('M',OLD),('L',PREVIOUS)]:
                x=np.column_stack([logit_score(load(parent,indexes[k][product],'score')[valid]),raw[...,7][valid],raw[...,11][valid],raw[...,12][valid]]).astype(np.float32)
                scores[k]=predict_logit(x,models[k]);winners[k]=load(parent,indexes[k][product],'nonshadow')[valid]
                pools[k].append(scores[k])
            preds=cross(scores,{k:m['threshold'] for k,m in models.items()},winners)
            for n in refs:
                ap=independent_AP(t==2,scores[n[0]])
                row=group_row(n,r,t,preds[n],refs[n][product]['ap'])
                assert (ap is None and row['ap'] is None) or abs(ap-row['ap'])<1e-12
                assert row==refs[n][product],('Phase81 baseline drift',n,product)
                rows[n].append(row)
            labels.append(t==2);print('Phase82 cached reference reproduction',len(labels),32,product,flush=True)
    for n in refs:
        previous=report['policies'][n]
        assert measures(np.sum([r['confusion'] for r in rows[n]],axis=0))['miou_percent']==previous['miou_percent']
        for field,key in [('ap','macro_ap'),('fpr','macro_fpr'),('miou_percent','macro_miou_percent')]:
            assert float(np.mean([r[field] for r in rows[n] if r[field] is not None]))==previous[key]['mean']
        assert abs(independent_AP(np.concatenate(labels),np.concatenate(pools[n[0]]))-previous['pooled_ap'])<1e-12
    write(RUNCONTROL/'baseline_reproduction.json',dict(status='all_96_reference_rows_exactly_reproduced',
        phase81_report_sha256=digest(PHASE81/'phase81_report.json'),phase80_calibration_lock_sha256=digest(PREVIOUS/'calibration_lock.json'),
        frozen_inputs=bindings,independent_AP=True,independent_boolean_confusion=True))

def extract():
    import torch
    sys.path.insert(0,str(REPO/'src'))
    from run_phase82_training import setup
    from phase82_kd import register_student
    register_student()
    setup()
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.registry import MODELS,DATASETS
    done=read(ROOT/'training_completed.json');ck=Path(done['best_checkpoint'])
    assert digest(ck)==done['best_checkpoint_sha256']
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase82_kd_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights()
    model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True)
    model.train();assert all(permitted_lora(n) for n,p in model.named_parameters() if p.requires_grad)
    model.eval().cuda();torch.manual_seed(65);np.random.seed(65)
    for cohort in ['fit_representatives','development_val']:
        dest=ROOT/cohort;dest.mkdir()
        selected=records(cohort);ds=DATASETS.build(dict(cfg.val_dataloader.dataset,cohort=cohort))
        rows=[];started=time.monotonic();torch.cuda.synchronize()
        for i in range(len(ds)):
            sample=ds[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy')
            r=selected[product];prepare_verify(r);truth=np.load(PREP/r['mask_path'])
            with torch.no_grad():p=model.test_step(pseudo_collate([sample]))[0]
            native=p.seg_logits.data.cpu().numpy();score=p.seg_logits.data.softmax(0).cpu().numpy()[2].astype(np.float32)
            pred=p.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            assert np.isfinite(native).all() and np.isfinite(score).all() and score.shape==truth.shape
            assert np.array_equal(pred,native.argmax(0))
            files={}
            for key,arr in dict(score=score,pred=pred,nonshadow=native[:2].argmax(0).astype(np.uint8)).items():
                path=dest/(product+'_'+key+'.npy');np.save(path,arr,allow_pickle=False)
                files[key]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
            rows.append(dict(product=product,group_id=r['group_id'],files=files,
                argmax_confusion=confusion(truth,pred).tolist()))
            print('Phase82 selected student forward',cohort,i+1,len(ds),product,flush=True)
        assert {r['product'] for r in rows}==set(selected)
        torch.cuda.synchronize()
        write(dest/'index.json',dict(rows=rows,manifest_sha256=MANIFEST_SHA,inference_seconds=time.monotonic()-started,
            checkpoint_sha256=done['best_checkpoint_sha256'],score_definition='softmax of aggregated three-class Mask2Former scores'))

def fit():
    # Exact reuse of the old deterministic sampling positions and spectral columns.
    base_check();fitrows=records('fit_representatives')
    li=rows_index(ROOT,'fit_representatives/index.json')
    previous=read(RECOVERY/'fit_feature_index.json');old={r['product']:r for r in previous['rows']}
    assert set(old)==set(li)==set(fitrows)
    dest=ROOT/'fit_features';dest.mkdir();xs=[];ys=[];ws=[];audit=[]
    for product,r in fitrows.items():
        prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;y=truth[valid]==2
        prior=old[product];path=RECOVERY/prior['feature_path'];assert digest(path)==prior['feature_sha256']
        spectral=np.load(path,mmap_mode='r')[:,2:5]
        x=np.column_stack([logit_score(load(ROOT,li[product],'score')[valid]),spectral]).astype(np.float32)
        assert np.isfinite(x).all() and len(x)==prior['valid_pixels']
        seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
        idx=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
        assert hashlib.sha256(idx.tobytes()).hexdigest()==prior['sample_indices_sha256']
        assert int(y[idx].sum())==prior['sample_positive']
        p=dest/(product+'.npy');np.save(p,x,allow_pickle=False)
        audit.append(dict(product=product,group_id=r['group_id'],feature_path=str(p.relative_to(ROOT)),
            feature_sha256=digest(p),old_feature_sha256=prior['feature_sha256'],
            sample_indices_sha256=prior['sample_indices_sha256'],sample_count=len(idx),sample_positive=int(y[idx].sum()),valid_pixels=len(x)))
        xs.append(x[idx]);ys.append(y[idx]);ws.append(np.full(len(idx),1/(153*len(idx))))
    x=np.concatenate(xs).astype(np.float64);y=np.concatenate(ys).astype(float);w=np.concatenate(ws)
    assert hashlib.sha256(y.tobytes()).hexdigest()==previous['sample_y_sha256']
    write(ROOT/'fit_input_audit.json',dict(rows=audit,sample_count=len(x),sample_positive=int(y.sum()),
          sample_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),sample_y_sha256=previous['sample_y_sha256'],equal_group_weights=True))
    models={}
    for name,cols in [('D+S',[0,1,2,3])]:
        model=fit_model(x[:,cols],y,w,1);neg=[]
        for r in audit:
            full=np.load(ROOT/r['feature_path'],mmap_mode='r')
            truth=np.load(PREP/fitrows[r['product']]['mask_path']);yy=truth[truth!=255]==2
            neg.append(predict_logit(full[:,cols],model)[~yy])
        model['threshold']=group_threshold(neg,.01)
        model['fit_macro_fpr']=float(np.mean([(v>=model['threshold']).mean() for v in neg]))
        assert model['fit_macro_fpr']<=.01+1e-10
        model['columns']=cols;models[name]=model;del neg
        print('Phase82 shallow frozen',name,model,flush=True)
    write(ROOT/'calibration_lock.json',dict(status='one_fit_only_model_frozen_before_development_shallow_evaluation',
        models=models,fit_input_audit_sha256=digest(ROOT/'fit_input_audit.json'),
        training_completed_sha256=digest(ROOT/'training_completed.json'),
        baseline_reproduction_sha256=digest(RUNCONTROL/'baseline_reproduction.json'),
        source_sha256=SOURCE_SHA,manifest_sha256=MANIFEST_SHA,original_split_sha256=SPLIT_SHA,
        recovery_lock_sha256=RECOVERY_SHA,phase67_lock_sha256=digest(CONTROL/'control_lock.json'),
        analysis_code_sha256=digest(Path(__file__)),math_sha256=digest(Path(__file__).with_name('phase65d_recovery_math.py')),
        full_fit_negative_threshold=True,fit_group_weights='1/153 per representative',development_fitting=False,
        no_confirmations_read=True,no_hyperparameter_search=True))

def aggregate(rows,pooled_ap):
    support=[r for r in rows if r['positive']];zero=[r for r in rows if not r['positive']]
    assert len(rows)==32 and len(support)==7 and len(zero)==25
    return dict(rows=rows,**measures(np.sum([r['confusion'] for r in rows],axis=0)),pooled_ap=pooled_ap,
        macro_ap=interval([r['ap'] for r in support]),macro_recall=interval([r['recall'] for r in support]),
        macro_fpr=interval([r['fpr'] for r in rows]),macro_miou_percent=interval([r['miou_percent'] for r in rows]),
        shadow_fpr=interval([r['fpr'] for r in support]),no_shadow_fpr=interval([r['fpr'] for r in zero]),
        zero_recall_groups=[r['product'] for r in support if r['recall']==0],
        fpr_above_one_percent_groups=sum(r['fpr']>.01 for r in rows),
        defined_iou_class_counts={str(i):sum(r['defined_iou_classes']==i for r in rows) for i in (1,2,3)})

def evaluate():
    lock=read(ROOT/'calibration_lock.json');assert digest(Path(__file__))==lock['analysis_code_sha256']
    assert set(lock['models'])=={'D+S'};model=lock['models']['D+S']
    dev=records('development_val');idx=rows_index(ROOT,'development_val/index.json')
    assert set(dev)==set(idx);rows=[];pool=[];ys=[];inputrows=[]
    score_dir=ROOT/'development_scores';score_dir.mkdir()
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in dev.items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;t=truth[valid]
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            x=np.column_stack([logit_score(load(ROOT,idx[product],'score')[valid]),raw[...,7][valid],raw[...,11][valid],raw[...,12][valid]]).astype(np.float32)
            score=predict_logit(x,model);ns=load(ROOT,idx[product],'nonshadow')[valid]
            pred=corrected(score,model['threshold'],ns);ap=curve(score,t==2)['ap'];independent=independent_AP(t==2,score)
            assert (ap is None and independent is None) or abs(ap-independent)<1e-12
            rows.append(group_row('D+S',r,t,pred,ap));pool.append(score);ys.append(t==2)
            p=score_dir/(product+'.npy');np.save(p,score,allow_pickle=False)
            inputrows.append(dict(product=product,group_id=r['group_id'],prepared_sha256=r['prepared_sha256'],
                 image_member_sha256=r['member_sha256']['image'],student_cache=idx[product],score_sha256=digest(p)))
            print('Phase82 frozen development evaluation',len(rows),32,product,flush=True)
    score=np.concatenate(pool);labels=np.concatenate(ys);poolap=curve(score,labels)['ap']
    assert abs(poolap-independent_AP(labels,score))<1e-12
    previous=read(PHASE81/'phase81_report.json');policies={}
    for n in ('MM','LL','LM'):policies[n]=aggregate(previous['policies'][n]['rows'],previous['policies'][n]['pooled_ap'])
    policies['D+S']=aggregate(rows,poolap);comparisons={};pairs=[]
    for n in ('LL','LM','MM'):
        rr=[]
        for x,y in zip(rows,policies[n]['rows']):
            assert x['group_id']==y['group_id']
            rr.append(dict(comparison='D+S minus '+n,product=x['product'],group_id=x['group_id'],
                delta_AP_pp=100*(x['ap']-y['ap']) if x['ap'] is not None else None,
                delta_recall_pp=100*(x['recall']-y['recall']) if x['recall'] is not None else None,
                delta_mIoU_pp=x['miou_percent']-y['miou_percent'],delta_FPR_pp=100*(x['fpr']-y['fpr'])))
        pairs.extend(rr)
        comparisons['D+S minus '+n]=dict(AP_pp=interval([r['delta_AP_pp'] for r in rr if r['delta_AP_pp'] is not None]),
             recall_pp=interval([r['delta_recall_pp'] for r in rr if r['delta_recall_pp'] is not None]),
             mIoU_pp=interval([r['delta_mIoU_pp'] for r in rr]),FPR_pp=interval([r['delta_FPR_pp'] for r in rr]),
             pooled_mIoU_pp=policies['D+S']['miou_percent']-policies[n]['miou_percent'],
             worst_AP_pp=min(r['delta_AP_pp'] for r in rr if r['delta_AP_pp'] is not None),
             worst_recall_pp=min(r['delta_recall_pp'] for r in rr if r['delta_recall_pp'] is not None),
             worst_mIoU_pp=min(r['delta_mIoU_pp'] for r in rr))
    c=comparisons['D+S minus LL'];gate=continuation_gate(c['pooled_mIoU_pp'],c['AP_pp']['mean'],c['recall_pp']['mean'],c['mIoU_pp']['mean'],policies['D+S']['macro_fpr']['mean'])
    write(ROOT/'development_input_audit.json',dict(rows=inputrows))
    write(ROOT/'phase82_report.json',dict(scope='one_seed_fixed_conditional_KD_fit_development_exploration',formal_confirmation=False,
        policies=policies,comparisons=comparisons,continuation_gate=gate,seed=82011,primary_comparison='D+S minus LL'))
    write(ROOT/'SHARED_PIXEL_VERIFIED.json',dict(status='passed',independent_AP_scene=32,independent_AP_pool=1,
        independent_boolean_confusions=32,reference_96_rows_verified_before_training=True))
    csvwrite(ROOT/'paired_differences.csv',pairs)
    table=[]
    for n,p in policies.items():
        for r in p['rows']:
            x={k:v for k,v in r.items() if k not in ('confusion','reference_support','iou_percent')};x['policy']=n
            for i,cl in enumerate(('Surface','Cloud','Shadow')):
                x[cl+'_reference_pixels']=r['reference_support'][i];x[cl+'_IoU_percent']=r['iou_percent'][i]
                for j,out in enumerate(('Surface','Cloud','Shadow')):x[cl+'_to_'+out]=r['confusion'][i][j]
            table.append(x)
    csvwrite(ROOT/'development_groups.csv',table)
    write(ROOT/'confusion_matrices.json',{n:dict(pooled=p['confusion'],rows=[dict(product=r['product'],confusion=r['confusion']) for r in p['rows']]) for n,p in policies.items()})
    write(ROOT/'COMPLETED.json',dict(status='completed',report_sha256=digest(ROOT/'phase82_report.json'),retain_fixed_scheme=gate['retain_fixed_scheme'],
         training_jobs=1,shallow_fits=1,student_fit_forwards=153,student_development_forwards=32,no_confirmations=True,no_work_dirs_sync=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');a=p.parse_args()
    if a.baseline:baseline()
    else:base_check();extract();fit();evaluate()

