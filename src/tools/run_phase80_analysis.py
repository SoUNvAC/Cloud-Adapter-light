"""Phase80: cached baseline reproduction, one LoRA forward per representative,
exactly two fit-only calibrations, then full fixed development comparison."""
import argparse,hashlib,io,os,sys,zipfile,csv
from pathlib import Path
import numpy as np
from phase80_common import *
from run_phase65d_recovery import records,prepare_verify,load,features,brief
from phase65d_recovery_math import logit_score,fit_model,predict_logit,group_threshold
from phase65d_metrics import curve,confusion,measures,corrected

RUNCONTROL=Path('/home/scv/shared/phase80_control_20261011')

def rows_index(root,rel):
    p=root/rel;d=read(p)
    assert d['manifest_sha256']==MANIFEST_SHA
    return {r['product']:r for r in d['rows']}

def baseline():
    base_check();base=read(RECOVERY/'recovery_lock.json');ctl=read(CONTROL/'control_lock.json')
    rr=read(RECOVERY/'recovery_report.json');cr=read(CONTROL/'delivery/phase67_report.json')
    assert ctl['recovery_lock_sha256']==RECOVERY_SHA
    dev=records('development_val')
    ss=rows_index(OLD,'development_val/index_source.json');ms=rows_index(OLD,'development_val/index_msre.json')
    assert set(ss)==set(ms)==set(dev)
    refs={'M':{r['product']:r for r in rr['policies']['msre_calibration']['rows']},
          'M+S':{r['product']:r for r in cr['policies']['MsRE_spectral']['rows']}}
    checked=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in dev.items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            x=features(raw,load(OLD,ms[product],'score'),load(OLD,ss[product],'score'),valid)
            ns=load(OLD,ms[product],'nonshadow')[valid]
            for name,model,cols in [('M',base['models']['msre_calibration'],[0]),
                                    ('M+S',ctl['models']['MsRE_spectral'],[0,2,3,4])]:
                score=predict_logit(x[:,cols],model);cm=confusion(truth[valid],corrected(score,model['threshold'],ns))
                ap=curve(score,truth[valid]==2)['ap'];ref=refs[name][product]
                assert np.array_equal(cm,np.asarray(ref['confusion'])),('Baseline confusion drift',name,product)
                assert ap==ref['ranking']['ap'],('Baseline AP drift',name,product)
                if name=='M':
                    previous=np.load(RECOVERY/'development_scores'/(product+'_msre_calibration.npy'),mmap_mode='r')
                    assert np.array_equal(score,previous),'M pixel score drift'
                checked.append(dict(policy=name,product=product,group_id=r['group_id'],ap=ap,confusion=cm.tolist()))
            print('Phase80 baseline reproduction',product,flush=True)
    write(RUNCONTROL/'baseline_reproduction.json',dict(status='64_baseline_scene_results_exactly_reproduced',rows=checked,
        recovery_lock_sha256=RECOVERY_SHA,control_lock_sha256=digest(CONTROL/'control_lock.json'),
        recovery_report_sha256=digest(RECOVERY/'recovery_report.json'),
        control_report_sha256=digest(CONTROL/'delivery/phase67_report.json')))

def extract():
    import torch
    sys.path.insert(0,str(REPO/'src'))
    from run_phase80_training import setup
    setup()
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.registry import MODELS,DATASETS
    done=read(ROOT/'training_completed.json');ck=Path(done['best_checkpoint'])
    assert digest(ck)==done['best_checkpoint_sha256']
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase80_lora_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights()
    model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True)
    model.train();assert all(permitted_lora(n) for n,p in model.named_parameters() if p.requires_grad)
    model.eval().cuda();torch.manual_seed(65);np.random.seed(65)
    for cohort in ['fit_representatives','development_val']:
        dest=ROOT/cohort;dest.mkdir()
        selected=records(cohort);ds=DATASETS.build(dict(cfg.val_dataloader.dataset,cohort=cohort))
        rows=[]
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
            print('Phase80 best LoRA forward',cohort,i+1,len(ds),product,flush=True)
        assert {r['product'] for r in rows}==set(selected)
        write(dest/'index.json',dict(rows=rows,manifest_sha256=MANIFEST_SHA,
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
    for name,cols in [('L',[0]),('L+S',[0,1,2,3])]:
        model=fit_model(x[:,cols],y,w,1);neg=[]
        for r in audit:
            full=np.load(ROOT/r['feature_path'],mmap_mode='r')
            truth=np.load(PREP/fitrows[r['product']]['mask_path']);yy=truth[truth!=255]==2
            neg.append(predict_logit(full[:,cols],model)[~yy])
        model['threshold']=group_threshold(neg,.01)
        model['fit_macro_fpr']=float(np.mean([(v>=model['threshold']).mean() for v in neg]))
        assert model['fit_macro_fpr']<=.01+1e-10
        model['columns']=cols;models[name]=model;del neg
        print('Phase80 shallow frozen',name,model,flush=True)
    write(ROOT/'calibration_lock.json',dict(status='two_fit_only_models_frozen_before_development_shallow_evaluation',
        models=models,fit_input_audit_sha256=digest(ROOT/'fit_input_audit.json'),
        training_completed_sha256=digest(ROOT/'training_completed.json'),
        baseline_reproduction_sha256=digest(RUNCONTROL/'baseline_reproduction.json'),
        source_sha256=SOURCE_SHA,manifest_sha256=MANIFEST_SHA,original_split_sha256=SPLIT_SHA,
        recovery_lock_sha256=RECOVERY_SHA,phase67_lock_sha256=digest(CONTROL/'control_lock.json'),
        analysis_code_sha256=digest(Path(__file__)),math_sha256=digest(Path(__file__).with_name('phase65d_recovery_math.py')),
        full_fit_negative_threshold=True,fit_group_weights='1/153 per representative',development_fitting=False,
        no_confirmations_read=True,no_hyperparameter_search=True))

def aggregate(rows,pool,labels):
    supported=[r for r in rows if r['positive']>0];zero=[r for r in rows if r['positive']==0]
    fprs=[r['fpr'] for r in rows if r['fpr'] is not None]
    return dict(rows=rows,**measures(np.sum([r['confusion'] for r in rows],axis=0)),
        ranking=brief(curve(np.concatenate(pool),labels)),
        macro_ap=interval([r['ranking']['ap'] for r in supported]),
        macro_recall=interval([r['recall'] for r in supported]),
        macro_precision=interval([r['precision'] for r in rows if r['precision'] is not None]),
        macro_miou_percent=interval([r['miou_percent'] for r in rows]),macro_fpr=interval(fprs),
        shadow_supported_fpr=interval([r['fpr'] for r in supported if r['fpr'] is not None]),
        no_shadow_fpr=interval([r['fpr'] for r in zero if r['fpr'] is not None]),
        fpr_above_one_percent_groups=sum(v>.01 for v in fprs),
        zero_recall_groups=[r['product'] for r in supported if r['recall']==0],
        n_groups=len(rows),shadow_supported_groups=len(supported),no_shadow_groups=len(zero))

def csvwrite(p,rows):
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def evaluate():
    lock=read(ROOT/'calibration_lock.json');assert digest(Path(__file__))==lock['analysis_code_sha256']
    base=read(RECOVERY/'recovery_lock.json');ctl=read(CONTROL/'control_lock.json')
    li=rows_index(ROOT,'development_val/index.json')
    ss=rows_index(OLD,'development_val/index_source.json');ms=rows_index(OLD,'development_val/index_msre.json')
    dev=records('development_val');assert set(dev)==set(li)==set(ss)==set(ms)
    names=['M','M+S','L','L+S','Source','MsRE','LoRA_argmax','L3']
    rows={n:[] for n in names};pools={n:[] for n in names};ys=[];input_audit=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for product,r in dev.items():
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;y=truth[valid]==2
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image']
            raw=np.load(io.BytesIO(b),allow_pickle=False)
            s=load(OLD,ss[product],'score');m=load(OLD,ms[product],'score');l=load(ROOT,li[product],'score')
            x=features(raw,m,s,valid)
            lx=np.column_stack([logit_score(l[valid]),x[:,2:5]]).astype(np.float32)
            ns=load(OLD,ms[product],'nonshadow')[valid];lns=load(ROOT,li[product],'nonshadow')[valid]
            scores={'Source':s[valid],'MsRE':m[valid],'LoRA_argmax':l[valid]}
            preds={'Source':load(OLD,ss[product],'pred')[valid],'MsRE':load(OLD,ms[product],'pred')[valid],
                   'LoRA_argmax':load(ROOT,li[product],'pred')[valid]}
            specs=[('M',x[:,[0]],base['models']['msre_calibration'],ns),
                   ('M+S',x[:,[0,2,3,4]],ctl['models']['MsRE_spectral'],ns),
                   ('L',lx[:,[0]],lock['models']['L'],lns),('L+S',lx,lock['models']['L+S'],lns),
                   ('L3',x,base['models']['msre_source_spectral'],ns)]
            for name,xx,model,other in specs:
                scores[name]=predict_logit(xx,model);preds[name]=corrected(scores[name],model['threshold'],other)
            for n in names:
                cm=confusion(truth[valid],preds[n]);ranking=brief(curve(scores[n],y))
                rows[n].append(dict(product=product,group_id=r['group_id'],policy=n,ranking=ranking,
                    predicted_shadow_fraction=float((preds[n]==2).mean()),**measures(cm)))
                pools[n].append(scores[n])
            ys.append(y);input_audit.append(dict(product=product,group_id=r['group_id'],
                prepared_sha256=r['prepared_sha256'],image_member_sha256=r['member_sha256']['image'],
                source_cache=ss[product],msre_cache=ms[product],lora_cache=li[product]))
            print('Phase80 frozen development evaluation',len(ys),32,product,flush=True)
    # Full-cache reproduction was done before training; recheck the final pipeline too.
    previous=read(RUNCONTROL/'baseline_reproduction.json')
    refs={(r['policy'],r['product']):r for r in previous['rows']}
    for n in ['M','M+S']:
        for r in rows[n]:
            ref=refs[n,r['product']];assert r['ranking']['ap']==ref['ap'] and r['confusion']==ref['confusion']
    labels=np.concatenate(ys);policies={n:aggregate(rows[n],pools.pop(n),labels) for n in names}
    assert all(p['shadow_supported_groups']==7 and p['n_groups']==32 for p in policies.values())
    comparisons={};pairs=[]
    for a,b in [('L','M'),('L+S','L'),('M+S','M'),('L+S','M+S')]:
        key=a+' minus '+b;allpairs=[]
        for ra,rb in zip(rows[a],rows[b]):
            assert ra['group_id']==rb['group_id']
            r=dict(comparison=key,product=ra['product'],group_id=ra['group_id'],positive=ra['positive'],negative=ra['negative'],
                delta_ap_pp=None if not ra['positive'] else 100*(ra['ranking']['ap']-rb['ranking']['ap']),
                delta_recall_pp=None if not ra['positive'] else 100*(ra['recall']-rb['recall']),
                delta_fpr_pp=100*(ra['fpr']-rb['fpr']),delta_miou_pp=ra['miou_percent']-rb['miou_percent'])
            allpairs.append(r);pairs.append(r)
        aps=[r['delta_ap_pp'] for r in allpairs if r['delta_ap_pp'] is not None]
        recalls=[r['delta_recall_pp'] for r in allpairs if r['delta_recall_pp'] is not None]
        comparisons[key]=dict(ap_pp=interval(aps),recall_pp=interval(recalls),fpr_pp=interval([r['delta_fpr_pp'] for r in allpairs]),
            miou_pp=interval([r['delta_miou_pp'] for r in allpairs]),improved=sum(v>1 for v in aps),stable=sum(abs(v)<=1 for v in aps),
            declined=sum(v< -1 for v in aps),worst_ap_pp=min(aps),worst_recall_pp=min(recalls))
    spectral=[]
    for l,ls,m,ms in zip(rows['L'],rows['L+S'],rows['M'],rows['M+S']):
        if l['positive']:
            spectral.append(dict(product=l['product'],group_id=l['group_id'],
                l_spectral_ap_pp=100*(ls['ranking']['ap']-l['ranking']['ap']),
                m_spectral_ap_pp=100*(ms['ranking']['ap']-m['ranking']['ap']),
                difference_of_increments_pp=100*((ls['ranking']['ap']-l['ranking']['ap'])-(ms['ranking']['ap']-m['ranking']['ap']))))
    write(ROOT/'development_input_audit.json',dict(rows=input_audit))
    write(ROOT/'phase80_report.json',dict(scope='single-seed fit/development adaptation exploration',
        formal_confirmation=False,primary_comparisons=['L minus M','L+S minus L','M+S minus M','L+S minus M+S'],
        calibration_lock_sha256=digest(ROOT/'calibration_lock.json'),policies=policies,comparisons=comparisons,
        spectral_increment_difference=interval([r['difference_of_increments_pp'] for r in spectral]),
        caveats=['7 supported development groups; selection uses these same development data',
                 'LoRA has different capacity and lacks MsRE target_head_delta; not a pure structure causal contrast',
                 'No unused confirmation pool was opened; old research stop conclusions remain']))
    table=[]
    for n in names:
        for r in rows[n]:table.append(dict(policy=n,product=r['product'],group_id=r['group_id'],positive=r['positive'],negative=r['negative'],
            ap=r['ranking']['ap'],recall=r['recall'],precision=r['precision'],fpr=r['fpr'],miou_percent=r['miou_percent'],
            surface_iou_percent=r['iou_percent'][0],cloud_iou_percent=r['iou_percent'][1],shadow_iou_percent=r['iou_percent'][2],
            shadow_to_surface=r['shadow_to_surface'],shadow_to_cloud=r['shadow_to_cloud'],
            predicted_shadow_fraction=r['predicted_shadow_fraction']))
    csvwrite(ROOT/'development_groups.csv',table);csvwrite(ROOT/'paired_differences.csv',pairs)
    csvwrite(ROOT/'spectral_increment_differences.csv',spectral)
    write(ROOT/'COMPLETED.json',dict(status='all_fixed_phase80_analyses_completed',report_sha256=digest(ROOT/'phase80_report.json'),
        no_additional_training=True,no_confirmation_access=True,no_alcd=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');a=p.parse_args()
    if a.baseline:baseline()
    else:base_check();extract();fit();evaluate()

if __name__=='__main__':main()
