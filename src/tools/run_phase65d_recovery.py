"""One fixed shallow fit/development recovery experiment, no heldout access."""
import argparse,csv,hashlib,io,json,os,sys,zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_phase65a_explore import digest,SOURCE_SHA,SPLIT_SHA
from phase65d_metrics import confusion,measures,curve,corrected
from phase65d_recovery_math import logit_score,fit_model,predict_logit,group_threshold,paired_interval

REPO=Path('/home/scv/Cloud-Adapter-light');DATA=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
PREP=DATA/'phase65a_explore_20261008';OLD=DATA/'phase65d_20261008_recovery01'
RUN=REPO/'src/work_dirs/phase65a_explore_20261008'
ADAPTED_SHA='82f88d235948ef0a9202cebaa026e7f9fb1e1ad5c0aa361d9045c3e6991ee16d'
LEVELS={'msre_calibration':1,'msre_source':2,'msre_source_spectral':5}
PROTOCOL=REPO/'src/research_plans/PHASE65D_RECOVERY_20261008.md'

def write(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)

def records(cohort):
    md=json.loads((PREP/'manifest.json').read_text())
    assert md['original_split_sha256']==SPLIT_SHA
    if cohort=='fit_representatives':rows=[r for r in md['rows'] if r['split']=='fit' and r['is_representative']];n=153
    elif cohort=='development_val':rows=[r for r in md['rows'] if r['split']==cohort];n=32
    else:raise ValueError('Forbidden cohort')
    assert len(rows)==len({r['group_id'] for r in rows})==n
    return {r['product']:r for r in rows}

def prepare_verify(r):
    prefix=r['rgb_path'].removesuffix('_rgb.npy')
    for suffix,sha in r['prepared_sha256'].items():assert digest(PREP/(prefix+suffix))==sha

def extract(root,name):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.utils import register_all_modules
    register_all_modules()
    import cloud_adapter.models,cloud_adapter.datasets.phase65_catalogue
    from mmseg.registry import MODELS,DATASETS
    ck=REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth' if name=='source' else RUN/'msre_seed65/best_mIoU_iter_2000.pth'
    expected=SOURCE_SHA if name=='source' else ADAPTED_SHA;assert digest(ck)==expected
    selected=records('fit_representatives')
    reference=json.loads((RUN/('source_evaluation.json' if name=='source' else 'adapted_evaluation.json')).read_text())
    old={r['product']:r for r in reference['scenes'] if r['cohort']=='fit_representatives'}
    assert set(old)==set(selected) and reference['checkpoint_sha256']==expected
    dest=root/('fit_'+name);dest.mkdir(exist_ok=False)
    os.environ['PHASE65_PREPARED_MANIFEST']=str(PREP/'manifest.json')
    os.environ['PHASE65_SOURCE_CHECKPOINT']=str(REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')
    torch.manual_seed(65);np.random.seed(65)
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase65a_explore_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights()
    missing,unexpected=model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=False)
    assert not unexpected and all(k.startswith(('backbone.target_msre.','backbone.target_head_delta.')) for k in missing)
    if name=='msre':assert not missing
    model.train();model.backbone.set_target_enabled(name=='msre');model.eval().cuda()
    dc=dict(cfg.val_dataloader.dataset);dc.update(cohort='fit_representatives',prepared_manifest=str(PREP/'manifest.json'))
    ds=DATASETS.build(dc);rows=[]
    for i in range(len(ds)):
        sample=ds[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy');r=selected[product];prepare_verify(r)
        truth=np.load(PREP/r['mask_path'])
        with torch.no_grad():p=model.test_step(pseudo_collate([sample]))[0]
        native=p.seg_logits.data.cpu().numpy();pred=p.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        score=p.seg_logits.data.softmax(0).cpu().numpy()[2].astype(np.float32)
        cm=confusion(truth,pred);assert np.array_equal(cm,np.asarray(old[product]['confusion'])),('argmax drift',product)
        assert np.isfinite(score).all() and score.shape==truth.shape
        files={}
        for key,array in {'score':score,'pred':pred,'nonshadow':native[:2].argmax(0).astype(np.uint8)}.items():
            path=dest/(product+'_'+key+'.npy');np.save(path,array);files[key]=dict(path=str(path.relative_to(root)),sha256=digest(path))
        rows.append(dict(product=product,group_id=r['group_id'],files=files,argmax_confusion=cm.tolist()))
        print('fit scores',name,i+1,len(ds),product,flush=True)
    assert len(rows)==153
    write(dest/'index.json',dict(checkpoint_sha256=expected,manifest_sha256=digest(PREP/'manifest.json'),rows=rows))

def indices(root):
    out=[]
    for name in ['source','msre']:
        d=json.loads((root/('fit_'+name)/'index.json').read_text())
        assert d['checkpoint_sha256']==(SOURCE_SHA if name=='source' else ADAPTED_SHA)
        assert d['manifest_sha256']==digest(PREP/'manifest.json')
        out.append({r['product']:r for r in d['rows']})
    assert set(out[0])==set(out[1])==set(records('fit_representatives'))
    return out

def load(root,row,key):
    f=row['files'][key];path=root/f['path'];assert digest(path)==f['sha256'];return np.load(path,mmap_mode='r')

def features(raw,msre,source,valid):
    return np.column_stack([logit_score(msre[valid]),logit_score(source[valid]),raw[...,7][valid],raw[...,11][valid],raw[...,12][valid]]).astype(np.float32)

def fit(root):
    if (root/'recovery_lock.json').exists():raise FileExistsError('Never refit frozen recovery')
    selected=records('fit_representatives');source,msre=indices(root)
    dest=root/'fit_features';dest.mkdir(exist_ok=False);sample_x=[];sample_y=[];sample_w=[];rows=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for i,(product,r) in enumerate(selected.items()):
            truth=np.load(PREP/r['mask_path']);valid=truth!=255;b=z.read('subscenes/'+product+'.npy')
            assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image'];raw=np.load(io.BytesIO(b))
            x=features(raw,load(root,msre[product],'score'),load(root,source[product],'score'),valid)
            assert np.isfinite(x).all()
            path=dest/(product+'.npy');np.save(path,x)
            seed=65081+int(hashlib.sha256(product.encode()).hexdigest()[:8],16)
            chosen=np.random.default_rng(seed).choice(len(x),min(8192,len(x)),replace=False)
            y=truth[valid]==2;sample_x.append(x[chosen]);sample_y.append(y[chosen]);sample_w.append(np.full(len(chosen),1/(153*len(chosen))))
            rows.append(dict(product=product,group_id=r['group_id'],feature_path=str(path.relative_to(root)),feature_sha256=digest(path),
                             sample_count=len(chosen),sample_positive=int(y[chosen].sum()),sample_indices_sha256=hashlib.sha256(chosen.tobytes()).hexdigest(),valid_pixels=len(x)))
            print('fit features',i+1,153,product,flush=True)
    x=np.concatenate(sample_x).astype(np.float64);y=np.concatenate(sample_y).astype(float);w=np.concatenate(sample_w)
    write(root/'fit_feature_index.json',dict(rows=rows,sample_x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),sample_y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
        sample_count=len(x),sample_positive=int(y.sum()),group_equal_weights=True))
    models={}
    for name,n in LEVELS.items():
        m=fit_model(x[:,:n],y,w,min(n,2));negatives=[]
        for r in rows:
            xx=np.load(root/r['feature_path'],mmap_mode='r');rr=selected[r['product']];truth=np.load(PREP/rr['mask_path']);yy=truth[truth!=255]==2
            logits=predict_logit(xx[:,:n],m);negatives.append(logits[~yy])
        m['threshold']=group_threshold(negatives,.01)
        m['fit_macro_fpr']=float(np.mean([(a>=m['threshold']).mean() for a in negatives]));assert m['fit_macro_fpr']<=.01+1e-10
        del negatives;models[name]=m;print('fit calibrated',name,m['fit_macro_fpr'],flush=True)
    hashes={str(p.relative_to(root)):digest(p) for p in [root/'fit_source/index.json',root/'fit_msre/index.json',root/'fit_feature_index.json']}
    write(root/'recovery_lock.json',dict(status='frozen_fit_only_before_development',models=models,level_inputs=LEVELS,
        fit_groups=153,fit_samples=len(x),sample_positive=int(y.sum()),fit_fpr_budget=.01,threshold_rule='equal-group full fit negative FPR, conservative exact ties',
        protocol_sha256=digest(PROTOCOL),code_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('phase65d_recovery_math.py')]},
        original_split_sha256=digest(REPO/'src/work_dirs/phase65a/split_lock.json'),source_checkpoint_sha256=SOURCE_SHA,msre_checkpoint_sha256=ADAPTED_SHA,
        prepared_manifest_sha256=digest(PREP/'manifest.json'),fit_input_sha256=hashes,development_fitting=False,development_prior_model_selection=True,
        no_t18_access=True,no_gt_proportions_in_predictors=True,phase65c_authorized=False))

def brief(c):return {k:v for k,v in c.items() if k not in ['precision','recall','fpr','thresholds']}

def evaluate(root):
    lock_path=root/'recovery_lock.json';lock=json.loads(lock_path.read_text());assert lock['status']=='frozen_fit_only_before_development'
    assert lock['protocol_sha256']==digest(PROTOCOL) and lock['original_split_sha256']==SPLIT_SHA
    for rel,h in lock['fit_input_sha256'].items():assert digest(root/rel)==h
    for name,h in lock['code_sha256'].items():assert digest(Path(__file__).with_name(name))==h
    selected=records('development_val');fitgroups={r['group_id'] for r in records('fit_representatives').values()}
    assert not fitgroups & {r['group_id'] for r in selected.values()}
    old=[];input_hashes={}
    for name in ['source','msre']:
        path=OLD/'development_val'/('index_'+name+'.json');d=json.loads(path.read_text());input_hashes[name]=digest(path)
        assert d['checkpoint_sha256']==(SOURCE_SHA if name=='source' else ADAPTED_SHA)
        assert d['manifest_sha256']==digest(PREP/'manifest.json');old.append({r['product']:r for r in d['rows']})
    assert set(old[0])==set(old[1])==set(selected)
    names=['source_original','msre_original']+list(LEVELS);policy_rows={n:[] for n in names};pool_scores={n:[] for n in names};pool_y=[]
    out=root/'development_scores';out.mkdir(exist_ok=False);table=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for i,(product,r) in enumerate(selected.items()):
            prepare_verify(r);truth=np.load(PREP/r['mask_path']);valid=truth!=255;yy=truth[valid]==2
            for prior in [old[0][product],old[1][product]]:
                assert digest(OLD/prior['truth_path'])==prior['truth_sha256'];assert np.array_equal(np.load(OLD/prior['truth_path']),truth)
            ss=load(OLD,old[0][product],'score');ms=load(OLD,old[1][product],'score');ns=load(OLD,old[1][product],'nonshadow')[valid]
            b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==r['member_sha256']['image'];raw=np.load(io.BytesIO(b))
            xx=features(raw,ms,ss,valid);scores={'source_original':ss[valid],'msre_original':ms[valid]}
            preds={'source_original':load(OLD,old[0][product],'pred')[valid],'msre_original':load(OLD,old[1][product],'pred')[valid]}
            for name,n in LEVELS.items():
                m=lock['models'][name];scores[name]=predict_logit(xx[:,:n],m);preds[name]=corrected(scores[name],m['threshold'],ns)
                np.save(out/(product+'_'+name+'.npy'),scores[name])
            for name in names:
                c=brief(curve(scores[name],yy));cm=confusion(truth[valid],preds[name]);row=dict(product=product,group_id=r['group_id'],policy=name,ranking=c,**measures(cm),
                    predicted_shadow_fraction=float((preds[name]==2).mean()))
                if name in ['source_original','msre_original']:
                    ref=old[0 if name=='source_original' else 1][product];assert np.array_equal(cm,np.asarray(ref['argmax_confusion']))
                policy_rows[name].append(row);pool_scores[name].append(scores[name]);table.append(dict(product=product,group_id=r['group_id'],policy=name,
                    ap=c['ap'],recall=row['recall'],fpr=row['fpr'],miou_percent=row['miou_percent'],shadow_iou_percent=row['iou_percent'][2],
                    recall_oracle_fpr01=c['matched_fpr'].get('0.01',{}).get('recall'),predicted_shadow_fraction=row['predicted_shadow_fraction']))
            pool_y.append(yy);print('evaluate frozen development',i+1,32,product,flush=True)
    policies={};plot_data={};labels=np.concatenate(pool_y)
    for name in names:
        rows=policy_rows[name];supported=[r for r in rows if r['ranking']['ap'] is not None]
        c=curve(np.concatenate(pool_scores[name]),labels);plot_data[name]=c
        policies[name]=dict(**measures(np.sum([r['confusion'] for r in rows],axis=0)),ranking=brief(c),rows=rows,
            n_groups=len(rows),shadow_supported_groups=len(supported),macro_ap=float(np.mean([r['ranking']['ap'] for r in supported])),
            macro_recall=float(np.mean([r['recall'] for r in supported])),macro_fpr=float(np.mean([r['fpr'] for r in rows])),
            macro_predicted_shadow_fraction=float(np.mean([r['predicted_shadow_fraction'] for r in rows])))
        print('pooled development',name,flush=True)
    del pool_scores
    comparisons={}
    for earlier,later in [('msre_original','msre_calibration'),('msre_calibration','msre_source'),('msre_source','msre_source_spectral')]:
        before={r['product']:r for r in policies[earlier]['rows']};pairs=[]
        for r in policies[later]['rows']:
            b=before[r['product']]
            if r['ranking']['ap'] is not None:
                pairs.append(dict(product=r['product'],group_id=r['group_id'],delta_ap=r['ranking']['ap']-b['ranking']['ap'],
                    delta_recall=r['recall']-b['recall'],delta_fpr=r['fpr']-b['fpr']))
        comparisons[later+'_minus_'+earlier]=dict(pairs=pairs,ap=paired_interval([r['delta_ap'] for r in pairs]),
            recall=paired_interval([r['delta_recall'] for r in pairs]))
    report=dict(status='completed_frozen_fit_development_recovery',policies=policies,comparisons=comparisons,
        fit_groups=153,development_groups=32,shadow_supported_development_groups=7,recovery_lock_sha256=digest(lock_path),
        original_split_sha256=SPLIT_SHA,development_cache_index_sha256=input_hashes,
        scope='Post-hoc development-only, not independent confirmation; no new segmenter training or T18/65b/sealed access',
        interpretation='AP and matched-FPR diagnostics use all valid pixels; statistical support is related groups. Fit FPR budget does not guarantee development FPR. Fixed three-level logistic; no interactions or selection on development.')
    write(root/'recovery_report.json',report);write(root/'plot_data.json',plot_data)
    with (root/'development_scenes.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','fit','evaluate']);p.add_argument('--model',choices=['source','msre']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.parent==DATA and root.name.startswith('phase65d_recovery_20261008') and root.is_dir()
    assert digest(REPO/'src/work_dirs/phase65a/split_lock.json')==SPLIT_SHA
    if a.action=='extract':extract(root,a.model)
    elif a.action=='fit':fit(root)
    else:evaluate(root)

if __name__=='__main__':main()
