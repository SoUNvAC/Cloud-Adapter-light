"""One authorized Phase72 confirmation. No fitting or threshold selection."""
import argparse,csv,hashlib,io,json,os,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from prepare_phase65a_explore import convert,digest
from phase65d_metrics import curve,confusion,measures,corrected
from phase65d_recovery_math import logit_score,predict_logit
from phase70_math import local_contrast,checked
from phase71_math import rgb_local
from phase72_statistics import interval,paired,decision
from freeze_phase66 import code_digest

REPO=Path('/home/scv/Cloud-Adapter-light')
DATA=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
ROOT=DATA/'phase72_20261010'
BUNDLE=REPO/'src/research_plans/phase72_frozen_20261010'
NAMES=['B','L3','RGB_local']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(d,f,indent=2,allow_nan=False)
def brief(r):return {k:v for k,v in r.items() if k not in ['precision','recall','fpr','thresholds']}

def verify():
    frozen=read(BUNDLE/'confirmation_lock.json');execution=read(BUNDLE/'execution_lock.json')
    assert digest(BUNDLE/'confirmation_lock.json')==execution['confirmation_lock_sha256']
    assert code_digest(REPO/'src/research_plans/PHASE72_INDEPENDENT_CONFIRMATION_20261010.md')==frozen['protocol_sha256']
    assert execution['authorization']=='user_20261010_explicit_execute_phase72'
    for name,sha in execution['code_sha256'].items():assert code_digest(REPO/name)==sha,('code drift',name)
    assert digest(REPO/'src/work_dirs/phase65a/split_lock.json')==frozen['original_split_sha256']
    for name in ['source','msre']:
        # Reuse already retained shared candidates; never copy work_dirs artifacts.
        path=DATA/'phase66_20261010/candidate'/(name+'.pth')
        assert digest(path)==frozen[name+'_checkpoint_sha256'],('shared checkpoint drift',name)
    policy=read(DATA/'phase71_20261010/policy_lock.json')
    assert digest(DATA/'phase71_20261010/policy_lock.json')=='ec409be4f85085c3cbdffdb0ab97ef0ec09b7c60d34351b397136dc7e19d834c'
    for n in NAMES:assert frozen['models'][n]==policy['models'][n]
    for local,sha in frozen['existing_input_locks'].items():
        assert digest(DATA/'phase71_20261010/delivery'/Path(local).name)==sha,('frozen input drift',local)
    status=read(DATA/'download_status.json');assert status['status']=='all_files_verified'
    for name,reference in frozen['official_files'].items():
        assert status['verified_files'][name]['sha256']==reference['sha256']
        assert (DATA/name).stat().st_size==reference['bytes']
    return frozen,execution

def prepare():
    frozen,execution=verify();assert not ROOT.exists(),'Preserve previous attempts; do not restart'
    import audit_phase72_metadata as audit
    import contextlib
    output=io.StringIO()
    with contextlib.redirect_stdout(output):audit.main()
    metadata=json.loads(output.getvalue())
    assert not metadata['prior_access_hits'] and not metadata['scan_errors'] and metadata['provenance_dependencies_match']
    assert metadata['selected_rows']==frozen['rows']
    # Reproduce frozen development results before accessing any confirmation pixels.
    import run_phase71 as old
    old.verify()
    index=old.read(old.ROOT/'feature_index.json');dev=old.records('development_val');sources,_=old.development_indices()
    previous=old.read(old.ROOT/'delivery/phase71_report.json')
    refs={n:{r['product']:r for r in previous['policies'][n]['rows']} for n in NAMES}
    reproduction=[]
    for entry in index['development']:
        product=entry['product'];r=dev[product];old.prepare_verify(r)
        truth=np.load(old.PREP/r['mask_path']);valid=truth!=255;ns=old.load(old.OLD,sources[1][product],'nonshadow')[valid]
        for name in NAMES:
            x,y=old.full_inputs(entry,name);m=frozen['models'][name];score=predict_logit(x,m)
            cm=confusion(truth[valid],corrected(score,m['threshold'],ns));rank=curve(score,y)
            assert np.array_equal(cm,refs[name][product]['confusion']) and rank['ap']==refs[name][product]['ranking']['ap']
        reproduction.append(product)
    assert len(reproduction)==32
    st=os.statvfs(DATA);assert st.f_bavail*st.f_frsize>8*1024**3,'Insufficient space; no automatic cleanup'
    ROOT.mkdir();(ROOT/'delivery').mkdir();(ROOT/'prepared').mkdir();(ROOT/'prepared/arrays').mkdir()
    write(ROOT/'preflight_metadata.json',metadata)
    write(ROOT/'development_reproduction.json',dict(all_three_policies_exact=True,products=reproduction))
    from importlib.metadata import version
    write(ROOT/'unseal_receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorization=execution['authorization'],confirmation_lock_sha256=digest(BUNDLE/'confirmation_lock.json'),
        execution_lock_sha256=digest(BUNDLE/'execution_lock.json'),authorized_cohort='65c_final',rows=frozen['rows'],
        package_versions={n:version(n) for n in ['numpy','torch','mmengine','mmsegmentation','mmcv']},
        preflight_metadata_sha256=digest(ROOT/'preflight_metadata.json'),old_sealed_test_accessed=False))
    rows=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as images,zipfile.ZipFile(DATA/'masks.zip') as masks:
        for g in frozen['rows']:
            product=g['representative'];rb=images.read('subscenes/'+product+'.npy');mb=masks.read('masks/'+product+'.npy')
            raw=np.load(io.BytesIO(rb),allow_pickle=False);rgb,truth,iv,_=convert(raw,np.load(io.BytesIO(mb),allow_pickle=False))
            assert (truth!=255).any(),'Empty representative; no replacement'
            selection=truth[iv]!=255
            # Context is image-valid only; reference mask selects evaluated centers, never neighbors.
            b=checked(local_contrast(raw,iv));rg=rgb_local(raw,iv)
            assert len(b)==len(rg)==int(iv.sum())
            prefix='arrays/'+product;arrays={'_rgb.npy':rgb,'_mask.npy':truth,'_image_valid.npy':iv,
                '_spectral.npy':raw[...,[7,11,12]].copy(),'_B.npy':b[selection],'_RGB.npy':rg[selection]}
            for suffix,a in arrays.items():np.save(ROOT/'prepared'/(prefix+suffix),a,allow_pickle=False)
            row=dict(product=product,group_id=g['group_id'],original_split='65c_final',public_support=g['h1_supported'],
                actual_shadow_support=bool((truth==2).any()),rgb_path=prefix+'_rgb.npy',mask_path=prefix+'_mask.npy',
                prefix=prefix,label_counts=np.bincount(truth[truth!=255],minlength=3).tolist(),
                member_sha256=dict(image=hashlib.sha256(rb).hexdigest(),mask=hashlib.sha256(mb).hexdigest()),
                prepared_sha256={suffix:digest(ROOT/'prepared'/(prefix+suffix)) for suffix in arrays})
            rows.append(row);print('Phase72 prepared',len(rows),64,product,flush=True)
    write(ROOT/'prepared/manifest.json',dict(scope='phase72_once_original_65c64',rows=rows))

def extract(name):
    frozen,_=verify();md=read(ROOT/'prepared/manifest.json');assert len(md['rows'])==64 and md['scope']=='phase72_once_original_65c64'
    out=ROOT/('scores_'+name);out.mkdir(exist_ok=False)
    import torch
    sys.path.insert(0,str(REPO/'src'))
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.utils import register_all_modules
    register_all_modules()
    import cloud_adapter.models,cloud_adapter.datasets.phase65_catalogue
    from mmseg.registry import MODELS
    from mmseg.datasets import BaseSegDataset
    class Dataset(BaseSegDataset):
        METAINFO=dict(classes=('surface_visible','cloud','shadow'),palette=[[80,130,70],[245,245,245],[70,70,70]])
        def load_data_list(self):
            return [dict(img_path=str(ROOT/'prepared'/r['rgb_path']),seg_map_path=str(ROOT/'prepared'/r['mask_path']),
                         label_map=None,reduce_zero_label=False,seg_fields=[]) for r in md['rows']]
    source=DATA/'phase66_20261010/candidate/source.pth';ck=DATA/'phase66_20261010/candidate'/(name+'.pth')
    os.environ['PHASE65_PREPARED_MANIFEST']=str(DATA/'phase65a_explore_20261008/manifest.json')
    os.environ['PHASE65_SOURCE_CHECKPOINT']=str(source)
    torch.manual_seed(65);np.random.seed(65)
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase65a_explore_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights()
    missing,unexpected=model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=False)
    assert not unexpected and all(k.startswith(('backbone.target_msre.','backbone.target_head_delta.')) for k in missing)
    if name=='msre':assert not missing
    model.train();model.backbone.set_target_enabled(name=='msre');model.eval().cuda()
    ds=Dataset(pipeline=cfg.test_pipeline,reduce_zero_label=False);rows=[]
    for i,r in enumerate(md['rows']):
        for suffix,sha in r['prepared_sha256'].items():assert digest(ROOT/'prepared'/(r['prefix']+suffix))==sha
        sample=ds[i];assert Path(sample['data_samples'].img_path).name==r['product']+'_rgb.npy'
        with torch.no_grad():p=model.test_step(pseudo_collate([sample]))[0]
        native=p.seg_logits.data.cpu().numpy()
        arrays=dict(score=p.seg_logits.data.softmax(0).cpu().numpy()[2].astype(np.float32),nonshadow=native[:2].argmax(0).astype(np.uint8))
        assert np.isfinite(arrays['score']).all() and arrays['score'].shape==np.load(ROOT/'prepared'/r['mask_path']).shape
        files={}
        for key,a in arrays.items():
            path=out/(r['product']+'_'+key+'.npy');np.save(path,a,allow_pickle=False);files[key]=dict(path=str(path.relative_to(ROOT)),sha256=digest(path))
        rows.append(dict(product=r['product'],group_id=r['group_id'],files=files));print('Phase72 scores',name,i+1,64,flush=True)
    write(out/'index.json',dict(checkpoint_sha256=frozen[name+'_checkpoint_sha256'],manifest_sha256=digest(ROOT/'prepared/manifest.json'),rows=rows))

def score_file(row,key):
    item=row['files'][key];p=ROOT/item['path'];assert digest(p)==item['sha256'];return np.load(p,allow_pickle=False)

def evaluate():
    frozen,_=verify();md=read(ROOT/'prepared/manifest.json');rows={n:[] for n in NAMES};pool={n:[] for n in NAMES};labels=[]
    indices=[]
    for name in ['source','msre']:
        idx=read(ROOT/('scores_'+name)/'index.json')
        assert idx['checkpoint_sha256']==frozen[name+'_checkpoint_sha256'] and idx['manifest_sha256']==digest(ROOT/'prepared/manifest.json')
        mapping={r['product']:r for r in idx['rows']};assert set(mapping)=={r['product'] for r in md['rows']};indices.append(mapping)
    for r in md['rows']:
        for suffix,sha in r['prepared_sha256'].items():assert digest(ROOT/'prepared'/(r['prefix']+suffix))==sha
        truth=np.load(ROOT/'prepared'/r['mask_path']);valid=truth!=255;t=truth[valid];y=t==2;product=r['product']
        ss=score_file(indices[0][product],'score')[valid];ms=score_file(indices[1][product],'score')[valid]
        spec=np.load(ROOT/'prepared'/(r['prefix']+'_spectral.npy'))[valid]
        original=np.column_stack([logit_score(ms),logit_score(ss),spec]).astype(np.float32)
        ns=score_file(indices[1][product],'nonshadow')[valid]
        for name in NAMES:
            extra=None if name=='L3' else np.load(ROOT/'prepared'/(r['prefix']+('_B.npy' if name=='B' else '_RGB.npy')))
            x=original if name=='L3' else np.column_stack([original,extra]);assert np.isfinite(x).all()
            model=frozen['models'][name];s=predict_logit(x,model);pred=corrected(s,model['threshold'],ns);cm=confusion(t,pred)
            metrics=measures(cm)
            strata={cls:dict(negative=int(cm[j].sum()),false_positive=int(cm[j,2]),
                fpr=float(cm[j,2]/cm[j].sum()) if cm[j].sum() else None) for j,cls in enumerate(['Surface','Cloud'])}
            rows[name].append(dict(product=product,group_id=r['group_id'],threshold=model['threshold'],ranking=brief(curve(s,y,fprs=(.01,.05))),
                **metrics,predicted_shadow_fraction=float((pred==2).mean()),class_fpr=strata))
            pool[name].append(s)
        labels.append(y);print('Phase72 policies',len(labels),64,flush=True)
    policies={};all_y=np.concatenate(labels)
    for name,rr in rows.items():
        positive=[r for r in rr if r['positive']];no_shadow=[r for r in rr if not r['positive']]
        strata={k:interval([r['fpr'] for r in subset if r['fpr'] is not None]) for k,subset in [('all',rr),('shadow',positive),('no_shadow',no_shadow)]}
        for cls in ['Surface','Cloud']:strata[cls]=interval([r['class_fpr'][cls]['fpr'] for r in rr if r['class_fpr'][cls]['fpr'] is not None])
        aggregate=measures(np.sum([r['confusion'] for r in rr],axis=0))
        policies[name]=dict(**aggregate,rows=rr,ranking=brief(curve(np.concatenate(pool.pop(name)),all_y,fprs=(.01,.05))),
            macro_ap=interval([r['ranking']['ap'] for r in rr if r['ranking']['ap'] is not None]),
            macro_recall=interval([r['recall'] for r in rr if r['recall'] is not None]),
            macro_precision=interval([r['precision'] for r in rr if r['precision'] is not None]),
            macro_miou_percent=interval([r['miou_percent'] for r in rr]),
            macro_iou_percent=[interval([r['iou_percent'][j] for r in rr if r['iou_percent'][j] is not None]) for j in range(3)],
            fpr_strata=strata,groups_fpr_above_1pct=sum(r['fpr']>.01 for r in rr if r['fpr'] is not None),
            zero_recall_group_ids=[r['group_id'] for r in positive if r['true_positive']==0],
            zero_recall_products=[r['product'] for r in positive if r['true_positive']==0],
            class_fpr_pooled={cls:dict(negative=sum(r['class_fpr'][cls]['negative'] for r in rr),false_positive=sum(r['class_fpr'][cls]['false_positive'] for r in rr)) for cls in ['Surface','Cloud']})
        print('Phase72 pooled',name,flush=True)
    comparisons={n:paired(rows['B'],rows[n]) for n in ['L3','RGB_local']}
    result=decision(comparisons['L3'],comparisons['RGB_local'],policies['B']['fpr_strata']['all'])
    report=dict(status='completed_once_independent_phase72',policies=policies,comparisons=comparisons,decision=result,
        groups=64,actual_shadow_support=sum(r['actual_shadow_support'] for r in md['rows']),
        public_shadow_support=sum(r['public_support'] for r in md['rows']),
        support_disagreements=[r['product'] for r in md['rows'] if r['actual_shadow_support']!=r['public_support']],
        confirmation_lock_sha256=digest(BUNDLE/'confirmation_lock.json'),execution_lock_sha256=digest(BUNDLE/'execution_lock.json'),
        prepared_manifest_sha256=digest(ROOT/'prepared/manifest.json'),unseal_receipt_sha256=digest(ROOT/'unseal_receipt.json'),
        score_index_sha256={n:digest(ROOT/('scores_'+n)/'index.json') for n in ['source','msre']},
        source_msre_forwards=128,new_fits=0,old_sealed_accessed=False,old_h1_failure_unchanged=True,
        limitations='Conditional group bootstrap; small positive support; retrospective access records cannot exclude undocumented reads; label reliability not certified.')
    write(ROOT/'delivery/phase72_report.json',report)
    flat=[]
    for n,rr in rows.items():
        for r in rr:
            cm=r['confusion']
            flat.append(dict(policy=n,product=r['product'],group_id=r['group_id'],ap=r['ranking']['ap'],threshold=r['threshold'],
                **{k:r[k] for k in ['positive','negative','true_positive','false_positive','recall','precision','fpr','miou_percent','shadow_to_surface','shadow_to_cloud','predicted_shadow_fraction']},
                **{'iou_'+str(j):r['iou_percent'][j] for j in range(3)},
                **{'cm_'+str(i)+str(j):cm[i][j] for i in range(3) for j in range(3)},
                surface_fpr=r['class_fpr']['Surface']['fpr'],cloud_fpr=r['class_fpr']['Cloud']['fpr'],
                oracle_recall_fpr01=r['ranking']['matched_fpr'].get('0.01',{}).get('recall'),
                oracle_recall_fpr05=r['ranking']['matched_fpr'].get('0.05',{}).get('recall')))
    with (ROOT/'delivery/groups.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    for name,path in [('confirmation_lock',BUNDLE/'confirmation_lock.json'),('execution_lock',BUNDLE/'execution_lock.json'),
                      ('unseal_receipt',ROOT/'unseal_receipt.json'),('preflight_metadata',ROOT/'preflight_metadata.json'),('development_reproduction',ROOT/'development_reproduction.json')]:write(ROOT/'delivery'/(name+'.json'),read(path))
    print('Phase72 COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','extract','evaluate']);p.add_argument('--model',choices=['source','msre']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='extract':assert a.model;extract(a.model)
    else:evaluate()
