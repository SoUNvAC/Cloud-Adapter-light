"""Reuse checkpoints on development; freeze report before post-hoc T18 inference."""
import argparse,json,os,sys,io,re,csv,zipfile,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_phase65a_explore import digest,SPLIT_SHA
from phase65d_metrics import curve,confusion,measures
from phase65d_followup_metrics import loo_ridge

REPO=Path('/home/scv/Cloud-Adapter-light');DATA=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
OLD=DATA/'phase65d_20261008_recovery01';PREP=DATA/'phase65a_explore_20261008'
RUN=REPO/'src/work_dirs/phase65a_explore_20261008/msre_seed65'
T18='S2A_MSIL1C_20180803T141051_N0206_R110_T18FYG_20180803T192500'
SHAS={3500:'db13c9c2278804c73ac45082e0a6d5bccd6a9ff3748082b6be4d89898812983a',4000:'57ccd89a722363cb9e2213524c42a689fbf6cac0917fcaa66720779586003187'}

def brief(c):return {k:v for k,v in c.items() if k not in ['precision','recall','fpr','thresholds']}

def previous(model,cohort='development_val'):
    p=OLD/cohort/('index_'+model+'.json');d=json.loads(p.read_text())
    return {r['product']:r for r in d['rows']}

def old_array(row,name):
    path=row['truth_path'] if name=='truth' else row['files'][name]['path']
    return np.load(OLD/path,mmap_mode='r')

def extract(root,step,cohort):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmseg.utils import register_all_modules
    register_all_modules()
    import cloud_adapter.models,cloud_adapter.datasets.phase65_catalogue
    from mmseg.registry import MODELS,DATASETS
    ck=RUN/f'iter_{step}.pth';assert digest(ck)==SHAS[step]
    prepared=PREP if cohort=='development_val' else DATA/'phase65b_explore_20261008'
    md=json.loads((prepared/'manifest.json').read_text());records={r['product']:r for r in md['rows'] if r['split']==cohort}
    if cohort!='development_val':
        assert (root/'development_report.json').exists();records={T18:records[T18]}
    dest=root/f'{cohort}_{step}';dest.mkdir(exist_ok=False)
    os.environ['PHASE65_PREPARED_MANIFEST']=str(PREP/'manifest.json')
    os.environ['PHASE65_SOURCE_CHECKPOINT']=str(REPO/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')
    torch.manual_seed(65);np.random.seed(65)
    cfg=Config.fromfile(str(REPO/'src/configs/protocol/phase65a_explore_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights()
    missing,unexpected=model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=False)
    assert not missing and not unexpected
    model.train();model.backbone.set_target_enabled(True);model.eval().cuda()
    dc=dict(cfg.val_dataloader.dataset);dc.update(cohort=cohort,prepared_manifest=str(prepared/'manifest.json'))
    if cohort!='development_val':dc['type']='Phase65EvidenceDataset'
    ds=DATASETS.build(dc);rows=[]
    for i in range(len(ds)):
        product=Path(ds.get_data_info(i)['img_path']).name.removesuffix('_rgb.npy')
        if product not in records:continue
        r=records[product];prefix=r['rgb_path'].removesuffix('_rgb.npy')
        for suffix,sha in r['prepared_sha256'].items():assert digest(prepared/(prefix+suffix))==sha
        sample=ds[i];truth=np.load(prepared/r['mask_path']);valid_image=np.load(prepared/r['image_valid_path'])
        with torch.no_grad():prediction=model.test_step(pseudo_collate([sample]))[0]
        score=prediction.seg_logits.data.softmax(0).cpu().numpy()[2].astype(np.float32)
        pred=prediction.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        path=dest/(product+'_score.npy');np.save(path,score)
        rows.append(dict(product=product,group_id=r['group_id'],score_path=str(path.relative_to(root)),score_sha256=digest(path),
            **measures(confusion(truth,pred)),observable_predicted_shadow_fraction=float((pred[valid_image]==2).mean())))
        print('trace',cohort,step,len(rows),len(records),product,flush=True)
    assert len(rows)==len(records)
    report=dict(step=step,checkpoint_sha256=digest(ck),manifest_sha256=digest(prepared/'manifest.json'),rows=rows,
        development_report_sha256=digest(root/'development_report.json') if cohort!='development_val' else None)
    (dest/'index.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    del model;torch.cuda.empty_cache()

def summarize_dev(root):
    records={r['product']:r for r in json.loads((PREP/'manifest.json').read_text())['rows'] if r['split']=='development_val'}
    source=previous('source');best=previous('msre');assert set(records)==set(source)==set(best)
    assert len({r['group_id'] for r in records.values()})==32
    features={r['product']:r['source_observable_features'] for r in json.loads((REPO/'src/work_dirs/phase65a_explore_20261008/source_evaluation.json').read_text())['scenes'] if r['cohort']=='development_val'}
    points=[];table=[];by_model={}
    for model in ['source','2000','3500','4000']:
        cache=source if model=='source' else best
        index=None if model in ['source','2000'] else json.loads((root/f'development_val_{model}/index.json').read_text())
        new={r['product']:r for r in index['rows']} if index else {}
        labels=[];scores=[];cms=[];scene=[];observed=[]
        for product,row in source.items():
            truth=old_array(row,'truth');valid=truth!=255
            score=old_array(cache[product],'score') if not index else np.load(root/new[product]['score_path'])
            cm=np.asarray(cache[product]['argmax_confusion']) if not index else np.asarray(new[product]['confusion'])
            c=brief(curve(score[valid],truth[valid]==2));m=measures(cm)
            if not index:
                image_valid=np.load(PREP/records[product]['image_valid_path'])
                fraction=float((old_array(cache[product],'pred')[image_valid]==2).mean())
            else:fraction=new[product]['observable_predicted_shadow_fraction']
            item=dict(product=product,group_id=row['group_id'],model=model,**m,ranking=c,observable_predicted_shadow_fraction=fraction)
            scene.append(item);scores.append(score[valid]);labels.append(truth[valid]==2);cms.append(cm);observed.append(fraction)
            table.append(dict(product=product,group_id=row['group_id'],model=model,ap=c['ap'],recall_fpr01=c['matched_fpr'].get('0.01',{}).get('recall'),predicted_shadow_fraction=fraction,miou_percent=m['miou_percent']))
        positive=[r for r in scene if r['ranking']['ap'] is not None]
        pool=brief(curve(np.concatenate(scores),np.concatenate(labels)));cm=np.sum(cms,axis=0)
        p=dict(model=model,**measures(cm),ranking=pool,macro_ap=float(np.mean([r['ranking']['ap'] for r in positive])),
            shadow_supported_groups=len(positive),macro_predicted_shadow_fraction=float(np.mean(observed)),
            pooled_predicted_shadow_fraction=float(cm[:,2].sum()/cm.sum()),rows=scene)
        points.append(p);by_model[model]={r['product']:r for r in scene}
        print('ranked trajectory',model,flush=True)
    supported=[p for p,r in by_model['source'].items() if r['ranking']['ap'] is not None]
    structure=[]
    for product in supported:
        base=by_model['source'][product]['ranking']['ap'];deltas={s:by_model[s][product]['ranking']['ap']-base for s in ['2000','3500','4000']}
        structure.append(dict(product=product,group_id=records[product]['group_id'],source_ap=base,delta_ap=deltas,
            category={s:'loss' if v<=-.01 else 'gain' if v>=.01 else 'stable' for s,v in deltas.items()},features=features[product]))
    x=np.asarray([r['features'] for r in structure]);y=np.asarray([r['delta_ap']['2000'] for r in structure])
    probes={'rgb_source':loo_ridge(x[:,:9],y),'rgb_source_nir_swir':loo_ridge(x,y)}
    log_paths=list(RUN.rglob('20261008_192141.log'))
    if len(log_paths)!=1:raise ValueError('Training log must match exactly one retained file')
    text=log_paths[0].read_text();last=0;logs=[]
    for line in text.splitlines():
        t=re.search(r'Iter\(train\)\s*\[\s*(\d+)/',line)
        if t:last=int(t.group(1))
        v=re.search(r'Iter\(val\).*mIoU:\s*([\d.]+)',line)
        if v:logs.append(dict(step=last,miou_percent=float(v.group(1))))
    report=dict(status='development_frozen_before_t18',selection_changed=False,points=points,structure=structure,
        persistent_loss_products=[r['product'] for r in structure if all(v=='loss' for v in r['category'].values())],
        loo_probes=probes,probe_product_order=supported,probe_target='2000-step minus Source AP, offline reference target only',
        feature_definition=['Source mean Surface/Cloud/Shadow score','Source entropy','Source predicted Shadow proportion','Source margin','RGB brightness mean/std/dark fraction','optional image B08/B11/B12 means'],
        feature_no_ground_truth_proportions=True,probe_support_limit='Only seven Shadow-supported related groups, exploratory LOO error; no deployment AUROC or significant spectral benefit claim',
        logged_validation_miou=logs,missing_checkpoints=[500,1000,1500,2500,3000],
        phase65d_lock_sha256=digest(OLD/'calibration_lock.json'),original_split_sha256=digest(REPO/'src/work_dirs/phase65a/split_lock.json'))
    assert report['original_split_sha256']==SPLIT_SHA
    (root/'development_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (root/'trajectory_scenes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

def spectral(root):
    report=json.loads((root/'development_report.json').read_text());rows=previous('source');msre=previous('msre')
    records={r['product']:r for r in json.loads((PREP/'manifest.json').read_text())['rows'] if r['split']=='development_val'}
    results=[]
    with zipfile.ZipFile(DATA/'subscenes.zip') as z:
        for r in report['structure']:
            product=r['product'];b=z.read('subscenes/'+product+'.npy');assert hashlib.sha256(b).hexdigest()==records[product]['member_sha256']['image']
            raw=np.load(io.BytesIO(b));truth=old_array(rows[product],'truth');shadow=truth==2;surface=truth==0
            brightness=raw[...,[3,2,1]].mean(-1);boundary=float(np.quantile(brightness[surface],.1))
            channels={'source':old_array(rows[product],'score'),'msre2000':old_array(msre[product],'score'),
                'negative_rgb_mean':-brightness,'negative_b08':-raw[...,7],'negative_b11':-raw[...,11],
                'negative_b12':-raw[...,12],'negative_nir_swir_mean':-raw[...,[7,11,12]].mean(-1)}
            item=dict(product=product,group_id=r['group_id'],category=r['category']['2000'],relative_brightness_boundary=boundary,strata={})
            for name,mask in [('surface_all',surface),('surface_lowest_decile',surface&(brightness<=boundary))]:
                v=shadow|mask;item['strata'][name]={n:brief(curve(a[v],truth[v]==2)) for n,a in channels.items()}
            results.append(item);print('spectral screen',product,flush=True)
    (root/'spectral_screen.json').write_text(json.dumps(dict(status='fixed_untrained_spectral_signal_screen',rows=results,
        development_report_sha256=digest(root/'development_report.json'),
        limitations='Standalone TOA darkness ranking only; no fusion, no model recovery, no tuned sign/weights/thresholds, no formal independent scene or footprint validation. Geometry and larger-context inputs not verified.'),indent=2,allow_nan=False))

def t18(root):
    original=json.loads((OLD/'diagnostic_report.json').read_text())['t18'];out={s:dict(ranking=original[m]['ranking']['score'],**original[m]['original']) for s,m in [('source','source'),('2000','msre')]}
    truth=old_array(previous('source','phase65b_evidence')[T18],'truth');valid=truth!=255
    for step in [3500,4000]:
        index=json.loads((root/f'phase65b_evidence_{step}/index.json').read_text());assert index['development_report_sha256']==digest(root/'development_report.json')
        row=index['rows'][0];score=np.load(root/row['score_path']);out[str(step)]=dict(row,ranking=brief(curve(score[valid],truth[valid]==2)))
    (root/'t18_posthoc.json').write_text(json.dumps(dict(product=T18,selection_use=False,development_report_sha256=digest(root/'development_report.json'),points=out),indent=2,allow_nan=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','development','spectral','t18']);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--step',type=int,choices=[3500,4000]);p.add_argument('--cohort',choices=['development_val','phase65b_evidence']);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(DATA);assert root.exists()
    if a.action=='extract':extract(root,a.step,a.cohort)
    else:
        fn={'development':summarize_dev,'spectral':spectral,'t18':t18}[a.action]
        name={'development':'development_report.json','spectral':'spectral_screen.json','t18':'t18_posthoc.json'}[a.action]
        if (root/name).exists():raise FileExistsError('Preserve completed diagnostic')
        fn(root)

if __name__=='__main__':main()
