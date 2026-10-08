"""Extract fixed model scores; development first, 65b only after calibration lock."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.utils import register_all_modules
register_all_modules()
import cloud_adapter.models
import cloud_adapter.datasets.phase65_catalogue
from mmseg.registry import MODELS,DATASETS
from prepare_phase65a_explore import SOURCE_SHA,digest
from phase65b_evidence_probe import ADAPTED_SHA,authorized
from phase65d_metrics import confusion


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['development_val','phase65b_evidence'],required=True)
    p.add_argument('--model',choices=['source','msre'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=authorized(a.output)
    root.mkdir(exist_ok=True);cohort=root/a.cohort;cohort.mkdir(exist_ok=True)
    index_path=cohort/('index_'+a.model+'.json')
    if index_path.exists():raise FileExistsError('Preserve completed scores')
    lock=root/'calibration_lock.json'
    if a.cohort=='phase65b_evidence' and not lock.exists():raise ValueError('Freeze development correction before any 65b score extraction')
    repo=Path('/home/scv/Cloud-Adapter-light');data=Path('/home/scv/shared/data/sentinel2_cloud_mask_catalogue_4172871')
    run=repo/'src/work_dirs/phase65a_explore_20261008'
    checkpoint=repo/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth' if a.model=='source' else run/'msre_seed65/best_mIoU_iter_2000.pth'
    expected=SOURCE_SHA if a.model=='source' else ADAPTED_SHA
    if digest(checkpoint)!=expected:raise ValueError('Frozen checkpoint mismatch')
    prepared=data/('phase65a_explore_20261008' if a.cohort=='development_val' else 'phase65b_explore_20261008')
    manifest=prepared/'manifest.json';md=json.loads(manifest.read_text())
    records={r['product']:r for r in md['rows'] if r['split']==a.cohort}
    expected_n=32 if a.cohort=='development_val' else 64
    if len(records)!=expected_n:raise ValueError('Frozen cohort size mismatch')
    reference_run=run if a.cohort=='development_val' else repo/'src/work_dirs/phase65b_explore_20261008'
    reference=reference_run/('source_evaluation.json' if a.model=='source' else 'adapted_evaluation.json')
    old=json.loads(reference.read_text());old_rows={r['product']:r for r in old['scenes'] if r['cohort']==a.cohort}
    if old['checkpoint_sha256']!=expected or set(old_rows)!=set(records):raise ValueError('Reference model/cohort mismatch')
    os.environ['PHASE65_PREPARED_MANIFEST']=str(data/'phase65a_explore_20261008/manifest.json')
    os.environ['PHASE65_SOURCE_CHECKPOINT']=str(repo/'src/work_dirs/phase64_source_parent/seed64/best_mIoU_iter_4000.pth')
    torch.manual_seed(65);np.random.seed(65)
    cfg=Config.fromfile(str(repo/'src/configs/protocol/phase65a_explore_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights()
    missing,unexpected=model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=False)
    if unexpected or any(not k.startswith(('backbone.target_msre.','backbone.target_head_delta.')) for k in missing):raise ValueError('State mismatch')
    if a.model=='msre' and missing:raise ValueError('Adapted checkpoint missing parameters')
    # Match the previous evaluator's module/freezing lifecycle exactly.
    model.train()
    model.backbone.set_target_enabled(a.model=='msre');model.eval().cuda()
    dc=dict(cfg.val_dataloader.dataset);dc.update(cohort=a.cohort,prepared_manifest=str(manifest))
    if a.cohort=='phase65b_evidence':dc['type']='Phase65EvidenceDataset'
    ds=DATASETS.build(dc);rows=[]
    for i in range(len(ds)):
        sample=ds[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy');r=records[product]
        prefix=r['rgb_path'].removesuffix('_rgb.npy')
        for suffix,sha in r['prepared_sha256'].items():
            if digest(prepared/(prefix+suffix))!=sha:raise ValueError('Prepared arrays changed')
        truth=np.load(prepared/r['mask_path']);rgb=np.load(prepared/r['rgb_path'])
        with torch.no_grad():prediction=model.test_step(pseudo_collate([sample]))[0]
        native=prediction.seg_logits.data.cpu().numpy().astype(np.float32)
        probability=prediction.seg_logits.data.softmax(0).cpu().numpy()[2].astype(np.float32)
        pred=prediction.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        if not np.isfinite(native).all() or probability.shape!=truth.shape:raise ValueError('Invalid score/grid')
        cm=confusion(truth,pred)
        if not np.array_equal(cm,np.asarray(old_rows[product]['confusion'])):
            print('argmax_drift',product,'observed',cm.tolist(),'reference',old_rows[product]['confusion'],flush=True)
            raise ValueError('Argmax result drift: '+product)
        dest=cohort/product;dest.mkdir(exist_ok=True)
        common={'truth':truth,'brightness':rgb.mean(-1,dtype=np.float32)/np.float32(255)}
        for name,array in common.items():
            path=dest/(name+'.npy')
            if path.exists():
                if not np.array_equal(np.load(path),array):raise ValueError('Shared truth/brightness changed')
            else:np.save(path,array)
        arrays={'score':probability,'native_score':native[2],
                'margin':native[2]-native[:2].max(0),'pred':pred,'nonshadow':native[:2].argmax(0).astype(np.uint8)}
        files={}
        for name,array in arrays.items():
            path=dest/(a.model+'_'+name+'.npy')
            if path.exists():raise FileExistsError('Preserve previous partial scores')
            np.save(path,array);files[name]=dict(path=str(path.relative_to(root)),sha256=digest(path))
        rows.append(dict(product=product,group_id=r['group_id'],files=files,
                         truth_path=str((dest/'truth.npy').relative_to(root)),truth_sha256=digest(dest/'truth.npy'),
                         brightness_path=str((dest/'brightness.npy').relative_to(root)),argmax_confusion=cm.tolist()))
        print('scores',a.cohort,a.model,i+1,len(ds),product,flush=True)
    report=dict(model=a.model,cohort=a.cohort,checkpoint_sha256=expected,manifest_sha256=digest(manifest),
                score_definition='softmax of three Mask2Former aggregated mask-class scores; not calibrated probability',
                native_score_definition='sum_q class_softmax(q,shadow)*mask_sigmoid(q,pixel)',
                calibration_lock_sha256=digest(lock) if lock.exists() else None,
                all_original_argmax_confusions_match=True,rows=rows)
    index_path.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
