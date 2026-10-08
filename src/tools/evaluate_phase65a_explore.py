"""Scene-representative descriptive evaluation; never confirmation inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmseg.utils import register_all_modules
register_all_modules()
import cloud_adapter.models
import cloud_adapter.datasets.phase65_catalogue
from mmseg.registry import MODELS, DATASETS


def metric(cm):
    c=np.asarray(cm,float); tp=np.diag(c); union=c.sum(0)+c.sum(1)-tp
    iou=np.divide(tp,union,out=np.full(3,np.nan),where=union>0)
    return dict(miou_percent=float(np.nanmean(iou)*100),
                iou_percent=[float(v*100) if np.isfinite(v) else None for v in iou],
                confusion=np.asarray(cm,int).tolist(),valid_pixels=int(c.sum()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--source-only',action='store_true');ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--phase65b-manifest',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous evaluation')
    torch.manual_seed(65);np.random.seed(65)
    cfg=Config.fromfile(str(Path(__file__).resolve().parents[1]/'configs/protocol/phase65a_explore_rgb.py'))
    model=MODELS.build(cfg.model);model.init_weights()
    ckpt=torch.load(a.checkpoint,map_location='cpu');state=ckpt['state_dict']
    missing,unexpected=model.load_state_dict(state,strict=False)
    permitted=('backbone.target_msre.','backbone.target_head_delta.')
    if unexpected or any(not k.startswith(permitted) for k in missing):
        raise ValueError(f'Checkpoint mismatch {missing} {unexpected}')
    model.train();names=[n for n,p in model.named_parameters() if p.requires_grad]
    if not names or any(not n.startswith(permitted) for n in names):raise ValueError('Unexpected trainable source parameter')
    model.backbone.set_target_enabled(not a.source_only);model.eval().cuda()
    manifest_path=a.phase65b_manifest or Path(os.environ['PHASE65_PREPARED_MANIFEST']);manifest=json.loads(manifest_path.read_text())
    cohorts=['fit_representatives','development_val']
    if a.phase65b_manifest:
        if manifest['scope']!='exploratory_phase65b_evidence_only':raise ValueError('Unauthorized manifest')
        expected=manifest['source_sha256'] if a.source_only else manifest['adapted_sha256']
        if hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen checkpoint mismatch')
        cohorts=['phase65b_evidence']
    records={r['product']:r for r in manifest['rows']}
    output=dict(scope=manifest['scope'],primary_confirmation_test=False,
                checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                prepared_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                trainable_names=names,source_path_disabled=a.source_only,scenes=[])
    for cohort in cohorts:
        dc=dict(cfg.val_dataloader.dataset);dc['cohort']=cohort
        if a.phase65b_manifest:
            dc.update(type='Phase65EvidenceDataset',prepared_manifest=str(manifest_path))
        ds=DATASETS.build(dc)
        for i in range(len(ds)):
            sample=ds[i];product=Path(sample['data_samples'].img_path).name.removesuffix('_rgb.npy')
            row=records[product];truth=np.load(manifest_path.parent/row['mask_path'])
            image_valid=np.load(manifest_path.parent/row['image_valid_path'])
            with torch.no_grad():pred=model.test_step(pseudo_collate([sample]))[0]
            label=pred.pred_sem_seg.data[0].cpu().numpy();probs=pred.seg_logits.data.softmax(0).cpu().numpy()
            if label.shape!=truth.shape or not np.isfinite(probs).all():raise ValueError('Inference grid/nonfinite mismatch')
            valid=truth!=255;cm=np.bincount(truth[valid]*3+label[valid],minlength=9).reshape(3,3)
            metrics=metric(cm);shadow=int(cm[2].sum());d=float((cm[2,1]-cm[2,0])/shadow) if shadow else None
            pv=probs[:,image_valid];entropy=-(pv*np.log(np.maximum(pv,1e-12))).sum(0)/np.log(3)
            ordered=np.sort(pv,axis=0);sp=row['observable_spectra']
            features=[*pv.mean(1).tolist(),float(entropy.mean()),float((label[image_valid]==2).mean()),
                      float((ordered[-1]-ordered[-2]).mean()),sp['brightness_mean'],sp['brightness_std'],
                      sp['dark_fraction'],sp['b8_mean'],sp['b11_mean'],sp['b12_mean']]
            output['scenes'].append(dict(product=product,group_id=row['group_id'],cohort=cohort,
                h1_public_support=row['h1_public_support'],actual_shadow_pixels=shadow,d=d,
                source_observable_features=features,**metrics))
            print('evaluated',cohort,i+1,len(ds),product,flush=True)
            if a.preflight:break
        if a.preflight:break
    if not a.preflight:
        output['aggregate']={cohort:metric(np.sum([s['confusion'] for s in output['scenes'] if s['cohort']==cohort],axis=0))
                             for cohort in cohorts}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':main()
