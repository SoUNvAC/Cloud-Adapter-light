"""After40k, compare original-source baseline to aligned-source retraining."""
from pathlib import Path
import sys,os,json,hashlib,importlib.util,time
import numpy as np
import torch
R=Path('/home/scv/shared/phase444_retrain_20261103');SRC=Path('/home/scv/Cloud-Adapter-light/src');ARM=sys.argv[1]
phase='429' if ARM=='baseline429' else '431';sys.path[:0]=[str(R/'code'),'/home/scv/shared/phase400_20261031/code'];os.chdir(SRC)
from mmengine.config import Config
from mmengine.dataset import Compose,pseudo_collate
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from report_phase400 import describe
import alignment444
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(m):
    h=hashlib.sha256()
    for n,t in m.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    register_all_modules();importlib.import_module('phase'+phase+'.integration');torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    spec=importlib.util.spec_from_file_location('catalogue444',SRC/'cloud_adapter/datasets/phase65_catalogue.py');spec.loader.exec_module(importlib.util.module_from_spec(spec))
    out=R/ARM;assert json.loads((out/'TRAIN_COMPLETE.json').read_text())['steps']==40000
    best=list(out.glob('best_mIoU_iter_*.pth'));assert len(best)==1
    old=Path('/home/scv/shared/phase'+phase+'_20261102')/('baseline/best_mIoU_iter_40000.pth' if phase=='429' else 'fused/best_mIoU_iter_32000.pth')
    rows=json.loads(Path('/home/scv/shared/phase443_20261103/INPUTS.json').read_text());summaries=[]
    for condition,checkpoint in [('original_training',old),('aligned_retraining',best[0])]:
        cfg=Config.fromfile(str(out/'CONFIG_LOCK.py'));model=MODELS.build(cfg.model);model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True);model.cuda().eval().requires_grad_(False);before=state_sha(model)
        totals={d:np.zeros((3,3),np.int64) for d in ['A','B']};per=[]
        sigroot=Path('/home/scv/shared/phase443_20261103' if condition=='aligned_retraining' else '/home/scv/shared/phase442_20261103');sigs={v['index']:v['input_tensor_sha'] for v in json.loads((sigroot/'official/SIGNATURES.json').read_text())}
        for i,row in enumerate(rows):
            d=row['domain'];assert sha(row['image'])==row['image_sha'] and sha(row['mask'])==row['mask_sha']
            pipeline=[dict(type='LoadImageFromFile' if d=='A' else 'LoadPhase65RGB'),dict(type='Resize',scale=(512,512)),dict(type='LoadAnnotations' if d=='A' else 'LoadPhase65Mask')]
            if condition=='aligned_retraining':pipeline.append(dict(type='AlignRGB444',domain=d))
            pipeline.append(dict(type='PackSegInputs'));sample=Compose(pipeline)(dict(img_path=row['image'],seg_map_path=row['mask'],seg_fields=[],label_map={0:0,1:1,2:1,3:2} if d=='A' else None,reduce_zero_label=False))
            y=sample['data_samples'].gt_sem_seg.data[0].numpy();valid=y!=255;assert np.bincount(y[valid],minlength=3).tolist()==row['counts']
            with torch.inference_mode():
                batch=model.data_preprocessor(pseudo_collate([sample]),training=False);assert hashlib.sha256(batch['inputs'].cpu().numpy().tobytes()).hexdigest()==sigs[row['index']]
                pred=model.predict(batch['inputs'],batch['data_samples'])[0].pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            cm=np.bincount(3*y[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
            assert np.array_equal(cm,np.array([[np.count_nonzero(valid&(y==a)&(pred==b)) for b in range(3)] for a in range(3)]))
            totals[d]+=cm;directory=out/'predictions'/condition;directory.mkdir(parents=True,exist_ok=True);path=directory/f"{row['index']:03d}.npz";np.savez_compressed(path,pred=pred)
            per.append(dict(index=row['index'],group=row['group'],domain=d,prediction_sha=sha(path),**describe(cm)))
            if i%20==0:(out/'EVAL_STATUS.json').write_text(json.dumps(dict(condition=condition,images=i+1,total=567)))
        assert state_sha(model)==before
        (out/(condition+'_PER_IMAGE.json')).write_text(json.dumps(per))
        summaries.append(dict(condition=condition,checkpoint=str(checkpoint),checkpoint_sha=sha(checkpoint),target_label_supervision_percent=0,BN='Native source training BN; no target recalibration',metrics={d:describe(c) for d,c in totals.items()},state_unchanged=True))
        del model;torch.cuda.empty_cache()
    (out/'EVAL_COMPLETE.json').write_text(json.dumps(dict(passed=True,forwards=1134,summary=summaries),indent=2))
if __name__=='__main__':main()
