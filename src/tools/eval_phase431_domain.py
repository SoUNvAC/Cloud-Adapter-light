"""Independent original Phase431 checkpoint / original CloudSEN535 evaluation."""
from pathlib import Path
import os,sys,json,hashlib,time
import numpy as np
import torch
P=Path('/home/scv/shared/phase431_20261102');SRC=Path('/home/scv/Cloud-Adapter-light/src');O=P/'cloudsen_original_recheck_20261106'
sys.path[:0]=[str(P/'code'),'/home/scv/shared/phase400_20261031/code'];os.chdir(SRC)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(m):
    h=hashlib.sha256()
    for n,t in m.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    from report_phase400 import describe
    register_all_modules();import phase431.integration
    O.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42901);torch.backends.cudnn.benchmark=False
    cfg=Config.fromfile(str(P/'code/phase431/config.py'));cfg.model.backbone.variant='fused';cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
    checkpoint=P/'fused/best_mIoU_iter_32000.pth';model=MODELS.build(cfg.model);model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True);model.cuda().eval().requires_grad_(False);before=digest(model)
    rows=[r for r in json.loads(Path('/home/scv/shared/phase442_20261103/INPUTS.json').read_text()) if r['domain']=='A'];assert len(rows)==535
    sigs={v['index']:v['input_tensor_sha'] for v in json.loads(Path('/home/scv/shared/phase442_20261103/official/SIGNATURES.json').read_text())}
    pipe=Compose([dict(type='LoadImageFromFile'),dict(type='Resize',scale=(512,512)),dict(type='LoadAnnotations'),dict(type='PackSegInputs')]);cm=np.zeros((3,3),np.int64);per=[];start=time.monotonic()
    for i,row in enumerate(rows):
        assert sha(row['image'])==row['image_sha'] and sha(row['mask'])==row['mask_sha']
        sample=pipe(dict(img_path=row['image'],seg_map_path=row['mask'],seg_fields=[],label_map={0:0,1:1,2:1,3:2},reduce_zero_label=False));y=sample['data_samples'].gt_sem_seg.data[0].numpy();valid=y!=255
        with torch.inference_mode():
            batch=model.data_preprocessor(pseudo_collate([sample]),training=False);assert hashlib.sha256(batch['inputs'].cpu().numpy().tobytes()).hexdigest()==sigs[row['index']]
            pred=model.predict(batch['inputs'],batch['data_samples'])[0].pred_sem_seg.data[0].cpu().numpy()
        assert y.shape==pred.shape and np.bincount(y[valid],minlength=3).tolist()==row['counts']
        c=np.bincount(3*y[valid].astype(np.int64)+pred[valid],minlength=9).reshape(3,3)
        assert np.array_equal(c,np.array([[np.count_nonzero(valid&(y==a)&(pred==b)) for b in range(3)] for a in range(3)]));cm+=c;per.append(dict(index=row['index'],group=row['group'],**describe(c)))
        if i%25==0:(O/'STATUS.json').write_text(json.dumps(dict(images=i+1,total=535,seconds=time.monotonic()-start)))
    assert digest(model)==before
    result=dict(passed=True,checkpoint=str(checkpoint),checkpoint_sha=sha(checkpoint),data='Original CloudSEN535 validation, no color mapping',target_label_supervision_percent=0,bn='Native source checkpoint BN; no target fit',forwards=535,training_updates=0,state_unchanged=True,seconds=time.monotonic()-start,**describe(cm))
    previous=json.loads(Path('/home/scv/shared/phase444_retrain_20261103/source431/EVAL_COMPLETE.json').read_text())['summary'][0]['metrics']['A'];result['previous_mIoU']=previous['mIoU'];result['delta_previous']=result['mIoU']-previous['mIoU'];assert abs(result['delta_previous'])<.002
    (O/'RESULT.json').write_text(json.dumps(result,indent=2));(O/'PER_IMAGE.json').write_text(json.dumps(per));print(json.dumps(result))
if __name__=='__main__':main()
