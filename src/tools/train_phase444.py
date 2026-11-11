"""Fresh40k source-only training in the common appearance space."""
from pathlib import Path
import sys,os,json,hashlib,time
import torch
R=Path('/home/scv/shared/phase444_retrain_20261103');SRC=Path('/home/scv/Cloud-Adapter-light/src')
ARM=sys.argv[1];assert ARM in ['baseline429','source431']
phase='429' if ARM=='baseline429' else '431';sys.path.insert(0,str(R/'code'));os.chdir(SRC)
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
import alignment444
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);register_all_modules();out=R/ARM;out.mkdir(exist_ok=False)
    cfg=Config.fromfile(str(R/'code'/('phase'+phase)/'config.py'));cfg.model.backbone.variant='baseline' if phase=='429' else 'fused'
    cfg.model.backbone.repo='/home/scv/shared/phase429_20261102/dinov3';cfg.model.backbone.weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth'
    cfg.work_dir=str(out);cfg.resume=False;cfg.load_from=None
    cfg.custom_imports.imports.append('alignment444')
    for name in ['train_dataloader','val_dataloader','test_dataloader']:
        ds=cfg[name].dataset;ds.data_root=str(SRC/'data/cloudsen12_high_l1c')
        assert ds.pipeline[-1]['type']=='PackSegInputs'
        ds.pipeline.insert(len(ds.pipeline)-1,dict(type='AlignRGB444',domain='A'))
        cfg[name].num_workers=4
    assert cfg.train_cfg.max_iters==40000 and cfg.train_dataloader.batch_size==2 and cfg.val_dataloader.batch_size==2
    lock=dict(protocol='Fresh40k source training on aligned inputs; no target labels or target BN fitting',target_label_supervision_percent=0,load_from=None,resume=False,
        weights_sha=sha(cfg.model.backbone.weights),mapping_sha=sha('/home/scv/shared/phase443_20261103/ALIGNMENT.npz'),mapping_fit='Existing Phase443 unlabeled source validation and target development image statistics; labels never used for alignment',
        code_sha={str(p.relative_to(R)):sha(p) for p in list((R/'code'/('phase'+phase)).glob('*.py'))+[R/'alignment444.py',R/'train_phase444.py']},
        augmentation='Original source crop/flip/photometric augmentation then frozen A map, before ImageNet normalization; float32 map output without extra clip',selection='Best aligned source validation mIoU only')
    (out/'INPUT_LOCK.json').write_text(json.dumps(lock,indent=2));cfg.dump(str(out/'CONFIG_LOCK.py'))
    runner=Runner.from_cfg(cfg);start=time.monotonic();runner.train();assert runner.iter==40000
    (out/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(passed=True,steps=runner.iter,seconds=time.monotonic()-start,target_label_supervision_percent=0)))
if __name__=='__main__':main()
