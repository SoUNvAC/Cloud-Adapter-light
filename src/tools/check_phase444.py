"""Verify all fixed evaluation inputs and training transform before40k."""
from pathlib import Path
import sys,os,json,hashlib,importlib.util
import torch,numpy as np
R=Path('/home/scv/shared/phase444_retrain_20261103');SRC=Path('/home/scv/Cloud-Adapter-light/src');sys.path.insert(0,str(R/'code'));os.chdir(SRC)
from mmseg.utils import register_all_modules
from mmseg.registry import MODELS
from mmengine.dataset import Compose,pseudo_collate
from mmengine.config import Config
import alignment444
def main():
    register_all_modules();spec=importlib.util.spec_from_file_location('cat444',SRC/'cloud_adapter/datasets/phase65_catalogue.py');spec.loader.exec_module(importlib.util.module_from_spec(spec))
    cfg=Config.fromfile(str(R/'code/phase429/config.py'));pre=MODELS.build(cfg.model.data_preprocessor).cuda().eval()
    a=Path('/home/scv/shared/phase443_20261103');rows=json.loads((a/'INPUTS.json').read_text());sig={v['index']:v['input_tensor_sha'] for v in json.loads((a/'official/SIGNATURES.json').read_text())}
    for row in rows:
        d=row['domain'];pipe=Compose([dict(type='LoadImageFromFile' if d=='A' else 'LoadPhase65RGB'),dict(type='Resize',scale=(512,512)),dict(type='AlignRGB444',domain=d),dict(type='PackSegInputs')]);sample=pipe(dict(img_path=row['image']))
        with torch.inference_mode():x=pre(pseudo_collate([sample]),training=False)['inputs']
        assert hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()==sig[row['index']]
    trainfiles=sorted((SRC/'data/cloudsen12_high_l1c/img_dir/train').glob('*.png'));assert trainfiles
    for p in trainfiles[:8]:
        info=dict(img_path=str(p),seg_map_path=str(SRC/'data/cloudsen12_high_l1c/ann_dir/train'/p.name),seg_fields=[],label_map={0:0,1:1,2:1,3:2},reduce_zero_label=False)
        before=Compose(cfg.train_pipeline[:-1])(info);labels=before['gt_seg_map'].copy();reference=before['img'].copy();out=alignment444.AlignRGB444().transform(before)
        assert np.array_equal(labels,out['gt_seg_map']) and out['img'].dtype==np.float32 and np.isfinite(out['img']).all()
        maps=dict(np.load(a/'ALIGNMENT.npz'))
        for c in range(3):assert np.array_equal(out['img'][...,2-c],np.interp(reference[...,2-c].astype(np.float32),maps['A'+str(c)+'_x'],maps['A'+str(c)+'_y']).astype(np.float32))
    (R/'PRECHECK.json').write_text(json.dumps(dict(passed=True,all567_aligned_inputs_exact=True,train_augmentation_then_map_8_images=True,labels_unchanged=True,training_images=len(trainfiles),target_label_supervision_percent=0),indent=2))
if __name__=='__main__':main()
