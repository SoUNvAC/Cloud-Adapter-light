import sys,json,hashlib,os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
from mmseg.registry import METRICS
R=Path('/home/scv/shared/phase500_20261111')
real=Path('/mnt/d/CloudAdapterExperiments/phase500_20261111')
assert not R.exists() and not real.exists()
real.mkdir();R.symlink_to(real,target_is_directory=True)
(R/'predictions496').mkdir()
sys.path[:0]=['/home/scv/shared/phase496_20261110/code','/home/scv/Cloud-Adapter-light/src']
from phase496.integration import ExactIoU496
register_all_modules();torch.set_num_threads(4)
rows=[];cm=np.zeros((3,3),dtype=np.int64)
@METRICS.register_module()
class Boundary500(ExactIoU496):
 def process(self,data_batch,data_samples):
  super().process(data_batch,data_samples)
  for s in data_samples:
   y=s['gt_sem_seg']['data'].squeeze().cpu().numpy();p=s['pred_sem_seg']['data'].squeeze().cpu().numpy()
   valid=y!=255;sh=(y==2)&valid;non=(y!=2)&valid
   cm[:]+=np.bincount((y[valid]*3+p[valid]).astype(np.int64),minlength=9).reshape(3,3)
   def distance(target):
    if not target.any():return np.full(y.shape,np.inf)
    return ndi.distance_transform_edt(np.pad(~target,1,constant_values=True))[1:-1,1:-1]
   inside=distance(non);outside=distance(sh);unknown=distance(~valid)
   fn=sh&(p!=2);fp=non&(p==2)
   row=dict(image=Path(s['img_path']).stem,FN=int(fn.sum()),FP=int(fp.sum()),shadow_pixels=int(sh.sum()),bands={},safe_bands={},components={})
   for name,low,high in [('within3',0,3),('3to8',3,8),('beyond8',8,np.inf)]:
    ib=(inside>low)&(inside<=high);ob=(outside>low)&(outside<=high)
    if name=='beyond8':ob=outside>8;ib=inside>8
    for key,safe in [('bands',valid),('safe_bands',valid&(unknown>8))]:
     row[key][name]=dict(FN=int((fn&ib&safe).sum()),FP=int((fp&ob&safe).sum()),shadow_pixels=int((sh&ib&safe).sum()),nonshadow_pixels=int((non&ob&safe).sum()),FN_to_surface=int((fn&ib&safe&(p==0)).sum()),FN_to_cloud=int((fn&ib&safe&(p==1)).sum()),FP_from_surface=int((fp&ob&safe&(y==0)).sum()),FP_from_cloud=int((fp&ob&safe&(y==1)).sum()))
   cc,n=ndi.label(sh,structure=np.ones((3,3)));areas=np.bincount(cc.ravel());sizes=areas[cc]
   for name,lo,hi in [('1to64',1,64),('65to256',65,256),('257to1024',257,1024),('beyond1024',1025,np.inf)]:
    mask=sh&(sizes>=lo)&(sizes<=hi)
    row['components'][name]=dict(shadow_pixels=int(mask.sum()),FN=int((fn&mask).sum()))
   rows.append(row)
   np.savez_compressed(R/'predictions496'/(row['image']+'.npz'),prediction=p.astype(np.uint8),label=y.astype(np.uint8))
cfg=Config.fromfile('/home/scv/shared/phase496_20261110/code/phase496/config.py')
cfg.model.backbone.update(variant='texture',repo='/home/scv/shared/phase429_20261102/dinov3',weights='/home/scv/shared/phase429_20261102/dinov3_splus_native.pth')
cfg.work_dir=str(R/'diagnosis496');cfg.load_from='/home/scv/shared/phase496_20261110/texture/iter_40000.pth'
cfg.test_dataloader=cfg.val_dataloader;cfg.test_dataloader.dataset.data_root='/home/scv/Cloud-Adapter-light/src/data/cloudsen12_high_l1c'
cfg.test_evaluator=dict(type='Boundary500',iou_metrics=['mIoU']);cfg.custom_hooks=[]
runner=Runner.from_cfg(cfg);assert len(runner.test_dataloader.dataset)==535
metrics=runner.test();assert len(rows)==535
out=dict(model='Phase496 texture final40K',target_supervision_percent=0,images=len(rows),metrics=metrics,confusion=cm.tolist(),FN=sum(r['FN'] for r in rows),FP=sum(r['FP'] for r in rows),shadow_pixels=sum(r['shadow_pixels'] for r in rows))
for key in ['bands','safe_bands','components']:
 out[key]={band:{k:sum(r[key][band][k] for r in rows) for k in rows[0][key][band]} for band in rows[0][key]}
out['FN_within8_fraction']=(out['bands']['within3']['FN']+out['bands']['3to8']['FN'])/out['FN']
out['shadow_within8_fraction']=(out['bands']['within3']['shadow_pixels']+out['bands']['3to8']['shadow_pixels'])/out['shadow_pixels']
reference=json.loads(Path('/home/scv/shared/phase496_20261110/RESULT.json').read_text())
ref=next(r['metrics'] for r in reference if r['variant']=='texture' and r['checkpoint']=='final')
assert all(abs(metrics[k]-ref[k])<1e-6 for k in ['IoU_surface','IoU_cloud','IoU_shadow','mIoU_exact'])
out['endpoint_reproduced']=True
out['distance_definition']='Euclidean distance to nearest opposite valid GT class pixel center; no image-edge artificial boundary; safe bands exclude within 8 pixels of ignore labels'
out['checkpoint_sha256']=hashlib.sha256(Path(cfg.load_from).read_bytes()).hexdigest()
(R/'DIAGNOSIS496.json').write_text(json.dumps(out,indent=2))
(R/'DIAGNOSIS496_SCENES.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(out,indent=2))
