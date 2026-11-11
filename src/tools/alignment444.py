"""Frozen Phase443 RGB map applied after original photometric augmentation."""
from pathlib import Path
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class AlignRGB444(BaseTransform):
    def __init__(self,domain='A',mapping='/home/scv/shared/phase443_20261103/ALIGNMENT.npz'):
        assert domain in ['A','B'];self.domain=domain;self.maps=dict(np.load(mapping))
    def transform(self,results):
        rgb=results['img'][...,::-1].astype(np.float32);out=np.empty_like(rgb)
        for c in range(3):out[...,c]=np.interp(rgb[...,c],self.maps[self.domain+str(c)+'_x'],self.maps[self.domain+str(c)+'_y'])
        results['img']=out[...,::-1].copy();return results
