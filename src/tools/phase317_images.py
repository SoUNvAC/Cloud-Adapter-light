"""Image-only readers: retain display black; no target reference mask used for valid."""
import numpy as np
from phase317_common import *
def display(row):
 import mmcv
 p=Path(row['image_path']);assert sha(p)==row['image_sha256']
 raw=mmcv.imread(str(p),flag='color',channel_order='bgr');assert raw.dtype==np.uint8 and raw.ndim==3 and raw.shape[2]==3
 resized=mmcv.imresize(raw,(512,512),interpolation='bilinear',backend='cv2')
 a=resized.transpose(2,0,1).astype(np.float32);valid=np.isfinite(a).all(0);assert valid.all()
 return a,valid
def image(pool,row,step=None):
 if pool=='CMC':
  from phase302_images import rgb
  a,v=rgb(row)
  if step is not None:
   t,l=step['top'],step['left'];a=a[:,t:t+512,l:l+512].copy();v=v[t:t+512,l:l+512].copy()
 else:a,v=display(row)
 if step and step['horizontal_flip']:a=a[:,:,::-1].copy();v=v[:,::-1].copy()
 return a,v
def pseudo(pool,row,step=None):
 path=OLD/'teacher'/(row['product']+'.npz') if pool=='CMC' else ROOT/pool/'teacher'/(row['name']+'.npz')
 with np.load(path,allow_pickle=False) as z:p=z['pseudo'].copy()
 if pool=='CMC' and step is not None:p=p[step['top']:step['top']+512,step['left']:step['left']+512].copy()
 if step and step['horizontal_flip']:p=p[:,::-1].copy()
 return p
