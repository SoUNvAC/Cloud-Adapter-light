"""CPU-only synthetic interpolation, grid and point mosaic checks; no Omni calls."""
import numpy as np
import ast
import inspect
from pathlib import Path
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation


def main():
    tree=ast.parse(Path(__file__).with_name('run_phase118.py').read_text())
    store=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observed_store')
    signature=inspect.Signature([inspect.Parameter(a.arg,inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in store.args.args])
    signature.bind(pred_batch=None,index_batch=None,pred_tracker=None,gradient=None,grad_tracker=None)
    moved=[(0,500,0,500),(0,500,0,500),(1,501,1,501)]
    assert list(dict.fromkeys(moved))==[moved[0],moved[2]]
    a=(np.arange(64).reshape(8,8)*13+100).astype(np.uint16)
    with MemoryFile() as f:
        with f.open(driver='GTiff',height=8,width=8,count=1,dtype='uint16',transform=from_origin(0,160,20,20),nodata=0) as ds:
            ds.write(a,1);b=ds.read(1,out_shape=(16,16),resampling=Resampling.bilinear)
    y,x=np.mgrid[:16,:16];fy,fx=y/2-.25,x/2-.25;y0=np.floor(fy).astype(int);x0=np.floor(fx).astype(int)
    wy,wx=fy-y0,fx-x0;y1=(y0+1).clip(0,7);x1=(x0+1).clip(0,7);y0=y0.clip(0,7);x0=x0.clip(0,7)
    ref=a[y0,x0]*(1-wy)*(1-wx)+a[y1,x0]*wy*(1-wx)+a[y0,x1]*(1-wy)*wx+a[y1,x1]*wy*wx
    assert np.max(np.abs(b.astype(float)-ref))<=1
    bad=np.zeros((8,8),bool);bad[3,3]=True
    conservative=np.repeat(np.repeat(binary_dilation(bad,structure=np.ones((3,3),bool)),2,0),2,1)
    influence=( (y0==3)&(x0==3) ) | ((y1==3)&(x0==3)) | ((y0==3)&(x1==3)) | ((y1==3)&(x1==3))
    assert np.all(conservative[influence])
    raw=np.arange(2*4*8*8,dtype=np.float32).reshape(2,4,8,8)
    yy=np.array([1,3,5]);xx=np.array([2,4,6]);extracted=raw[0,:,yy,xx].T
    assert extracted.shape==(4,3) and np.array_equal(extracted,np.stack([raw[0,:,y,x] for y,x in zip(yy,xx)],1))
    axis=np.arange(1,9,dtype=np.float32)/8;g=axis[:,None]*axis[None]
    dense=np.zeros((4,8,8),np.float32);dense+=raw[0]*g;dense+=raw[1]*g
    sparse=(raw[0,:,yy,xx].T+raw[1,:,yy,xx].T)*g[yy,xx][None]
    assert np.allclose(sparse,dense[:,yy,xx],atol=1e-5)
    print('Synthetic CPU bilinear, conservative support, grid and sparse-point raw mosaic passed; Omni calls0')


if __name__=='__main__':main()
