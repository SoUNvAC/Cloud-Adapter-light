"""Independent single-point integer-sum CPU path; no main raster operator import."""
import argparse
import os
os.environ.setdefault('GDAL_NUM_THREADS','1');os.environ.setdefault('OPJ_NUM_THREADS','1')
import numpy as np
import rasterio
from rasterio.windows import Window
from phase104_common import *

def main(root, coordinate):
    positions=coords(coordinate);byid={r['point_id']:r for r in positions}
    lock=read(RESULT/'task_lock.json');selected=sample48(positions)
    assert selected==lock['CPU_point_ids']
    original=read(root/'point_inputs_private.json');old={r['point_id']:r for r in original['rows']}
    products={r['product_id']:r for r in read(RESULT/'object_manifest.json')['products']}
    reports=[]
    for pid in selected:
        r=byid[pid];m=products[r['product_id']];g=m['grid20']
        # Re-derive from the original projected coordinate, not stored20m indices.
        x=float(r['native_x_m']);y=float(r['native_y_m'])
        col=int(np.floor((x-g['ULX'])/g['XDIM']));row=int(np.floor((y-g['ULY'])/g['YDIM']))
        assert (row,col)==(int(r['row20']),int(r['col20']))
        y0=row//1020*1020;x0=col//1020*1020
        assert (y0,x0,row-y0,col-x0)==tuple(old[pid]['window'][k] for k in ('y','x','point_y','point_x'))
        valid=True;maxerr=0.
        for b in BANDS:
            scale=2 if b in ('B02','B03','B04','B08') else 1
            with rasterio.open(root/'imagery'/r['product_id']/(b+'.jp2')) as ds:
                a=ds.read(1,window=Window(col*scale,row*scale,scale,scale))
            raw=[int(v) for line in a for v in line];assert len(raw)==scale*scale
            mean=sum(raw)/len(raw)
            native_toa=[float(np.float32(v)/np.float32(10000)) for v in raw]
            toa=np.float32(sum(native_toa)/len(raw))
            was=old[pid]['bands'][b];assert a.tolist()==was['DN'] and mean==was['mean_DN']
            err=abs(float(toa)-was['TOA']);maxerr=max(maxerr,err);assert err==0
            vb=all(v not in (0,65535) for v in raw);assert vb==was['valid'];valid=valid and vb
        assert valid==old[pid]['input_valid']
        reports.append(dict(point_id=pid,product_id=r['product_id'],all_six_original_DN_exact=True,
            arithmetic_TOA_max_abs_error=maxerr,grid_index_exact=True,window_exact=True,input_valid=valid))
        print('Independent CPU point',len(reports),48,flush=True)
    write(root/'CPU_48_verification.json',dict(status='passed',selection_uses_labels_or_image_results=False,
          selection_rule='SHA256(PHASE104_INPUT|point_id) first48 of9500',rows=reports,
          path='independent per-point native raster read, integer DN sum/4; native float32 TOA then Python double mean; original coordinates',
          point_inputs_sha256=sha(root/'point_inputs_private.json'),models_loaded=0,forwards=0),True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--coordinates',type=Path,required=True)
    a=ap.parse_args()
    with rasterio.Env(GDAL_CACHEMAX=64*1024*1024):main(a.root,a.coordinates)
