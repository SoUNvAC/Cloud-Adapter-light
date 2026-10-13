"""Raster/grid/point acceptance, no labels, no models, no scores."""
import argparse
import collections
import os
os.environ.setdefault('GDAL_NUM_THREADS','1')
os.environ.setdefault('OPJ_NUM_THREADS','1')
import numpy as np
import rasterio
from rasterio.windows import Window
from phase104_common import *

def main(root, coordinate):
    manifest=read(RESULT/'object_manifest.json');lock=read(RESULT/'task_lock.json')
    assert sha(RESULT/'object_manifest.json')==lock['object_manifest_sha256']
    status=read(root/'download_status.json');assert status['status']=='all_files_verified' and len(status['files'])==96
    rows=coords(coordinate);byproduct=collections.defaultdict(list)
    for r in rows:byproduct[r['product_id']].append(r)
    records={r['point_id']:dict(point_id=r['point_id'],product_id=r['product_id'],group_id=r['group_id'],
              row20=int(r['row20']),col20=int(r['col20']),bands={},invalid_reasons=[]) for r in rows}
    windows=[];grids=[]
    for product in manifest['products']:
        pid=product['product_id'];rr=byproduct[pid];blocks=collections.defaultdict(list)
        for r in rr:
            y,x=int(r['row20']),int(r['col20']); assert 0<=y<5490 and 0<=x<5490
            assert int(r['row10'])//2==y and int(r['col10'])//2==x
            w=window(y,x);blocks[(w['y'],w['x'])].append(r)
            records[r['point_id']]['window']=w
        for (y,x),points in sorted(blocks.items()):
            wh=min(1020,5490-y);ww=min(1020,5490-x)
            windows.append(dict(product_id=pid,group_id=product['group_id'],y=y,x=x,height=wh,width=ww,
                          right_padding=1020-ww,bottom_padding=1020-wh,points=len(points),selection_uses_labels=False))
        for band in BANDS:
            fine=band in BANDS[:4];scale=2 if fine else 1
            path=root/'imagery'/pid/(band+'.jp2');key=pid+'/'+band+'.jp2'
            assert sha(path)==status['files'][key]['sha256'], 'Stored raster SHA mismatch'
            with rasterio.open(path) as ds:
                g=product['grid10' if fine else 'grid20']
                expected=(g['XDIM'],0,g['ULX'],0,g['YDIM'],g['ULY'])
                assert tuple(ds.transform)[:6]==expected,(pid,band,'transform')
                assert ds.crs.to_string()==product['crs'] and ds.count==1
                assert (ds.height,ds.width)==(g['NROWS'],g['NCOLS'])
                assert ds.dtypes==('uint16',),(pid,band,ds.dtypes)
                grids.append(dict(product_id=pid,band=band,crs=ds.crs.to_string(),transform=list(ds.transform)[:6],
                                  height=ds.height,width=ds.width,xml_match=True,sha256=status['files'][key]['sha256']))
                for (y,x),points in sorted(blocks.items()):
                    h=min(1020,5490-y);w=min(1020,5490-x)
                    a=ds.read(1,window=Window(x*scale,y*scale,w*scale,h*scale))
                    assert a.shape==(h*scale,w*scale)
                    for r in points:
                        local_y=(int(r['row20'])-y)*scale;local_x=(int(r['col20'])-x)*scale
                        dn=a[local_y:local_y+scale,local_x:local_x+scale]
                        assert dn.shape==(scale,scale)
                        finite=bool(np.isfinite(dn).all());nodata=bool(np.any(dn==0));sat=bool(np.any(dn==65535))
                        mean=np.mean(dn.astype(np.float32),dtype=np.float32)
                        # Exact Phase89 order: native float32 DN->TOA, float64 area mean, cast float32.
                        toa=np.float32(np.mean(dn.astype(np.float32)/np.float32(product['quantification']),dtype=np.float64))
                        v=finite and not nodata and not sat and bool(np.isfinite(toa))
                        point=records[r['point_id']]
                        point['bands'][band]=dict(DN=dn.tolist(),mean_DN=float(mean),TOA=float(toa),valid=v)
                        if not finite or not np.isfinite(toa):point['invalid_reasons'].append(band+':nonfinite')
                        if nodata:point['invalid_reasons'].append(band+':NODATA')
                        if sat:point['invalid_reasons'].append(band+':SATURATED')
            print('Grid/points',pid,band,'accepted',flush=True)
    validity=[]
    for r in rows:
        p=records[r['point_id']];assert set(p['bands'])==set(BANDS)
        p['input_valid']=not p['invalid_reasons']
        p['RGB_TOA_times255']=[float(np.float32(p['bands'][b]['TOA'])*np.float32(255)) for b in ('B04','B03','B02')]
        validity.append(dict(point_id=p['point_id'],product_id=p['product_id'],group_id=p['group_id'],
              row20=p['row20'],col20=p['col20'],input_valid=p['input_valid'],
              invalid_reasons=json.dumps(p['invalid_reasons'],separators=(',',':')),
              window_y=p['window']['y'],window_x=p['window']['x']))
    write(root/'point_inputs_private.json',dict(rows=list(records.values()),no_point_labels=True),True)
    csvwrite(root/'point_validity_no_labels.csv',validity)
    write(root/'window_index.json',dict(rows=windows,selection='all fixed9500 points; no labels',
          dense_window_cache_created=False,lazy_sources_verified=True,size20=1020,stride20=1020),True)
    write(root/'grid_acceptance.json',dict(rows=grids,products=16,rasters=96,all_grids_match_original_XML=True),True)
    write(root/'input_acceptance.json',dict(status='all_fixed_points_processed',point_count=9500,
          input_valid=sum(r['input_valid'] for r in records.values()),input_invalid=sum(not r['input_valid'] for r in records.values()),
          native10_cell_count=len({(r['product_id'],r['row10'],r['col10']) for r in rows}),
          native20_cell_count=len({(r['product_id'],r['row20'],r['col20']) for r in rows}),
          window_count=len(windows),coordinate_sha256=COORD_SHA,point_inputs_sha256=sha(root/'point_inputs_private.json'),
          point_validity_sha256=sha(root/'point_validity_no_labels.csv'),window_index_sha256=sha(root/'window_index.json'),
          grid_acceptance_sha256=sha(root/'grid_acceptance.json'),labels_accessed=False,models_loaded=0,forwards=0),True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--coordinates',type=Path,required=True)
    a=ap.parse_args()
    with rasterio.Env(GDAL_CACHEMAX=128*1024*1024):main(a.root,a.coordinates)
