"""Metadata and coordinate identity only. Never calculate truth-based selector quality."""
from phase319_common import *
import re,zipfile,math
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from affine import Affine
from rasterio.warp import transform
from phase319_geodesy import inverse,forward,qa as geodesy_qa
def project(src,dst,x,y):
 xx,yy=transform(src,dst,[x],[y]);return xx[0],yy[0]
def mtl(path):
 values={}
 for line in Path(path).read_text().splitlines():
  m=re.match(r'^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$',line)
  if m:values[m[1]]=m[2].strip('"')
 return values
def matrix(crs,affine,c,r):
 x,y=affine*(c,r);lon,lat=project(crs,'EPSG:4326',x,y);columns=[]
 for dc,dr in [(1,0),(0,1)]:
  xx,yy=affine*(c+dc,r+dr);lo,la=project(crs,'EPSG:4326',xx,yy);a,d=inverse(lon,lat,lo,la);columns.append([d*math.sin(math.radians(a)),d*math.cos(math.radians(a))])
 return np.array(columns).T,(lon,lat)
def gridcheck(crs,af):
 M,center=matrix(crs,af,256,256);errs=[];scales=[]
 for c in [0,256,512]:
  for r in [0,256,512]:
   N,_=matrix(crs,af,c,r)
   for j in range(2):
    a=np.degrees(np.arctan2(M[0,j],M[1,j]));b=np.degrees(np.arctan2(N[0,j],N[1,j]));errs.append(abs((a-b+180)%360-180));scales.append(abs(np.linalg.norm(N[:,j])/np.linalg.norm(M[:,j])-1))
 checks=[]
 for a in [0,90,180,270]:
  lo,la=forward(*center,a,1000);xx,yy=project('EPSG:4326',crs,lo,la);cc,rr=(~af)*(xx,yy);v=M@np.array([cc-256,rr-256]);aa=np.degrees(np.arctan2(v[0],v[1]));checks.append(dict(azimuth=a,angle_error_deg=abs((aa-a+180)%360-180),distance_error_m=abs(np.linalg.norm(v)-1000)))
 assert all(c['angle_error_deg']<.1 and c['distance_error_m']<1 for c in checks),'independent geographic forward/inverse grid check failed'
 return dict(matrix_east_north_per_output_pixel=M.tolist(),center_lonlat=center,max_angle_deviation_deg=max(errs),max_relative_scale_deviation=max(scales),independent_forward_inverse=checks,qualified=max(errs)<=.1 and max(scales)<=.001)
def main():
 guard();pre=load(ROOT/'PREREGISTRATION.json');assert not (ROOT/'GEOMETRY_INPUTS.json').exists()
 write(ROOT/'GEODESY_QA.json',geodesy_qa());pins={};headers=[];rows=[];archives=[];rawpins={}
 def pp(p,k):
  p=Path(p)
  if str(p) not in pins:pins[str(p)]=pin(p,k)
  return pins[str(p)]['sha256']
 # Inspect the actual archive member inventory, not an inferred Landsat convention.
 zp=Path('/home/scv/shared/data/sparcs/l8cloudmasks.zip');zh=pp(zp,'SPARCS original archive SHA and member index only')
 assert zh=='5cde604615ee241950b5a0b641ae5de82b27738b51e39aa724d18faec0bbaa9a'
 z=zipfile.ZipFile(zp);members=z.namelist();write(ROOT/'SPARCS_ARCHIVE_METADATA.json',dict(path=str(zp),sha256=zh,members=members,metadata_members=[n for n in members if n.lower().endswith(('.txt','.xml','.json'))]))
 axes=load(PRIOR/'REFERENCE_LOCK.json')['L8_axes'];assert axes in ['x_col_y_row','x_row_y_col'];lut=np.zeros(256,np.uint8);lut[64]=1;lut[192]=2;lut[255]=3
 for pool in POOLS:
  for row in inputs(pool):
   guard();n=row['name'];rec=dict(pool=pool,name=n,scene=row['scene'],biome=row.get('biome','NA'),status='unqualified',missing=[],image_path=row['image_path'],image_sha256=pp(row['image_path'],'fit image byte identity; no quality'),mask_path=row['mask_path'],mask_sha256=pp(row['mask_path'],'coordinate identity/header only'))
   assert rec['image_sha256']==row['image_sha256']
   if pool=='L8':
    d=Path('/home/scv/shared/data/l8_biome_raw/l8biome')/row['biome']/row['scene'];t=d/(row['scene']+'.TIF');m=d/(row['scene']+'_MTL.txt');rawmask=d/(row['scene']+'_fixedmask.TIF')
   else:t=Path(row['multispectral_path']);d=t.parent;m=d/(row['scene']+'_mtl.txt');rawmask=None
   rec.update(mtl_path=str(m),native_tiff=str(t))
   if not m.is_file():rec['missing'].append('same-scene MTL absent at exact product directory')
   else:
    rec['mtl_sha256']=pp(m,'actual same-scene sun metadata');v=mtl(m);rec['mtl_identity']=v.get('LANDSAT_SCENE_ID');rec['acquired']=v.get('DATE_ACQUIRED');rec['center_time']=v.get('SCENE_CENTER_TIME');rec['azimuth_deg']=float(v['SUN_AZIMUTH'])%360;rec['elevation_deg']=float(v['SUN_ELEVATION']);rec['sun_units']='degrees; clockwise from geographic north, scene-center'
    assert v['LANDSAT_SCENE_ID']==row['scene'] and 0<rec['elevation_deg']<=90 and math.isfinite(rec['azimuth_deg'])
    from datetime import datetime
    assert datetime.strptime(row['scene'][9:16],'%Y%j').strftime('%Y-%m-%d')==rec['acquired'],'acquisition date not bound to scene ID'
   if not t.is_file():rec['missing'].append('original multispectral TIFF missing')
   else:
    rec['native_tiff_sha256']=pp(t,'original TIFF SHA and header only')
    with rasterio.open(t) as h:
     rec.update(native_shape=list(h.shape),crs=str(h.crs),native_affine=list(h.transform)[:6],tiff_tags=h.tags())
     if not h.crs:rec['missing'].append('TIFF CRS absent')
     if pool=='SPARCS':
      with Image.open(row['image_path']) as im:photo_shape=im.size
      with Image.open(row['mask_path']) as im:mask_shape=im.size
      assert photo_shape==mask_shape==(h.width,h.height)==(1000,1000)
      evidence={}
      for p in [Path(row['image_path']),Path(row['mask_path']),t,m]:
       member='sending/'+p.name;assert member in members
       # Stream member bytes and compare hashes, no extraction and no truth metric.
       import hashlib
       hh=hashlib.sha256()
       with z.open(member) as f:
        for b in iter(lambda:f.read(1048576),b''):hh.update(b)
       assert hh.hexdigest()==pp(p,'original same-index SPARCS bundle identity');evidence[member]=hh.hexdigest()
      af=h.transform*Affine.scale(h.width/512,h.height/512);rec.update(index_identity='same released product basename and equal native photo/mask/data extent, exact archive bytes; fixed Resize512',bundle_member_sha256=evidence,output_affine=list(af)[:6],resize_scale=h.width/512)
     else:
      assert rawmask.is_file();rec['raw_mask_path']=str(rawmask);rec['raw_mask_sha256']=pp(rawmask,'coordinate identity only; no selector quality');x,y=int(row['x']),int(row['y']);col,rr=(x,y) if axes=='x_col_y_row' else (y,x)
      with rasterio.open(rawmask) as mh:
       assert mh.shape==h.shape and mh.crs==h.crs and mh.transform==h.transform
       raw=mh.read(1,window=Window(col,rr,512,512),boundless=True,fill_value=0)
      png=np.asarray(Image.open(row['mask_path']));assert np.isin(raw,[0,64,128,192,255]).all();assert np.array_equal(lut[raw],png),'original converter coordinate identity failed; no alternate registration'
      af=h.transform*Affine.translation(col,rr);rec.update(index_identity='exact original raw ordinal mapping reproduces every PNG value; original axis '+axes,axes=axes,col=col,row=rr,output_affine=list(af)[:6],resize_scale=1)
     if h.crs:
      gc=gridcheck(h.crs,af);rec['gridcheck']=gc
      if not gc['qualified']:rec['missing'].append('bounded center-grid approximation exceeds preregistered angle/scale tolerance')
   if not rec['missing']:rec['status']='qualified'
   rows.append(rec);print(pool,n,rec['status'],flush=True)
 z.close();write(ROOT/'GEOMETRY_INPUTS.json',rows);write(ROOT/'INPUT_SHA256.json',pins);write(ROOT/'COORDINATE_ACCEPTANCE.json',dict(status='complete',products=len(rows),L8_original_axes=axes,L8_fit_masks_exactly_reproduced=100,no_truth_quality_computed=True,no_registration_search=True,resources=guard()));write(ROOT/'STATUS.json',dict(status='coordinate_checked',epoch=time.time()))
if __name__=='__main__':main()
